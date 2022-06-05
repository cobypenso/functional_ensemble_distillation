'''
This file contains api functions for inference with the Ensemble captured using cSGHMC.

The following functions are provided:
1.csghmc_predict - predict over a dataset
2.csghmc_predict_unsupervised - predict over a dataset without labels (for example mixup)
3.csghmc_predict_with_idxs - predict over a dataset using only a subset of the models in the ensemble (providing idxs of the models)
4.csghmc_predict_single_image - predict over a single image
5.csghmc_predict_single_batch - predict over a single batch
'''

import sys
sys.path.append('..')
import os
import torch
import random
import argparse
import numpy as np
from models import *
import torch.nn as nn
from data import get_cifar10
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from timeit import default_timer as timer


device_id = 0
seed = 1
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
             right += 1.0
    return right/len(truth)

def test_unsupervised(net, testloader):
    net.eval()

    pred_list = []
    logits = []
    with torch.no_grad():
        for _, images in enumerate(testloader): 
            if len(images) == 2:
                images = images[0]
            if use_cuda:
                images = images.cuda(device_id)
            outputs = net(images.to(dtype=torch.float)) 
            outputs = outputs.cpu()
            logits.append(outputs)
            pred_list.append(F.softmax(outputs,dim=1))

            _, predicted = torch.max(outputs.data, 1)
    
    pred_list = torch.cat(pred_list,0)
    logits = torch.cat(logits, 0)
    return pred_list, logits

def test(net, testloader, criterion, verbose = True):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_list = []
    logits = []
    truth_res = []
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            truth_res += list(targets.data)
            outputs = net(inputs)
            logits.append(outputs.cpu())
            pred_list.append(F.softmax(outputs,dim=1).cpu())
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            inputs, targets = inputs.cpu(), targets.cpu() 
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/len(testloader), correct, total,
        100. * correct.item() / total))
    pred_list = torch.cat(pred_list,0)
    logits = torch.cat(logits, 0)
    return pred_list,truth_res, logits

def csghmc_predict(net, testset, checkpoints_dir, ensemble_size = 120, cuda_use=None, batch_size = 1024, verbose = True):
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    if use_cuda or cuda_use:
        net.cuda(device_id)
        cudnn.benchmark = True
        cudnn.deterministic = True

    criterion = nn.CrossEntropyLoss()

    pred_list = []
    logits_list = []
    labels_list = []
    num_model = ensemble_size
    true_num_model = 0
    for m in range(num_model):
        if not os.path.isfile(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)):
            continue
        true_num_model += 1
        net.load_state_dict(torch.load(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)))
        pred, truth_res, logits = test(net, testloader, criterion, verbose)
        labels_list.append(torch.argmax(pred, dim = -1))
        pred_list.append(pred)
        logits_list.append(logits)
    
    logits = sum(logits_list) / true_num_model
    fake = sum(pred_list) / true_num_model
    values, pred_label = torch.max(fake,dim = 1)
    pred_res = list(pred_label.data)
    logits = logits.data
    acc = get_accuracy(truth_res, pred_res)
    if verbose:
        print('Ensemble accuracy: ' +str(acc))

    return {"full_predictions":pred_list,
            "full_logits": logits_list,
            "full_labels": labels_list,
            "ensemble":pred_res,
            "ensemble_logits": logits,
            "ensemble_softmax":fake}
            

def csghmc_predict_with_idxs(net, example, checkpoints_dir, model_idxs, cuda_use=None, verbose = True):
    testloader = torch.utils.data.DataLoader([example], batch_size=1, shuffle=False)
    if use_cuda or cuda_use:
        net.cuda(device_id)
        cudnn.benchmark = True
        cudnn.deterministic = True

    criterion = nn.CrossEntropyLoss()

    pred_list = []
    logits_list = []
    labels_list = []
    true_num_model = 0
    for m in model_idxs:
        if not os.path.isfile(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)):
            continue
        true_num_model += 1
        net.load_state_dict(torch.load(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)))
        pred, truth_res, logits = test(net, testloader, criterion, verbose)
        labels_list.append(torch.argmax(pred, dim = -1))
        pred_list.append(pred)
        logits_list.append(logits)
    
    logits = sum(logits_list) / true_num_model
    fake = sum(pred_list) / true_num_model
    values, pred_label = torch.max(fake,dim = 1)
    pred_res = list(pred_label.data)
    logits = logits.data
    acc = get_accuracy(truth_res, pred_res)
    if verbose:
        print('Ensemble accuracy: ' +str(acc))

    return {"full_predictions":pred_list,
            "full_logits": logits_list,
            "full_labels": labels_list,
            "ensemble":pred_res,
            "ensemble_logits": logits,
            "ensemble_softmax":fake}


def csghmc_predict_unsupervised(net, images, checkpoints_dir, ensemble_size = 120, cuda_use=None, batch_size = 256):

    testloader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4)

    if use_cuda or cuda_use:
        net.cuda(device_id)
        cudnn.benchmark = True
        cudnn.deterministic = True

    pred_list = []
    logits_list = []
    labels_list = []
    num_model = ensemble_size
    true_num_model = 0
    for m in range(num_model):
        if not os.path.isfile(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)):
            continue
        true_num_model += 1
        net.load_state_dict(torch.load(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)))
        pred, logits = test_unsupervised(net, testloader)
        labels_list.append(torch.argmax(pred, dim = -1))
        pred_list.append(pred)
        logits_list.append(logits)
    
    logits = sum(logits_list)/true_num_model
    fake = sum(pred_list)/true_num_model
    values, pred_label = torch.max(fake,dim = 1)
    pred_res = list(pred_label.data)
    logits = logits.data
    
    return {"full_predictions":pred_list,
            "full_logits": logits_list,
            "full_labels": labels_list,
            "ensemble":pred_res,
            "ensemble_logits": logits,
            "ensemble_softmax":fake}


def predict_with_single_model(net, testloader):
    net.cuda().eval()
    import ipdb; ipdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    pred, truth_res, logits = test(net, testloader, criterion, False)

    net.cpu()

    return {'probs': pred,
            'true_labels': truth_res,
            'logits': logits}

def csghmc_predict_single_image(net, image, checkpoints_dir, ensemble_size = 120, label_size = 10, cuda_use=None):
    if use_cuda or cuda_use:
        net.cuda(device_id)
        cudnn.benchmark = True
        cudnn.deterministic = True


    pred_list = torch.zeros((ensemble_size,1)).cuda()
    prob_list = torch.zeros((ensemble_size, label_size)).cuda()
    softmax = nn.Softmax(dim=0)
    start = timer()
    with torch.no_grad():
        image = image.cuda()
        for m in range(ensemble_size):
            net.load_state_dict(torch.load(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)))
            net.eval()
            outputs = net(image[None,...])
            probs = softmax(outputs)
            # _, pred = torch.argmax(probs) 
            prob_list[m,:] = probs
            # pred_list.append(pred)
    
    prob = sum(prob_list) / ensemble_size
    prediction = torch.argmax(prob)
    end = timer()   

    return prediction, prob, end - start


def csghmc_predict_single_batch(net, batch, checkpoints_dir, ensemble_size = 120, label_size = 10, cuda_use=None):
    if use_cuda or cuda_use:
        net.cuda(device_id)
        cudnn.benchmark = True
        cudnn.deterministic = True


    pred_list = torch.zeros(size = (ensemble_size, len(batch))).cuda()
    prob_list = torch.zeros(size = (ensemble_size, len(batch), label_size)).cuda()
    softmax = nn.Softmax(dim=1)
    start = timer()
    with torch.no_grad():
        for m in range(ensemble_size):
            net.load_state_dict(torch.load(checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)))
            net.eval()
            outputs = net(batch)
            probs = softmax(outputs)
            # _, pred = torch.max(outputs.data, 1)
            prob_list[m] = probs
            # pred_list.append(pred)
    
    prob = torch.sum(prob_list, dim = 0) / ensemble_size
    prediction = torch.argmax(prob, dim = -1)
    end = timer()   

    return prediction, prob, end - start