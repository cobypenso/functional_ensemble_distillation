'''
This file contains api functions for inference using the generator.

Two functions available:
1. gen_predict - inference over a dataset.
2. gen_predict_single_image - inference over a single image.
'''

import sys
sys.path.append('../')
import torch
from data import *
from training_helper import *
from models import *
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from classification_generators.generators import *


def gen_predict(generator, dataset, device, batch_size, ensemble_size, latent_std, return_all_preds = False):
    '''
        Generator predict function.
        
        @param generator - trained generator for prediction.
        @param dataset - dataset to predict
        @param device, batch_size, ensemble_size, latent_std 
        @param return_all_preds - If True, return all prediction vectors (every z for every input)

        ens_preds, ens_softmax, all_softmax
        @returns ens_preds - argmax prediction
        @returns ens_softmax - final probability vector.
        @returns all_softmax- probability vectors of all z's (only if return_all_preds = True)

    '''
    ens_preds = []
    ens_softmax = []
    all_softmax = []
    generator.eval()

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    softmax = nn.Softmax(dim=2)
    for _, sample in enumerate(dataloader):
        X = sample[0]
        batch_size = len(X)
        #------- [B,3,32,32] --------#
        image = X.to(device, non_blocking = True)
        #------- [B,M,3,32,32] ------#
        image = image.repeat_interleave(ensemble_size).view(X.shape[0], *X.shape[1:], ensemble_size).permute(0,4,1,2,3)
         #----- z:[B,M,3,32,32] ------- image:[B,M,3,32,32] ------#
        image = image.reshape(-1, *image.shape[2:])
        #----- z:[B*M,3,32,32] ------- image:[B*M,3,32,32] ------#
        z = Variable(torch.normal(mean = torch.zeros(*image.shape), std = latent_std * torch.ones(*image.shape)))
        z = z.to(device)
        with torch.no_grad():
            gen_classes_vector = generator(z.float(), image.float()).squeeze()

        gen_classes_vector = gen_classes_vector.view(batch_size, ensemble_size, -1)
        gen_preds = softmax(gen_classes_vector)
        ensemble_simplex = torch.sum(gen_preds, dim = 1)/ensemble_size
        preds = torch.argmax(ensemble_simplex, dim = 1)

        # append results.
        for i in range(len(ensemble_simplex)):
            ens_softmax.append(ensemble_simplex[i])
            ens_preds.append(preds[i])    
            all_softmax.append(gen_preds[i])
    
    if return_all_preds:
            return ens_preds, ens_softmax, all_softmax
    return ens_preds, ens_softmax


def gen_predict_single_image(generator, image, device, ensemble_size, latent_std):
    '''
        Generator predict on a single input function.
        
        @param generator - trained generator for prediction.
        @param image - image to predict.
        @param device, ensemble_size, latent_std 

        @returns gen_preds - probability vectors of all z's
        @returns ensemble_simplex - final probability vector.
        @returns preds - argmax prediction
        @returns (end - start) - time took for prediction.
    '''
    generator.eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        start = timer()
        image = image.to(device, non_blocking = True)
        #------- [B,M,3,32,32] ------#
        image = image.repeat_interleave(ensemble_size).view(*image.shape, ensemble_size).permute(3,0,1,2)

        #----- z:[B*M,3,32,32] ------- image:[B*M,3,32,32] ------#
        z = Variable(torch.normal(mean = torch.zeros(*image.shape, device = device), std = latent_std * torch.ones(*image.shape, device = device)))

        gen_classes_vector = generator(z.float(), image.float())

        gen_preds = softmax(gen_classes_vector)
        ensemble_simplex = torch.sum(gen_preds, dim = 0)/ensemble_size
        preds = torch.argmax(ensemble_simplex, dim = 0)
        end = timer()

    return gen_preds, ensemble_simplex, preds, end - start