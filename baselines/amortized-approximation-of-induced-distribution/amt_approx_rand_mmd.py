import sys

sys.path.append('../')
sys.path.append('../../')
from models import get_net
from data import *  
from utils import *


import torch
# from utils.vgg import *
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
from torchvision.datasets import CIFAR10, LSUN
import numpy as np
import matplotlib.pyplot as plt

num_samples = 120 # ensemble size

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample[0].to(device)
            target = sample[2].to(device) # ??

            # data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output,dim=1), target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def polynomial(x, y, B, c, d):
    # calculate mmd with polynomial kernel in batch

    xx, yy, zz = torch.bmm(x, x.permute(0,2,1)), \
                 torch.bmm(y, y.permute(0,2,1)), \
                 torch.bmm(x, y.permute(0,2,1))

    K = torch.pow((xx + c), d)
    L = torch.pow((yy + c), d)
    P = torch.pow((zz + c), d)

    beta = (1. / (B * (B - 1)))
    gamma = (2. / (B * B))

    return beta * (torch.sum(K, [1,2]) + torch.sum(L, [1,2])) - gamma * torch.sum(P, [1,2])

def batch_mmd(x, y, B, alpha):
    # calculate mmd with RBF kernel in batch
    xx, yy, zz = torch.bmm(x, x.permute(0,2,1)), \
                 torch.bmm(y, y.permute(0,2,1)), \
                 torch.bmm(x, y.permute(0,2,1))

    rx = (xx.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(xx))
    ry = (yy.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(yy))

    K = torch.exp(- alpha * (rx.permute(0,2,1) + rx - 2 * xx))
    L = torch.exp(- alpha * (ry.permute(0,2,1) + ry - 2 * yy))
    P = torch.exp(- alpha * (rx.permute(0,2,1) + ry - 2 * zz))

    beta = (1. / (B * (B - 1)))
    gamma = (2. / (B * B))

    return beta * (torch.sum(K, [1,2]) + torch.sum(L, [1,2])) - gamma * torch.sum(P, [1,2])

def train_approx(args, fmodel, gmodel, device, approx_loader, f_optimizer, g_optimizer, epoch):
    gmodel.train()
    fmodel.train()
    for batch_idx, sample in enumerate(approx_loader):
        
        data = sample[0].to(device)
        output_samples = sample[1].to(device) # logits of the teacher ensemble
        target = sample[2].to(device) # ??

        f_optimizer.zero_grad()

        with torch.no_grad():
            # To be consistant with KL, the exp() function is changed to softplus,
            # i.e., alpha0 = softplus(g).
            # Note that, for mmd, the exp() function can be used directly for faster convergence,
            # without tuning hyper-parameters.
            g_out = F.softplus(gmodel(data))
            output = F.softmax(output_samples, dim=2).clamp(0.0001, 0.9999)

        f_out = F.softmax(fmodel(data), dim=1)

        pi = f_out.mul(g_out)

        s1 = torch.distributions.Dirichlet(pi, validate_args=False).rsample((num_samples,)).permute(1,0,2)


        loss = (batch_mmd(output, s1, num_samples, args.alpha) #1e5
                + 1e-1 * polynomial(output, s1, num_samples, 1, 3)
                + 1e-2 * polynomial(output, s1, num_samples, 1, 4)
                ).mean()

        loss.backward()
        f_optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(
                epoch, loss.item()))

        g_optimizer.zero_grad()

        g_out = F.softplus(gmodel(data))

        with torch.no_grad():
            output = F.softmax(output_samples, dim=2).clamp(0.0001, 0.9999)

        with torch.no_grad():
            f_out = F.softmax(fmodel(data), dim=1)

        pi = f_out.mul(g_out)
        s1 = torch.distributions.Dirichlet(pi, validate_args=False).rsample((num_samples,)).permute(1,0,2)

        loss = (batch_mmd(output, s1, num_samples, args.alpha)
                + 1e-1 * polynomial(output, s1, num_samples, 1, 3)
                + 1e-2 * polynomial(output, s1, num_samples, 1, 4)
                ).mean()

        loss.backward()
        g_optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(
                epoch, loss.item()))

def eval_approx_on_ood(args, smean, sconc, device, test_loader):
    smean.eval()
    sconc.eval()

    batch_idx = 0
    all_preds = []
    all_images = []
    all_labels = []
    with torch.no_grad():
        # for data, target in test_loader:
        for sample in test_loader:
            data = sample[0].to(device)
            target = sample[1].to(device)    
            # data, target = data.to(device), target.to(device)
            # data = data.view(data.shape[0], -1)

            g_out = F.softplus(sconc(data))
            f_out = F.softmax(smean(data), dim=1) 
            pi_q = f_out.mul(g_out)


            # THE PREDICTIONS :
            approx_result = torch.argmax(pi_q, dim=1)
            # Generate samples from the dirichlet distribution
            approx_samples = torch.distributions.Dirichlet(pi_q,validate_args=False).rsample((num_samples,))
            approx_samples = approx_samples.permute(1,0,2)

            for i in range(len(approx_samples)):
                all_preds.append(approx_samples[i])
                all_images.append(data[i].cpu())
                all_labels.append(target[i].cpu())

            batch_idx += 1

    return all_preds, all_images, all_labels

def eval_approx(args,  smean, sconc, device, test_loader):
    smean.eval()
    sconc.eval()
    miscls_origin = []
    miscls_approx = []
    entros_origin_1 = []
    fentros_approx_1 = []
    entros_approx_1 = []
    maxp_origin_1 = []
    maxp_approx_1 = []
    gvalue_approx_1 = []
    mi_approx_1 = []

    batch_idx = 0
    all_preds = []
    all_images = []
    all_labels = []
    with torch.no_grad():
        # for data, target in test_loader:
        for sample in test_loader:
            data = sample[0].to(device)
            teacher_test_samples = sample[1].to(device)
            target = sample[2].to(device)    
            # data, target = data.to(device), target.to(device)
            # data = data.view(data.shape[0], -1)

            g_out = F.softplus(sconc(data))
            f_out = F.softmax(smean(data), dim=1) 
            pi_q = f_out.mul(g_out)

            samples_p_pi = F.softmax(teacher_test_samples, dim=2).to(device)
            avg_origin_output = torch.mean(samples_p_pi, dim=1)

            pi_p_avg_batch = avg_origin_output
            # THE PREDICTIONS :
            origin_result = torch.argmax(pi_p_avg_batch, dim=1)
            approx_result = torch.argmax(pi_q, dim=1)
            # Generate samples from the dirichlet distribution
            approx_samples = torch.distributions.Dirichlet(pi_q, validate_args=False).rsample((num_samples,))
            approx_samples = approx_samples.permute(1,0,2)

            for i in range(len(approx_samples)):
                all_preds.append(approx_samples[i])
                all_images.append(data[i].cpu())
                all_labels.append(target[i].cpu())


            miscls_approx.append((1- (approx_result == target).float()).cpu().numpy())
            miscls_origin.append((1- (origin_result == target).float()).cpu().numpy())

            # entro_origin = (-torch.bmm(pi_p_avg_batch.view(data.shape[0], 1, -1),
            #                           torch.log(pi_p_avg_batch.view(data.shape[0], -1, 1)))).view(-1)

            # fentro_approx = (-torch.bmm(f_out.view(data.shape[0], 1, -1),
            #                           torch.log(f_out.view(data.shape[0], -1, 1)))).view(-1)

            # alpha = pi_q
            # alpha0 = alpha.sum(1)

            # entro_approx = torch.lgamma(alpha).sum(1) \
            #                - torch.lgamma(alpha0) \
            #                + (alpha0 - 10).mul(torch.digamma(alpha0)) \
            #                - ((alpha - 1 ).mul(torch.digamma(alpha))).sum(1)

            # mi_approx = torch.sum((alpha/alpha0.unsqueeze(1))*(torch.log(alpha/alpha0.unsqueeze(1))
            #                                          -torch.digamma(alpha+1.)+torch.digamma(alpha0.unsqueeze(1)+1.)), dim=1)
            # mi_approx_1.append(1./mi_approx.cpu().numpy())
            # entros_origin_1.append(entro_origin.cpu().numpy())
            # fentros_approx_1.append(fentro_approx.cpu().numpy())
            # entros_approx_1.append(entro_approx.cpu().numpy())

            # maxp_origin = 1./torch.max(pi_p_avg_batch, dim=1)[0]
            # maxp_approx = 1./torch.max(f_out, dim=1)[0]

            # maxp_origin_1.append(maxp_origin.cpu().numpy())
            # maxp_approx_1.append(maxp_approx.cpu().numpy())
            # gvalue_approx_1.append(1./g_out.cpu().numpy()) 
            batch_idx += 1

    
    # num_labels = all_preds[0].shape[-1]
    # all_preds = torch.cat(all_preds).view(-1, num_samples, num_labels) # here change to #number of classes instead of 10

    miscls_approx = np.concatenate(miscls_approx)
    miscls_origin = np.concatenate(miscls_origin)
    # mi_approx_1 = np.concatenate(mi_approx_1)
    # entros_origin_1 = np.concatenate(entros_origin_1)
    # fentros_approx_1 = np.concatenate(fentros_approx_1)
    # entros_approx_1 = np.concatenate(entros_approx_1)
    # maxp_origin_1 = np.concatenate(maxp_origin_1)
    # maxp_approx_1 = np.concatenate(maxp_approx_1)
    # gvalue_approx_1 = np.concatenate(gvalue_approx_1)
    correct_approx = np.sum(1-miscls_approx)
    correct_ensemble = np.sum(1-miscls_origin)

    print('approx ACC :', correct_approx / (len(test_loader.dataset)))
    print('ensemble ACC :', correct_ensemble / (len(test_loader.dataset)))

    return all_preds, all_images, all_labels

    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Amortized approximation on Cifar10')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--approx_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to approx (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dropout-rate', type=float, default=0.5, metavar='p_drop',
                        help='dropout rate')
    
    parser.add_argument('--alpha', type=float, default=1e5) 

    parser.add_argument('--model_path', type=str, default='./checkpoint/', metavar='N',
                        help='path where the model params are saved.')
    parser.add_argument('--arch', type=str, help='distilled model arch.')
    parser.add_argument('--gen_ds_train_path', type=str, help='path where the data is saved.')
    parser.add_argument('--gen_ds_test_path', type=str, help='path where the data is saved.')
    parser.add_argument('--ds_format', type=str, default='file', help='ds format.')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='ds name.')
    parser.add_argument('--output_dataset_path', type=str, default='prediction.pt', help='output ds path .')
    parser.add_argument('--from-approx-model', type=int, default=0,
                        help='if our model is loaded or trained')
    parser.add_argument('--ood', type=bool, default = False)
    parser.add_argument('--ood_data_path', type=str)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}

    if not args.ood:
        if args.ds_format == 'file':
            gen_dataset, gen_dataloader = load_dataset_for_gm_training(args.batch_size, args.gen_ds_train_path, 'augment')
        else:
            gen_dataset, gen_dataloader = load_dataset_for_gm_training_from_folder(args.batch_size, args.gen_ds_train_path, 'augment', False)
        
        gen_testset, gen_testloader = load_dataset_for_gm_training(args.test_batch_size, args.gen_ds_test_path, 'no_augment', drop_last = False)
        
    # if args.from_approx_model == 0:
    #     # output_samples = torch.load('./cifar10-vgg19rand-tr-samples.pt')
    #     output_samples = torch.load(args.train_ds_path)

    # --------------- training approx ---------

    print('approximating ...')
    fmodel = get_net(args.arch).to(device)
    gmodel = get_net(args.arch, concentration=True).to(device)

    if args.from_approx_model == 0:
        f_optimizer = optim.SGD(fmodel.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        g_optimizer = optim.SGD(gmodel.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        best_acc = 0
        for epoch in range(1, args.approx_epochs + 1):
            train_approx(args, fmodel, gmodel, device, gen_dataloader, f_optimizer, g_optimizer, epoch)

            if epoch % 5 == 0:
                acc = test(args, fmodel, device, gen_testloader)
                if acc > best_acc:
                    torch.save(fmodel.state_dict(), args.model_path + args.dataset_name + '_lr' + str(args.lr) + '_alpha' + str(args.alpha) +'rand-mean-mmd.pt')
                    torch.save(gmodel.state_dict(), args.model_path + args.dataset_name + '_lr' + str(args.lr) + '_alpha' + str(args.alpha) +'rand-conc-mmd.pt')
                    best_acc = acc

    else:
        fmodel.load_state_dict(torch.load(args.model_path + args.dataset_name + '_lr' + str(args.lr) + '_alpha' + str(args.alpha) +'rand-mean-mmd.pt'))
        gmodel.load_state_dict(torch.load(args.model_path + args.dataset_name + '_lr' + str(args.lr) + '_alpha' + str(args.alpha) +'rand-conc-mmd.pt'))


    # fitting individual Dirichlet is not in the sample code as it's time-consuming
    if args.ood:
        _, _, ood_dataset, extraset = get_svhn(args.ood_data_path, split = 1)
        gen_testloader = DataLoader(ood_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers = 4,
                            pin_memory = False,
                            prefetch_factor = 8,
                            persistent_workers = True)
        all_preds, all_images, all_labels = eval_approx_on_ood(args, fmodel, gmodel, device, gen_testloader)
    else:
        all_preds, all_images, all_labels = eval_approx(args, fmodel, gmodel, device, gen_testloader)

    def create_dataset(predictions, images, labels, dataset_path):
        '''
            predictions - 
            dataset - (image,label)
            dataset_path - path to which to save the new dataset
        '''
        predictions_tensor = torch.cat(predictions,0).view(-1, *predictions[0].shape).cpu()
        images_tensor = torch.cat(images,0).view(-1, *images[0].shape).cpu()
        labels_tensor = torch.tensor(labels).cpu()

        print (predictions_tensor.shape)
        print (images_tensor.shape)
        print (labels_tensor.shape)
        new_dataset = TensorDataset(images_tensor, predictions_tensor, labels_tensor) # create your datset
        torch.save(new_dataset, dataset_path)
        

    # save the predictions for analysis
    create_dataset(all_preds, all_images, all_labels, args.output_dataset_path)

if __name__ == '__main__':
    main()
