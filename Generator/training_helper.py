'''
This file responsible for the generator training itself. 
There is no direct access to this file (helper file for generator_training.py) 

In order to train the generator --> see generator_training.py 
'''
import time
import wandb
import torch
import numpy as np
import torch.nn as nn
from mmd_loss import MMD
import torch.distributed as dist
from torch.autograd import Variable

from utils import save_model, get_optimizer, get_scheduler


def y_filtered_by_idx(Y, bagging, virtual_ensemble_size, ensemble_size):
    '''
        Pick probability vector of specific idx list - corresponding to the models idx in the ensemble.

        @note - used for virtual_ensemble_size support. i.e in case we would like to sample each
                iteration only a subset of the models in the ensemble. (space consideration).
    '''
    if bagging:
        new_Y = torch.zeros((len(Y), virtual_ensemble_size, len(Y[0][0])))
        for i in range(len(Y)):
            idxs = np.random.choice(np.linspace(0, len(Y[i])-1, len(Y[i]), dtype=int), virtual_ensemble_size)
            new_Y[i] = Y[i][idxs,:]
        return new_Y
    else:
        idxs = np.random.choice(np.linspace(0, ensemble_size-1, ensemble_size, dtype=int), virtual_ensemble_size)
        return Y[:,idxs,:]
    

def train_iteration(X, Y, generator, loss, batch_size, virtual_ensemble_size, latent_std, device):
    '''
    Train iteration

    @param X - inputs - size [B, ...], for cifar10 - [B, 3, 32, 32]
    @param Y - proability vectors - size [B, M, C], for cifar10 - [B, M, 10]
    @param generator
    @param loss - mmd loss
    @param batch_size - B
    @param virtual_ensemble_size - M
    @param latent_std - std of the noise concat to the input. 
    @param device - 'cuda' or 'cpu'
    '''

    #----- change dims to z:[B*M,3,32,32], image:[B*M,3,32,32] ------#
    image = X.repeat_interleave(Y.shape[1]).view(X.shape[0], *X.shape[1:], Y.shape[1])
    dim_idxs = np.arange(image.dim())
    dim_idxs = [dim_idxs[0], dim_idxs[-1], *dim_idxs[1:-1]]
    image = image.permute(dim_idxs) #(0,4,1,2,3)
    image = image.reshape(-1, *image.shape[2:])
    z = torch.normal(mean = torch.zeros(*image.shape, device = device), std = latent_std * torch.ones(*image.shape, device = device))
    
    # ---- forward pass ---- #
    Y_gen = generator(z, image)
    Y_gen = Y_gen.view(batch_size, virtual_ensemble_size, Y.shape[-1])

    softmax = nn.Softmax(dim=2)
    target = Y.permute(1,0,2).reshape(-1,batch_size * Y.shape[-1])
    Y_gen_softmax = softmax(Y_gen)
    source = Y_gen_softmax.permute(1,0,2).reshape(-1,batch_size * Y.shape[-1])

    #------ MMD takes [M,B*C],[M,B*C] ------#
    mmd = loss(target, source)
    return mmd, Y_gen_softmax

    
# in case different training iteration is needed - can control it from here.
train_func_dict = {
    'regular': train_iteration,
    'combination': train_iteration,
}

def train(generator, trainloader, testloader, optimizer, train_type, device='cuda', args=None, gpu = 0):
    '''
        Train generator
    '''
    args.ensemble_size = trainloader.dataset[0][1].shape[0]
    if args.bagging: # there is not fixed size ensemble size!
        args.ensemble_size = None
    wandb.config.update(args)

    loss = MMD(bandwidth_range=args.kernel_bw, kernel = args.kernel_type, device = device)
    if args.empty_loss:
        loss = torch.nn.MSELoss()
        
    g_optimizer = get_optimizer(optimizer, args.generator_lr, args.weight_decay, generator, args.opt_ckpt)
    lr_scheduler = get_scheduler(g_optimizer)

    train_iteration = train_func_dict[train_type]
    virutal_batch_size = int(args.generator_batch_size / args.gradient_accumulate)

    generator.train()
    for p in generator.parameters():
        p.requires_grad = True
    
    for epoch in range(args.start_epoch, args.epochs):
        acc = 0
        items = 0

        # make sure all processes start epoch together 
        if args.mp:
            trainloader.sampler.set_epoch(epoch)
            dist.barrier() 
        
        print('Starting epoch {}'.format(epoch))
        print (time.asctime( time.localtime(time.time()) ))
        mmd_track = []
        idx = 0
        for _, sample in enumerate(trainloader):
        
            if args.bagging: # there is not fixed size ensemble size!
                ensemble_size = len(sample[1])
            else:
                ensemble_size = args.ensemble_size
                
            g_optimizer.zero_grad()
            mmd_sum = 0
            # import ipdb; ipdb.set_trace()
            for i in range(args.gradient_accumulate):
                # ---- X[B,3,32,32] ---- Y[B,M,C]---- #
                
                X = sample[0][i:i+virutal_batch_size].to(device, non_blocking = True).float()
                Y = sample[1][i:i+virutal_batch_size]
                items += X.shape[0]
                Y_filtered = y_filtered_by_idx(Y, args.bagging, args.virtual_ensemble_size, ensemble_size).to(device, non_blocking = True).float()

                mmd, Y_gen_softmax = train_iteration(X, Y_filtered, generator, loss, virutal_batch_size,
                                    args.virtual_ensemble_size, args.latent_std, device)
                mmd_sum += mmd
                mmd.backward()
                y_true = torch.argmax(torch.sum(Y_filtered, dim = 1) / args.virtual_ensemble_size, dim = -1)
                y_pred = torch.argmax(torch.sum(Y_gen_softmax, dim = 1) / args.virtual_ensemble_size, dim = -1)
                acc += (y_true == y_pred).sum()

            g_optimizer.step()

            mmd_track.append(mmd_sum / args.gradient_accumulate)

        loss_calc = sum(mmd_track) / len(mmd_track)
        acc = acc / items

        wandb.log({
            "Epoch": epoch,
            "MMD_Loss": loss_calc,
            "Accuracy": acc})

        if epoch % args.epoch_per_checkpoint == 0 and epoch != 0:
            if (args.mp and gpu==0) or (not args.mp):
                save_model(args.save_dir + "/epoch{}".format(epoch), generator)
            if args.mp:
                dist.barrier()

        if epoch % args.epochs_per_valid == 0 and epoch != 0:
            print ('Starting evaluation')
            print (time.asctime( time.localtime(time.time()) ))
            validation_step(generator, testloader, device, epoch, 8, 'Test', args, loss, train_type)
            print ('End evaluation')
            print (time.asctime( time.localtime(time.time()) ))
            generator.train()

        lr_scheduler.step()

    if (args.mp and gpu==0) or (not args.mp):
        save_model(args.save_dir + "/epoch{}".format(epoch), generator)
        save_model(args.save_dir + "/opt_epoch{}".format(epoch), g_optimizer)
    if args.mp:
        dist.barrier()   
    return generator


def train_with_combination(generator, trainloader, trainloader_secondary, testloader, optimizer, train_type, device='cuda', args=None, gpu = 0):
    '''
        Train generator with with combination of primary and secondary dataset
    '''
    args.ensemble_size = trainloader.dataset[0][1].shape[0]
    if args.bagging: # there is not fixed size ensemble size!
        args.ensemble_size = None
    wandb.config.update(args)

    loss = MMD(bandwidth_range=args.kernel_bw, kernel = args.kernel_type, device = device)
    g_optimizer = get_optimizer(optimizer, args.generator_lr, args.weight_decay, generator, args.opt_ckpt)
    lr_scheduler = get_scheduler(g_optimizer)

    train_iteration = train_func_dict[train_type]
    virutal_batch_size = int(args.generator_batch_size / args.gradient_accumulate)

    generator.train()
    for p in generator.parameters():
        p.requires_grad = True
        
    for epoch in range(args.start_epoch, args.epochs):
        acc = 0
        items = 0
        # make sure all processes start epoch together

        if args.mp:
            trainloader.sampler.set_epoch(epoch)
            dist.barrier() 
        
        print('Starting epoch {}'.format(epoch))
        mmd_track = []
        mmd_secondary_track = []
        tot_loss_track = []
        secondary_iterator = iter(trainloader_secondary)

        for _, sample in enumerate(trainloader):
            
        # get next batch from the auxilary dataset
            try:
                secondary_sample = next(secondary_iterator)
            except StopIteration:
                secondary_iterator = iter(trainloader_secondary)
                secondary_sample = next(secondary_iterator)
            
            if args.bagging: # there is not fixed size ensemble size!
                ensemble_size = len(sample[1])
            else:
                ensemble_size = args.ensemble_size
                
            g_optimizer.zero_grad()
            mmd_sum = 0
            mmd_secondary_sum = 0
            tot_loss_sum = 0

            for i in range(args.gradient_accumulate):
                # -- On Original Dataset -- #
                # Enable batch norm layers
                if args.stat_only_for_main_data:
                    for module in (generator.modules()):
                        if isinstance(module, nn.BatchNorm2d):
                            module.track_running_stats = True
                            bn = module
                
                X = sample[0][i:i+virutal_batch_size].to(device, non_blocking = True)
                Y = sample[1][i:i+virutal_batch_size]
                items += X.shape[0]

                Y_filtered = y_filtered_by_idx(Y, args.bagging, args.virtual_ensemble_size, ensemble_size).to(device, non_blocking = True).float()
                mmd, Y_gen_softmax = train_iteration(X, Y_filtered, generator, loss, virutal_batch_size,
                                                    args.virtual_ensemble_size, args.latent_std, device)
                
                mmd_sum += mmd
                mmd.backward()
                # -- On Mixup Dataset -- #
                # Disable batch norm layers
                if args.stat_only_for_main_data:
                    for module in (generator.modules()):
                        if isinstance(module, nn.BatchNorm2d):
                            module.track_running_stats = False
                            bn = module
                
                X_secondary = secondary_sample[0][i:i+virutal_batch_size].to(device, non_blocking = True).float()
                Y_secondary = secondary_sample[1][i:i+virutal_batch_size]
                Y_secondary_filtered = y_filtered_by_idx(Y_secondary, args.bagging, args.virtual_ensemble_size, ensemble_size).to(device, non_blocking = True).float()
                mmd_secondary, _ = train_iteration(X_secondary, Y_secondary_filtered, generator, loss, virutal_batch_size,
                                                   args.virtual_ensemble_size, args.latent_std, device)
                mmd_secondary_coeff = args.secondary_loss_coeff * mmd_secondary
                
                mmd_secondary_sum += mmd_secondary
                mmd_secondary_coeff.backward()
                tot_loss = mmd + args.secondary_loss_coeff * mmd_secondary

                tot_loss_sum += tot_loss

                y_true = torch.argmax(torch.sum(Y_filtered, dim = 1) / args.virtual_ensemble_size, dim = -1)
                y_pred = torch.argmax(torch.sum(Y_gen_softmax, dim = 1) / args.virtual_ensemble_size, dim = -1)
                acc += (y_true == y_pred).sum().cpu()
                

            g_optimizer.step()

            mmd_track.append(mmd_sum.detach().cpu() / args.gradient_accumulate)
            mmd_secondary_track.append(mmd_secondary_sum.detach().cpu() / args.gradient_accumulate)
            tot_loss_track.append(tot_loss_sum.detach().cpu() / args.gradient_accumulate)



        mmd_loss_calc = sum(mmd_track)/len(mmd_track)
        tot_loss_calc = sum(tot_loss_track)/len(tot_loss_track)
        secondary_loss_calc = sum(mmd_secondary_track)/len(mmd_secondary_track)
        acc = acc / items

        wandb.log({
            "Epoch": epoch,
            "MMD_Loss": mmd_loss_calc,
            "MMD_Mixup_Loss":secondary_loss_calc,
            "Total_Loss": tot_loss_calc,
            "Accuracy": acc})

        if epoch % args.epoch_per_checkpoint == 0 and epoch != 0:
            if (args.mp and gpu==0) or (not args.mp):
                save_model(args.save_dir + "/epoch{}".format(epoch), generator)
                save_model(args.save_dir + "/opt_epoch{}".format(epoch), g_optimizer)
            if args.mp:
                dist.barrier()

        if epoch % args.epochs_per_valid == 0:
            # validation_on_trainset(generator, trainloader, latent_size, Y.shape[1], device, epoch, batch_size, args.virutal_ensemble_size, 'Train')
            validation_step(generator, testloader, device, epoch, 8, 'Test', args, loss, train_type)
            generator.train()

        lr_scheduler.step()

    if (args.mp and gpu==0) or (not args.mp):
        save_model(args.save_dir + "/epoch{}".format(epoch), generator)
        
    if args.mp:
        dist.barrier()   
    return generator



def validation_step(generator, dataloader, device, epoch, batch_size, log_name, args, mmd_loss, train_option):
    '''
        Validation function.
    '''
    
    generator.eval()
    cel_loss = nn.NLLLoss()

    simi = []
    mmd_track = []
    cel_track = []
    correct = 0
    softmax = nn.Softmax(dim=2)

    with torch.no_grad():
        for batch_ndx, sample in enumerate(dataloader):
            X = sample[0]
            Y = sample[1]
            true_labels = sample[2].to(device, non_blocking = True)
            batch_size = len(X)
            #------- [B,3,32,32] --------#
            image = X.to(device, non_blocking = True)
            #------- [B,M,3,32,32] ------#
            image = image.repeat_interleave(Y.shape[1]).view(X.shape[0], *X.shape[1:], Y.shape[1])
            dim_idxs = np.arange(image.dim()) 
            dim_idxs = [dim_idxs[0], dim_idxs[-1], *dim_idxs[1:-1]]
            image = image.permute(dim_idxs) #(0,4,1,2,3)
            #------- [B,M,C] -------#
            classes_vector = Y.to(device, non_blocking = True)
          
            #----- z:[B,M,3,32,32] ------- image:[B,M,3,32,32] ------#
            image = image.reshape(-1, *image.shape[2:])
            #----- z:[B*M,3,32,32] ------- image:[B*M,3,32,32] ------#

            if train_option == 'modes':
                stds = generator.stds.repeat(batch_size)[:,None,None,None].expand_as(image)
                means = generator.means.repeat(batch_size)[:,None,None,None].expand_as(image)    
                z = means + torch.normal(mean = torch.zeros(*image.shape, device = device), std = torch.ones(*image.shape, device = device)) * stds
            else:
                z = Variable(torch.normal(mean = torch.zeros(*image.shape, device = device), std = args.latent_std * torch.ones(*image.shape, device = device)))
            
            
            gen_classes_vector = generator(z, image).squeeze()

            gen_classes_vector = gen_classes_vector.view(batch_size, Y.shape[1], Y.shape[-1])
            # z = z.cpu()
            gen_preds = softmax(gen_classes_vector)
            #MMD Calc
            if train_option == 'option1':
                target = classes_vector.reshape(-1, Y.shape[1] * Y.shape[-1])
                source = gen_preds.reshape(-1, Y.shape[1] * Y.shape[-1])
            elif train_option in ['option2', 'secondary', 'gan', 'modes']:
                target = classes_vector.permute(1,0,2).reshape(-1, batch_size * Y.shape[-1])
                source = gen_preds.permute(1,0,2).reshape(-1, batch_size * Y.shape[-1])
            
            mmd = mmd_loss(target, source)
            #CE Calc
            ensemble_simplex = torch.sum(gen_preds, dim = 1)/Y.shape[1]
            cel = cel_loss(torch.log(ensemble_simplex) ,true_labels)
            mmd_track.append(mmd)
            cel_track.append(cel)
            #Argmax on the histogram (ensebmle of epsilons)
            ensemble_predictions = torch.sum(gen_preds, dim = 1)/Y.shape[1]
            preds = torch.argmax(ensemble_predictions, dim = 1)
            correct += preds.eq(true_labels).sum()
            
            fake = torch.sum(classes_vector, dim=1)/Y.shape[1]
            values, posterior = torch.max(fake,dim = 1)
            similarity = np.equal(posterior.cpu(), preds.cpu()).sum() / len(posterior)
            simi.append(similarity.detach())

    generator.train()
    acc = correct / (batch_ndx*batch_size)
    mmd_loss_calc = sum(mmd_track)/len(mmd_track)
    cel_loss_calc = sum(cel_track)/len(cel_track)

    wandb.log({log_name + "_Similarity":sum(simi)/len(simi), 
            log_name + "_Accuracy" : acc, 
            log_name + '_MMD_Loss':mmd_loss_calc,
            log_name + '_CE_Loss' :cel_loss_calc, 
            "Epoch":epoch})

    del mmd_track, cel_track, z
    torch.cuda.empty_cache()

    return similarity