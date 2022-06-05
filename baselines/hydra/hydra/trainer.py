import argparse
import logging
import sys
from matplotlib import testing

from numpy import save
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
import numpy as np
import torch.nn.functional as F
import torch.utils.data
from tqdm import trange
import hydra.models as models
from utils import *
from data import get_svhn

from hydra.utils import get_device, set_logger, set_seed, CE_with_probs

parser = argparse.ArgumentParser(description="Hydra")

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-epochs", type=int, default=200)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="optimizer")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--test-batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
parser.add_argument('--network', type=str, default='ResNet18', help="network to use")
parser.add_argument('--num-heads', type=int, default=120, help="number of total classifiers")
parser.add_argument('--num-heads-in-batch', type=int, default=8, help="number of classifiers to take in each batch")
parser.add_argument('--temp', type=int, default=5, help="temperature during training")
parser.add_argument('--eval-temp', type=int, default=1, help="temperature during evaluation")
parser.add_argument('--switch-phase', type=int, default=100, help="epoch to move from training one head to multiple heads")


#############################
#       General args        #
#############################
parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
parser.add_argument('--num-workers', default=0, type=int, help='num wortkers')
parser.add_argument("--eval-every", type=int, default=1, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output", help="dir path for output file")
parser.add_argument("--seed", type=int, default=42, help="seed value")

parser.add_argument("--train_path", type=str, default = './train.pt')
parser.add_argument("--test_path", type=str, default = './test.pt')
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_classes", type=int)
parser.add_argument('--predict', type=bool, default=False)
parser.add_argument('--predict_net', type=bool, default=False)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--ood', type=bool, default=False)

args = parser.parse_args()
        
set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
num_classes = args.num_classes

exp_name = f'Hydra_seed_{args.seed}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))

################################
# get data from teacher
###############################
# generate fictitious input

gen_dataset, gen_dataloader = load_dataset_for_gm_training(args.batch_size, args.train_path, 'augment')
gen_testset, gen_testloader = load_dataset_for_gm_training(args.test_batch_size, args.test_path, 'no_augment', drop_last = False)


train_preds = torch.load(args.train_path)
test_preds = torch.load(args.test_path)

###############################
# Init network with Multiple heads
###############################
network = getattr(models, args.network)(num_classes=num_classes, num_heads=args.num_heads, testing = args.predict_net)
network.to(device)
softmax = torch.nn.Softmax(dim=2)

###############################
# optimizer
###############################
param_group = []
param_group.append({'params': network.parameters()})
optimizer = torch.optim.SGD(param_group, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True) \
           if args.optimizer == 'sgd' else torch.optim.Adam(param_group, lr=args.lr, weight_decay=args.wd)


def predict(network, gen_testloader, save_path, ood = False):
    network.eval()

    image_list = []
    pred_list = []
    label_list = []

    with torch.no_grad():
        for k, batch in enumerate(gen_testloader):
            batch = (t.to(device) for t in batch)
            if ood:
                train_data, clf_labels = batch
                student_logits = torch.zeros(size = (len(train_data), 120, args.num_classes))
            else:  
                train_data, teacher_logits, clf_labels = batch
                student_logits = torch.zeros_like(teacher_logits)
            
            for j in range(120):
                # if train_data.shape == torch.Size([len(train_data), 3,96,96]):
                #     train_data = F.interpolate(train_data, size = (32,32))
                student_logits[:, j, :] = network(train_data, [j]).squeeze()

            for i in range(len(train_data)):
                image_list.append(train_data[i].cpu())
                pred_list.append(student_logits[i].cpu())
                label_list.append(clf_labels[i].cpu())
    
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
    
    create_dataset(pred_list, image_list, label_list, save_path)


if args.train:
    ###############################
    # Train
    ###############################
    criteria = torch.nn.CrossEntropyLoss()

    best_loss = best_val_nll = best_test_nll = np.inf
    best_epoch = 0
    best_val_acc = best_test_acc = 0
    epoch_iter = trange(args.num_epochs)
    best_test_labels_vs_preds = best_test_clf_report = None

    best_labels_vs_preds_val = None
    best_val_loss = -1

    for epoch in epoch_iter:
        network.train()
        cumm_loss = num_samples = 0
        correct = 0.
        # logging.info(torch.cuda.memory_allocated(device=device))
        # select several clients
        for k, batch in enumerate(gen_dataloader):

            optimizer.zero_grad()
            batch = (t.to(device) for t in batch)
            train_data, teacher_logits, clf_labels = batch

            if epoch < args.switch_phase:
                head_ids = torch.zeros(1, device=device, dtype=torch.int64)
            else:
                if epoch == args.switch_phase:
                    network.init_heads()

                head_ids = torch.tensor(np.random.choice(range(args.num_heads), size=args.num_heads_in_batch,
                                                        replace=False), device=device)
                # assign None to gradients of all heads to obtain lower memory footprint
                for n, p in network.named_parameters():
                    if 'classifiers' in n:
                        p.grad = None

            student_logits = network(train_data, head_ids)

            teacher_probs = softmax(teacher_logits[:, head_ids, :])
            student_probs = softmax(student_logits / args.temp)
            loss = (args.temp ** 2) * CE_with_probs(teacher_probs, student_probs)

            loss.backward()
            optimizer.step()
            
            correct += (torch.argmax(torch.mean(student_probs, dim = 1), dim = -1) == clf_labels).sum()

            epoch_iter.set_description(f"[{epoch} {k}] loss: {loss:.3f}")
            cumm_loss += loss * clf_labels.shape[0]
            num_samples += clf_labels.shape[0]

        print ('Train acc:', correct / num_samples)
        cumm_loss /= num_samples
        logging.info(f"\nEpoch loss: {cumm_loss:.4f}")

        if epoch % 25 == 0 and epoch > args.switch_phase:
            # eval
            # acc = eval(network, gen_testloader)
            torch.save(network.state_dict(), args.save_path + '_model.pt')

    torch.save(network.state_dict(), args.save_path + '_model.pt')

if args.predict:
    
    network.load_state_dict(torch.load(args.save_path + '_model.pt'))
    predict(network, gen_testloader, args.save_path+ '_test_predictions.pt')

if args.ood:
    _, _, ood_dataset, extraset = get_svhn('./datasets/svhn/', split = 1)
    gen_testloader = DataLoader(ood_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        drop_last=False, 
                        num_workers = 4,
                        pin_memory = False,
                        prefetch_factor = 8,
                        persistent_workers = True)
    network.load_state_dict(torch.load(args.save_path + '_model.pt'))
    predict(network, gen_testloader, args.save_path+ '_ood_predictions.pt', ood = True)  