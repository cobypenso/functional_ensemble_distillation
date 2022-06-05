"""Train and make predictions with distilled network parameterising a Dirichlet distribution over ensemble output"""
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from data import get_svhn
from models import get_net
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import h5py
from torch.utils.data import TensorDataset

from src import utils 
from src import metrics
# from src import utils
# from src import metrics
from src.dataloaders import cifar10_corrupted
from src.dataloaders import cifar10_ensemble_pred
from src.ensemble import ensemble_wrapper
from src.distilled import cifar_resnet_dirichlet
from src.experiments.cifar10 import resnet_utils

LOGGER = logging.getLogger(__name__)


def train_distilled_network_dirichlet(model_dir="models/distilled_model_cifar10_dirichlet", model_name = "ensemble_predictions.h5"):
    """Distill ensemble with distribution distillation (Dirichlet) """

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)


    # New Data loading --- #
    train_data = cifar10_ensemble_pred.Cifar10Data(augmentation=True, data_dir = model_dir, ensemble_path = model_name)


    train_loader = torch.utils.data.DataLoader(train_data.set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)


    test_data = cifar10_ensemble_pred.Cifar10Data(train=False, data_dir = model_dir, ensemble_path = model_name)

    test_loader = torch.utils.data.DataLoader(test_data.set,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=0)

    ensemble_size = args.num_ensemble_members

    ensemble = ensemble_wrapper.EnsembleWrapper(output_size=args.output_size)
    
    device = utils.torch_settings(args.seed, args.gpu)
    distilled_model = cifar_resnet_dirichlet.CifarResnetDirichlet(ensemble,
                                                                  resnet_utils.Bottleneck,
                                                                  [2, 2, 2, 2],
                                                                  device=device,
                                                                  learning_rate=args.lr,
                                                                  output_size=args.output_size,
                                                                  conv_type = args.conv_type,
                                                                  norm_type = args.norm_type)

    loss_metric = metrics.Metric(name="Mean loss", function=distilled_model.calculate_loss)
    distilled_model.add_metric(loss_metric)

    distilled_model.train(train_loader, num_epochs=args.num_epochs, validation_loader=test_loader)
    
    distilled_model.eval_mode()
    counter = 0
    model_acc = 0

    for batch in test_loader:
        
        inputs, labels = batch
        inputs, labels = inputs[0].to(distilled_model.device), labels.to(distilled_model.device)
        predicted_distribution = distilled_model.predict(inputs.float(), num_samples=args.num_ensemble_members).mean(axis=1)
        model_acc += metrics.accuracy(predicted_distribution.to(distilled_model.device), labels.long())
        counter += 1

    LOGGER.info("Test accuracy is {}".format(model_acc / counter))
    
    torch.save(distilled_model.state_dict(), model_dir + model_name[:-3] +'_ep_' + str(args.num_epochs)+'_norm_'+args.norm_type + '_seed_' + str(args.seed))


def predictions_dirichlet(model_dir="../models/distilled_model_cifar10_dirichlet",
                          model_name = "ensemble_predictions.h5"):
    """Make and save predictions on train and test data with distilled model at model_dir"""

    args = utils.parse_args()

    train_data = cifar10_ensemble_pred.Cifar10Data(data_dir = model_dir, ensemble_path = model_name)
    test_data = cifar10_ensemble_pred.Cifar10Data(train=False, data_dir = model_dir, ensemble_path = model_name)

    ensemble = ensemble_wrapper.EnsembleWrapper(output_size=args.output_size)

    device = utils.torch_settings(args.seed, args.gpu)

    distilled_model = cifar_resnet_dirichlet.CifarResnetDirichlet(ensemble,
                                                                  resnet_utils.Bottleneck,
                                                                  [2, 2, 2, 2],
                                                                  learning_rate=args.lr,
                                                                  device=device,
                                                                  output_size=args.output_size,
                                                                  conv_type = args.conv_type,
                                                                  norm_type = args.norm_type)

    distilled_model.load_state_dict(torch.load(model_dir + model_name[:-3] +'_ep_' + str(args.num_epochs)+'_norm_'+args.norm_type + '_seed_' + str(args.seed), map_location=torch.device(device)))
    distilled_model.eval_mode() 

    data_list = [test_data]
    labels = ["test"]

    if args.save_format == 'h5':
        file_dir = './predictions'
        hf = h5py.File(file_dir + '.h5', 'w')

    for data_set, label in zip(data_list, labels):

        data, pred_samples, alpha, teacher_predictions, targets = \
            [], [], [], [], []

        data_loader = torch.utils.data.DataLoader(data_set.set,
                                                  batch_size=8,
                                                  shuffle=False,
                                                  num_workers=0)

        for batch in data_loader:
            inputs, labels = batch
            img = inputs[0].to(distilled_model.device)
            
            a, probs = distilled_model.predict(img, num_samples=args.num_ensemble_members, return_params=True)
            alpha.append(a.data.cpu().numpy())
            data.append(img.data.cpu().numpy())
            targets.append(labels.data.cpu().numpy())
            teacher_predictions.append(inputs[1].data.cpu().numpy())
            pred_samples.append(probs.data.cpu().numpy())

        data = np.concatenate(data, axis=0)
        pred_samples = np.concatenate(pred_samples, axis=0)
        teacher_predictions = np.concatenate(teacher_predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        alpha = np.concatenate(alpha, axis=0)

        preds = np.argmax(np.mean(pred_samples, axis=1), axis=-1)

        # Check accuracy
        acc = np.mean(preds == targets)
        LOGGER.info("Accuracy on {} data set is: {}".format(label, acc))

        # Check accuracy relative teacher
        teacher_preds = np.argmax(np.mean(teacher_predictions, axis=1), axis=-1)
        rel_acc = np.mean(preds == teacher_preds)
        LOGGER.info("Accuracy on {} data set relative teacher is: {}".format(label, rel_acc))

        if args.save_format == 'h5':
            grp = hf.create_group(label)
            grp.create_dataset("data", data=data)
            grp.create_dataset("predictions", data=pred_samples)
            grp.create_dataset("teacher-predictions", data=teacher_predictions)
            grp.create_dataset("targets", data=targets)
            grp.create_dataset("alpha", data=alpha)
        elif args.save_format == 'pt':
            
            new_dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(pred_samples), torch.from_numpy(targets))
            torch.save(new_dataset, model_dir + model_name[:-3] +'_ep_' + str(args.num_epochs)+'_norm_'+args.norm_type + '_seed_' + str(args.seed) + '_' + label + '.pt')

    return pred_samples








def predictions_ood(model_dir, model_name, save_path):
    """Make and save predictions on train and test data with distilled model at model_dir"""

    args = utils.parse_args()

    trainset, validset, testset, extraset = get_svhn('../datasets/svhn/', split = 0.8)

    ensemble = ensemble_wrapper.EnsembleWrapper(output_size=args.output_size)

    distilled_model = cifar_resnet_dirichlet.CifarResnetDirichlet(ensemble,
                                                                  resnet_utils.Bottleneck,
                                                                  [2, 2, 2, 2],
                                                                  learning_rate=args.lr,
                                                                  device='cuda',
                                                                  output_size=args.output_size,
                                                                  conv_type = args.conv_type,
                                                                  norm_type = args.norm_type)

    distilled_model.load_state_dict(torch.load(model_dir + model_name[:-3] +'_ep_' + str(args.num_epochs)+'_norm_'+args.norm_type + '_seed_' + str(args.seed), map_location=torch.device('cuda')))
    distilled_model.eval_mode() 
    
    data_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=4,
                                                  shuffle=False,
                                                  num_workers=0)

    images_list = []
    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for batch in data_loader:
                inputs, labels = batch
                img = inputs.to(distilled_model.device)
                
                a, probs = distilled_model.predict(img, num_samples=args.num_ensemble_members, return_params=True)
                for i in range(len(inputs)):
                    images_list.append(inputs[i])
                    predictions_list.append(probs[i].cpu())
                    labels_list.append(labels[i])
    

    def create_dataset(predictions, images, labels, dataset_path):
        '''
            predictions - 
            dataset - (image,label)
            dataset_path - path to which to save the new dataset
        '''
        predictions_tensor = torch.cat(predictions,0).view(-1, *predictions[0].shape)
        images_tensor = torch.cat(images,0).view(-1, *images[0].shape)
        labels_tensor = torch.tensor(labels)

        print (predictions_tensor.shape)
        print (images_tensor.shape)
        print (labels_tensor.shape)
        new_dataset = TensorDataset(images_tensor, predictions_tensor, labels_tensor) # create your datset
        torch.save(new_dataset, dataset_path)
    

    create_dataset(predictions_list, images_list, labels_list, save_path)




def predictions_corrupted_data_dirichlet(model_dir="models/distilled_model_cifar10_dirichlet",
                               file_dir="../../dataloaders/data/distilled_model_predictions_corrupted_data_dirichlet.h5"):
    """Make and save predictions on corrupted data with distilled model at model_dir"""

    args = utils.parse_args()

    # Load model
    ensemble = ensemble_wrapper.EnsembleWrapper(output_size=10)

    distilled_model = cifar_resnet_dirichlet.CifarResnetDirichlet(ensemble,
                                                                  resnet_utils.Bottleneck,
                                                                  [2, 2, 2, 2],
                                                                  learning_rate=args.lr,
                                                                  conv_type = args.conv_type,
                                                                  norm_type = args.norm_type)

    distilled_model.load_state_dict(torch.load(model_dir, map_location=torch.device(distilled_model.device)))

    distilled_model.eval_mode()

    corruption_list = ["test", "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                       "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "motion_blur", "pixelate",
                       "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]

    hf = h5py.File(file_dir, 'w')

    for i, corruption in enumerate(corruption_list):
        corr_grp = hf.create_group(corruption)

        if corruption == "test":
            intensity_list = [0]
        else:
            intensity_list = [1, 2, 3, 4, 5]

        for intensity in intensity_list:
            # Load the data
            data_set = cifar10_corrupted.Cifar10DataCorrupted(corruption=corruption, intensity=intensity,
                                                              data_dir="../../")
            dataloader = torch.utils.data.DataLoader(data_set.set,
                                                     batch_size=100,
                                                     shuffle=False,
                                                     num_workers=0)

            # data = []
            predictions, targets, alpha = [], [], []

            for j, batch in enumerate(dataloader):
                inputs, labels = batch
                targets.append(labels.data.numpy())
                # data.append(inputs.data.numpy())

                inputs, labels = inputs.to(distilled_model.device), labels.to(distilled_model.device)

                a, preds = distilled_model.predict(inputs, return_params=True)
                alpha.append(a.to(torch.device("cpu")).data.numpy())
                predictions.append(preds.to(torch.device("cpu")).data.numpy())

            sub_grp = corr_grp.create_group("intensity_" + str(intensity))

            # data = np.concatenate(data, axis=0)
            # sub_grp.create_dataset("data", data=data)

            predictions = np.concatenate(predictions, axis=0)
            sub_grp.create_dataset("predictions", data=predictions)

            targets = np.concatenate(targets, axis=0)
            sub_grp.create_dataset("targets", data=targets)

            preds = np.argmax(np.mean(predictions, axis=1), axis=-1)

            acc = np.mean(preds == targets)
            LOGGER.info("Accuracy on {} data set with intensity {} is {}".format(corruption, intensity, acc))

            alpha = np.concatenate(alpha, axis=0)
            sub_grp.create_dataset("alpha", data=alpha)

    hf.close()


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))
    
    print(args.model_path)
    print(args.model_name)

    if args.distill:
        train_distilled_network_dirichlet(model_dir = args.model_path, model_name = args.model_name)
    if args.predict:
        predictions_dirichlet(model_dir = args.model_path, model_name = args.model_name)
    if args.ood:
        predictions_ood(model_dir = args.model_path, model_name = args.model_name, save_path = args.predictions_save_path)



if __name__ == "__main__":
    main()
