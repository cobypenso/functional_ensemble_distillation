"""Train and make predictions with distilled network parameterising a Dirichlet distribution over ensemble output"""
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

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


def predictions_dirichlet(model_dir, model_name, data_dir, save_dir):
    """Make and save predictions on train and test data with distilled model at model_dir"""

    args = utils.parse_args()
    
    # train_data = cifar10_ensemble_pred.Cifar10Data()
    # test_data = cifar10_ensemble_pred.Cifar10Data(train=False)
    train_data = cifar10_ensemble_pred.Cifar10Data(data_dir = data_dir, ensemble_path = model_name)
    test_data = cifar10_ensemble_pred.Cifar10Data(train=False, data_dir = data_dir, ensemble_path = model_name)

    ensemble = ensemble_wrapper.EnsembleWrapper(output_size=args.output_size)

    distilled_model = cifar_resnet_dirichlet.CifarResnetDirichlet(ensemble,
                                                                  resnet_utils.Bottleneck,
                                                                  [2, 2, 2, 2],
                                                                  learning_rate=args.lr,
                                                                  output_size=args.output_size,
                                                                  conv_type = args.conv_type,
                                                                  norm_type = args.norm_type)

    distilled_model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    distilled_model.eval_mode()

    #data_list = [test_data, train_data]
    data_list = [test_data]
    #labels = ["test", "train"]
    labels = ["test"]

    if args.save_format == 'h5':
        hf = h5py.File(save_dir + '.h5', 'w')

    for data_set, label in zip(data_list, labels):

        data, pred_samples, alpha, teacher_predictions, targets = \
            [], [], [], [], []

        data_loader = torch.utils.data.DataLoader(data_set.set,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=0)

        for batch in data_loader:
            inputs, labels = batch
            img = inputs[0].to(distilled_model.device)
            data.append(img.data.numpy())
            targets.append(labels.data.numpy())
            teacher_predictions.append(inputs[1].data.numpy())
            
            a, probs = distilled_model.predict(img, num_samples=args.num_ensemble_members, return_params=True)
            alpha.append(a.data.numpy())
            pred_samples.append(probs.data.numpy())

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
        import ipdb; ipdb.set_trace()
        if args.save_format == 'h5':
            grp = hf.create_group(label)
            grp.create_dataset("data", data=data)
            grp.create_dataset("predictions", data=pred_samples)
            grp.create_dataset("teacher-predictions", data=teacher_predictions)
            grp.create_dataset("targets", data=targets)
            grp.create_dataset("alpha", data=alpha)
        elif args.save_format == 'pt':
            
            new_dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(pred_samples), torch.from_numpy(targets))
            torch.save(new_dataset, save_dir + '_' + label + '.pt')

    return pred_samples


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))
    
    print(args.model_path)
    print(args.model_name)
    print(args.predictions_save_path)
    print(args.data_dir)
    predictions_dirichlet(args.model_path, args.model_name,
                          args.data_dir,
                          args.predictions_save_path)



if __name__ == "__main__":
    main()
