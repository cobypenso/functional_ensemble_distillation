"""Train and make predictions with distilled network parameterising a Gaussian distribution over ensemble logits"""

import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import h5py

from src import utils
from src import metrics
from src.dataloaders import cifar10_corrupted
from src.dataloaders import cifar10_ensemble_pred
from src.ensemble import ensemble_wrapper
from src.distilled import cifar_resnet_distilled
from src.experiments.cifar10 import resnet_utils

LOGGER = logging.getLogger(__name__)


def train_distilled_network_gaussian(model_dir="models/distilled_model_cifar10"):
    """Distill ensemble either with distribution distillation (Gaussian)"""

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    data_ind = np.load("training_files/training_data_indices.npy")
    num_train_points = 40000
    train_ind = data_ind[:num_train_points]
    valid_ind = data_ind[num_train_points:]

    train_data = cifar10_ensemble_pred.Cifar10Data(ind=train_ind, augmentation=True)
    valid_data = cifar10_ensemble_pred.Cifar10Data(ind=valid_ind)

    train_loader = torch.utils.data.DataLoader(train_data.set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_data.set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    test_data = cifar10_ensemble_pred.Cifar10Data(train=False)

    test_loader = torch.utils.data.DataLoader(test_data.set,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=0)

    ensemble_size = 10

    # Note that the ensemble predictions are assumed to have been saved to file (see ensemble_predictions.py),
    # ensemble_indices.npy contains the order of the ensemble members such that ind[:ensemble_size] are the indices
    # of the first ensemble, ind[ensemble_size:2*ensemble_size] are the indices of the second ensemble and so on
    ind = np.load("training_files/ensemble_indices.npy")[((args.rep - 1) * ensemble_size):(args.rep * ensemble_size)]
    ensemble = ensemble_wrapper.EnsembleWrapper(
        output_size=10, indices=ind)

    device = utils.torch_settings(args.seed, args.gpu)
    distilled_model = cifar_resnet_distilled.CifarResnetLogits(ensemble,
                                                               resnet_utils.Bottleneck,
                                                               [2, 2, 2, 2],
                                                               device=device,
                                                               learning_rate=args.lr)

    loss_metric = metrics.Metric(name="Mean loss", function=distilled_model.calculate_loss)
    distilled_model.add_metric(loss_metric)

    distilled_model.train(train_loader, num_epochs=args.num_epochs, validation_loader=valid_loader)

    distilled_model.eval_mode()
    counter = 0
    model_acc = 0

    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs[0].to(distilled_model.device), labels.to(distilled_model.device)

        predicted_distribution = distilled_model.predict(inputs)[0].mean(axis=1)

        model_acc += metrics.accuracy(predicted_distribution.to(distilled_model.device), labels.long())
        counter += 1

    LOGGER.info("Test accuracy is {}".format(model_acc / counter))

    torch.save(distilled_model.state_dict(), model_dir)


def predictions_gaussian(model_dir="models/distilled_model_cifar10",
                         file_dir="../../dataloaders/data/distilled_model_predictions.h5"):
    """Make and save predictions on train and test data with distilled model at model_dir"""

    args = utils.parse_args()

    train_data = cifar10_ensemble_pred.Cifar10Data()
    test_data = cifar10_ensemble_pred.Cifar10Data(train=False)

    ensemble = ensemble_wrapper.EnsembleWrapper(output_size=10)

    distilled_model = cifar_resnet_distilled.CifarResnetLogits(ensemble,
                                                               resnet_utils.Bottleneck,
                                                               [2, 2, 2, 2],
                                                               learning_rate=args.lr)

    distilled_model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    distilled_model.eval_mode()

    data_list = [test_data, train_data]
    labels = ["test", "train"]

    hf = h5py.File(file_dir, 'w')

    for data_set, label in zip(data_list, labels):

        data, pred_samples, mean, var, logits, teacher_logits, teacher_predictions, targets = \
            [], [], [], [], [], [], [], []

        data_loader = torch.utils.data.DataLoader(data_set.set,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=0)

        for batch in data_loader:
            inputs, labels = batch
            img = inputs[0].to(distilled_model.device)
            data.append(img.data.numpy())
            targets.append(labels.data.numpy())
            teacher_logits.append(inputs[2].data.numpy())
            teacher_predictions.append(inputs[1].data.numpy())

            m, v, _, logs, probs = distilled_model.predict(img, return_raw_data=True, return_logits=True)
            mean.append(m.data.numpy())
            var.append(v.data.numpy())
            logits.append(logs.data.numpy())

            pred_samples.append(probs.data.numpy())

        data = np.concatenate(data, axis=0)
        pred_samples = np.concatenate(pred_samples, axis=0)
        teacher_logits = np.concatenate(teacher_logits, axis=0)
        teacher_predictions = np.concatenate(teacher_predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        mean = np.concatenate(mean, axis=0)
        var = np.concatenate(var, axis=0)
        logits = np.concatenate(logits, axis=0)
        preds = np.argmax(np.mean(pred_samples, axis=1), axis=-1)

        # Check accuracy
        acc = np.mean(preds == targets)
        LOGGER.info("Accuracy on {} data set is: {}".format(label, acc))

        # Check accuracy relative teacher
        teacher_preds = np.argmax(np.mean(teacher_predictions, axis=1), axis=-1)
        rel_acc = np.mean(preds == teacher_preds)
        LOGGER.info("Accuracy on {} data set relative teacher is: {}".format(label, rel_acc))

        grp = hf.create_group(label)
        grp.create_dataset("data", data=data)
        grp.create_dataset("predictions", data=pred_samples)
        grp.create_dataset("teacher-logits", data=teacher_logits)
        grp.create_dataset("teacher-predictions", data=teacher_predictions)
        grp.create_dataset("targets", data=targets)
        grp.create_dataset("mean", data=mean)
        grp.create_dataset("var", data=var)
        grp.create_dataset("logits", data=logits)


def predictions_corrupted_data_gaussian(model_dir="models/distilled_model_cifar10",
                                        file_dir="../../dataloaders/data/distilled_model_predictions.h5"):
    """Make predictions on corrupted data with distilled model at model_dir"""

    args = utils.parse_args()

    # Load model
    ensemble = ensemble_wrapper.EnsembleWrapper(output_size=10)

    distilled_model = cifar_resnet_distilled.CifarResnetLogits(ensemble,
                                                               resnet_utils.Bottleneck,
                                                               [2, 2, 2, 2],
                                                               learning_rate=args.lr)

    distilled_model.load_state_dict(torch.load(model_dir, map_location=distilled_model.device))

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
            predictions, targets, logits, mean, var, raw_output = [], [], [], [], [], []

            for j, batch in enumerate(dataloader):
                inputs, labels = batch
                targets.append(labels.data.numpy())
                # data.append(inputs.data.numpy())

                inputs, labels = inputs.to(distilled_model.device), labels.to(distilled_model.device)

                m, v, raw, logs, preds = distilled_model.predict(inputs, return_raw_data=True, return_logits=True,
                                                                 comp_fix=True)
                logits.append(logs.data.numpy())
                mean.append(m.data.numpy())
                var.append(v.data.numpy())
                raw_output.append(raw.data.numpy())

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

            logits = np.concatenate(logits, axis=0)
            sub_grp.create_dataset("logits", data=logits)

            mean = np.concatenate(mean, axis=0)
            sub_grp.create_dataset("mean", data=mean)

            var = np.concatenate(var, axis=0)
            sub_grp.create_dataset("var", data=var)

            raw_output = np.concatenate(raw_output, axis=0)
            sub_grp.create_dataset("raw_output", data=raw_output)

    hf.close()


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))

    train_distilled_network_gaussian()
    predictions_corrupted_data_gaussian()
    #predictions_gaussian()


if __name__ == "__main__":
    main()
