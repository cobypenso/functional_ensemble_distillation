import logging
import torch
import h5py
import numpy as np


class Cifar10DataPredictions:
    """Saved predictions for (corrupted) CIFAR10 data, wrapper. Files created through distillation
    experiments from src.experiments.cifar10 or obtained from Ovadia et. al. (2019)
    """

    def __init__(self, model, corruption, intensity, rep=None, data_dir="data/", ensemble_indices=None,
                 extract_logits_info=False):

        self._log = logging.getLogger(self.__class__.__name__)

        model_list = ["distilled", "dropout", "dropout_nofirst", "ensemble", "ensemble_new", "ll_dropout", "ll_svi",
                      "svi", "temp_scaling", "vanilla", "mixture_distill", "dirichlet_distill"]
        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                           "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "pixelate",
                           "saturate", "shot_noise", "spatter", "speckle_noise", "zoom_blur", "test"]
        intensity_list = [0, 1, 2, 3, 4, 5]

        if (model not in model_list) or (intensity not in intensity_list) or (corruption not in corruption_list):
            print("Data not found: model, corruption or intensity does not exist")

        elif rep is not None and rep > 5:
            print("Rep has to be between 1 and 5")

        else:

            if model == "ensemble_new":
                # Ensemble file created from own source

                if intensity == 0:
                    filepath = data_dir + "ensemble_predictions/ensemble_predictions.h5"
                else:
                    filepath = data_dir + "ensemble_predictions/ensemble_predictions_corrupted_data.h5"

                with h5py.File(filepath, 'r') as f:

                        if intensity == 0:
                            sub_grp = f["test"]
                        else:
                            grp = f[corruption]
                            sub_grp = grp["intensity_" + str(intensity)]

                        predictions = sub_grp["predictions"][()]
                        targets = sub_grp["targets"][()]

                        if extract_logits_info:
                            self.logits = sub_grp["logits"][()]

                if rep is not None:

                    if ensemble_indices is None:
                        ensemble_size = 10
                        predictions = predictions[:, (rep-1)*ensemble_size:rep*ensemble_size, :]

                    else:
                        predictions = predictions[:, ensemble_indices, :]

                else:
                    targets = np.repeat([targets], 5, axis=0).reshape(-1)

            elif model in ["distilled", "mixture_distill", "dirichlet_distill"]:

                spec = ""
                if model == "mixture_distill":
                    spec = "mixture_"
                elif model == "dirichlet_distill":
                    spec = "dirichlet_"

                filepath = data_dir + "distilled_model_" + spec + str(rep) + "_predictions_corrupted_data.h5"

                with h5py.File(filepath, 'r') as f:

                    if intensity == 0:
                        grp = f["test"]
                    else:
                        grp = f[corruption]

                    sub_grp = grp["intensity_" + str(intensity)]

                    predictions = sub_grp["predictions"][()]
                    targets = sub_grp["targets"][()]

                    if extract_logits_info:
                        self.mean = sub_grp["mean"][()]
                        self.var = sub_grp["var"][()]

            else:

                filepath = data_dir + "cifar_model_predictions.hdf5"

                with h5py.File(filepath, 'r') as f:
                    grp = f[model]

                    if intensity == 0:
                        sub_grp = grp["test"]
                    else:
                        sub_grp = grp["corrupt-static-" + corruption + "-" + str(intensity)]

                    predictions = sub_grp["probs"][()]
                    targets = np.squeeze(sub_grp["labels"][()])

                    if rep is None:
                        predictions = predictions.reshape(-1, predictions.shape[-1])
                        targets = targets.reshape(-1)

                    else:
                        predictions = predictions[rep-1, :, :]
                        targets = targets[rep-1, :]

            self.set = CustomSet(predictions, targets)

            self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                            "horse", "ship", "truck")
            self.num_classes = len(self.classes)

            self.set = CustomSet(predictions, targets)


class CustomSet:

    def __init__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        self.length = self.predictions.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (prediction, label) where label is index of the target class.
        """
        preds = torch.tensor(self.predictions[index, :])
        target = torch.tensor(self.targets[index], dtype=torch.int64)

        return preds, target


def main():
    """Entry point for debug visualisation"""
    data = Cifar10DataPredictions("dropout", "brightness", 1)
    loader = torch.utils.data.DataLoader(data.set,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)
    dataiter = iter(loader)
    predictions, targets = dataiter.next()
    acc = np.mean(np.argmax(predictions.data.numpy(), axis=-1) == targets.data.numpy())
    print("Accuracy is: {}".format(acc))


if __name__ == "__main__":
    main()
