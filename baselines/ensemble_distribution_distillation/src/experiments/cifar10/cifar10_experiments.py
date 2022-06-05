import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.lines as matplot_lines
from matplotlib import pyplot as plt
import tikzplotlib

from src import utils
from src import metrics
from src.dataloaders import cifar10_benchmark_model_predictions


LOGGER = logging.getLogger(__name__)


def make_boxplot(data_list, file_dir, label="ACC", model_list=None, colors=None, max_y=1.0):
    """Make boxplot over data in data_list
    M = number of models, N = number of data_sets, I = number of intensities, R = number of repetitions
    data_list: list of length I with matrices of size (N*R, M)
    """

    num_intensities = len(data_list)
    xlab = np.arange(0, num_intensities)
    fig, axis = plt.subplots(2, int(np.ceil(num_intensities/2)))

    for i, (data, ax) in enumerate(zip(data_list, axis.reshape(-1)[0:num_intensities])):

        bplot = ax.boxplot(data, whis=10000, patch_artist=True)  # Setting "whis" to high value forces plot to show min/max

        if i == 0 or i == 3:
            ax.set_ylabel(label)

        ax.set_xlabel("Intensity " + str(xlab[i]))

        if colors is not None:
            assert len(colors) == data.shape[-1]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        if i == 0:
            for l in range(len(colors)):
                plt.setp(bplot['boxes'][l], color=colors[l])
                plt.setp(bplot['medians'][l], color=colors[l])
        else:
            plt.setp(bplot['medians'], color='black')

        if colors is not None:
            custom_lines = []
            for color in colors:
                custom_lines.append(matplot_lines.Line2D([0], [0], color=color, lw=2))

        if i == 2:
            if model_list is not None:
                assert len(model_list) == len(custom_lines)
                ax.legend(custom_lines, model_list, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_ylim([0.0, max_y])

        tikzplotlib.save(file_dir)

    plt.show()


def ood_data_experiment():
    """Repeat experiment from Ovidia et al. (2019), plotting boxplots with accuracy and ece on corrupted data"""

    data_dir = "../../dataloaders/data/"

    model_list = ["distilled", "dirichlet_distill", "mixture_distill", "ensemble_new", "vanilla", "temp_scaling",
                  "dropout_nofirst", "ll_dropout", "svi", "ll_svi"]
    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                       "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "pixelate",
                       "saturate", "shot_noise", "spatter", "speckle_noise", "zoom_blur"]
    intensity_list = [0, 1, 2, 3, 4, 5]
    rep_list = [1, 2, 3, 4, 5]
    ensemble_inds = np.load("data/ensemble_indices.npy")
    ensemble_size = 10

    acc_list = []
    ece_list = []
    for intensity in intensity_list:
        print(intensity)
        acc = np.zeros((len(corruption_list), len(rep_list), len(model_list)))
        ece = np.zeros((len(corruption_list), len(rep_list), len(model_list)))

        for i, corruption in enumerate(corruption_list):
            print(corruption)
            for j, model in enumerate(model_list):

                for k, rep in enumerate(rep_list):
                    if model == "ensemble_new":
                        inds = ensemble_inds[((rep-1) * ensemble_size):(rep * ensemble_size)]
                    else:
                        inds = None

                    data = cifar10_benchmark_model_predictions.Cifar10DataPredictions(model=model,
                                                                                      corruption=corruption,
                                                                                      intensity=intensity,
                                                                                      data_dir=data_dir,
                                                                                      rep=rep,
                                                                                      ensemble_indices=inds)

                    if model in ["distilled", "ensemble_new", "dirichlet_distill"]:
                        data.set.predictions = np.mean(data.set.predictions, axis=1)

                    acc[i, k-1, j] = metrics.accuracy(torch.tensor(data.set.predictions),
                                                      torch.tensor(data.set.targets, dtype=torch.long))
                    ece[i, k-1, j] = metrics.ece(data.set.predictions, data.set.targets)

        acc_list.append(acc.reshape(-1, acc.shape[-1]))
        ece_list.append(ece.reshape(-1, ece.shape[-1]))

    model_list_text = ["Gaussian Distilled", "Dirichlet Distilled", "Mixture Distilled", "Ensemble", "Vanilla",
                       "Temp Scaling", "Dropout", "LL Dropout", "SVI", "LL SVI"]
    colors = ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6",
              "#6A3D9A"]
    make_boxplot(acc_list, "data/fig/acc_benchmark_experiments_test.tikz", model_list=model_list_text, colors=colors)
    make_boxplot(ece_list, "data/fig/ece_benchmark_experiments_test.tikz", label="ECE", model_list=model_list_text,
                 colors=colors, max_y=0.8)


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    ood_data_experiment()


if __name__ == "__main__":
    main()
