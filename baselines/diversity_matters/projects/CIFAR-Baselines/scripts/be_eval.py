import os
import sys
import torch
import argparse
from tqdm import tqdm
from tabulate import tabulate
sys.path.append("./")


from giung2.config import get_cfg
from giung2.data.build import build_dataloaders
from giung2.modeling.build import build_model
from giung2.evaluation import (
    evaluate_acc, evaluate_nll, evaluate_ece,
    compute_ent, compute_kld, get_optimal_temperature,
)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=None, required=True, metavar="FILE",
                        help="path to config file")
    parser.add_argument("--weight-file", default=None, required=True, metavar="FILE",
                        help="path to weight file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options at the end of the command.")
    args = parser.parse_args()
    print("Command Line Args:", args)
    print()

    # load config file
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file, allow_unsafe=True)
    cfg.merge_from_list(args.opts)
    cfg.NUM_GPUS = 1

    # build dataloaders
    dataloaders = build_dataloaders(cfg, root="./datasets")

    # build model
    model = build_model(cfg).cuda().eval()
    model.load_state_dict(
        torch.load(
            args.weight_file, map_location="cpu"
        )["model_state_dict"]
    )

    # disable grad
    torch.set_grad_enabled(False)

    # make predictions on valid split
    true_labels = []
    pred_logits = []
    for images, labels in tqdm(dataloaders["val_loader"], desc="valid examples", leave=False):
        images = images.repeat(cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE, 1, 1, 1)
        logits = model(images)["logits"]
        logits = torch.stack(
            torch.split(
                logits, logits.size(0) // cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE
            ), dim=1
        )
        pred_logits.append(logits.cpu())
        true_labels.append(labels.cpu())
    val_pred_logits = torch.cat(pred_logits) # [num_examples, ens_size, num_classes]
    val_true_labels = torch.cat(true_labels) # [num_examples,]
    val_confidences = torch.softmax(val_pred_logits, dim=2)

    # make predictions on test split
    true_labels = []
    pred_logits = []
    for images, labels in tqdm(dataloaders["tst_loader"], desc="test examples", leave=False):
        images = images.repeat(cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE, 1, 1, 1)
        logits = model(images)["logits"]
        logits = torch.stack(
            torch.split(
                logits, logits.size(0) // cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE
            ), dim=1
        )
        pred_logits.append(logits.cpu())
        true_labels.append(labels.cpu())
    tst_pred_logits = torch.cat(pred_logits) # [num_examples, ens_size, num_classes]
    tst_true_labels = torch.cat(true_labels) # [num_examples,]
    tst_confidences = torch.softmax(tst_pred_logits, dim=2)

    # evaluate standard metrics
    DATA = []
    for member_idx in tqdm(range(cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE), desc="standard metrics", leave=False):
        DATA.append([
            f"Member #{member_idx}",
            evaluate_acc(tst_confidences[:, member_idx, :], tst_true_labels),
            evaluate_nll(tst_confidences[:, member_idx, :], tst_true_labels),
            evaluate_ece(tst_confidences[:, member_idx, :], tst_true_labels),
            compute_ent( tst_confidences[:, member_idx, :]                 ),
            0.0
        ])
    DATA.append([
        "Ensemble",
        evaluate_acc(tst_confidences.mean(1), tst_true_labels),
        evaluate_nll(tst_confidences.mean(1), tst_true_labels),
        evaluate_ece(tst_confidences.mean(1), tst_true_labels),
        compute_ent( tst_confidences.mean(1)                 ),
        compute_kld( tst_confidences                         ),
    ])
    print("Standard metrics on test examples:")
    print(tabulate(DATA, headers=["Label", "ACC", "NLL", "ECE", "ENT", "KLD"], floatfmt=".4f"))
    print()

    # evaluate calibrated metrics
    DATA = []
    for member_idx in tqdm(range(cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE), desc="calibrated metrics", leave=False):
        t_opt = get_optimal_temperature(
            confidences = val_confidences[:, member_idx, :],
            true_labels = val_true_labels,
        )
        _tst_confidences = torch.softmax(
            torch.log(1e-12 + tst_confidences[:, member_idx, :]) / t_opt, dim=1
        )
        DATA.append([
            f"Member #{member_idx}",
            evaluate_acc(_tst_confidences, tst_true_labels),
            evaluate_nll(_tst_confidences, tst_true_labels),
            evaluate_ece(_tst_confidences, tst_true_labels),
            compute_ent( _tst_confidences                 ),
            t_opt,
        ])
    t_opt = get_optimal_temperature(
        confidences = val_confidences.mean(1),
        true_labels = val_true_labels,
    )
    _tst_confidences = torch.softmax(
        torch.log(1e-12 + tst_confidences.mean(1)) / t_opt, dim=1
    )
    DATA.append([
        "Ensemble",
        evaluate_acc(_tst_confidences, tst_true_labels),
        evaluate_nll(_tst_confidences, tst_true_labels),
        evaluate_ece(_tst_confidences, tst_true_labels),
        compute_ent( _tst_confidences                 ),
        t_opt,
    ])
    print("Calibrated metrics on test examples:")
    print(tabulate(DATA, headers=["# Ens", "ACC", "NLL", "ECE", "ENT", "TS"], floatfmt=".4f"))
    print()
