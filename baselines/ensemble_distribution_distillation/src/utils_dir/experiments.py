"""Utilities for experiments"""
import sys
import re
from datetime import datetime
import numpy as np
import git

ACTIVE_BRANCH_REGEX = re.compile(r"(.+)/(.+)/(.+)")


def experiment_info(model, args, misc=None):
    repo = git.Repo(search_parent_directories=True)
    git_info = {
        "branch": repo.active_branch.name,
        "commit_hash": repo.head.object.hexsha
    }
    return {
        "date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info,
        "model": model.info(),
        "args": _args_to_string(args),
        "misc": misc
    }


def _args_to_string(args):
    return {
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "ensemble_size": args.num_ensemble_members,
        "seed": args.seed,
        "gpu": args.gpu
    }


def unpack_results(result):
    """Specific experiment function"""
    ens, dist = result
    ens, ens_header = gen_arr(ens)
    dist, dist_header = gen_arr(dist)
    return (ens, ens_header), (dist, dist_header)


def gen_arr(sub_result):
    """Specific experiment function"""
    (rmse, nll, ause) = sub_result

    return np.column_stack((rmse, nll, ause)), ["rmse", "nll", "ause"]


def csv_result(result, header=False, file=None):
    if file is None:
        file_ens = sys.stdout
        file_dist = sys.stdout
    else:
        file_ens = file.parent / (file.stem + "_ens" + file.suffix)
        file_dist = file.parent / (file.stem + "_dist" + file.suffix)
    delimiter = ";"
    (ens, ens_header), (dist, dist_header) = unpack_results(result)
    np.savetxt(file_ens, ens, delimiter=";", header=delimiter.join(ens_header))
    np.savetxt(file_dist,
               dist,
               delimiter=";",
               header=delimiter.join(dist_header))
