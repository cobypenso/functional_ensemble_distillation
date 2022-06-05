"""
https://github.com/facebookresearch/detectron2
"""
import os
import sys
sys.path.append('./')
import logging
import PIL
import torch
import torch.nn as nn
import torchvision
from tabulate import tabulate

import engine.utils as utils



__all__ = [
    "create_ddp_model",
    "default_setup",
]


def create_ddp_model(
        model: nn.Module,
        *,
        fp16_compression=False,
        **kwargs,
    ) -> nn.Module:
    """
    Create a DistributedDataParallel model if there exists more than one process.

    Args:
        model (nn.Module)
        fp16_compression (bool)
        kwargs: arguments of `DistributedDataParallel`
    """
    if utils.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [utils.get_local_rank()]
    ddp = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def default_setup(cfg, args, rank=None):
    """
    Perform some basic common setups at the beginning of a job.

    Args:
        cfg (CfgNode)
        args (argparse.NameSpace)
        rank (int)
    """
    if rank is None:
        rank = utils.get_rank()
    
    logger = logging.getLogger("giung2")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger_fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging for master
    if rank == 0:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logger_fmt)
        logger.addHandler(sh)

    # file logging for all
    filename = os.path.join(cfg.OUTPUT_DIR, "log.txt")
    if rank > 0:
        filename = os.path.join(cfg.OUTPUT_DIR, f"log.txt.rank{rank}")
    fh = logging.FileHandler(filename)
    fh.setFormatter(logger_fmt)
    logger.addHandler(fh)

    # log environments
    logger.info("Rank of current process: {}. World size: {}".format(rank, utils.get_world_size()))

    log_str = "Development Environments:\n"
    log_str += tabulate([
        ("sys.platform", sys.platform),
        ("Python", sys.version.replace("\n", "")),
        ("PyTorch", torch.__version__ + " @" + os.path.dirname(torch.__file__)),
        ("PyTorch debug build", torch.version.debug),
        ("torchvision", torchvision.__version__ + " @" + os.path.dirname(torchvision.__file__)),
        ("Pillow", PIL.__version__),
    ]) + "\n"
    log_str += torch.__config__.show()
    logger.info(log_str)

    log_str = f"Configuration from '{args.config_file}':\n"
    log_str += cfg.dump()
    logger.info(log_str)

    log_str = "Command Line Arguments:\n"
    log_str += str(args) + "\n"
    logger.info(log_str)

    if utils.get_rank() == 0:
        path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    seed = cfg.SEED
    utils.seed_all_rng(None if seed < 0 else seed + rank)
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN_DETERMINISTIC
