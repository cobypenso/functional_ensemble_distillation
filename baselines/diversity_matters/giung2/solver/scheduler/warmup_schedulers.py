import math
from bisect import bisect_right
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


__all__ = [
    "build_warmup_simple_cosine_lr",
    "build_warmup_cyclical_cosine_lr",
    "build_warmup_multi_step_lr",
    "build_warmup_linear_decay_lr",
]


def build_warmup_simple_cosine_lr(optimizer: Optimizer, **kwargs) -> _LRScheduler:

    NUM_EPOCHS    = kwargs.pop("NUM_EPOCHS", None)
    WARMUP_EPOCHS = kwargs.pop("WARMUP_EPOCHS", None)
    WARMUP_METHOD = kwargs.pop("WARMUP_METHOD", None)
    WARMUP_FACTOR = kwargs.pop("WARMUP_FACTOR", None)

    def _lr_sched(epoch):

        # start from one
        epoch += 1

        # warmup
        if epoch < WARMUP_EPOCHS:
            if WARMUP_METHOD == "linear":
                return (1.0 - WARMUP_FACTOR) / (WARMUP_EPOCHS - 1) * (epoch - 1) + WARMUP_FACTOR
            elif WARMUP_METHOD == "constant":
                return WARMUP_FACTOR

        # cosine decays
        else:
            return 0.5 * (
                1.0 + math.cos(
                    math.pi * (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS + 1.0)
                )
            )

    return LambdaLR(optimizer, lr_lambda=_lr_sched)


def build_warmup_cyclical_cosine_lr(optimizer: Optimizer, **kwargs) -> _LRScheduler:

    NUM_EPOCHS    = kwargs.pop("NUM_EPOCHS", None)
    WARMUP_EPOCHS = kwargs.pop("WARMUP_EPOCHS", None)
    WARMUP_METHOD = kwargs.pop("WARMUP_METHOD", None)
    WARMUP_FACTOR = kwargs.pop("WARMUP_FACTOR", None)
    PRETRAIN_EPOCHS = kwargs.pop("PRETRAIN_EPOCHS", None)
    REPEATED_EPOCHS = kwargs.pop("REPEATED_EPOCHS", None)

    def _lr_sched(epoch):

        # start from one
        epoch += 1

        # warmup
        if epoch < WARMUP_EPOCHS:
            if WARMUP_METHOD == "linear":
                return (1.0 - WARMUP_FACTOR) / (WARMUP_EPOCHS - 1) * (epoch - 1) + WARMUP_FACTOR
            elif WARMUP_METHOD == "constant":
                return WARMUP_FACTOR

        # cosine decays
        elif epoch <= REPEATED_EPOCHS[1]:
            return 0.5 * (
                1.0 + math.cos(
                    math.pi * (epoch - WARMUP_EPOCHS) / (PRETRAIN_EPOCHS - WARMUP_EPOCHS + 1.0)
                )
            )

        else:
            return _lr_sched(epoch - REPEATED_EPOCHS[1] + REPEATED_EPOCHS[0] - 2)

    return LambdaLR(optimizer, lr_lambda=_lr_sched)


def build_warmup_multi_step_lr(optimizer: Optimizer, **kwargs) -> _LRScheduler:

    MILESTONES    = kwargs.pop("MILESTONES", None)
    WARMUP_METHOD = kwargs.pop("WARMUP_METHOD", None)
    WARMUP_FACTOR = kwargs.pop("WARMUP_FACTOR", None)
    GAMMA         = kwargs.pop("GAMMA", None)

    def _lr_sched(epoch):

        # start from one
        epoch += 1

        # warmup
        if epoch < MILESTONES[0]:
            if WARMUP_METHOD == "linear":
                return (1.0 - WARMUP_FACTOR) / (MILESTONES[0] - 1) * (epoch - 1) + WARMUP_FACTOR
            elif WARMUP_METHOD == "constant":
                return WARMUP_FACTOR

        # step decays
        else:
            return GAMMA ** bisect_right(MILESTONES[1:], epoch)

    return LambdaLR(optimizer, lr_lambda=_lr_sched)


def build_warmup_linear_decay_lr(optimizer: Optimizer, **kwargs) -> _LRScheduler:

    MILESTONES    = kwargs.pop("MILESTONES", None)
    WARMUP_METHOD = kwargs.pop("WARMUP_METHOD", None)
    WARMUP_FACTOR = kwargs.pop("WARMUP_FACTOR", None)
    GAMMA         = kwargs.pop("GAMMA", None)

    def _lr_sched(epoch):

        # start from one
        epoch += 1

        # warmup
        if epoch < MILESTONES[0]:
            if WARMUP_METHOD == "linear":
                return (1.0 - WARMUP_FACTOR) / (MILESTONES[0] - 1) * (epoch - 1) + WARMUP_FACTOR
            elif WARMUP_METHOD == "constant":
                return WARMUP_FACTOR

        # high constant
        elif epoch < MILESTONES[1]:
            return 1.0

        # linear decay
        elif epoch < MILESTONES[2]:
            return (GAMMA - 1.0) / (MILESTONES[2] - MILESTONES[1]) * (epoch - MILESTONES[2]) + GAMMA

        # low constant
        else:
            return GAMMA

    return LambdaLR(optimizer, lr_lambda=_lr_sched)
