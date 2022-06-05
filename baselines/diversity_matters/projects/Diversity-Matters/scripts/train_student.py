import os
import sys
import shutil
sys.path.append("./")


import torch
import torch.nn.functional as F
import logging
import argparse
from tabulate import tabulate


from giung2.config import get_cfg
from giung2.engine import launch, create_ddp_model, default_setup
from giung2.engine.utils import synchronize, get_rank
from giung2.data.build import build_dataloaders
from giung2.modeling.build import build_model
from giung2.solver.build import build_optimizer, build_scheduler


def train(args, cfg, logger, dataloaders, model, teacher_models):

    # build optimizer
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # build DDP
    model = create_ddp_model(
        model=model,
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    )

    # setup training
    epoch_idx = 0
    best_acc1, best_acc1_epoch = float( 0.0 ), int( -1 )
    best_loss, best_loss_epoch = float("inf"), int( -1 )
    while epoch_idx < cfg.SOLVER.NUM_EPOCHS:

        # ---------------------------------------------------------------------- #
        # Training
        # ---------------------------------------------------------------------- #
        epoch_idx += 1
        run_epoch(
            args, cfg, epoch_idx, model, dataloaders, logger,
            optimizer=optimizer, scheduler=scheduler, is_train=True, teacher_models=teacher_models,
        )

        # ---------------------------------------------------------------------- #
        # Validation
        # ---------------------------------------------------------------------- #
        _valid_epochs = []
        _valid_epochs += [1,]
        _valid_epochs += list( range(0, cfg.SOLVER.NUM_EPOCHS - cfg.SOLVER.VALID_FINALE + 1, cfg.SOLVER.VALID_FREQUENCY) )
        _valid_epochs += list( range(cfg.SOLVER.NUM_EPOCHS - cfg.SOLVER.VALID_FINALE, cfg.SOLVER.NUM_EPOCHS + 1, 1) )
        if epoch_idx in _valid_epochs:
            val_loss, val_acc1 = run_epoch(
                args, cfg, epoch_idx, model, dataloaders, logger,
                optimizer=None, scheduler=None, is_train=False, teacher_models=teacher_models,
            )

            if get_rank() == 0:
                is_best_loss = val_loss < best_loss
                is_best_acc1 = val_acc1 > best_acc1
                best_loss = min(val_loss, best_loss)
                best_acc1 = max(val_acc1, best_acc1)

                checkpoint = {
                    "epoch_idx": epoch_idx,
                    "model_state_dict": model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc1": best_acc1,
                    "best_loss": best_loss,
                }

                if args.checkpoint_last_only:
                    filename = os.path.join(cfg.OUTPUT_DIR, "checkpoint.pth.tar".format(epoch_idx))
                else:
                    filename = os.path.join(cfg.OUTPUT_DIR, "model_e{:03d}.pth.tar".format(epoch_idx))
                torch.save(checkpoint, filename)
                logger.info(f"[Checkpoint Epoch {epoch_idx}] Save {filename}")

                if is_best_loss:
                    best_loss_epoch = epoch_idx
                    bestname = os.path.join(cfg.OUTPUT_DIR, "best_loss.pth.tar")
                    shutil.copyfile(filename, bestname)
                    logger.info(f"[Checkpoint Epoch {epoch_idx}] Save {bestname}")

                if is_best_acc1:
                    best_acc1_epoch = epoch_idx
                    bestname = os.path.join(cfg.OUTPUT_DIR, "best_acc1.pth.tar")
                    shutil.copyfile(filename, bestname)
                    logger.info(f"[Checkpoint Epoch {epoch_idx}] Save {bestname}")

    # log the best achievement
    log_dict = {
        "best_acc1"      : best_acc1,
        "best_acc1_epoch": best_acc1_epoch,
        "best_loss"      : best_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    log_str = "Summary: best_acc1={:.4f} @ best_acc1_epoch={:d}, best_loss={:.4f} @ best_loss_epoch={:d}".format(
        log_dict["best_acc1"], log_dict["best_acc1_epoch"], log_dict["best_loss"], log_dict["best_loss_epoch"],
    )
    logger.info(log_str)


def run_epoch(args, cfg, epoch_idx, model, dataloaders, logger,
              optimizer=None, scheduler=None, is_train=False, teacher_models=None):

    prev_grad_enabled = torch.is_grad_enabled()

    if is_train:
        model.train()
        dataloader = dataloaders["dataloader"]
        log_str = "[Train Epoch {:d}/{:d}] Start training.".format(epoch_idx, cfg.SOLVER.NUM_EPOCHS)
        if cfg.NUM_GPUS > 1:
            dataloader.sampler.set_epoch(epoch_idx)

    else:
        model.eval()
        dataloader = dataloaders["val_loader"]
        log_str = "[Valid Epoch {:d}/{:d}] Start validation.".format(epoch_idx, cfg.SOLVER.NUM_EPOCHS)

    logger.info(log_str)

    def update_meter(meter, v, n):
        meter["sum"] += v * n
        meter["cnt"] += n
        meter["avg"] = meter["sum"] / meter["cnt"]
    loss_meter = {"sum": 0.0, "avg": 0.0, "cnt": 0,}
    loss_ce_meter = {"sum": 0.0, "avg": 0.0, "cnt": 0,}
    loss_kd_meter = {"sum": 0.0, "avg": 0.0, "cnt": 0,}
    acc1_meter = {"sum": 0.0, "avg": 0.0, "cnt": 0,}
    acc5_meter = {"sum": 0.0, "avg": 0.0, "cnt": 0,}

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):

        torch.set_grad_enabled(is_train)

        images = images.cuda()
        labels = labels.cuda()
        ensemble_size = cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE

        if args.kd_method_name is None:
            
            # preprocess batches
            labels = labels.repeat(ensemble_size)
            images = images.repeat(ensemble_size, 1, 1, 1)
            images_split = torch.split(images, images.size(0) // ensemble_size)
            
            # forward networks
            with torch.no_grad():
                t_logits = torch.cat([
                    _model(_images)["logits"] for _model, _images in zip(teacher_models, images_split)
                ])
            outputs = model(images, labels)
            
            # compute loss
            loss_ce = F.cross_entropy(outputs["logits"], outputs["labels"], reduction="mean")
            loss_kd = F.kl_div(
                F.log_softmax(outputs["logits"] / args.kd_temperature, dim=1),
                F.softmax(t_logits / args.kd_temperature, dim=1).detach(),
                log_target=False, reduction="batchmean",
            ) * args.kd_temperature * args.kd_temperature
            loss = (1 - args.kd_alpha) * loss_ce + args.kd_alpha * loss_kd

        elif args.kd_method_name == "gaussian":
            
            # make Gaussian-perturbed batches
            eps = torch.randn_like(images) * (args.kd_method_step_size / 255.0)
            images_eps = torch.clamp(images + eps, min=0.0, max=1.0).repeat(ensemble_size, 1, 1, 1)
            images_eps_split = torch.split(images_eps, images_eps.size(0) // ensemble_size)

            # preprocess batches
            labels = labels.repeat(ensemble_size)
            images = images.repeat(ensemble_size, 1, 1, 1)
            
            # forward networks
            with torch.no_grad():
                t_logits = torch.cat([
                    _model(_images)["logits"] for _model, _images in zip(teacher_models, images_eps_split)
                ])
            outputs = model(images, labels)
            outputs_eps = model(images_eps, labels)
            
            # compute loss
            loss_ce = F.cross_entropy(outputs["logits"], outputs["labels"], reduction="mean")
            loss_kd = F.kl_div(
                F.log_softmax(outputs_eps["logits"] / args.kd_temperature, dim=1),
                F.softmax(t_logits / args.kd_temperature, dim=1).detach(),
                log_target=False, reduction="batchmean",
            ) * args.kd_temperature * args.kd_temperature
            loss = (1 - args.kd_alpha) * loss_ce + args.kd_alpha * loss_kd

        elif args.kd_method_name in ["ods_l2", "c_ods_l2"]:

            # make ODS-perturbed batches
            torch.set_grad_enabled(True)

            x = images.detach().clone()
            x.requires_grad = True
            x.grad = None
            
            teacher_idx = torch.randint(0, ensemble_size, (1,)).item()
            teacher_outputs = teacher_models[teacher_idx](x)
            y = torch.softmax(teacher_outputs["logits"] / args.kd_temperature, dim=1)
            
            grad = torch.autograd.grad(
                torch.sum(torch.empty_like(y).uniform_(-1.0, 1.0) * y), x,
                retain_graph=True, create_graph=True,
            )[0].detach()
            grad = (
                x.view(x.size(0), -1).size(1) ** 0.5
            ) * F.normalize(
                grad.view(grad.size(0), -1), p=2, dim=1
            ).view(grad.size())
            
            eps = (args.kd_method_step_size / 255.0) * grad
            if args.kd_method_name == "c_ods_l2":
                eps *= torch.max(y, dim=1)[0].view(-1, 1, 1, 1)
            images_eps = torch.clamp(images + eps, min=0.0, max=1.0).detach().repeat(ensemble_size, 1, 1, 1)
            images_eps_split = torch.split(images_eps, images_eps.size(0) // ensemble_size)
            
            torch.set_grad_enabled(is_train)

            # preprocess batches
            labels = labels.repeat(ensemble_size)
            images = images.repeat(ensemble_size, 1, 1, 1)
            
            # forward networks
            with torch.no_grad():
                t_logits = torch.cat([
                    _model(_images)["logits"] for _model, _images in zip(teacher_models, images_eps_split)
                ])
            outputs = model(images, labels)
            outputs_eps = model(images_eps, labels)
            
            # compute loss
            loss_ce = F.cross_entropy(outputs["logits"], outputs["labels"], reduction="mean")
            loss_kd = F.kl_div(
                F.log_softmax(outputs_eps["logits"] / args.kd_temperature, dim=1),
                F.softmax(t_logits / args.kd_temperature, dim=1).detach(),
                log_target=False, reduction="batchmean",
            ) * args.kd_temperature * args.kd_temperature
            loss = (1 - args.kd_alpha) * loss_ce + args.kd_alpha * loss_kd

        else:
            raise NotImplementedError()

        # optimize weights
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.set_grad_enabled(False)

        # compute accuracies
        _, pred = outputs["logits"].topk(5, 1, True, True)
        correct = pred.t().eq(outputs["labels"].view(1, -1).expand_as(pred.t()))
        acc1 = correct[:1].reshape(-1).float().sum(0, keepdim=True).div_(outputs["labels"].size(0)).item()
        acc5 = correct[:5].reshape(-1).float().sum(0, keepdim=True).div_(outputs["labels"].size(0)).item()
        loss = loss.detach().item()

        # update meters
        update_meter(loss_meter, loss, outputs["labels"].size(0))
        update_meter(loss_ce_meter, loss_ce, outputs["labels"].size(0))
        update_meter(loss_kd_meter, loss_kd, outputs["labels"].size(0))
        update_meter(acc1_meter, acc1, outputs["labels"].size(0))
        update_meter(acc5_meter, acc5, outputs["labels"].size(0))

        # log train progress
        _logging_batches = []
        _logging_batches += list(range(0, len(dataloader) + 1, len(dataloader) // cfg.LOG_FREQUENCY))
        _logging_batches += [len(dataloader),]
        if is_train and (batch_idx in _logging_batches):
            log_str = "[Train Epoch {:d}/{:d}] [Batch {:d}/{:d}] ".format(
                epoch_idx, cfg.SOLVER.NUM_EPOCHS, batch_idx, len(dataloader)
            )
            log_str += "loss: {:.4e} ({:.4e}), loss_ce: {:.4e} ({:.4e}), loss_kd: {:.4e} ({:.4e}), acc1: {:.4f} ({:.4f}), acc5: {:.4f} ({:.4f}), lr: {:.8f}, max_mem: {:.0f}M".format(
                loss, loss_meter["avg"],
                loss_ce, loss_kd_meter["avg"],
                loss_kd, loss_ce_meter["avg"],
                acc1, acc1_meter["avg"],
                acc5, acc5_meter["avg"],
                scheduler.get_last_lr()[0],
                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            )
            logger.info(log_str)

    # logging
    log_dict = {
        "epoch_idx": epoch_idx,
        "trn/loss" if is_train else "val/loss": loss_meter["avg"],
        "trn/loss_ce" if is_train else "val/loss_ce": loss_ce_meter["avg"],
        "trn/loss_kd" if is_train else "val/loss_kd": loss_kd_meter["avg"],
        "trn/acc1" if is_train else "val/acc1": acc1_meter["avg"],
        "trn/acc5" if is_train else "val/acc5": acc5_meter["avg"],
    }

    log_str = "[{} Epoch {:d}/{:d}] ".format("Train" if is_train else "Valid", epoch_idx, cfg.SOLVER.NUM_EPOCHS)
    log_str += "loss: {:.4e}, loss_ce: {:.4e}, loss_kd: {:.4e}, acc1: {:.4f}, acc5: {:.4f}".format(
        log_dict["trn/loss" if is_train else "val/loss"],
        log_dict["trn/loss_ce" if is_train else "val/loss_ce"],
        log_dict["trn/loss_kd" if is_train else "val/loss_kd"],
        log_dict["trn/acc1" if is_train else "val/acc1"],
        log_dict["trn/acc5" if is_train else "val/acc5"],
    )
    logger.info(log_str)

    synchronize()

    if is_train:
        scheduler.step()
    torch.set_grad_enabled(prev_grad_enabled)

    return loss_meter["avg"], acc1_meter["avg"]


def main(args, cfg):

    default_setup(cfg, args)
    logger = logging.getLogger("giung2")

    # build dataloaders
    dataloaders = build_dataloaders(cfg, root="./datasets")
    log_str = "Build dataloaders:\n"
    log_str += tabulate([
        (
            k,
            len(dataloaders[k].dataset),
            len(dataloaders[k]),
            dataloaders[k].batch_size,
            type(dataloaders[k].sampler).__name__,
        ) for k in dataloaders
    ], headers=["Key", "# Examples", "# Batches", "Batch Size", "Sampler"])
    logger.info(log_str + "\n")

    # overview dataloaders
    for k in dataloaders:
        log_str = f"Overview {k}:\n"
        log_str += dataloaders[k].dataset.describe()
        logger.info(log_str + "\n")

    # build model
    model = build_model(cfg).cuda()
    log_str = "Build networks:\n"
    log_str += str(model)
    logger.info(log_str)

    # build teacher models
    _cfg = get_cfg()
    _cfg.merge_from_file(args.kd_teacher_config_file, allow_unsafe=True)
    ensemble_size = cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE
    teacher_models = []
    for idx in range(ensemble_size):
        teacher_models.append(build_model(_cfg))
        teacher_models[idx].load_state_dict(
            torch.load(
                os.path.join(
                    os.path.dirname(f"{args.kd_teacher_weight_file}")[:-1] + f"{idx}",
                    os.path.basename(f"{args.kd_teacher_weight_file}")
                ), map_location="cpu"
            )["model_state_dict"]
        )
        # teacher_models[idx].cuda().eval()
        teacher_models[idx].cpu().eval()

    # train model
    train(args, cfg, logger, dataloaders, model, teacher_models)

    # finished
    logger.info("Finished.")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=None, required=True, metavar="FILE",
                        help="path to config file")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12345", metavar="URL",
                        help="URL for pytorch distributed backend")
    parser.add_argument("--checkpoint-last-only", default=False, action="store_true",
                        help="save 'checkpoint.pth.tar' as the last checkpoint")
    
    # KD settings
    parser.add_argument("--kd-teacher-config-file", default=None, required=True, metavar="FILE")
    parser.add_argument("--kd-teacher-weight-file", default=None, required=True, metavar="FILE")
    parser.add_argument("--kd-alpha", default=0.9, type=float)
    parser.add_argument("--kd-temperature", default=4.0, type=float)
    parser.add_argument("--kd-method-name", default=None, type=str,
                        choices=["gaussian", "ods_l2", "c_ods_l2",])
    parser.add_argument("--kd-method-step-size", default=1.0, type=float,)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options at the end of the command.")
    args = parser.parse_args()
    print("Command Line Args:", args)

    # load config file
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file, allow_unsafe=True)
    cfg.merge_from_list(args.opts)

    # check the number of GPUs
    if cfg.NUM_GPUS > torch.cuda.device_count():
        raise AssertionError(
            f"Try to use cfg.NUM_GPUS={cfg.NUM_GPUS}, "
            f"but there exists only {torch.cuda.device_count()} GPUs..."
        )

    launch(
        main_func=main,
        num_gpus_per_machine=cfg.NUM_GPUS,
        num_machines=1,
        machine_rank=0,
        dist_url=args.dist_url,
        args=(args, cfg,),
    )
