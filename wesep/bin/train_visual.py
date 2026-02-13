# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import re
from pprint import pformat

import fire
import matplotlib.pyplot as plt
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wesep.utils.schedulers as schedulers
from wesep.dataset.dataset import Dataset, tse_collate_fn, tse_collate_fn_2spk
from wesep.models import get_model
from wesep.utils.checkpoint import (
    load_checkpoint,
    load_pretrained_model,
    save_checkpoint,
)
from wesep.utils.executor_av import Executor
from wesep.utils.file_utils import (
    load_speaker_embeddings,
    read_label_file,
    read_vec_scp_file,
)
from wesep.utils.losses import parse_loss
from wesep.utils.utils import parse_config_or_kwargs, set_seed, setup_logger
from wesep.dataset.av_dataset import DatasetUtilsAV, tse_collate_fn_av_2spk

MAX_NUM_log_files = 100  # The maximum number of log-files to be kept
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def train(config="/mnt/afs/250010237/my_projects/wesep_wenet/examples/audio/voxceleb2/config/tse_bsrnn_av.yaml", **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    # print(kwargs)
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get("checkpoint", None)
    if checkpoint is not None:
        checkpoint = os.path.realpath(checkpoint)
    find_unused_parameters = configs.get("find_unused_parameters", False)

    # dist configs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(configs["gpus"][rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl")

    # Log rotation
    model_dir = os.path.join(configs["exp_dir"], "models")
    logger = setup_logger(rank, configs["exp_dir"], gpu, MAX_NUM_log_files)

    print("-------------------", dist.get_rank(), world_size)
    if world_size > 1:
        logger.info("training on multiple gpus, this gpu {}".format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs["exp_dir"]))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split("\n"):
            logger.info(line)

    # seed
    set_seed(configs["seed"] + rank)

    # loss
    criterion = configs.get("loss", None)
    if criterion:
        criterion = parse_loss(criterion)
    else:
        criterion = [
            parse_loss("SISDR"),
        ]
    loss_posi = configs["loss_args"].get(
        "loss_posi",
        [[
            0,
        ]],
    )
    loss_weight = configs["loss_args"].get(
        "loss_weight",
        [[
            1.0,
        ]],
    )
    loss_args = (loss_posi, loss_weight)

    # embeds
    # =========================
    # AV-only: do NOT read utt2spk/spk2utt/enroll files
    # Strategy: one epoch ~= one full pass of CSV (like USEV)
    # =========================
    import math

    def _count_csv_partition(csv_path: str, part: str) -> int:
        # CSV line format: first column is partition, e.g. "train,..."
        n = 0
        with open(csv_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                if ln.split(",")[0] == part:
                    n += 1
        return n

    joint_training = configs["model_args"]["tse_model"].get("joint_training", False)
    multi_task = configs["model_args"]["tse_model"].get("multi_task", False)

    # For your AV pipeline, we don't need speaker label dict unless you do CE multi-task.
    # Keep dict_spk empty; spk_label in av_dataset is placeholder anyway.
    dict_spk = {}

    train_part = configs["dataset_args"].get("partition", "train")
    val_part = configs["dataset_args"].get("val_partition", "val")

    num_train_mix = _count_csv_partition(configs["train_data"], train_part)
    num_val_mix = _count_csv_partition(configs["val_data"], val_part)

    batch_size = configs["dataloader_args"]["batch_size"]
    drop_last = configs["dataloader_args"].get("drop_last", True)

    # sample_num_per_epoch: if set, use it; else use full train partition (one pass)
    if configs["dataset_args"].get("sample_num_per_epoch", 0) > 0:
        sample_num_per_epoch = int(configs["dataset_args"]["sample_num_per_epoch"])
    else:
        sample_num_per_epoch = int(num_train_mix)

    # Each rank only sees ~1/world_size of samples due to DistributedSampler in DataList
    # So epoch_iter should be computed on per-rank sample count.
    per_rank_train = max(1, sample_num_per_epoch // world_size)
    per_rank_val = max(1, num_val_mix // world_size)

    if drop_last:
        epoch_iter = max(1, per_rank_train // batch_size)
        val_iter = max(1, per_rank_val // batch_size)
    else:
        epoch_iter = max(1, math.ceil(per_rank_train / batch_size))
        val_iter = max(1, math.ceil(per_rank_val / batch_size))

    if rank == 0:
        logger.info(f"[AV] train_part={train_part}, val_part={val_part}")
        logger.info(f"[AV] num_train_mix={num_train_mix}, num_val_mix={num_val_mix}")
        logger.info(f"[AV] sample_num_per_epoch={sample_num_per_epoch}")
        logger.info(f"[AV] epoch_iter={epoch_iter}, val_iter={val_iter}")


    # dataset and dataloader
    train_dataset = DatasetUtilsAV(
        data_list_file=configs["train_data"],
        configs=configs["dataset_args"],
        state="train",
        repeat_dataset=configs.get("repeat_dataset", True),
    )
    val_cfg = dict(configs["dataset_args"])
    val_cfg["partition"] = val_cfg.get("val_partition", "val")
    val_dataset = DatasetUtilsAV(
        data_list_file=configs["val_data"],
        configs=val_cfg,
        state="val",
        repeat_dataset=True,
    )

    cue_mode = configs["dataset_args"].get("cue_mode", "both")  # enroll / visual / both

    train_dataloader = DataLoader(
        train_dataset,
        **configs["dataloader_args"],
        collate_fn=lambda b: tse_collate_fn_av_2spk(b, cue_mode=cue_mode),
    )

    val_dataloader = DataLoader(
        val_dataset,
        **configs["dataloader_args"],
        collate_fn=lambda b: tse_collate_fn_av_2spk(b, cue_mode=cue_mode),
    )

    batch_size = configs["dataloader_args"]["batch_size"]
    if configs["dataset_args"].get("sample_num_per_epoch", 0) > 0:
        sample_num_per_epoch = configs["dataset_args"]["sample_num_per_epoch"]
    else:
        sample_num_per_epoch = len(tr_lines) // 2
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    val_iter = val_lines // 2 // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info("epoch iteration number: {}".format(epoch_iter))
        logger.info("val iteration number: {}".format(val_iter))

    # model
    model_list = []
    scheduler_list = []
    optimizer_list = []

    logger.info("<== Model ==>")
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    num_params = sum(param.numel() for param in model.parameters())

    if rank == 0:
        logger.info("tse_model size: {:.2f} M".format(num_params / 1e6))
        # print model
        for line in pformat(model).split("\n"):
            logger.info(line)

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=find_unused_parameters)
    device = torch.device("cuda")

    if rank == 0:
        logger.info("<== TSE Model Loss ==>")
        logger.info("loss criterion is: " + str(configs["loss"]))

    configs["optimizer_args"]["tse_model"]["lr"] = configs["scheduler_args"][
        "tse_model"]["initial_lr"]
    optimizer = getattr(torch.optim, configs["optimizer"]["tse_model"])(
        ddp_model.parameters(), **configs["optimizer_args"]["tse_model"])
    if rank == 0:
        logger.info("<== TSE Model Optimizer ==>")
        logger.info("optimizer is: " + configs["optimizer"]["tse_model"])

    # scheduler
    configs["scheduler_args"]["tse_model"]["num_epochs"] = configs[
        "num_epochs"]
    configs["scheduler_args"]["tse_model"]["epoch_iter"] = epoch_iter
    configs["scheduler_args"]["scale_ratio"] = 1.0

    scheduler = getattr(schedulers, configs["scheduler"]["tse_model"])(
        optimizer, **configs["scheduler_args"]["tse_model"])
    if rank == 0:
        logger.info("<== TSE Model Scheduler ==>")
        logger.info("scheduler is: " + configs["scheduler"]["tse_model"])

    if configs["model_init"]["tse_model"] is not None:
        logger.info("Load initial model from {}".format(
            configs["model_init"]["tse_model"]))
        load_pretrained_model(ddp_model, configs["model_init"]["tse_model"])
    elif checkpoint is None:
        logger.info("Train model from scratch ...")

    for c in criterion:
        c = c.to(device)

    # append to list
    model_list.append(ddp_model)
    optimizer_list.append(optimizer)
    scheduler_list.append(scheduler)
    # scaler = torch.amp.GradScaler('cuda', enabled=configs["enable_amp"])
    # For torch 1.6+
    scaler = torch.cuda.amp.GradScaler(enabled=configs["enable_amp"])

    # If specify checkpoint, load some info from checkpoint.
    if checkpoint is not None:
        load_checkpoint(model_list, optimizer_list, scheduler_list, scaler,
                        checkpoint)
        start_epoch = (
            int(re.findall(r"(?<=checkpoint_)\d*(?=.pt)", checkpoint)[0]) + 1)
        logger.info("Load checkpoint: {}".format(checkpoint))
    else:
        start_epoch = 1
    logger.info("start_epoch: {}".format(start_epoch))

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs["exp_dir"], "config.yaml")
        with open(saved_config_path, "w") as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ["Train/Val", "Epoch", "iter", "Loss", "LR"]
        for line in tp.header(header, width=10, style="grid").split("\n"):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    executor = Executor()
    executor.step = 0

    train_losses = []
    val_losses = []
    for epoch in range(start_epoch, configs["num_epochs"] + 1):
        train_dataset.set_epoch(epoch)

        # train_loss_com
        train_loss, _ = executor.train(
            train_dataloader,
            model_list,
            epoch_iter,
            optimizer_list,
            criterion,
            scheduler_list,
            scaler=scaler,
            epoch=epoch,
            logger=logger,
            enable_amp=configs["enable_amp"],
            clip_grad=configs["clip_grad"],
            log_batch_interval=configs["log_batch_interval"],
            device=device,
            se_loss_weight=loss_args,
            multi_task=multi_task,
            SSA_enroll_prob=configs["dataset_args"].get("SSA_enroll_prob", 0),
            fbank_args=configs["dataset_args"].get('fbank_args', None),
            sample_rate=configs["dataset_args"]['resample_rate'],
            speaker_feat=configs["dataset_args"].get('speaker_feat', True))

        val_loss, _ = executor.cv(
            val_dataloader,
            model_list,
            val_iter,
            criterion,
            epoch=epoch,
            logger=logger,
            enable_amp=configs["enable_amp"],
            log_batch_interval=configs["log_batch_interval"],
            device=device,
        )

        if rank == 0:
            logger.info("Epoch {} Train info train_loss {}".format(
                epoch, train_loss))
            logger.info("Epoch {} Val info val_loss {}".format(
                epoch, val_loss))
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            best_loss = val_loss
            scheduler.best = best_loss
            # plot
            plt.figure()
            plt.title("Loss of Train and Validation")
            x = list(range(start_epoch, epoch + 1))
            plt.plot(x, train_losses, "b-", label="Train Loss", linewidth=0.8)
            plt.plot(x,
                     val_losses,
                     "c-",
                     label="Validation Loss",
                     linewidth=0.8)
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.xticks(range(start_epoch, epoch + 1, 1))
            plt.savefig(
                f"{configs['exp_dir']}/{configs['model']['tse_model']}.png")
            plt.close()

        if rank == 0:
            if (epoch % configs["save_epoch_interval"] == 0
                    or epoch >= configs["num_epochs"] - configs["num_avg"]):
                save_checkpoint(
                    model_list,
                    optimizer_list,
                    scheduler_list,
                    scaler,
                    os.path.join(model_dir, "checkpoint_{}.pt".format(epoch)),
                )
                try:
                    os.symlink(
                        "checkpoint_{}.pt".format(epoch),
                        os.path.join(model_dir, "latest_checkpoint.pt"),
                    )
                except FileExistsError:
                    os.remove(os.path.join(model_dir, "latest_checkpoint.pt"))
                    os.symlink(
                        "checkpoint_{}.pt".format(epoch),
                        os.path.join(model_dir, "latest_checkpoint.pt"),
                    )

    if rank == 0:
        os.symlink(
            "checkpoint_{}.pt".format(configs["num_epochs"]),
            os.path.join(model_dir, "final_checkpoint.pt"),
        )
        logger.info(tp.bottom(len(header), width=10, style="grid"))


if __name__ == "__main__":
    fire.Fire(train)
