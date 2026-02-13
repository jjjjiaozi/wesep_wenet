# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

from contextlib import nullcontext

import tableprint as tp

# if your python version < 3.7 use the below one
import torch

from wesep.utils.funcs import clip_gradients, compute_fbank, apply_cmvn
import random


class Executor:

    def __init__(self):
        self.step = 0

    def train(self,
              dataloader,
              models,
              epoch_iter,
              optimizers,
              criterion,
              schedulers,
              scaler,
              epoch,
              enable_amp,
              logger,
              clip_grad=5.0,
              log_batch_interval=100,
              device=torch.device("cuda"),
              se_loss_weight=1.0,
              multi_task=False,
              SSA_enroll_prob=0,
              fbank_args=None,
              sample_rate=16000,
              speaker_feat=True):
        """Train one epoch"""
        model = models[0]
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        model.train()
        log_interval = log_batch_interval
        accum_grad = 1
        losses = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):
                features = batch["wav_mix"]
                targets = batch["wav_targets"]
                enroll = batch.get("spk_embeds", None)
                visual_feat = batch.get("visual_feat", None)
                spk_label = batch["spk_label"]

                features = features.float().to(device)
                targets = targets.float().to(device)
                spk_label = spk_label.to(device)

                if enroll is not None:
                    enroll = enroll.float().to(device)
                if visual_feat is not None:
                    visual_feat = visual_feat.float().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    # move / cast (确保 device 和 dtype 一致)
                    features = features.float().to(device)
                    targets = targets.float().to(device)
                    spk_label = spk_label.to(device)

                    if enroll is not None:
                        enroll = enroll.float().to(device)
                    if visual_feat is not None:
                        visual_feat = visual_feat.float().to(device)

                    # SSA only when enroll exists
                    if (enroll is not None) and (SSA_enroll_prob > 0) and (SSA_enroll_prob > random.random()):
                        with torch.no_grad():
                            outputs = model(features, enroll=enroll, visual_feat=visual_feat)
                            if not isinstance(outputs, (list, tuple)):
                                outputs = [outputs]
                            est_speech = outputs[0]

                            # 如果你确定模型能吃 fbank enroll，就保留；否则建议先直接用 est_speech 做 enroll（或直接禁用 SSA）
                            self_enroll = enroll
                            if (fbank_args is not None) and speaker_feat:
                                self_enroll = compute_fbank(est_speech, **fbank_args, sample_rate=sample_rate)
                                self_enroll = apply_cmvn(self_enroll)

                        outputs = model(features, enroll=self_enroll, visual_feat=visual_feat)
                    else:
                        outputs = model(features, enroll=enroll, visual_feat=visual_feat)

                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]

                    loss = 0
                    for ii in range(len(criterion)):
                        for ji in range(len(se_loss_weight[0][ii])):
                            if multi_task and criterion[ii].__class__.__name__ == "CrossEntropyLoss":
                                loss += se_loss_weight[1][ii][ji] * (
                                    criterion[ii](outputs[se_loss_weight[0][ii][ji]], spk_label).mean() / accum_grad
                                )
                            else:
                                loss += se_loss_weight[1][ii][ji] * (
                                    criterion[ii](outputs[se_loss_weight[0][ii][ji]], targets).mean() / accum_grad
                                )


                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                # updata the model
                optimizer.zero_grad()
                # scaler does nothing here if enable_amp=False
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_grad)
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            (
                                "TRAIN",
                                epoch,
                                i + 1,
                                total_loss_avg * accum_grad,
                                optimizer.param_groups[0]["lr"],
                            ),
                            width=10,
                            style="grid",
                        ))
                if (i + 1) == epoch_iter:
                    break
            total_loss_avg = sum(losses) / len(losses)
            return total_loss_avg, 0
    def cv(
        self,
        dataloader,
        models,
        val_iter,
        criterion,
        epoch,
        enable_amp,
        logger,
        log_batch_interval=100,
        device=torch.device("cuda"),
    ):
        """Cross validation"""
        model = models[0]
        model.eval()
        losses = []
        log_interval = log_batch_interval  # ✅ 修复 A

        total_loss_avg = float("inf")      # ✅ 修复 B（防止空集）
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                features = batch["wav_mix"].float().to(device)
                targets  = batch["wav_targets"].float().to(device)

                enroll = batch.get("spk_embeds", None)
                visual_feat = batch.get("visual_feat", None)

                if enroll is not None:
                    enroll = enroll.float().to(device)
                if visual_feat is not None:
                    visual_feat = visual_feat.float().to(device)
                if i == 1 and (visual_feat is not None):
                    print("[DBG] batch visual_feat:", visual_feat.shape)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    outputs = model(features, enroll=enroll, visual_feat=visual_feat)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    loss = criterion[0](outputs[0], targets).mean()

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            ("VAL", epoch, i + 1, total_loss_avg, "-"),
                            width=10,
                            style="grid",
                        )
                    )
                if (i + 1) == val_iter:
                    break

        return total_loss_avg, 0


    # def cv(
    #         self,
    #         dataloader,
    #         models,
    #         val_iter,
    #         criterion,
    #         epoch,
    #         enable_amp,
    #         logger,
    #         log_batch_interval=100,
    #         device=torch.device("cuda"),
    # ):
    #     """Cross validation on"""
    #     model = models[0]
    #     model.eval()
    #     losses = []

    #     with torch.no_grad():
    #         for i, batch in enumerate(dataloader):
    #             features = batch["wav_mix"].float().to(device)
    #             targets  = batch["wav_targets"].float().to(device)

    #             enroll = batch.get("spk_embeds", None)
    #             visual_feat = batch.get("visual_feat", None)

    #             if enroll is not None:
    #                 enroll = enroll.float().to(device)
    #             if visual_feat is not None:
    #                 visual_feat = visual_feat.float().to(device)

    #             with torch.cuda.amp.autocast(enabled=enable_amp):
    #                 outputs = model(features, enroll=enroll, visual_feat=visual_feat)
    #                 if not isinstance(outputs, (list, tuple)):
    #                     outputs = [outputs]
    #                 loss = criterion[0](outputs[0], targets).mean()

    #             losses.append(loss.item())
    #             total_loss_avg = sum(losses) / len(losses)

    #             if (i + 1) % log_interval == 0:
    #                 logger.info(
    #                     tp.row(
    #                         ("VAL", epoch, i + 1, total_loss_avg, "-"),
    #                         width=10,
    #                         style="grid",
    #                     ))
    #             if (i + 1) == val_iter:
    #                 break
    #     return total_loss_avg, 0
