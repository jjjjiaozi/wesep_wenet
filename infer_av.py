from __future__ import print_function

import os
import time
import re

import fire
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import (
    generate_enahnced_scp,
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)

# AV dataset
from wesep.dataset.av_dataset import DatasetUtilsAV, tse_collate_fn_av_2spk


def _pick_gpu(gpus_cfg):
    """
    Accepts:
      - int (0)
      - str ("0,1")
      - list/tuple ([0,1] or ["0","1"])
    """
    if isinstance(gpus_cfg, (list, tuple)):
        return int(gpus_cfg[0])
    if isinstance(gpus_cfg, str):
        # could be "0,1" or "[0,1]"
        s = gpus_cfg.strip()
        s = re.sub(r"[\[\]\s]", "", s)
        if "," in s:
            return int(s.split(",")[0])
        return int(s)
    return int(gpus_cfg)


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", s)


def infer(config="config/tse_bsrnn_av.yaml", **kwargs):
    start = time.time()
    total_sisnr = 0.0
    total_sisnri = 0.0
    total_cnt = 0
    accept_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)

    sign_save_wav = configs.get("save_wav", True)
    rank = 0
    set_seed(configs["seed"] + rank)

    # device
    gpu = _pick_gpu(configs.get("gpus", 0))
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    # sample rate (prefer dataset_args.resample_rate)
    sample_rate = configs.get("fs", None)
    if sample_rate is None:
        sample_rate = configs.get("dataset_args", {}).get("resample_rate", 16000)
    if isinstance(sample_rate, str):
        sample_rate = 16000 if sample_rate.lower() in ("16k", "16000") else 8000
    sample_rate = int(sample_rate)

    # model
    if "spk_model_init" in configs.get("model_args", {}).get("tse_model", {}):
        configs["model_args"]["tse_model"]["spk_model_init"] = False

    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])

    model_path = os.path.realpath(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], "infer_av.log")
    logger.info(f"Load checkpoint from {model_path}")

    save_audio_dir = os.path.join(configs["exp_dir"], "audio_av")
    if sign_save_wav:
        os.makedirs(save_audio_dir, exist_ok=True)
        logger.info(f"Save wavs to: {save_audio_dir}")
    else:
        logger.info("Do NOT save wav outputs (save_wav=false).")

    model = model.to(device)
    model.eval()

    # ---- AV dataset/dataloader ----
    dataset_args = dict(configs.get("dataset_args", {}))
    cue_mode = dataset_args.get("cue_mode", "both")  # enroll / visual / both

    # choose test partition
    # (your CSV col0 is partition: train/val/test)
    test_part = configs.get("test_partition", None) or dataset_args.get("test_partition", None)
    if test_part is None:
        test_part = dataset_args.get("partition", "test")  # fallback
    dataset_args["partition"] = test_part

    test_dataset = DatasetUtilsAV(
        data_list_file=configs["test_data"],
        configs=dataset_args,
        state="test",
        repeat_dataset=configs.get("repeat_dataset", False),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=configs.get("infer_num_workers", 2),
        pin_memory=True,
        collate_fn=lambda b: tse_collate_fn_av_2spk(b, cue_mode=cue_mode, mode="min"),
    )

    logger.info(f"[AV] cue_mode={cue_mode}, test_partition={test_part}")
    logger.info(f"[AV] test_data={configs['test_data']}")

    # ---- run ----
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # batch is already 2spk-expanded: B==2 typically
            features = batch["wav_mix"].float().to(device)        # (B,T)
            targets = batch["wav_targets"].float().to(device)     # (B,T)
            enroll = batch.get("spk_embeds", None)
            visual_feat = batch.get("visual_feat", None)

            if enroll is not None:
                enroll = enroll.float().to(device)
            if visual_feat is not None:
                visual_feat = visual_feat.float().to(device)

            outputs = model(features, enroll=enroll, visual_feat=visual_feat)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # take waveform output

            # outputs: (B,T)
            # optional normalize for saving/listening
            if outputs.dim() == 2 and torch.min(outputs.max(dim=1).values) > 0:
                denom = outputs.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
                outputs_np = (outputs / denom * 0.9).cpu().numpy()
            else:
                outputs_np = outputs.cpu().numpy()

            ref_np = targets.cpu().numpy()
            mix_np = features.cpu().numpy()

            spk_list = batch.get("spk", ["spk"] * outputs_np.shape[0])
            key_list = batch.get("key", ["utt"] * outputs_np.shape[0])

            # save + score each item in the expanded batch
            B = outputs_np.shape[0]
            for bi in range(B):
                est = outputs_np[bi]
                ref = ref_np[bi]
                mix = mix_np[bi]

                # align lengths
                end = min(est.size, ref.size, mix.size)
                est = est[:end]
                ref = ref[:end]
                mix = mix[:end]

                sisnr, sisnri = cal_SISNRi(est, ref, mix)

                total_sisnr += float(sisnr)
                total_sisnri += float(sisnri)
                total_cnt += 1
                if sisnri > 1:
                    accept_cnt += 1

                key_s = _sanitize_name(str(key_list[bi]))
                spk_s = _sanitize_name(str(spk_list[bi]))

                logger.info(
                    f"Num={total_cnt:06d} | Key={key_s} | Target={spk_s} | "
                    f"SI-SNR={sisnr:.2f} | SI-SNRi={sisnri:.2f}"
                )

                if sign_save_wav:
                    out_path = os.path.join(save_audio_dir, f"{total_cnt:06d}-{key_s}-T{spk_s}.wav")
                    sf.write(out_path, est, sample_rate)

            # 可选：只跑前 N 条
            max_utts = configs.get("max_test_utts", 0)
            if max_utts and total_cnt >= int(max_utts):
                break

    end = time.time()

    # generate enhanced scp for scoring
    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info(f"Time Elapsed: {end - start:.1f}s")
    if total_cnt > 0:
        logger.info(f"Average SI-SNR : {total_sisnr / total_cnt:.2f}")
        logger.info(f"Average SI-SNRi: {total_sisnri / total_cnt:.2f}")
        logger.info(f"Acceptance rate (SI-SNRi>1 dB): {accept_cnt / total_cnt * 100:.2f}%")
    else:
        logger.info("No test samples were processed (check partition/test_data path).")


if __name__ == "__main__":
    fire.Fire(infer)
