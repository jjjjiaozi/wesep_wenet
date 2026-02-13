import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from wesep.dataset.dataset import DataList, Processor


def _read_wav(path, normalize=False):
    """Return torch.FloatTensor (T,)"""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)  # (C,T)
        wav = wav.mean(dim=0).to(torch.float32)  # -> (T,)
    except Exception:
        import soundfile as sf
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav).to(torch.float32)

    if normalize:
        m = wav.abs().max().clamp(min=1e-8)
        wav = wav / m
    return wav


def _read_npy(path):
    """Return torch.FloatTensor (Tv, Dv)"""
    arr = np.load(path)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return torch.from_numpy(arr).to(torch.float32)


def _utils_like_mix_path(line: str, mixture_direc: str, partition: str) -> str:
    # utils.py: mixture_direc + partition + '/' + line.replace(',','_').replace('/','_') + '.wav'
    fname = line.replace(",", "_").replace("/", "_") + ".wav"
    return os.path.join(mixture_direc, partition, fname)


def _utils_like_spk_paths(parts, c, audio_direc, visual_direc):
    """
    utils.py:
      audio = audio_direc + parts[c*4+1] + '/' + parts[c*4+2] + '/' + parts[c*4+3] + '.wav'
      visual= visual_direc+ parts[c*4+1] + '/' + parts[c*4+2] + '/' + parts[c*4+3] + '.npy'
    where parts[c*4+4] is an extra field (offset) NOT used for path.
    """
    base = c * 4 + 1
    lvl1 = parts[base]         # e.g. "train"
    spk_id = parts[base + 1]   # e.g. "id04705"
    rel = parts[base + 2]      # e.g. "8PgY43tYJrw/00006" (may contain '/')
    offset = parts[base + 3]   # e.g. "0" or "2.84..." (not used for path)

    rel_parts = rel.split("/") if rel else []

    # Build nested path exactly like string concatenation
    a_base = os.path.join(audio_direc, lvl1, spk_id, *rel_parts)
    v_base = os.path.join(visual_direc, lvl1, spk_id, *rel_parts)

    audio_path = a_base + ".wav"
    visual_path = v_base + ".npy"

    return audio_path, visual_path, spk_id, offset


def parse_utils_av(
    iterator,
    audio_direc,
    visual_direc,
    mixture_direc,
    partition,
    C=2,
    sampling_rate=16000,
    fps=25,
    max_length=None,
    normalize_audio=False,
):
    """
    Convert DataList items -> sample dict (WeSep style keys),
    while keeping utils-like CSV parsing + path building.
    """
    for item in iterator:
        line = item["src"].strip()
        if not line:
            continue

        parts = line.split(",")

        # utils.py uses column 0 for partition filter
        # and last column as duration (sec).
        line_partition = parts[0]
        if line_partition != partition:
            continue

        duration_sec = float(parts[-1])

        # mixture path: based on FULL line string
        mix_path = _utils_like_mix_path(line, mixture_direc, partition)

        # read mixture
        mix_wav = _read_wav(mix_path, normalize=normalize_audio)

        # optional max_length clip (utils does min_length + max_length; here we enforce max_length only)
        if max_length is not None:
            max_samp = int(max_length * sampling_rate)
            mix_wav = mix_wav[:max_samp]

        sample = {
            "key": line,
            "wav_mix": mix_wav.unsqueeze(0),  # (1,T)
            "num_speaker": C,
            "duration_sec": duration_sec,
        }

        # speakers
        for c in range(C):
            audio_path, visual_path, spk_id, offset = _utils_like_spk_paths(
                parts, c, audio_direc, visual_direc
            )

            wav = _read_wav(audio_path, normalize=normalize_audio)
            vis = _read_npy(visual_path)

            if max_length is not None:
                max_samp = int(max_length * sampling_rate)
                max_fr = int(max_length * fps)
                wav = wav[:max_samp]
                vis = vis[:max_fr]

            si = c + 1
            sample[f"spk{si}"] = spk_id
            sample[f"spk{si}_offset"] = offset  # keep it (utils CSV has it)
            sample[f"wav_spk{si}"] = wav.unsqueeze(0)      # (1,T)
            sample[f"embed_spk{si}"] = wav.unsqueeze(0)    # TEMP: use enroll wav as cue
            sample[f"visual_spk{si}"] = vis                # (Tv,Dv)

        # keep distributed fields if present
        for k in ["rank", "world_size", "worker_id", "num_workers"]:
            if k in item:
                sample[k] = item[k]

        yield sample


def tse_collate_fn_av_2spk(batch, cue_mode="both", mode="min"):
    """
    WeSep-like 2spk expand:
      output keys: wav_mix, wav_targets, spk_embeds (or None), visual_feat (or None), spk, key, spk_label
    cue_mode: "enroll" | "visual" | "both"
    mode: "min" or "max" for length alignment
    """
    new_batch = {}

    wav_mix = []
    wav_targets = []
    spk_embeds = []
    visual_feat = []

    spk = []
    key = []
    spk_label = []  # keep placeholder
    length_spk = []
    length_vis = []

    def _append_one(s, idx):
        wav_mix.append(s["wav_mix"])
        wav_targets.append(s[f"wav_spk{idx}"])
        spk.append(s[f"spk{idx}"])
        key.append(s["key"])

        if cue_mode in ("enroll", "both"):
            e = s[f"embed_spk{idx}"]
            spk_embeds.append(e)
            length_spk.append(e.shape[-1])

        if cue_mode in ("visual", "both"):
            v = s[f"visual_spk{idx}"]
            visual_feat.append(v)
            length_vis.append(v.shape[0])

    for s in batch:
        _append_one(s, 1)
        _append_one(s, 2)

    # ---- align wav_mix length within batch ----
    # mix_lens = [x.shape[-1] for x in wav_mix]  # each x is (1,T)
    # if len(set(mix_lens)) != 1:
    #     if mode == "min":
    #         L = min(mix_lens)
    #         wav_mix = [x[..., :L] for x in wav_mix]
    #         wav_targets = [y[..., :L] for y in wav_targets]
    #         # enroll wav (if present)
    #         if cue_mode in ("enroll", "both") and len(spk_embeds) > 0 and spk_embeds[0].dim() == 2:
    #             spk_embeds = [e[..., :L] for e in spk_embeds]
    #     else:
    #         L = max(mix_lens)
    #         wav_mix = [F.pad(x, (0, L - x.shape[-1])) for x in wav_mix]
    #         wav_targets = [F.pad(y, (0, L - y.shape[-1])) for y in wav_targets]
    #         if cue_mode in ("enroll", "both") and len(spk_embeds) > 0 and spk_embeds[0].dim() == 2:
    #             spk_embeds = [F.pad(e, (0, L - e.shape[-1])) for e in spk_embeds]
    # ---- align length within batch (use mix as reference) ----
    # each x in wav_mix / wav_targets is (1, T)
    mix_lens = [x.shape[-1] for x in wav_mix]
    tgt_lens = [y.shape[-1] for y in wav_targets]

    if mode == "min":
        # 取 mix 和 target 的共同最短，避免越界
        L = min(min(mix_lens), min(tgt_lens))
        wav_mix = [x[..., :L] for x in wav_mix]
        wav_targets = [y[..., :L] for y in wav_targets]
    else:  # mode == "max"
        # 取 mix 和 target 里最长的，全部 pad 到同一长度
        L = max(max(mix_lens), max(tgt_lens))
        wav_mix = [F.pad(x, (0, L - x.shape[-1])) for x in wav_mix]
        wav_targets = [F.pad(y, (0, L - y.shape[-1])) for y in wav_targets]

    # enroll wav (if present and is waveform-like tensor)
    if cue_mode in ("enroll", "both") and len(spk_embeds) > 0 and spk_embeds[0].dim() == 2:
        # (1, T_enroll) -> align to L as well
        if mode == "min":
            spk_embeds = [e[..., :L] for e in spk_embeds]
        else:
            spk_embeds = [F.pad(e, (0, L - e.shape[-1])) for e in spk_embeds]


    new_batch["wav_mix"] = torch.cat(wav_mix, dim=0)       # (B,T)
    new_batch["wav_targets"] = torch.cat(wav_targets, dim=0)

    # enroll alignment
    if cue_mode in ("enroll", "both"):
        if len(length_spk) > 0 and (len(set(length_spk)) != 1):
            if mode == "min":
                L = min(length_spk)
                spk_embeds = [x[..., :L] for x in spk_embeds]
            else:
                L = max(length_spk)
                spk_embeds = [F.pad(x, (0, L - x.shape[-1])) for x in spk_embeds]
        new_batch["spk_embeds"] = torch.cat(spk_embeds, dim=0)  # (B,Te)
    else:
        new_batch["spk_embeds"] = None

    # visual alignment
    if cue_mode in ("visual", "both"):
        if len(length_vis) > 0 and (len(set(length_vis)) != 1):
            if mode == "min":
                Tv = min(length_vis)
                visual_feat = [v[:Tv] for v in visual_feat]
            else:
                Tv = max(length_vis)
                visual_feat = [F.pad(v, (0, 0, 0, Tv - v.shape[0])) for v in visual_feat]
        new_batch["visual_feat"] = torch.stack(visual_feat, dim=0)  # (B,Tv,Dv)
    else:
        new_batch["visual_feat"] = None

    new_batch["spk"] = spk
    new_batch["key"] = key
    new_batch["spk_label"] = torch.as_tensor(spk_label) if len(spk_label) > 0 else torch.zeros(len(spk), dtype=torch.long)
    return new_batch


def DatasetUtilsAV(
    data_list_file,
    configs,
    state="train",
    repeat_dataset=False,
):
    """
    Build WeSep-style IterableDataset but using utils-style CSV + dir structure.
    Required configs keys:
      audio_direc, visual_direc, mixture_direc
    Optional:
      partition, num_speakers, resample_rate, fps, max_length, normalize_audio, shuffle
    """
    audio_direc = configs["audio_direc"]
    visual_direc = configs["visual_direc"]
    mixture_direc = configs["mixture_direc"]

    partition = configs.get("partition", "train")
    C = configs.get("num_speakers", 2)

    sampling_rate = configs.get("resample_rate", 16000)
    fps = configs.get("fps", 25)
    max_length = configs.get("max_length", None)
    normalize_audio = configs.get("normalize_audio", False)

    # read CSV lines (keep full line string)
    with open(data_list_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # IMPORTANT: filter by column 0 == partition (same as utils.py)
    lines = [ln for ln in lines if ln.split(",")[0] == partition]

    # shuffle for training
    shuffle = configs.get("shuffle", state == "train")
    dataset = DataList(lines, shuffle=shuffle, repeat_dataset=repeat_dataset)

    # parse using utils-style logic
    dataset = Processor(
        dataset,
        parse_utils_av,
        audio_direc,
        visual_direc,
        mixture_direc,
        partition,
        C,
        sampling_rate,
        fps,
        max_length,
        normalize_audio,
    )

    return dataset
