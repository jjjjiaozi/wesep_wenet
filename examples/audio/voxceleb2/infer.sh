#!/bin/bash
set -euo pipefail

# ===== user edit =====
GPU=0
CONFIG="config/tse_bsrnn_av.yaml"

# 你训练出来的 checkpoint（例如 latest_checkpoint.pt 或 checkpoint_XX.pt）
CKPT="/mnt/afs/250010237/my_projects/wesep_wenet/examples/audio/voxceleb2/exp/BSRNN_AV/voxceleb2_av/models/final_checkpoint.pt"

# 你的 AV CSV（同 train_visual.py 的 train_data/val_data 一样，这里用 test_data）
TEST_CSV="/mnt/afs/wwu/vox2_mp4/preprocess/mixture_data_list_2mix.csv"

# exp_dir（infer 会把 wav 输出保存到 ${exp_dir}/audio_av）
EXP_DIR="exp/BSRNN_AV/voxceleb2"

# 选择 cue_mode: visual / enroll / both（必须与你训练时一致才好比）
CUE_MODE="visual"

# test partition（CSV 第 0 列）：test / val / train
TEST_PART="test"

# 可选：只跑前 N 条（0 表示全跑）
MAX_TEST_UTTS=0
# ====================

export CUDA_VISIBLE_DEVICES=${GPU}

python /mnt/afs/250010237/my_projects/wesep_wenet/infer_av.py \
  --config ${CONFIG} \
  --exp_dir ${EXP_DIR} \
  --checkpoint ${CKPT} \
  --test_data ${TEST_CSV} \
  --test_partition ${TEST_PART} \
  --save_wav true \
  --max_test_utts ${MAX_TEST_UTTS} \
  --dataset_args.cue_mode ${CUE_MODE} \
  --gpus ${GPU}
