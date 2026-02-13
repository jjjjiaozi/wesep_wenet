#!/usr/bin/env bash
set -euo pipefail

# =========================
# AV-TSE (BSRNN) training launcher (visual-only / cue_mode switch in yaml)
# =========================

# ---- GPU setting ----
# Use format like: gpus="[0]" or gpus="[0,1]"
gpus="${gpus:-[0,1,2,3,4,5,6,7]}"

# ---- Config / exp ----
config="${config:-config/tse_bsrnn_av.yaml}"
exp_dir="${exp_dir:-exp/BSRNN_AV/voxceleb2_av}"

# ---- Train script ----
train_script="${train_script:-/mnt/afs/250010237/my_projects/wesep_wenet/train_visual.py}"

# ---- Resume checkpoint (optional) ----
checkpoint="${checkpoint:-}"
if [[ -z "${checkpoint}" && -f "${exp_dir}/models/latest_checkpoint.pt" ]]; then
  checkpoint="${exp_dir}/models/latest_checkpoint.pt"
fi

# ---- Misc ----
num_avg="${num_avg:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---- Parse number of gpus from "[0,1,2]" ----
gpu_list="$(echo "${gpus}" | tr -d '[]' | tr -d ' ')"
if [[ -z "${gpu_list}" ]]; then
  echo "[ERROR] gpus is empty. Example: gpus=\"[0]\" or gpus=\"[0,1]\""
  exit 1
fi
num_gpus="$(echo "${gpu_list}" | awk -F',' '{print NF}')"

echo "[INFO] gpus=${gpus}  (num_gpus=${num_gpus})"
echo "[INFO] config=${config}"
echo "[INFO] exp_dir=${exp_dir}"
echo "[INFO] train_script=${train_script}"
echo "[INFO] checkpoint=${checkpoint:-<none>}"

mkdir -p "${exp_dir}"

# ---- Run ----
torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
  "${train_script}" \
  --config "${config}" \
  --exp_dir "${exp_dir}" \
  --gpus "${gpus}" \
  --num_avg "${num_avg}" \
  ${checkpoint:+--checkpoint "${checkpoint}"}
