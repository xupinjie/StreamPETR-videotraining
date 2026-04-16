#!/usr/bin/env bash
# run_bs_sweep.sh
# Batch-size sweep on 8 GPUs, 200 steps per job.
# Calls run_experiment.sh for each (variant, bs) combination.
#
# Variants:
#   image        — JPEG baseline: DataLoader workers decode JPEG images on CPU
#   stream       — video stream access: DataLoader workers seek+decode video on GPU (NVDEC)
#   worker       — demuxer-free GOP: DataLoader workers read pre-extracted GOP packets, decode on GPU (NVDEC)
#   mainprocess  — demuxer-free GOP: workers send GOP bytes to main process, main process decodes on GPU (serial)
#   prefetch     — demuxer-free GOP + prefetch: same as mainprocess, but decode runs on a background
#                  thread with a dedicated CUDA stream, overlapping with backward pass
#   prefetch_pool — prefetch + SharedGOPPool: cross-worker shared GOP cache with zero-copy reads
#   opencv       — video OpenCV: DataLoader workers decode video frames on CPU using cv2.VideoCapture
#
# Usage:
#   bash tools/run_bs_sweep.sh                     # all variants
#   bash tools/run_bs_sweep.sh --variant image
#   bash tools/run_bs_sweep.sh --variant prefetch

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VARIANT=""

for arg in "$@"; do
  case $arg in
    --variant) shift; VARIANT="$1"; shift ;;
  esac
done

run_variant() {
  local variant=$1
  shift
  local bs_list=("$@")
  for bs in "${bs_list[@]}"; do
    echo "=== ${variant} bs=${bs} 8GPU ==="
    bash "$SCRIPT_DIR/run_experiment.sh" \
      --variant "$variant" --bs "$bs" --ngpu 8 --iters 200
  done
}

if [ -z "$VARIANT" ] || [ "$VARIANT" = "image" ]; then
  run_variant image 1 2 4 8 16
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "stream" ]; then
  run_variant stream 1 2 4 8
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "worker" ]; then
  run_variant worker 1 2 4 8
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "mainprocess" ]; then
  run_variant mainprocess 1 2 4 8 16
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "prefetch" ]; then
  run_variant prefetch 1 2 4 8 16
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "prefetch_pool" ]; then
  run_variant prefetch_pool 1 2 4 8 16
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "opencv" ]; then
  run_variant opencv 1 2 4 8 16
fi

echo "BS sweep done."
