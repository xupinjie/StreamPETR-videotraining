#!/usr/bin/env bash
# run_gpu_scale.sh
# GPU scaling test with fixed per-GPU bs=8, 200 steps per job.
# Calls run_experiment.sh for each (variant, ngpu) combination.
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
#   bash tools/run_gpu_scale.sh                        # all variants
#   bash tools/run_gpu_scale.sh --variant image
#   bash tools/run_gpu_scale.sh --variant prefetch

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
  for ngpu in 1 2 4 8; do
    echo "=== ${variant} bs=8 ${ngpu}GPU ==="
    bash "$SCRIPT_DIR/run_experiment.sh" \
      --variant "$variant" --bs 8 --ngpu "$ngpu" --iters 200
  done
}

if [ -z "$VARIANT" ] || [ "$VARIANT" = "image" ]; then
  run_variant image
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "stream" ]; then
  run_variant stream
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "worker" ]; then
  run_variant worker
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "mainprocess" ]; then
  run_variant mainprocess
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "prefetch" ]; then
  run_variant prefetch
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "prefetch_pool" ]; then
  run_variant prefetch_pool
fi
if [ -z "$VARIANT" ] || [ "$VARIANT" = "opencv" ]; then
  run_variant opencv
fi

echo "GPU scale done."
