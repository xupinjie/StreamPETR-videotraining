#!/usr/bin/env bash
# run_experiment.sh — run a single training experiment with configurable
# variant, batch size, GPU count, and iteration count.
#
# Usage:
#   bash tools/run_experiment.sh --variant prefetch --bs 16 --ngpu 8 --iters 200
#   bash tools/run_experiment.sh --variant image --bs 8 --ngpu 1 --iters 50
#   bash tools/run_experiment.sh --variant image --bs 8 --ngpu 8 --iters 200 \
#       --work-dir stream_petr_r50_flash_704_bs8_seq_24e_8gpu_minix8
#
# Variants:
#   image        — JPEG baseline: DataLoader workers decode JPEG images on CPU
#   stream       — video stream access: DataLoader workers seek+decode video on GPU (NVDEC)
#   worker       — demuxer-free GOP: DataLoader workers read pre-extracted GOP packets, decode on GPU (NVDEC)
#   mainprocess  — demuxer-free GOP: workers send GOP bytes to main process, main process decodes on GPU (serial)
#   prefetch     — demuxer-free GOP + prefetch: same as mainprocess, but decode runs on a background
#                  thread with a dedicated CUDA stream, overlapping with backward pass
#   opencv       — video OpenCV: DataLoader workers decode video frames on CPU using cv2.VideoCapture

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$ROOT_DIR/projects/configs/StreamPETR_video"
GEN_DIR="$ROOT_DIR/projects/configs/_generated"

# ── Defaults ──────────────────────────────────────────────────────
VARIANT=""
BS=2
NGPU=8
ITERS=200
WORK_DIR_OVERRIDE=""

# ── Parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --variant)     VARIANT="$2";  shift 2 ;;
    --bs)          BS="$2";       shift 2 ;;
    --ngpu)        NGPU="$2";     shift 2 ;;
    --iters)       ITERS="$2";    shift 2 ;;
    --work-dir)    WORK_DIR_OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$VARIANT" ]; then
  echo "Error: --variant is required (image|stream|worker|mainprocess|prefetch)"
  exit 1
fi

# ── Template config per variant ───────────────────────────────────
case $VARIANT in
  image)
    TEMPLATE="$CFG_DIR/base_image_minix8.py"
    ;;
  stream)
    TEMPLATE="$CFG_DIR/base_video_stream_minix8.py"
    ;;
  worker)
    TEMPLATE="$CFG_DIR/base_video_demuxer_free_minix8.py"
    ;;
  mainprocess)
    TEMPLATE="$CFG_DIR/base_video_demuxer_free_mainprocess_minix8.py"
    ;;
  prefetch)
    TEMPLATE="$CFG_DIR/base_video_demuxer_free_mainprocess_prefetch_minix8.py"
    ;;
  opencv)
    TEMPLATE="$CFG_DIR/base_video_opencv_minix8.py"
    ;;
  *)
    echo "Error: unknown variant '$VARIANT'"
    echo "Choose from: image, stream, worker, mainprocess, prefetch, opencv"
    exit 1
    ;;
esac

if [ ! -f "$TEMPLATE" ]; then
  echo "Error: template not found: $TEMPLATE"
  exit 1
fi

# ── Generate config ───────────────────────────────────────────────
mkdir -p "$GEN_DIR"
GEN_CFG="$GEN_DIR/${VARIANT}_bs${BS}_${NGPU}gpu_${ITERS}step.py"
if [ -n "$WORK_DIR_OVERRIDE" ]; then
  WORK_DIR="$ROOT_DIR/work_dirs/$WORK_DIR_OVERRIDE"
else
  WORK_DIR="$ROOT_DIR/work_dirs/${VARIANT}_bs${BS}_${NGPU}gpu_${ITERS}step"
fi

RUNNER_TYPE=$(grep '^runner' "$TEMPLATE" | sed "s/.*type='\([^']*\)'.*/\1/")

sed -e "s/^batch_size = .*/batch_size = ${BS}/" \
    -e "s/^num_gpus = .*/num_gpus = ${NGPU}/" \
    -e "s/^runner = .*/runner = dict(type='${RUNNER_TYPE}', max_iters=${ITERS})/" \
    "$TEMPLATE" > "$GEN_CFG"

# ── Run ───────────────────────────────────────────────────────────
echo "========================================"
echo "VARIANT : $VARIANT"
echo "BS      : $BS"
echo "NGPU    : $NGPU"
echo "ITERS   : $ITERS"
echo "CONFIG  : $GEN_CFG"
echo "WORKDIR : $WORK_DIR"
echo "========================================"

PORT=${PORT:-29501} bash "$SCRIPT_DIR/dist_train.sh" \
  "$GEN_CFG" "$NGPU" \
  --work-dir "$WORK_DIR"
