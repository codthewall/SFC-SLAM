#!/usr/bin/env bash
# 跑一次 TUM 序列并直接给出 ATE/RPE 评测。
#
# 为什么要有它：以前验收靠肉眼看日志里的"路径总长"，而路径总长对单帧跳变
# 极端敏感（fr1_desk2 曾报 478m，真实约 5–12m），既不能判断位姿对错，也无法
# 跨次比较。这个脚本把「跑 -> 写轨迹 -> 对齐 -> 算 ATE/RPE」串起来，产出一份
# 可比较的数字。
#
# 用法：
#   scripts/run_and_eval_tum.sh <dataset_dir> [camera_yaml] [extra args...]
# 例：
#   scripts/run_and_eval_tum.sh ~/datasets/fr1_desk2 config/camera_fr1.yaml
#   scripts/run_and_eval_tum.sh ~/datasets/fr2_xyz   config/camera_fr2.yaml --max_frames 500
#
# 前置：已构建 test_tum_rgbd（见 src/sfc_core/CMakeLists.txt 的 test_tum_rgbd target），
#       二进制路径可用 TEST_BIN 覆盖（默认 ./build/test_tum_rgbd）。
set -uo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "usage: $0 <dataset_dir> [camera_yaml] [extra args...]" >&2
    exit 1
fi
shift || true

CAMERA="config/camera_fr1.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CAMERA="$1"
    shift
fi

TEST_BIN="${TEST_BIN:-./build/test_tum_rgbd}"
OUT_DIR="${OUT_DIR:-./eval_out}"
mkdir -p "$OUT_DIR"

SEQ_NAME="$(basename "$DATASET")"
TRAJ="$OUT_DIR/${SEQ_NAME}_traj.tum"
LOG="$OUT_DIR/${SEQ_NAME}_run.log"

if [[ ! -x "$TEST_BIN" ]]; then
    echo "ERROR: 找不到可执行文件 $TEST_BIN" >&2
    echo "       先用 TEST_BIN=/path/to/test_tum_rgbd 指定，或先构建。" >&2
    exit 1
fi
if [[ ! -f "$DATASET/groundtruth.txt" ]]; then
    echo "ERROR: $DATASET/groundtruth.txt 不存在，无法评测。" >&2
    echo "       TUM 序列目录应包含 rgb.txt / depth.txt / groundtruth.txt。" >&2
    exit 1
fi

echo "=== 1/2 运行 SLAM ($SEQ_NAME) ==="
"$TEST_BIN" --dataset "$DATASET" --camera "$CAMERA" --output "$TRAJ" "$@" 2>&1 | tee "$LOG"
RUN_RC=${PIPESTATUS[0]}
if [[ $RUN_RC -ne 0 ]]; then
    echo "ERROR: 运行失败 (exit $RUN_RC)，日志 $LOG" >&2
    exit $RUN_RC
fi
if [[ ! -s "$TRAJ" ]]; then
    echo "ERROR: 轨迹文件为空: $TRAJ" >&2
    exit 1
fi

echo
echo "=== 2/2 评测 ATE/RPE ==="
# --align：SLAM 输出与真值不在同一世界系，必须做一次 Umeyama 对齐再比。
# 不加 --align-scale：RGB-D 的尺度是深度图给的，对齐时再估尺度会把尺度错误藏起来。
python3 "$(dirname "$0")/evaluate_trajectory.py" \
    --est "$TRAJ" \
    --ref "$DATASET/groundtruth.txt" \
    --align \
    --save-aligned "$OUT_DIR/${SEQ_NAME}_aligned.tum"

echo
echo "产物："
echo "  轨迹       $TRAJ"
echo "  对齐轨迹   $OUT_DIR/${SEQ_NAME}_aligned.tum"
echo "  运行日志   $LOG"
