#!/bin/bash
# convert_and_play_tum.sh
# 将 TUM-RGBD .bag（ROS 1）转换为 ROS 2 bag 并播放
#
# 用法：
#   chmod +x scripts/convert_and_play_tum.sh
#   ./scripts/convert_and_play_tum.sh /path/to/rgbd_dataset_freiburg1_desk2.bag

set -e

BAG_FILE="$1"

if [ -z "$BAG_FILE" ]; then
    echo "用法: $0 <path_to.bag>"
    echo "示例: $0 ~/Downloads/rgbd_dataset_freiburg1_desk2.bag"
    exit 1
fi

if [ ! -f "$BAG_FILE" ]; then
    echo "文件不存在: $BAG_FILE"
    exit 1
fi

# 检测 bag 格式
echo "检测 bag 格式..."
FILE_HEADER=$(xxd -l 4 "$BAG_FILE" 2>/dev/null | head -1)
if echo "$FILE_HEADER" | grep -q "724f 732e 6261 67"; then
    echo "  → 检测为 ROS 1 bag"
    IS_ROS1=true
elif echo "$FILE_HEADER" | grep -q "726f 732e 6261 6732"; then
    echo "  → 检测为 ROS 2 bag"
    IS_ROS1=false
else
    echo "  → 未知格式，尝试作为 ROS 2 bag 打开"
    IS_ROS1=false
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONVERTED_BAG="/tmp/tum_rgbd_ros2"

if [ "$IS_ROS1" = true ]; then
    echo ""
    echo "ROS 1 bag 需要先转换为 ROS 2..."
    echo ""

    # 检查 ros1_bridge
    if ! ros2 pkg list 2>/dev/null | grep -q ros1_bridge; then
        echo "警告：ros1_bridge 未安装，ROS 1 bag 无法直接桥接到 ROS 2"
        echo ""
        echo "解决方案（选择一种）："
        echo ""
        echo "方案 A【推荐】：在 ROS 1 环境中播放 bag，通过 ros1_bridge 桥接到 ROS 2"
        echo "  1. 安装 ros1_bridge："
        echo "       sudo apt install ros-humble-ros1-bridge"
        echo "  2. 启动 ros1_bridge："
        echo "       ros2 run ros1_bridge dynamic_bridge --bridge-all-topics &"
        echo "  3. 在另一个终端播放 bag："
        echo "       source /opt/ros/noetic/setup.bash"
        echo "       rosbag play $BAG_FILE"
        echo "  4. 在第三个终端运行你的 VO："
        echo "       ros2 run sfc_ros_humble sfc_slam_node"
        echo ""
        echo "方案 B【离线模式】：将 bag 导出为图片序列，再用 test_tum_rgbd"
        echo "  1. 用 ROS 1 的 image_view 批量导出："
        echo "       rosrun image_view extract_images _filename_format:='frame%04i.png' image:=/camera/rgb/image_color"
        echo "       rosrun image_view extract_images _filename_format:='depth%04i.png' image:=/camera/depth/image"
        echo "  2. 从 bag 提取时间戳（见下方脚本）"
        echo ""
        echo "方案 C【推荐】：在 ROS 1 侧直接用你的 VO（不需要桥接）"
        echo "  将 VO 改写为 ROS 1 nodelet 模式运行"
        echo ""
        exit 0
    fi

    # 转换
    echo "正在转换 $BAG_FILE ..."
    rm -rf "$CONVERTED_BAG"
    ros2 bag convert \
        -s rosbag_v2 \
        -i "$BAG_FILE" \
        -o "$CONVERTED_BAG"

    BAG_TO_PLAY="$CONVERTED_BAG"
else
    echo "ROS 2 bag，直接播放"
    BAG_TO_PLAY="$BAG_FILE"
fi

echo ""
echo "Bag 已就绪: $BAG_TO_PLAY"
echo ""
echo "========================================"
echo "播放步骤"
echo "========================================"
echo ""
echo "终端 1：播放 bag"
echo "  ros2 bag play $BAG_TO_PLAY --rate 1.0"
echo ""
echo "终端 2：运行 VO"
echo "  ros2 run sfc_ros_humble sfc_slam_node"
echo ""
echo "终端 3：保存轨迹（VO 运行后执行）"
echo "  ros2 topic echo /sfc/trajectory --csv > ~/sfc_traj.tum"
echo ""
echo "评估（安装 evo）："
echo "  pip install evo --upgrade --no-binary evo"
echo "  evo_ape tum ~/datasets/freiburg1_desk2/groundtruth.txt ~/sfc_traj.tum \\"
echo "      -as --correct_scale -va --plot --plot_mode xyz"
echo ""
echo "========================================"
echo "详细：查看 bag 中 topic"
echo "  ros2 bag info $BAG_TO_PLAY"
echo "  ros2 topic list"
echo "========================================"
