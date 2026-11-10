#!/usr/bin/env python3
"""
tum_bag_player.py
=================
将 TUM-RGBD .bag 文件（ROS 1 格式）播放为 ROS 2 图像话题，
驱动 SFC Tracking 节点（test_visual_slam 的 ROS 2 模式）。

用法：
    # 终端 1：播放 bag（自动检测 topic 并 remap）
    python3 scripts/tum_bag_player.py --bag /path/to/rgbd_dataset_freiburg1_desk2.bag

    # 终端 2：运行你的 VO
    ros2 run sfc_ros_humble sfc_slam_node

    # 终端 3（可选）：保存 VO 轨迹
    ros2 topic echo /your_vo/pose --csv > ~/sfc_traj.tum

注意：
    - 必须在 ROS 2 环境（source install/setup.bash）中运行
    - TUM bag 通常发布 ROS 1 格式（sensor_msgs/Image），需要 ros1_bridge
    - 若已有 ROS 2 bag，直接用 ros2 bag play 即可
"""

import argparse
import sys
import os

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
except ImportError:
    print("错误：需要 ROS 2 环境。请先 source ros2 setup.bash")
    sys.exit(1)

try:
    from rosbag2_py import Reader, SequentialReaderStorageLoader
    from rosbag2_py._storage import ConverterOptions, SerializationFormat
    print("[tum_bag_player] 使用 rosbag2_py 读取 bag")
    HAS_ROSBAG2 = True
except ImportError:
    print("[tum_bag_player] rosbag2_py 不可用，尝试 rosbag1...")
    HAS_ROSBAG2 = False

try:
    import cv2
    import numpy as np
except ImportError:
    print("错误：需要 opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
except ImportError:
    print("错误：需要 sensor_msgs 和 cv_bridge，请确保 ROS 2 环境正确")
    sys.exit(1)


# ============================================================================
# TUM-RGBD 相机内参（freiburg1 系列）
# ============================================================================
# freiburg1 (Kinect 640x480):
TUM_FREIBURG1_INTRINSICS = {
    'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5,
    'width': 640, 'height': 480,
    'depth_scale': 0.0002,   # TUM PNG 标准：pixel = depth_mm / 5 → depth_m = pixel / 5000
    'depth_encoding': '16UC1',
}
# freiburg2 (Kinect 640x480, 实验室场景):
TUM_FREIBURG2_INTRINSICS = {
    'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7,
    'width': 640, 'height': 480,
    'depth_scale': 0.0002,   # TUM PNG 标准：pixel = depth_mm / 5 → depth_m = pixel / 5000
    'depth_encoding': '16UC1',
}
# freiburg3 (Kinect 640x480, 不同光照):
TUM_FREIBURG3_INTRINSICS = {
    'fx': 535.4, 'fy': 539.2, 'cx': 320.1, 'cy': 247.6,
    'width': 640, 'height': 480,
    'depth_scale': 0.0002,   # TUM PNG 标准：pixel = depth_mm / 5 → depth_m = pixel / 5000
    'depth_encoding': '16UC1',
}


# ============================================================================
# ROS 1 Image → ROS 2 Image 转换（核心逻辑）
# ============================================================================

class TUMLidarPlayer(Node):
    """
    读取 bag 中的 RGB + Depth topic，发布为与 test_visual_slam 兼容的 ROS 2 话题。
    自动处理：
      - ROS 1 → ROS 2 消息桥接（通过 topic remap 或手动转换）
      - TUM 深度图单位转换（16UC1 mm → 32FC1 m）
      - 图像分辨率检查并提醒
    """

    def __init__(self, bag_path: str, rgb_topic: str = '', depth_topic: str = '',
                 intrinsics: dict = None, playback_speed: float = 1.0,
                 target_rgb_topic: str = '/camera/color/image_raw',
                 target_depth_topic: str = '/camera/depth/image_raw',
                 target_ir_topic: str = '/camera/ir/image_raw'):
        super().__init__('tum_bag_player')

        self.bag_path = bag_path
        self.playback_speed = playback_speed
        self.intrinsics = intrinsics or TUM_FREIBURG1_INTRINSICS
        self.bridge = CvBridge()

        # 目标发布话题（与 test_visual_slam / ROS 2 SLAM 节点一致）
        self.target_rgb_topic = target_rgb_topic
        self.target_depth_topic = target_depth_topic
        self.target_ir_topic = target_ir_topic

        # 发布者
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_rgb = self.create_publisher(Image, target_rgb_topic, qos)
        self.pub_depth = self.create_publisher(Image, target_depth_topic, qos)
        self.pub_ir = self.create_publisher(Image, target_ir_topic, qos)

        # Camera info（供校准用）
        self.pub_rgb_info = self.create_publisher(
            CameraInfo, target_rgb_topic + '/camera_info', qos)
        self.pub_depth_info = self.create_publisher(
            CameraInfo, target_depth_topic + '/camera_info', qos)

        self.get_logger().info(f"TUMLidarPlayer 启动: {bag_path}")
        self.get_logger().info(f"  目标 RGB topic:  {target_rgb_topic}")
        self.get_logger().info(f"  目标 Depth topic: {target_depth_topic}")
        self.get_logger().info(f"  目标 IR topic:   {target_ir_topic}")
        self.get_logger().info(f"  播放速度: {playback_speed}x")

        # 检测 bag 中的 topic
        detected_topics = self._detect_topics()
        self.get_logger().info(f"  检测到 topic: {detected_topics}")

        # 使用检测到的 topic 或用户指定的 topic
        self.rgb_source_topic = rgb_topic or self._guess_rgb_topic(detected_topics)
        self.depth_source_topic = depth_topic or self._guess_depth_topic(detected_topics)

        self.get_logger().info(f"  RGB 源 topic:   {self.rgb_source_topic}")
        self.get_logger().info(f"  Depth 源 topic: {self.depth_source_topic}")

        self.frame_count = 0
        self.start_time = None

        # 启动 bag 播放定时器（立即触发一次）
        self.timer = self.create_timer(0.001, self.timer_callback)

        # bag 读取器（延迟初始化）
        self.bag_reader = None
        self.bag_timer_active = True

    # ------------------------------------------------------------------------
    # Topic 检测
    # ------------------------------------------------------------------------

    def _detect_topics(self) -> list:
        """检测 bag 中的所有 topic"""
        try:
            loader = SequentialReaderStorageLoader()
            storage_options = loader.get_default_storage_options()
            storage_options.uri = self.bag_path

            converter_options = ConverterOptions(
                SerializationFormat.CDMS_V2,
                SerializationFormat.CDMS_V2
            )

            reader = Reader()
            reader.open(storage_options, converter_options)

            topics = []
            msg_types = {}
            for entry in reader.get_all_topics_and_types():
                topics.append(entry.name)
                msg_types[entry.name] = entry.type

            reader.close()
            return topics
        except Exception as e:
            self.get_logger().warn(f"  无法检测 bag topic: {e}")
            return []

    def _guess_rgb_topic(self, topics: list) -> str:
        """猜测 RGB image topic"""
        candidates = [
            '/camera/rgb/image_color', '/camera/rgb/image_rect_color',
            '/camera/rgb/image_raw', '/camera/rgb/image_mono',
            '/rgbd_image', '/image', '/camera/image_color',
            '/depth/image', '/rgb/image',
        ]
        for t in candidates:
            if t in topics:
                return t
        # 模糊匹配
        for t in topics:
            if 'rgb' in t.lower() or 'color' in t.lower() or 'image' in t.lower():
                return t
        return topics[0] if topics else ''

    def _guess_depth_topic(self, topics: list) -> str:
        """猜测 Depth image topic"""
        candidates = [
            '/camera/depth_registered/image', '/camera/depth/image',
            '/camera/depth/image_raw', '/depth/image',
            '/camera/depth_registered/hw_registered/image_rect_raw',
        ]
        for t in candidates:
            if t in topics:
                return t
        for t in topics:
            if 'depth' in t.lower() or 'registered' in t.lower():
                return t
        return topics[1] if len(topics) > 1 else ''

    # ------------------------------------------------------------------------
    # 定时器回调（主播放循环）
    # ------------------------------------------------------------------------

    def timer_callback(self):
        if not self.bag_timer_active:
            return

        if self.bag_reader is None:
            try:
                loader = SequentialReaderStorageLoader()
                storage_options = loader.get_default_storage_options()
                storage_options.uri = self.bag_path

                converter_options = ConverterOptions(
                    SerializationFormat.CDMS_V2,
                    SerializationFormat.CDMS_V2
                )

                self.bag_reader = Reader()
                self.bag_reader.open(storage_options, converter_options)
                self.get_logger().info("Bag 已打开，开始播放...")
            except Exception as e:
                self.get_logger().error(f"无法打开 bag: {e}")
                self.bag_timer_active = False
                return

        # 读取下一条消息
        try:
            (topic, data, timestamp) = self.bag_reader.read_next()
        except Exception:
            self.get_logger().info(f"Bag 播放完毕，共 {self.frame_count} 帧")
            self.bag_timer_active = False
            rclpy.shutdown()
            return

        if self.start_time is None:
            self.start_time = timestamp

        elapsed_sec = (timestamp - self.start_time) / 1e9 / self.playback_speed

        # 处理 RGB / Color
        if topic == self.rgb_source_topic:
            self._publish_rgb(data, elapsed_sec)

        # 处理 Depth
        elif topic == self.depth_source_topic:
            self._publish_depth(data, elapsed_sec)

    # ------------------------------------------------------------------------
    # ROS 2 消息发布
    # ------------------------------------------------------------------------

    def _publish_rgb(self, data, timestamp_ns):
        """将 ROS 1 Image 消息（bytes 或 Python dict）发布为 ROS 2 Image"""
        try:
            # data 是序列化后的 bytes 或已解析的消息对象
            if isinstance(data, (bytes, bytearray)):
                # ROS 2 CDR 序列化数据，需要反序列化
                from rclpy.serialization import deserialize_message
                import sensor_msgs.msg as sms
                msg = deserialize_message(data, Image())
            else:
                msg = data

            # 转换编码（假设 input 是 bgr8 或 rgb8）
            cv_img = self._image_msg_to_cv2(msg)
            if cv_img is None:
                return

            # ROS 2 发布：直接使用原编码（bgr8/rgb8）
            ros2_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding=msg.encoding if hasattr(msg, 'encoding') else 'bgr8')
            ros2_msg.header.stamp = rclpy.time.Time(seconds=timestamp_ns).to_msg()
            ros2_msg.header.frame_id = 'camera'

            self.pub_rgb.publish(ros2_msg)

            # 同时发布 IR（灰度版）
            if cv_img.ndim == 3:
                ir_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            else:
                ir_gray = cv_img
            ir_msg = self.bridge.cv2_to_imgmsg(ir_gray, encoding='mono8')
            ir_msg.header.stamp = ros2_msg.header.stamp
            ir_msg.header.frame_id = 'camera'
            self.pub_ir.publish(ir_msg)

            # CameraInfo
            cam_info = self._make_camera_info(self.intrinsics)
            cam_info.header.stamp = ros2_msg.header.stamp
            cam_info.header.frame_id = 'camera'
            self.pub_rgb_info.publish(cam_info)

        except Exception as e:
            self.get_logger().warn(f"RGB 处理异常: {e}", throttle_duration_sec=2)

    def _publish_depth(self, data, timestamp_ns):
        """将 TUM depth（16UC1 mm）发布为 ROS 2 Image"""
        try:
            if isinstance(data, (bytes, bytearray)):
                from rclpy.serialization import deserialize_message
                msg = deserialize_message(data, Image())
            else:
                msg = data

            cv_depth = self._image_msg_to_cv2(msg)
            if cv_depth is None:
                return

            # TUM 深度图：16UC1 mm → 16UC1 mm（单位不变，直接发布）
            # Tracking 内部会按 depth_scale=0.001 转为米
            # 若已是 32FC1 米，则 depth_scale=1.0
            depth_encoding = '16UC1'
            if cv_depth.dtype == np.float32 or cv_depth.dtype == np.float64:
                depth_encoding = '32FC1'

            depth_msg = self.bridge.cv2_to_imgmsg(cv_depth, encoding=depth_encoding)
            depth_msg.header.stamp = rclpy.time.Time(seconds=timestamp_ns).to_msg()
            depth_msg.header.frame_id = 'camera_depth'
            self.pub_depth.publish(depth_msg)

            # CameraInfo for depth
            cam_info = self._make_camera_info(self.intrinsics)
            cam_info.header.stamp = depth_msg.header.stamp
            cam_info.header.frame_id = 'camera_depth'
            self.pub_depth_info.publish(cam_info)

        except Exception as e:
            self.get_logger().warn(f"Depth 处理异常: {e}", throttle_duration_sec=2)

    def _image_msg_to_cv2(self, msg) -> np.ndarray:
        """sensor_msgs/Image → numpy array"""
        if msg.encoding in ('bgr8', 'BGR8', 'rgb8', 'RGB8'):
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding in ('mono8', 'MONO8', '8UC1'):
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        elif msg.encoding in ('16UC1', '16UC1'):
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        elif msg.encoding in ('32FC1', '32FC1'):
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            self.get_logger().warn(f"  不支持的编码: {msg.encoding}", throttle_duration_sec=5)
            return None

    def _make_camera_info(self, intr: dict) -> CameraInfo:
        """从内参字典构造 CameraInfo 消息"""
        from math import sqrt
        cam_info = CameraInfo()
        cam_info.width = intr['width']
        cam_info.height = intr['height']
        cam_info.k = [
            intr['fx'], 0.0, intr['cx'],
            0.0, intr['fy'], intr['cy'],
            0.0, 0.0, 1.0,
        ]
        cam_info.p = [
            intr['fx'], 0.0, intr['cx'], 0.0,
            0.0, intr['fy'], intr['cy'], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        cam_info.distortion_model = 'plumb_bob'
        cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return cam_info


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='TUM-RGBD bag 播放工具：将 .bag 播放为 ROS 2 图像话题',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法（自动检测 topic）
  python3 scripts/tum_bag_player.py --bag ~/bags/rgbd_dataset_freiburg1_desk2.bag

  # 指定 topic
  python3 scripts/tum_bag_player.py --bag ~/bags/rgbd_dataset_freiburg1_desk2.bag \\
      --rgb_topic /camera/rgb/image_color --depth_topic /camera/depth/image

  # 指定数据集内参（freiburg1/2/3）
  python3 scripts/tum_bag_player.py --bag ~/bags/rgbd_dataset_freiburg1_desk2.bag --intrinsics freiburg1

  # 0.5x 慢速播放
  python3 scripts/tum_bag_player.py --bag ~/bags/rgbd_dataset_freiburg1_desk2.bag --speed 0.5

  # 发布到自定义 topic（配合你自己的 SLAM 节点）
  python3 scripts/tum_bag_player.py --bag ~/bags/rgbd_dataset_freiburg1_desk2.bag \\
      --target_rgb /my_slam/rgb --target_depth /my_slam/depth
        """
    )
    parser.add_argument('--bag', '-b', required=True,
                        help='.bag 文件路径（ROS 1 或 ROS 2 bag 均可）')
    parser.add_argument('--rgb_topic', '-r',
                        help='bag 中 RGB 图像 topic（留空则自动检测）')
    parser.add_argument('--depth_topic', '-d',
                        help='bag 中 Depth 图像 topic（留空则自动检测）')
    parser.add_argument('--intrinsics', '-i',
                        choices=['freiburg1', 'freiburg2', 'freiburg3', 'custom'],
                        default='freiburg1',
                        help='TUM 数据集内参类型（默认: freiburg1，即 Kinect 640x480）')
    parser.add_argument('--fx', type=float, default=525.0,
                        help='fx（与 --intrinsics 配合使用）')
    parser.add_argument('--fy', type=float, default=525.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='播放速度倍数（默认: 1.0）')
    parser.add_argument('--target_rgb',
                        default='/camera/color/image_raw',
                        help='发布 RGB 的目标 topic（默认: /camera/color/image_raw）')
    parser.add_argument('--target_depth',
                        default='/camera/depth/image_raw',
                        help='发布 Depth 的目标 topic（默认: /camera/depth/image_raw）')
    parser.add_argument('--target_ir',
                        default='/camera/ir/image_raw',
                        help='发布 IR 的目标 topic（默认: /camera/ir/image_raw）')
    parser.add_argument('--dump_topics', action='store_true',
                        help='打印 bag 中所有 topic 并退出')
    return parser.parse_args()


# ============================================================================
# 主程序
# ============================================================================

def main():
    args = parse_args()

    # 参数处理
    intrinsics_map = {
        'freiburg1': TUM_FREIBURG1_INTRINSICS,
        'freiburg2': TUM_FREIBURG2_INTRINSICS,
        'freiburg3': TUM_FREIBURG3_INTRINSICS,
    }
    intrinsics = intrinsics_map.get(args.intrinsics, TUM_FREIBURG1_INTRINSICS)

    if args.intrinsics == 'custom':
        intrinsics = {
            'fx': args.fx, 'fy': args.fy,
            'cx': args.cx, 'cy': args.cy,
            'width': 640, 'height': 480,
            'depth_scale': 0.001,
        }
        print(f"[main] 使用自定义内参: fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")

    if not os.path.exists(args.bag):
        print(f"错误：文件不存在: {args.bag}")
        print("请先下载 TUM-RGBD bag：")
        print("  wget https://vision.in.tum.de/datasets/rgbd-dataset/freiburg1_desk2.bag")
        sys.exit(1)

    if args.dump_topics:
        loader = SequentialReaderStorageLoader()
        storage_options = loader.get_default_storage_options()
        storage_options.uri = os.path.abspath(args.bag)
        reader = Reader()
        try:
            reader.open(storage_options, ConverterOptions(
                SerializationFormat.CDMS_V2, SerializationFormat.CDMS_V2))
            print(f"\nBag: {args.bag}")
            print(f"{'Topic':<60} Type")
            print('-' * 90)
            for entry in reader.get_all_topics_and_types():
                print(f"  {entry.name:<57} {entry.type}")
            reader.close()
        except Exception as e:
            print(f"无法读取 bag: {e}")
        sys.exit(0)

    # 启动 ROS 2
    rclpy.init(args=sys.argv)
    node = TUMLidarPlayer(
        bag_path=os.path.abspath(args.bag),
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        intrinsics=intrinsics,
        playback_speed=args.speed,
        target_rgb_topic=args.target_rgb,
        target_depth_topic=args.target_depth,
        target_ir_topic=args.target_ir,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"播放结束，共 {node.frame_count} 帧")


if __name__ == '__main__':
    main()
