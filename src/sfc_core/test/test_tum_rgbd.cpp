/**
 * test_tum_rgbd.cpp
 *
 * TUM-RGBD Benchmark Player (Pure VO Mode)
 * ========================================
 * 读取 TUM-RGBD 数据集（rgb/ + depth/ + timestamp files），
 * 以真实时间戳驱动 SFC Tracking (VO only)，输出 TUM 格式轨迹文件，
 * 配合 evo 工具进行 APE / RPE 评估。
 *
 * 编译后使用示例：
 *   # fr1/desk（使用 TUM 默认内参）
 *   ./test_tum_rgbd --dataset /path/to/fr1_desk
 *
 *   # 指定相机内参（如果用自己的相机标定结果）
 *   ./test_tum_rgbd --dataset /path/to/fr1_desk --fx 525.0 --fy 525.0 --cx 319.5 --cy 239.5
 *
 *   # 评估（先下载 groundtruth.txt 到数据集目录）
 *   evo_ape tum /path/to/fr1_desk/groundtruth.txt sfc_traj.tum \
 *       -a --no_warnings
 *
 *   evo_rpe tum /path/to/fr1_desk/groundtruth.txt sfc_traj.tum \
 *       --delta 1 --delta_unit m -a -va --plot
 */

#ifdef sfc_core
#undef sfc_core
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <random>
#include <numeric>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "sfc_core/frontend/tracking.hpp"
#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/config/config_loader.hpp"

using namespace sfc_core;

// ============================================================================
// TUM-RGBD 数据格式
// ============================================================================
// timestamp filename
//   1305031102.172958 1305031102_172958.png
//
// groundtruth.txt:
// timestamp tx ty tz qx qy qz qw

struct TUMFrame {
    double timestamp;   // 秒
    std::string rgb_name;
    std::string depth_name;
    cv::Mat rgb;
    cv::Mat depth;      // 16UC1 mm
};

struct TUMGroundtruth {
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

// ============================================================================
// 文件系统工具
// ============================================================================

#ifdef _WIN32
#include <windows.h>
#define stat_struct struct _stat
#define stat_func _stat
#else
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#define stat_struct struct stat
#define stat_func stat
#endif

static bool fileExists(const std::string& path) {
    stat_struct st;
    return stat_func(path.c_str(), &st) == 0;
}

static std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

// ============================================================================
// TUM 时间戳解析
// ============================================================================

static double parseTUMTimestamp(const std::string& s) {
    return std::stod(s);
}

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, delim)) {
        parts.push_back(token);
    }
    return parts;
}

// ============================================================================
// TUM 数据集加载
// ============================================================================

static bool loadTUMDataset(const std::string& dataset_dir,
                           std::vector<TUMFrame>& frames,
                           std::vector<TUMGroundtruth>& groundtruth,
                           double start_offset = 0.0) {
    std::string rgb_assoc_file = joinPath(dataset_dir, "rgb.txt");
    std::string depth_assoc_file = joinPath(dataset_dir, "depth.txt");
    std::string gt_file = joinPath(dataset_dir, "groundtruth.txt");

    std::map<double, std::string> rgb_map;
    {
        std::ifstream f(rgb_assoc_file);
        if (!f.is_open()) {
            std::cerr << "[loadTUMDataset] 无法打开: " << rgb_assoc_file << "\n";
            return false;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto parts = split(line, ' ');
            if (parts.size() < 2) continue;
            double ts = parseTUMTimestamp(parts[0]);
            if (ts < start_offset) continue;
            rgb_map[ts] = parts[1];
        }
    }

    std::map<double, std::string> depth_map;
    {
        std::ifstream f(depth_assoc_file);
        if (!f.is_open()) {
            std::cerr << "[loadTUMDataset] 无法打开: " << depth_assoc_file << "\n";
            return false;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto parts = split(line, ' ');
            if (parts.size() < 2) continue;
            double ts = parseTUMTimestamp(parts[0]);
            depth_map[ts] = parts[1];
        }
    }

    std::string rgb_dir = joinPath(dataset_dir, "rgb");
    std::string depth_dir = joinPath(dataset_dir, "depth");

    for (auto& rgb_entry : rgb_map) {
        double ts = rgb_entry.first;
        std::string rgb_name = rgb_entry.second;
        if (rgb_name.size() > 4 &&
            ((rgb_name[3] == '/' || rgb_name[3] == '\\') &&
             rgb_name[0] == 'r' && rgb_name[1] == 'g' && rgb_name[2] == 'b')) {
            rgb_name = rgb_name.substr(4);
        }
        if (rgb_name.empty()) continue;

        auto it = depth_map.lower_bound(ts);
        double best_ts = ts;
        std::string best_depth;
        if (it != depth_map.end()) {
            best_depth = it->second;
            best_ts = it->first;
        }
        if (it != depth_map.begin()) {
            auto it_prev = std::prev(it);
            if (std::fabs(it_prev->first - ts) < std::fabs(best_ts - ts)) {
                best_depth = it_prev->second;
                best_ts = it_prev->first;
            }
        }

        if (best_depth.empty()) continue;
        if (best_depth.size() > 6 &&
            ((best_depth[5] == '/' || best_depth[5] == '\\') &&
             best_depth[0] == 'd' && best_depth[1] == 'e' &&
             best_depth[2] == 'p' && best_depth[3] == 't' && best_depth[4] == 'h')) {
            best_depth = best_depth.substr(6);
        }
        if (best_depth.empty()) continue;

        TUMFrame frame;
        frame.timestamp = ts;
        frame.rgb_name = rgb_name;
        frame.depth_name = best_depth;

        frame.rgb = cv::imread(joinPath(rgb_dir, rgb_name), cv::IMREAD_COLOR);
        if (frame.rgb.empty()) {
            std::cerr << "[loadTUMDataset] 无法读取 RGB: " << joinPath(rgb_dir, rgb_name) << "\n";
            continue;
        }

        frame.depth = cv::imread(joinPath(depth_dir, best_depth), cv::IMREAD_UNCHANGED);
        if (frame.depth.empty()) {
            std::cerr << "[loadTUMDataset] 无法读取 Depth: " << joinPath(depth_dir, best_depth) << "\n";
            continue;
        }

        frames.push_back(std::move(frame));
    }

    if (fileExists(gt_file)) {
        std::ifstream f(gt_file);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto parts = split(line, ' ');
            if (parts.size() < 8) continue;
            TUMGroundtruth gt;
            gt.timestamp = parseTUMTimestamp(parts[0]);
            gt.t = Eigen::Vector3d(std::stod(parts[1]), std::stod(parts[2]), std::stod(parts[3]));
            gt.q = Eigen::Quaterniond(std::stod(parts[7]),
                                      std::stod(parts[4]), std::stod(parts[5]), std::stod(parts[6]));
            if (!gt.q.isApprox(Eigen::Quaterniond::Identity(), 1e-6)) {
                gt.q.normalize();
            }
            groundtruth.push_back(gt);
        }
        std::cout << "[loadTUMDataset] Groundtruth 已加载: " << groundtruth.size() << " 条\n";
    } else {
        std::cout << "[loadTUMDataset] groundtruth.txt 未找到，跳过对比评估\n";
    }

    if (frames.empty()) {
        std::cerr << "[loadTUMDataset] 未加载到任何帧！\n";
        return false;
    }

    std::cout << "[loadTUMDataset] 共加载 " << frames.size() << " 帧\n";
    std::cout << "  RGB 目录: " << rgb_dir << "\n";
    std::cout << "  Depth 目录: " << depth_dir << "\n";
    std::cout << "  首帧时间戳: " << std::fixed << std::setprecision(6) << frames.front().timestamp << " s\n";
    std::cout << "  末帧时间戳: " << std::fixed << std::setprecision(6) << frames.back().timestamp << " s\n";
    return true;
}

// ============================================================================
// 相机内参加载（支持命令行覆盖）
// ============================================================================

struct CameraK {
    double fx, fy, cx, cy;
    int width, height;
    cv::Mat K;
    cv::Mat D;
    CameraK() : fx(0.0), fy(0.0), cx(0.0), cy(0.0), width(640), height(480) {
        K = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
        D = cv::Mat(1, 5, CV_64F, cv::Scalar(0));
    }
};

// 内参优先级（高→低）：命令行显式参数 > --camera 指定的 yaml > 报错退出。
// 这条链是本文件唯一的取参路径，任何内参都不得再以字面量形式出现（R-019）。
static bool resolveIntrinsics(const std::string& camera_yaml,
                               double cli_fx, double cli_fy, double cli_cx, double cli_cy,
                               int cli_width, int cli_height,
                               CameraK& cam) {
    sfc_core::config::CameraIntrinsicsConfig ic;
    if (!camera_yaml.empty()) {
        ic = sfc_core::config::loadCameraIntrinsicsConfig(camera_yaml);
        if (!ic.isValid()) {
            std::cerr << "[resolveIntrinsics] 无法从 " << camera_yaml
                      << " 读取有效内参（fx/fy 必须为正）\n";
            return false;
        }
        std::cout << "[resolveIntrinsics] 内参来源: " << camera_yaml << "\n";
    } else {
        std::cerr << "[resolveIntrinsics] 未指定 --camera，且无内置默认内参\n"
                  << "  用 config/camera_fr1.yaml / camera_fr2.yaml / camera_fr3.yaml 之一，\n"
                  << "  或显式传 --fx --fy --cx --cy。\n";
        return false;
    }

    // 命令行覆盖：仅在显式给出（>0）时生效
    if (cli_fx > 0.0) ic.fx = cli_fx;
    if (cli_fy > 0.0) ic.fy = cli_fy;
    if (cli_cx > 0.0) ic.cx = cli_cx;
    if (cli_cy > 0.0) ic.cy = cli_cy;
    if (cli_width > 0) ic.width = cli_width;
    if (cli_height > 0) ic.height = cli_height;

    cam.fx = ic.fx;
    cam.fy = ic.fy;
    cam.cx = ic.cx;
    cam.cy = ic.cy;
    cam.width = ic.width;
    cam.height = ic.height;
    cam.K = ic.makeK();
    cam.D = ic.makeD();

    std::cout << "[resolveIntrinsics] fx=" << cam.fx << " fy=" << cam.fy
              << " cx=" << cam.cx << " cy=" << cam.cy
              << " (" << cam.width << "x" << cam.height << ")\n"
              << "  k1=" << ic.k1 << " k2=" << ic.k2
              << " p1=" << ic.p1 << " p2=" << ic.p2 << " k3=" << ic.k3 << "\n";
    return true;
}

// ============================================================================
// TUM 格式轨迹写入
// ============================================================================

static bool writeTUMTrajectory(const std::string& path,
                                const std::vector<double>& timestamps,
                                const std::vector<Eigen::Vector3d>& traj_t,
                                const std::vector<Eigen::Quaterniond>& traj_q,
                                const std::vector<int>& inliers,
                                double scale = 1.0,
                                const std::vector<int>* kf_indices = nullptr) {
    if (traj_t.size() != timestamps.size() || traj_t.size() != traj_q.size()) {
        std::cerr << "[writeTUMTrajectory] 长度不匹配！\n";
        return false;
    }

    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[writeTUMTrajectory] 无法写入: " << path << "\n";
        return false;
    }
    f << "# timestamp tx ty tz qx qy qz qw\n";

    if (kf_indices != nullptr && !kf_indices->empty()) {
        for (int fi : *kf_indices) {
            if (fi < 0 || fi >= static_cast<int>(traj_t.size())) continue;
            const auto& q = traj_q[fi];
            Eigen::Vector3d t_scaled = traj_t[fi] * scale;
            f << std::fixed << std::setprecision(9) << timestamps[fi] << " "
              << std::setprecision(9) << t_scaled.x() << " " << t_scaled.y() << " " << t_scaled.z() << " "
              << std::setprecision(9) << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
    } else {
        for (size_t i = 0; i < traj_t.size(); ++i) {
            const auto& q = traj_q[i];
            Eigen::Vector3d t_scaled = traj_t[i] * scale;
            f << std::fixed << std::setprecision(9) << timestamps[i] << " "
              << std::setprecision(9) << t_scaled.x() << " " << t_scaled.y() << " " << t_scaled.z() << " "
              << std::setprecision(9) << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
    }
    f.close();
    std::cout << "[writeTUMTrajectory] 已保存: " << path << "（" << traj_t.size() << " 帧）\n";
    return true;
}

// ============================================================================
// 参数解析
// ============================================================================

struct Args {
    std::string dataset_dir;
    std::string output_traj = "sfc_traj.tum";
    std::string vocabulary_path;  // ORB 词汇表路径，用于 BoW 匹配
    std::string camera_yaml = "config/camera_fr1.yaml";  // 相机内参来源（R-019）
    std::string config_dir = "config";                   // 其余运行参数来源（R-019）
    int max_frames = 0;
    double start_offset = 0.0;

    // 0 表示"未显式指定"，由 resolveIntrinsics 从 camera_yaml 填充
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    int img_width = 0, img_height = 0;

    int n_features = 0;      // 0 = 用 config/tracking.yaml
    int min_features = 0;    // 0 = 用 config/tracking.yaml
    double min_depth = -1.0; // <0 = 用 config/tracking.yaml
    double max_depth = -1.0; // <0 = 用 config/tracking.yaml
    double depth_scale = -1.0; // <0 = 用 config/tracking.yaml
    double kf_trans_thresh = -1.0; // <0 = 用 config/tracking.yaml
    double kf_rot_thresh = -1.0;   // <0 = 用 config/tracking.yaml
    double trajectory_scale = 1.0;
    bool verbose = false;
    // 默认纯 VO，保持与既有诊断日志/基线可复现；--slam 才启用回环检测 + PGO。
    bool enable_slam = false;
};

static Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) args.dataset_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.output_traj = argv[++i];
        else if (arg == "--config_dir" && i + 1 < argc) args.config_dir = argv[++i];
        else if (arg == "--camera" && i + 1 < argc) args.camera_yaml = argv[++i];
        else if (arg == "--max_frames" && i + 1 < argc) args.max_frames = std::atoi(argv[++i]);
        else if (arg == "--start_offset" && i + 1 < argc) args.start_offset = std::atof(argv[++i]);
        else if (arg == "--fx" && i + 1 < argc) args.fx = std::atof(argv[++i]);
        else if (arg == "--fy" && i + 1 < argc) args.fy = std::atof(argv[++i]);
        else if (arg == "--cx" && i + 1 < argc) args.cx = std::atof(argv[++i]);
        else if (arg == "--cy" && i + 1 < argc) args.cy = std::atof(argv[++i]);
        else if (arg == "--width" && i + 1 < argc) args.img_width = std::atoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) args.img_height = std::atoi(argv[++i]);
        else if (arg == "--n_features" && i + 1 < argc) args.n_features = std::atoi(argv[++i]);
        else if (arg == "--kf_trans" && i + 1 < argc) args.kf_trans_thresh = std::atof(argv[++i]);
        else if (arg == "--kf_rot" && i + 1 < argc) args.kf_rot_thresh = std::atof(argv[++i]);
        else if (arg == "--depth_scale" && i + 1 < argc) args.depth_scale = std::atof(argv[++i]);
        else if (arg == "--vocabulary" && i + 1 < argc) args.vocabulary_path = argv[++i];
        else if (arg == "--verbose" || arg == "-v") args.verbose = true;
        else if (arg == "--slam") args.enable_slam = true;
        else if (arg == "--trajectory_scale" && i + 1 < argc) args.trajectory_scale = std::atof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "TUM-RGBD Benchmark Runner (Pure VO)\n\n";
            std::cout << "Usage: test_tum_rgbd --dataset /path/to/fr1_desk [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --dataset      <path>   数据集根目录（含 rgb/ depth/）\n";
            std::cout << "  --config_dir   <dir>    yaml 配置目录（默认: config）\n";
            std::cout << "  --camera       <yaml>  相机内参 yaml（默认: config/camera_fr1.yaml）\n";
            std::cout << "  --output       <file>   输出轨迹 TUM 文件（默认: sfc_traj.tum）\n";
            std::cout << "  --max_frames   <N>      最大帧数（0=不限）\n";
            std::cout << "  --start_offset <s>      跳过开头若干秒\n";
            std::cout << "  --fx/fy/cx/cy          覆盖 yaml 中的相机内参（不传则用 yaml）\n";
            std::cout << "  --width/height         覆盖 yaml 中的图像分辨率\n";
            std::cout << "  --n_features   <N>      覆盖 yaml 中的每帧特征数\n";
            std::cout << "  --kf_trans     <m>      覆盖 yaml 中的关键帧平移阈值\n";
            std::cout << "  --kf_rot       <rad>    覆盖 yaml 中的关键帧旋转阈值\n";
            std::cout << "  --depth_scale  <float>  覆盖 yaml 中的深度缩放（TUM PNG: 0.0002）\n";
            std::cout << "  --vocabulary   <path>   ORB 词汇表 .txt 路径（影响 BoW 跟踪质量）\n";
            std::cout << "  --trajectory_scale <float> 轨迹事后尺度校正（默认: 1.0）\n";
            std::cout << "  --slam                  启用回环检测 + 位姿图优化（默认纯 VO）\n";
            std::cout << "\n不传任何覆盖参数时，全部运行参数取自 config/*.yaml（R-019）。\n";
            std::cout << "\n示例:\n";
            std::cout << "  test_tum_rgbd --dataset ~/datasets/fr1_desk\n";
            std::cout << "  test_tum_rgbd --dataset ~/datasets/fr2_desk --camera config/camera_fr2.yaml --max_frames 500\n";
            std::cout << "  test_tum_rgbd --dataset ~/datasets/fr1_room --kf_trans 0.1\n";
            std::exit(0);
        }
    }
    return args;
}

// ============================================================================
// 主程序
// ============================================================================

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    if (args.dataset_dir.empty()) {
        std::cerr << "错误: 必须指定 --dataset 参数\n";
        std::cerr << "用法: test_tum_rgbd --dataset /path/to/fr1_desk\n";
        std::cerr << "查看帮助: test_tum_rgbd --help\n";
        return 1;
    }

    std::cout << "============================================================\n";
    std::cout << "  TUM-RGBD Benchmark Runner (Pure VO)\n";
    std::cout << "============================================================\n";
    std::cout << "  数据集: " << args.dataset_dir << "\n";
    std::cout << "  输出轨迹: " << args.output_traj << "\n";

    std::vector<TUMFrame> frames;
    std::vector<TUMGroundtruth> groundtruth;
    if (!loadTUMDataset(args.dataset_dir, frames, groundtruth, args.start_offset)) {
        return 1;
    }

    CameraK cam;
    if (!resolveIntrinsics(args.camera_yaml, args.fx, args.fy, args.cx, args.cy,
                           args.img_width, args.img_height, cam)) {
        return 1;
    }

    int actual_w = frames.front().rgb.cols;
    int actual_h = frames.front().rgb.rows;
    bool need_resize = (actual_w != cam.width || actual_h != cam.height);
    if (need_resize) {
        std::cout << "[main] 图像分辨率 " << actual_w << "x" << actual_h
                  << " ≠ 内参分辨率 " << cam.width << "x" << cam.height
                  << "，将在处理时 resize\n";
        double sx = static_cast<double>(actual_w) / cam.width;
        double sy = static_cast<double>(actual_h) / cam.height;
        cam.fx *= sx; cam.fy *= sy;
        cam.cx *= sx; cam.cy *= sy;
        cam.width = actual_w; cam.height = actual_h;
        cam.K.at<double>(0, 0) = cam.fx;
        cam.K.at<double>(1, 1) = cam.fy;
        cam.K.at<double>(0, 2) = cam.cx;
        cam.K.at<double>(1, 2) = cam.cy;
    }

    // ---- 运行参数：唯一来源是 config/*.yaml（R-019）----
    // 命令行只做覆盖，不提供任何"默认值"。缺文件时 loadAllConfigs 会告警并回退
    // 结构体默认值，因此这里不再重复写一份等价的字面量。
    sfc_core::config::AllConfigs all_cfg = sfc_core::config::loadAllConfigs(args.config_dir);
    sfc_core::frontend::TrackingConfig tcfg = all_cfg.tracking;
    sfc_core::feature::OctreeORBConfig& ext_cfg = tcfg.extractor_config;
    sfc_core::feature::MatcherConfig& mat_cfg = tcfg.matcher_config;

    tcfg.K = cam.K;
    tcfg.D = cam.D;

    // 命令行覆盖（只在显式给出时生效；0 / 负值代表"用 yaml"）
    if (args.n_features > 0) {
        ext_cfg.n_features = args.n_features;
    }
    if (args.min_features > 0) {
        tcfg.min_features = args.min_features;
    }
    if (args.min_depth > 0.0) {
        tcfg.min_depth = args.min_depth;
    }
    if (args.max_depth > 0.0) {
        tcfg.max_depth = args.max_depth;
    }
    if (args.depth_scale > 0.0) {
        tcfg.depth_scale = args.depth_scale;
    }
    if (args.kf_trans_thresh > 0.0) {
        tcfg.min_keyframe_translation = args.kf_trans_thresh;
    }
    if (args.kf_rot_thresh > 0.0) {
        tcfg.min_keyframe_rotation = args.kf_rot_thresh;
    }

    // R3：这里不再强制打开 disable_undistort。
    // 原先的顾虑是「去畸变后 compute3DPoints 用归一化坐标当像素用 → 深度全读成 0」，
    // 但现在的 compute3DPoints 把两件事分开了：深度采样一律用**畸变像素**
    // （深度图与畸变 RGB 对齐），只有反投影才用理想像素。所以
    // disable_undistort=false（做去畸变）是正确配置，且 TUM fr1 的强畸变必须去畸变，
    // 否则 2D/3D 两侧都在畸变坐标里"自洽"，方向偏差会被掩盖。
    // 仍然保留一条告警：真有人在 D 非空时把 disable_undistort 打开，说明配置错位。
    if (tcfg.disable_undistort && !tcfg.D.empty()) {
        std::cerr << "[main] 警告: " << args.config_dir
                  << "/tracking.yaml 的 disable_undistort=true 但 D 非空（fr1 有强畸变）。"
                  << "畸变会被静默忽略，建议改成 false。\n";
    }
    tcfg.verbose_diag = args.verbose;

    std::cout << "\n[Config] 来源: " << args.config_dir << "/tracking.yaml\n"
              << "  n_features=" << ext_cfg.n_features
              << " min_features=" << tcfg.min_features
              << " depth=[" << tcfg.min_depth << ", " << tcfg.max_depth << "]"
              << " depth_scale=" << tcfg.depth_scale << "\n"
              << "  kf_trans=" << tcfg.min_keyframe_translation
              << " kf_rot=" << tcfg.min_keyframe_rotation
              << " forced_kf_interval=" << tcfg.forced_keyframe_interval << "\n"
              << "  fast=[" << ext_cfg.ini_threshold_fast << ", " << ext_cfg.min_threshold_fast
              << "] lowe=" << mat_cfg.lowe_ratio
              << " max_hamming=" << mat_cfg.max_hamming_dist
              << " cross_check=" << (mat_cfg.cross_check ? 1 : 0) << "\n";

    auto tracking = std::make_unique<sfc_core::frontend::Tracking>(tcfg);

    // 加载 ORB 词汇表（用于 BoW 特征匹配和重定位）
    // 不加载时所有帧的 TrackRefKF bow_matches=0，跟踪会退化成全靠 RELOC
    if (!args.vocabulary_path.empty()) {
        tracking->setBoWVocabulary(args.vocabulary_path);
    } else {
        std::cerr << "[main] 警告: 未指定 --vocabulary，TrackRefKF 将跳过 BoW 匹配\n"
                  << "[main] 建议下载 ORB 词汇表并使用 --vocabulary 指定路径\n";
    }

    // ---- 完整 SLAM：回环检测 + 位姿图优化 ----
    // 只有显式 --slam 才启用。默认纯 VO 是为了让已有诊断日志和基线可复现：
    // 启用回环后同一条序列的轨迹会被 PGO 改写，与旧日志逐帧对比就失去意义。
    if (args.enable_slam) {
        if (args.vocabulary_path.empty()) {
            // LoopDetector 内部有自己的 PlaceRecognition：它会从已登记关键帧的描述子里
            // 增量构建词表（collectForVocabulary → tryBuildVocabulary），词表没建好时
            // detect() 退回简单 BoW 共同词计数。所以外部词表不是硬性前提，
            // 但有词表时位置识别的候选质量明显更好——这里只告警，不拦。
            std::cerr << "[main] 提示: 未指定 --vocabulary，回环检测将依赖内部增量词表"
                      << "（共同词计数），候选质量低于 ORB 词表\n";
        }
        // 内参由 initLoopDetection 内部从 TrackingConfig 的 K/D 填入（
        // LoopDetectorConfig 的 camera_K/camera_D 不是 yaml 字段，loader 不读）。
        tracking->initLoopDetection(all_cfg.loop_detector, all_cfg.pose_graph_optimizer);
        std::cout << "[Config] 模式: 完整 SLAM（回环检测 + PGO）"
                  << " min_loop_separation=" << all_cfg.loop_detector.min_loop_separation
                  << " min_geometric_inliers=" << all_cfg.loop_detector.min_geometric_inliers
                  << " pgo_max_iterations=" << all_cfg.pose_graph_optimizer.max_iterations
                  << "\n";
    } else {
        std::cout << "[Config] 模式: 纯 VO（未启用回环检测，加 --slam 开启）\n";
    }
    int n = (args.max_frames > 0)
                ? std::min(args.max_frames, static_cast<int>(frames.size()))
                : static_cast<int>(frames.size());

    std::vector<double> timestamps;
    std::vector<Eigen::Vector3d> traj_t;
    std::vector<Eigen::Quaterniond> traj_q;
    std::vector<int> inliers;
    std::vector<bool> is_kf;
    double base_ts = frames.front().timestamp;

    std::cout << "\n[Processing] 共 " << n << " 帧...\n";
    std::cout << "  内参: fx=" << cam.fx << " fy=" << cam.fy
              << " cx=" << cam.cx << " cy=" << cam.cy << "\n";
    const double kf_trans_used = tcfg.min_keyframe_translation;
    const double kf_rot_used = tcfg.min_keyframe_rotation;
    std::cout << "  关键帧阈值: trans=" << kf_trans_used
              << " m  rot=" << kf_rot_used << " rad\n";
    std::cout << std::string(70, '-') << "\n";

    auto last_log_time = std::chrono::steady_clock::now();

    traj_t.reserve(n);
    traj_q.reserve(n);

    for (int i = 0; i < n; ++i) {
        TUMFrame& fd = frames[i];

        cv::Mat rgb = fd.rgb.clone();
        cv::Mat depth = fd.depth.clone();

        if (need_resize) {
            cv::resize(rgb, rgb, cv::Size(cam.width, cam.height));
            cv::resize(depth, depth, cv::Size(cam.width, cam.height));
        }

        cv::Mat ir;
        cv::cvtColor(rgb, ir, cv::COLOR_BGR2GRAY);

        double ts = fd.timestamp - base_ts;
        sfc_core::frontend::Pose pose = tracking->track(rgb, depth, ir, ts);

        // 关键帧标记必须来自 Tracking 内部的真实判定，不能再用 i % N == 0 推测：
        // 那只是"理论上该建了"，与 needNewKeyFrame() 的实际决策无关，
        // 会让 _kf 轨迹与真实地图内容脱节，评测时看不出关键帧是否真的稀疏。
        const bool this_is_kf = tracking->isLastFrameKeyFrame();

        timestamps.push_back(fd.timestamp);
        traj_t.push_back(pose.t);
        traj_q.push_back(pose.q);
        inliers.push_back(tracking->getLastInlierCount());
        is_kf.push_back(this_is_kf);

        // 前 30 帧逐帧诊断，仅在 --verbose 时输出（R-017）
        if (args.verbose && i < 30) {
            std::clog << "  [DBG " << i << "] pose=(" << std::fixed << std::setprecision(4)
                      << pose.t.x() << "," << pose.t.y() << "," << pose.t.z()
                      << ") inliers=" << tracking->getLastInlierCount()
                      << " nGood=" << tracking->getLastNGood()
                      << " state=" << tracking->getTrackingState()
                      << (this_is_kf ? " KF" : "")
                      << std::endl;
        }

        auto now = std::chrono::steady_clock::now();
        if (i == 0 || i == n - 1 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count() >= 2000) {
            double frame_dt = (i > 0) ? (ts - timestamps[i - 1]) : 0.0;
            Eigen::Vector3d frame_delta = (i > 0) ? (pose.t - traj_t[i - 1]) : Eigen::Vector3d::Zero().eval();

            std::string gt_info;
            if (!groundtruth.empty()) {
                const double TIME_TOL = 0.1;
                double best_dt = 1e9;
                Eigen::Vector3d gt_t_curr;
                for (const auto& gt : groundtruth) {
                    double dt = std::abs(gt.timestamp - ts);
                    if (dt < best_dt) { best_dt = dt; gt_t_curr = gt.t; }
                }
                if (best_dt < TIME_TOL) {
                    double prev_ts = timestamps[i - 1];
                    double best_dt2 = 1e9;
                    Eigen::Vector3d gt_t_prev;
                    for (const auto& gt : groundtruth) {
                        double dt = std::fabs(gt.timestamp - prev_ts);
                        if (dt < best_dt2) { best_dt2 = dt; gt_t_prev = gt.t; }
                    }
                    if (best_dt2 < TIME_TOL) {
                        double gt_delta = (gt_t_curr - gt_t_prev).norm();
                        gt_info = " | GT_frame_delta=" + std::to_string(gt_delta).substr(0, 8) + "m";
                        if (frame_delta.norm() > 0.001) {
                            gt_info += " (SLAM/GT=" + std::to_string(frame_delta.norm() / gt_delta).substr(0, 6) + "x)";
                        }
                    }
                }
            }

            std::cout << "  [" << std::setw(5) << i << "/" << n << "]"
                      << " t=" << std::fixed << std::setprecision(3) << ts << " s"
                      << " |t|=" << std::fixed << std::setprecision(5) << frame_delta.norm() << " m"
                      << " pose=(" << std::fixed << std::setprecision(3)
                      << pose.t.x() << ", " << pose.t.y() << ", " << pose.t.z() << ")"
                      << " inliers=" << std::setw(4) << inliers.back()
                      << " kf=" << (is_kf.back() ? "Y" : "N")
                      << gt_info
                      << "\n";
            last_log_time = now;
        }

        if (i % 30 == 0 && !rgb.empty()) {
            // ★★★ 调试：统计关键点分布（使用去畸变坐标，更准确反映特征在全图的真实分布）★★★
            auto kps_undist = tracking->getCurrentUndistortedKeypoints();
            const auto& kps = (!kps_undist.empty() ? kps_undist : tracking->getCurrentKeypoints());
            if (args.verbose && !kps.empty()) {
                int ul = 0, ur = 0, ll = 0, lr = 0;
                int center_x = rgb.cols / 2, center_y = rgb.rows / 2;
                for (const auto& kp : kps) {
                    if (kp.pt.x < center_x && kp.pt.y < center_y) ul++;
                    else if (kp.pt.x >= center_x && kp.pt.y < center_y) ur++;
                    else if (kp.pt.x < center_x && kp.pt.y >= center_y) ll++;
                    else lr++;
                }
                std::cerr << "\n[VIZ-DBG] Frame " << i
                          << " img=" << rgb.cols << "x" << rgb.rows
                          << " kps=" << kps.size()
                          << " UL=" << ul << "(" << static_cast<int>(static_cast<double>(ul) * 100.0 / kps.size()) << "%)"
                          << " UR=" << ur << "(" << static_cast<int>(static_cast<double>(ur) * 100.0 / kps.size()) << "%)"
                          << " LL=" << ll << "(" << static_cast<int>(static_cast<double>(ll) * 100.0 / kps.size()) << "%)"
                          << " LR=" << lr << "(" << static_cast<int>(static_cast<double>(lr) * 100.0 / kps.size()) << "%)"
                          << std::endl;
                if (ul > kps.size() * 0.6) {
                    std::cerr << "  !!! >60% keypoints in UL - coordinate problem!" << std::endl;
                }
            }

            // 可视化：优先使用去畸变后的关键点坐标
            auto kps_vis = tracking->getCurrentUndistortedKeypoints();
            const auto& vis_kps = (!kps_vis.empty() ? kps_vis : tracking->getCurrentKeypoints());
            cv::Mat vis = rgb.clone();
            if (!vis_kps.empty()) {
                cv::drawKeypoints(vis, vis_kps, vis, cv::Scalar(0, 255, 100),
                                  cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
            }
            std::ostringstream ss;
            ss << "Frame: " << i << "/" << n;
            cv::putText(vis, ss.str(), cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::imshow("TUM-RGBD Tracking", vis);
#ifdef _WIN32
            cv::waitKey(30);
#else
            cv::waitKey(1);
#endif
        }
    }

    cv::destroyAllWindows();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  处理完成: " << n << " 帧\n";

    int kf_count = 0;
    for (bool k : is_kf) if (k) ++kf_count;

    if (traj_t.size() >= 2) {
        double path_length = 0;
        for (size_t i = 1; i < traj_t.size(); ++i)
            path_length += (traj_t[i] - traj_t[i - 1]).norm();

        Eigen::Vector3d start = traj_t.front();
        Eigen::Vector3d end = traj_t.back();
        double drift = (end - start).norm();
        double drift_ratio = (path_length > 0.01) ? (drift / path_length) : 0.0;

        std::cout << "  路径总长: " << std::fixed << std::setprecision(3) << path_length << " m\n";
        std::cout << "  起终点漂移: " << std::fixed << std::setprecision(3) << drift << " m\n";
        std::cout << "  漂移率: " << std::fixed << std::setprecision(1) << drift_ratio * 100 << "%\n";
        std::cout << "  关键帧数: " << kf_count << "\n";
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  Groundtruth 对比提示（请在终端运行 evo）：\n\n";
    std::cout << "  # 绝对位姿误差（APE）\n";
    std::cout << "  evo_ape tum " << args.dataset_dir << "/groundtruth.txt " << args.output_traj
              << " -a --no_warnings\n";
    std::cout << "  # 相对位姿误差（RPE）\n";
    std::cout << "  evo_rpe tum " << args.dataset_dir << "/groundtruth.txt " << args.output_traj
              << " --delta 1 --delta_unit m -a -va --plot\n";

    std::vector<int> kf_indices;
    for (int fi = 0; fi < static_cast<int>(is_kf.size()); ++fi) {
        if (is_kf[fi]) kf_indices.push_back(fi);
    }

    writeTUMTrajectory(args.output_traj, timestamps, traj_t, traj_q,
                       inliers, args.trajectory_scale, &kf_indices);

    if (!kf_indices.empty()) {
        std::string kf_tum = args.output_traj;
        size_t dot = kf_tum.rfind('.');
        if (dot != std::string::npos) kf_tum.insert(dot, "_kf");
        else kf_tum += "_kf";
        writeTUMTrajectory(kf_tum, timestamps, traj_t, traj_q,
                            inliers, args.trajectory_scale, &kf_indices);
    }

    if (!traj_t.empty() && !kf_indices.empty()) {
        double cumsum_kf = 0, max_t_kf = 0;
        int prev_kf_idx = -1;
        for (int fi = 0; fi < static_cast<int>(traj_t.size()); ++fi) {
            if (!is_kf[fi]) continue;
            if (prev_kf_idx >= 0) {
                double t_mag = (traj_t[fi] - traj_t[prev_kf_idx]).norm();
                cumsum_kf += t_mag;
                if (t_mag > max_t_kf) max_t_kf = t_mag;
            }
            prev_kf_idx = fi;
        }
        double avg_t_kf = kf_count > 1 ? cumsum_kf / (kf_count - 1) : 0;

        std::cout << "\n[VO TrackDiag] ==========================================\n";
        std::cout << "[VO TrackDiag] 关键帧数量 = " << kf_count << "\n";
        std::cout << "[VO TrackDiag] 关键帧路径总长 = " << std::fixed << std::setprecision(2) << cumsum_kf << " m\n";
        std::cout << "[VO TrackDiag] 平均关键帧间隔 = " << std::fixed << std::setprecision(4) << avg_t_kf << " m\n";
        std::cout << "[VO TrackDiag] 最大关键帧间隔 = " << std::fixed << std::setprecision(4) << max_t_kf << " m\n";
        std::cout << "[VO TrackDiag] 起终点漂移 = " << std::fixed << std::setprecision(4)
                  << (traj_t.front() - traj_t.back()).norm() << " m\n";
        if (avg_t_kf > 0.5) {
            std::cout << "[VO TrackDiag] *** 警告：平均关键帧间隔过大 ***\n";
        }
        std::cout << "[VO TrackDiag] ==========================================\n";
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  Done.\n";

    return 0;
}
