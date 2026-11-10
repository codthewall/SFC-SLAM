/**
 * test_tum_module_diag.cpp
 *
 * TUM-RGBD 分模块诊断测试 v2
 * ================================================================
 * 完全复刻 original Tracking pipeline，逐个模块打印诊断数据。
 *
 * 关键修复（v1 → v2）：
 *   v1 错误：直接用 camera-coordinate 3D 点喂 PnP
 *   v2 正确：先把 camera-coordinate 点变到 world-coordinate（用 prev_pose），
 *            再喂 PnP。PnP 输出就是 world 坐标系下的 pose。
 *
 * 模块：
 *   M0  - PNG 深度原值
 *   M1  - 深度尺度转换
 *   M2  - 3D 点（camera frame）
 *   M2b - 3D 点（world frame，用 prev_pose 变换）
 *   M3  - ORB 特征提取
 *   M4  - 帧间特征匹配
 *   M5  - RobustPnP（original tracking pipeline + init_pose）
 *   M7  - cam_world 直接取 pose.t
 *   M8  - 累积轨迹
 *   M9  - GT 对比
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
#include <climits>
#include <sys/stat.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

// g2o（在命名空间外引入）
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#ifdef sfc_core
#undef sfc_core
#endif

// 项目头文件
#include "sfc_core/config/config_loader.hpp"
#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/frontend/robust_pnp_solver.hpp"
#include "sfc_core/feature/feature_matcher.hpp"

// ============================================================================
// 工具函数
// ============================================================================

static bool fileExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, delim)) parts.push_back(token);
    return parts;
}

static double parseDouble(const std::string& s) { return std::stod(s); }

// ============================================================================
// TUM 数据
// ============================================================================

struct TUMFrame {
    double timestamp = 0;
    std::string rgb_name, depth_name;
    cv::Mat rgb, depth;
};

struct TUMGroundtruth {
    double timestamp = 0;
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

static bool loadTUMDataset(const std::string& dataset_dir,
                           std::vector<TUMFrame>& frames,
                           std::vector<TUMGroundtruth>& groundtruth) {
    std::string rgb_assoc_file = joinPath(dataset_dir, "rgb.txt");
    std::string depth_assoc_file = joinPath(dataset_dir, "depth.txt");
    std::string gt_file = joinPath(dataset_dir, "groundtruth.txt");

    std::map<double, std::string> rgb_map, depth_map;

    { std::ifstream f(rgb_assoc_file);
      if (!f.is_open()) { std::cerr << "[load] Cannot open " << rgb_assoc_file << "\n"; return false; }
      std::string line;
      while (std::getline(f, line)) {
          if (line.empty() || line[0] == '#') continue;
          auto parts = split(line, ' ');
          if (parts.size() < 2) continue;
          rgb_map[parseDouble(parts[0])] = parts[1];
      }}
    { std::ifstream f(depth_assoc_file);
      if (!f.is_open()) { std::cerr << "[load] Cannot open " << depth_assoc_file << "\n"; return false; }
      std::string line;
      while (std::getline(f, line)) {
          if (line.empty() || line[0] == '#') continue;
          auto parts = split(line, ' ');
          if (parts.size() < 2) continue;
          depth_map[parseDouble(parts[0])] = parts[1];
      }}

    std::string rgb_dir = joinPath(dataset_dir, "rgb");
    std::string depth_dir = joinPath(dataset_dir, "depth");

    for (auto& rgb_entry : rgb_map) {
        double ts = rgb_entry.first;
        std::string rgb_name = rgb_entry.second;
        if (rgb_name.size() > 4 && rgb_name[3] == '/' &&
            rgb_name[0] == 'r' && rgb_name[1] == 'g' && rgb_name[2] == 'b')
            rgb_name = rgb_name.substr(4);

        auto it = depth_map.lower_bound(ts);
        double best_ts = ts;
        std::string best_depth;
        if (it != depth_map.end()) { best_depth = it->second; best_ts = it->first; }
        if (it != depth_map.begin()) {
            auto it_prev = std::prev(it);
            if (std::fabs(it_prev->first - ts) < std::fabs(best_ts - ts)) {
                best_depth = it_prev->second; best_ts = it_prev->first;
            }
        }
        if (best_depth.empty()) continue;
        if (best_depth.size() > 6 && best_depth[5] == '/' &&
            best_depth[0] == 'd' && best_depth[1] == 'e' &&
            best_depth[2] == 'p' && best_depth[3] == 't' && best_depth[4] == 'h')
            best_depth = best_depth.substr(6);

        TUMFrame frame;
        frame.timestamp = ts;
        frame.rgb_name = rgb_name;
        frame.depth_name = best_depth;
        frame.rgb = cv::imread(joinPath(rgb_dir, rgb_name), cv::IMREAD_COLOR);
        if (frame.rgb.empty()) continue;
        frame.depth = cv::imread(joinPath(depth_dir, best_depth), cv::IMREAD_UNCHANGED);
        if (frame.depth.empty()) continue;
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
            gt.timestamp = parseDouble(parts[0]);
            gt.t = Eigen::Vector3d(parseDouble(parts[1]), parseDouble(parts[2]), parseDouble(parts[3]));
            gt.q = Eigen::Quaterniond(parseDouble(parts[7]),
                                       parseDouble(parts[4]), parseDouble(parts[5]), parseDouble(parts[6]));
            if (!gt.q.isApprox(Eigen::Quaterniond::Identity(), 1e-6)) gt.q.normalize();
            groundtruth.push_back(gt);
        }
        std::cout << "[load] Groundtruth: " << groundtruth.size() << " entries\n";
    } else {
        std::cout << "[load] No groundtruth — skipping GT comparison\n";
    }

    std::cout << "[load] Frames: " << frames.size() << "\n";
    return !frames.empty();
}

static TUMGroundtruth queryGT(double timestamp, const std::vector<TUMGroundtruth>& gt, double tol = 0.1) {
    TUMGroundtruth empty{};
    double best_dt = 1e9;
    for (const auto& g : gt) {
        double dt = std::fabs(g.timestamp - timestamp);
        if (dt < best_dt) { best_dt = dt; empty = g; }
    }
    return (best_dt < tol) ? empty : TUMGroundtruth{};
}

// ============================================================================
// 相机内参
// ============================================================================

static cv::Mat makeK(double fx, double fy, double cx, double cy) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx; K.at<double>(0, 2) = cx;
    K.at<double>(1, 1) = fy; K.at<double>(1, 2) = cy;
    return K;
}

// ============================================================================
// 诊断参数
// ============================================================================

struct Args {
    std::string dataset;
    // 运行参数的默认来源是 config/*.yaml（R-019）。
    // 这里的 0 / 负值是"未指定，用 yaml"的哨兵，不再是真实参数值。
    std::string config_dir = "config";
    std::string camera_yaml = "config/camera_fr1.yaml";
    double depth_scale = -1.0;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    int width = 0, height = 0;
    double min_depth = -1.0, max_depth = -1.0;
    int max_frames = 0;
    int diag_interval = 20;
    int compare_interval = 10;
    bool verbose_pnp = false;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--dataset" && i + 1 < argc) a.dataset = argv[++i];
        else if (s == "--config_dir" && i + 1 < argc) a.config_dir = argv[++i];
        else if (s == "--camera" && i + 1 < argc) a.camera_yaml = argv[++i];
        else if (s == "--depth_scale" && i + 1 < argc) a.depth_scale = std::atof(argv[++i]);
        else if (s == "--fx" && i + 1 < argc) a.fx = std::atof(argv[++i]);
        else if (s == "--fy" && i + 1 < argc) a.fy = std::atof(argv[++i]);
        else if (s == "--cx" && i + 1 < argc) a.cx = std::atof(argv[++i]);
        else if (s == "--cy" && i + 1 < argc) a.cy = std::atof(argv[++i]);
        else if (s == "--min_depth" && i + 1 < argc) a.min_depth = std::atof(argv[++i]);
        else if (s == "--max_depth" && i + 1 < argc) a.max_depth = std::atof(argv[++i]);
        else if (s == "--max_frames" && i + 1 < argc) a.max_frames = std::atoi(argv[++i]);
        else if (s == "--diag_interval" && i + 1 < argc) a.diag_interval = std::atoi(argv[++i]);
        else if (s == "--compare_interval" && i + 1 < argc) a.compare_interval = std::atoi(argv[++i]);
        else if (s == "--verbose_pnp") a.verbose_pnp = true;
        else if (s == "-h" || s == "--help") {
            std::cout << "Usage: test_tum_module_diag --dataset /path/to/rgbd_dataset\n"
                      << "  --config_dir <dir>       Default config（运行参数唯一来源，R-019）\n"
                      << "  --camera <yaml>          Default config/camera_fr1.yaml\n"
                      << "  --depth_scale <float>    覆盖 tracking.yaml 的 depth_scale\n"
                      << "  --min_depth <float>      覆盖 tracking.yaml 的 min_depth\n"
                      << "  --max_depth <float>      覆盖 tracking.yaml 的 max_depth\n"
                      << "  --fx/--fy/--cx/--cy      覆盖内参 yaml 的值\n"
                      << "  --max_frames <int>       Default all\n"
                      << "  --diag_interval <int>    Print detail every N frames\n"
                      << "  --compare_interval <int>  Compare GT every N frames\n"
                      << "  --verbose_pnp             Print RobustPnP verbose output\n";
            std::exit(0);
        }
    }
    return a;
}

// 把 Args 里的哨兵值解析成真正生效的参数：yaml 打底，命令行只做覆盖。
// 与 test_tum_rgbd 的 resolveIntrinsics 同源，避免两个诊断程序各写一套内参（R-019）。
// out_cfg 回传完整配置，供 PnP / 匹配器复用同一份 yaml。
static bool resolveRuntimeParams(Args& a, sfc_core::config::AllConfigs& out_cfg) {
    out_cfg = sfc_core::config::loadAllConfigs(a.config_dir);
    const sfc_core::frontend::TrackingConfig& tcfg = out_cfg.tracking;

    sfc_core::config::CameraIntrinsicsConfig ic =
        sfc_core::config::loadCameraIntrinsicsConfig(a.camera_yaml);
    if (!ic.isValid()) {
        std::cerr << "ERROR: 无法从 " << a.camera_yaml << " 取得有效内参（fx/fy 必须为正）\n";
        return false;
    }

    a.depth_scale = (a.depth_scale > 0.0) ? a.depth_scale : tcfg.depth_scale;
    a.min_depth = (a.min_depth > 0.0) ? a.min_depth : tcfg.min_depth;
    a.max_depth = (a.max_depth > 0.0) ? a.max_depth : tcfg.max_depth;
    a.fx = (a.fx > 0.0) ? a.fx : ic.fx;
    a.fy = (a.fy > 0.0) ? a.fy : ic.fy;
    a.cx = (a.cx > 0.0) ? a.cx : ic.cx;
    a.cy = (a.cy > 0.0) ? a.cy : ic.cy;
    a.width = ic.width;
    a.height = ic.height;
    return true;
}

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);
    if (args.dataset.empty()) {
        std::cerr << "ERROR: --dataset required\n";
        return 1;
    }

    sfc_core::config::AllConfigs all_cfg;
    if (!resolveRuntimeParams(args, all_cfg)) {
        return 1;
    }

    std::cout << "============================================================\n";
    std::cout << "  TUM-RGBD Module Diagnostic v2 (original Tracking pipeline)\n";
    std::cout << "============================================================\n";
    std::cout << "  Dataset   : " << args.dataset << "\n";
    std::cout << "  Config    : " << args.config_dir << " + " << args.camera_yaml << "\n";
    std::cout << "  depth_scale = " << args.depth_scale << "\n";
    std::cout << "  K = [" << args.fx << " 0 " << args.cx << "; 0 " << args.fy << " " << args.cy << "]\n";
    std::cout << "  min_depth=" << args.min_depth << " max_depth=" << args.max_depth << "\n";
    std::cout << "============================================================\n\n";

    // --- 加载数据 ---
    std::vector<TUMFrame> frames;
    std::vector<TUMGroundtruth> gt;
    if (!loadTUMDataset(args.dataset, frames, gt)) return 1;

    int N = args.max_frames > 0 ? std::min(args.max_frames, (int)frames.size()) : (int)frames.size();

    cv::Mat K_cv = makeK(args.fx, args.fy, args.cx, args.cy);
    cv::Mat D_cv = cv::Mat::zeros(1, 5, CV_64F);

    // --- RobustPnP 配置：基线取自 robust_pnp_solver.yaml（R-019）---
    // 只有下面这一行是本诊断程序刻意偏离基线的地方：
    // 关掉扩张轮次，只看纯 EPnP + BA，用来隔离"扩张轮次把外点吃进内点集"这一类问题。
    // 偏离写在代码里显式说明，而不是在别处再放一份硬编码默认值。
    sfc_core::frontend::RobustPnPConfig pnp_config = all_cfg.robust_pnp_solver;
    pnp_config.verbose = args.verbose_pnp;
    pnp_config.expansion_rounds = 0;

    sfc_core::frontend::RobustPnPSolver pnp_solver(K_cv, pnp_config);
    pnp_solver.setCamera(K_cv, D_cv);

    // --- 特征匹配器：来源 feature_matcher.yaml（R-019）---
    sfc_core::feature::FeatureMatcher matcher(all_cfg.feature_matcher);

    // =========================================================================
    // 诊断数据结构
    // =========================================================================
    struct FrameDiag {
        int id = 0;
        double ts = 0;

        // M0/M1
        double raw_depth_center = 0;
        double depth_mean = 0;

        // M2
        int n_pts3d_cam = 0;
        double pt3d_z_mean = 0;

        // M2b
        int n_pts3d_world = 0;

        // M3
        int n_features = 0;

        // M4
        int n_matches_raw = 0, n_matches_inlier = 0;

        // M5
        Eigen::Vector3d pnp_t = Eigen::Vector3d::Zero();
        double pnp_t_norm = 0;
        double pnp_inlier_ratio = 0;
        std::string pnp_quality;
        bool pnp_success = false;

        // M7
        Eigen::Vector3d cam_world = Eigen::Vector3d::Zero();

        // GT
        Eigen::Vector3d gt_cam = Eigen::Vector3d::Zero();
        double gt_frame_dt = 0;
        double slam_frame_dt = 0;
        double scale_ratio = 0;
    };

    std::vector<FrameDiag> diags;
    diags.reserve(N);

    // =========================================================================
    // 首帧全局统计
    // =========================================================================
    {
        const cv::Mat& D = frames[0].depth;
        int64_t sum = 0, cnt = 0;
        uint16_t vmin = 0xFFFF, vmax = 0;
        for (int r = 0; r < D.rows; r++) {
            for (int c = 0; c < D.cols; c++) {
                uint16_t uv;
                if (D.type() == CV_32FC1) uv = static_cast<uint16_t>(D.at<float>(r, c));
                else uv = D.at<uint16_t>(r, c);
                if (uv > 0) { sum += uv; cnt++; if (uv < vmin) vmin = uv; if (uv > vmax) vmax = uv; }
            }
        }
        double avg_raw = cnt > 0 ? static_cast<double>(sum) / cnt : 0;
        int cx = D.cols / 2, cy = D.rows / 2;
        double center_raw;
        if (D.type() == CV_32FC1) center_raw = D.at<float>(cy, cx);
        else center_raw = D.at<uint16_t>(cy, cx);

        std::cout << "[M0] === 首帧 PNG 统计 ===\n";
        std::cout << "[M0] dtype=" << D.type() << " PNG raw: min=" << vmin << " max=" << vmax
                  << " mean=" << std::fixed << std::setprecision(1) << avg_raw << "\n";
        std::cout << "[M0] center pixel: " << center_raw << "\n";
        if (D.type() != CV_32FC1) {
            double if_mm = avg_raw * args.depth_scale;
            double if_cm  = avg_raw * args.depth_scale * 10;
            std::cout << "[M0] 解释为 mm (scale=" << args.depth_scale << "): depth=" << if_mm << "m\n";
            std::cout << "[M0] 解释为 cm (scale=" << args.depth_scale * 10 << "): depth=" << if_cm << "m\n";
            if (avg_raw > 3000) std::cout << "[M0] ★ 判定: PNG 是 mm（标准 TUM）\n";
            else if (avg_raw > 300) std::cout << "[M0] ★ 判定: PNG 是 cm\n";
            else std::cout << "[M0] ★ 判定: PNG 是 m\n";
        }
        std::cout << "\n";
    }

    // =========================================================================
    // 逐帧处理（完全复刻 original Tracking pipeline）
    // =========================================================================

    // 当前帧 pose（world 坐标系）
    sfc_core::frontend::Pose curr_pose;
    // 上一帧 3D 点（camera 坐标系 + 对应特征点坐标）
    std::vector<cv::KeyPoint> prev_kps_undist;
    std::vector<Eigen::Vector3d> prev_pts3d_cam;  // camera 坐标
    std::vector<sfc_core::frontend::Pose> prev_pose_history;

    double cum_path_slam = 0;
    double cum_path_gt = 0;

    std::vector<double> scale_ratios;

    std::cout << "----------------------------------------------------------\n";
    std::cout << "  开始逐帧诊断（完全复刻 original Tracking pipeline）\n";
    std::cout << "----------------------------------------------------------\n\n";

    for (int fi = 0; fi < N; fi++) {
        const TUMFrame& F = frames[fi];

        FrameDiag d;
        d.id = fi; d.ts = F.timestamp;

        // ---- M0: PNG 原值 ----
        {
            int cx = F.depth.cols / 2, cy = F.depth.rows / 2;
            if (F.depth.type() == CV_32FC1) d.raw_depth_center = F.depth.at<float>(cy, cx);
            else d.raw_depth_center = F.depth.at<uint16_t>(cy, cx);
        }

        // ---- M3: ORB 特征提取 ----
        cv::Mat gray, gray_prev;
        cv::cvtColor(F.rgb, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> kps_curr;
        cv::Mat desc_curr;
        cv::ORB::create(2000, 1.2f, 8)->detectAndCompute(gray, cv::noArray(), kps_curr, desc_curr);
        d.n_features = (int)kps_curr.size();

        // ---- M1: 深度转米 + M2: 3D 点（camera frame）----
        std::vector<Eigen::Vector3d> curr_pts3d_cam;
        curr_pts3d_cam.reserve(kps_curr.size());
        double zsum = 0, zcnt = 0;
        for (const auto& kp : kps_curr) {
            int u = cvRound(kp.pt.x), v = cvRound(kp.pt.y);
            if (u < 0 || u >= F.depth.cols || v < 0 || v >= F.depth.rows) continue;
            double d_raw;
            if (F.depth.type() == CV_32FC1) d_raw = F.depth.at<float>(v, u);
            else {
                uint16_t uv = F.depth.at<uint16_t>(v, u);
                if (uv == 0) continue;
                d_raw = uv * args.depth_scale;
            }
            if (std::isnan(d_raw) || std::isinf(d_raw) ||
                d_raw <= args.min_depth || d_raw > args.max_depth) continue;
            double x = (u - args.cx) * d_raw / args.fx;
            double y = (v - args.cy) * d_raw / args.fy;
            curr_pts3d_cam.emplace_back(x, y, d_raw);
            zsum += d_raw; zcnt++;
        }
        d.n_pts3d_cam = (int)curr_pts3d_cam.size();
        d.pt3d_z_mean = zcnt > 0 ? zsum / zcnt : 0;
        d.depth_mean = d.pt3d_z_mean;

        // ---- M2b: camera → world 变换 ----
        // 复刻 original tracking.cpp：
        //   P_world = R_cw^T * (P_cam - t_cw)
        //   其中 R_cw, t_cw 是上一帧的 pose（相机在世界的位置/旋转）
        std::vector<Eigen::Vector3d> curr_pts3d_world;
        curr_pts3d_world.reserve(curr_pts3d_cam.size());
        if (fi > 0) {
            Eigen::Matrix3d R_cw = curr_pose.q.toRotationMatrix();
            Eigen::Vector3d t_cw = curr_pose.t;
            for (const auto& pc : curr_pts3d_cam) {
                // P_world = R^T * (P_cam - t)
                Eigen::Vector3d pw = R_cw.transpose() * (pc - t_cw);
                curr_pts3d_world.push_back(pw);
            }
        }
        d.n_pts3d_world = (int)curr_pts3d_world.size();

        // ---- 第一帧：初始化 ----
        if (fi == 0) {
            curr_pose = sfc_core::frontend::Pose();  // identity

            std::cout << "[M0] frame=0 raw_center=" << std::fixed << std::setprecision(1) << d.raw_depth_center << "\n";
            std::cout << "[M1] frame=0 depth_mean=" << std::fixed << std::setprecision(4) << d.depth_mean << "m"
                      << " (scale=" << args.depth_scale << ")\n";
            std::cout << "[M2] frame=0 camera_pts=" << d.n_pts3d_cam << " z_mean=" << d.pt3d_z_mean << "m\n";
            std::cout << "[M3] frame=0 features=" << d.n_features << "\n";
            std::cout << "[M7] frame=0 cam_world=(" << curr_pose.t.transpose() << ") ← 原点\n";
            std::cout << "[M8] frame=0 cum_path=0.000m\n\n";

            prev_kps_undist = kps_curr;
            prev_pts3d_cam = curr_pts3d_cam;
            d.cam_world = curr_pose.t;
            d.gt_cam = gt.empty() ? Eigen::Vector3d::Zero() : gt[0].t;
            prev_pose_history.push_back(curr_pose);
            diags.push_back(d);
            continue;
        }

        // ---- M4: 特征匹配 ----
        const TUMFrame& F_prev = frames[fi - 1];
        cv::cvtColor(F_prev.rgb, gray_prev, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> kps_prev;
        cv::Mat desc_prev;
        cv::ORB::create(2000, 1.2f, 8)->detectAndCompute(gray_prev, cv::noArray(), kps_prev, desc_prev);

        std::vector<cv::DMatch> raw_matches;
        matcher.match(desc_prev, desc_curr, raw_matches);

        std::vector<cv::DMatch> inlier_matches;
        matcher.removeOutliers(raw_matches, kps_prev, kps_curr, inlier_matches);

        d.n_matches_raw = (int)raw_matches.size();
        d.n_matches_inlier = (int)inlier_matches.size();

        // ---- 构建 PnP 输入（world 3D + curr 2D）----
        // 复刻 original tracking：用 prev_pose 把 camera 点变到 world，再匹配 curr 2D
        std::vector<cv::Point3f> pts3d_pnp;
        std::vector<cv::Point2f> pts2d_pnp;

        // 上一帧的 pose
        const auto& prev_pose = prev_pose_history.back();
        Eigen::Matrix3d R_prev = prev_pose.q.toRotationMatrix();
        Eigen::Vector3d t_prev = prev_pose.t;

        for (const auto& m : inlier_matches) {
            int i_prev = m.queryIdx;
            int i_curr = m.trainIdx;

            // 上一帧的 camera 3D 点
            if (i_prev >= (int)prev_pts3d_cam.size()) continue;
            const auto& pc_prev = prev_pts3d_cam[i_prev];
            if (pc_prev.norm() < 1e-6 || std::isnan(pc_prev.norm())) continue;
            if (pc_prev.z() < args.min_depth || pc_prev.z() > args.max_depth) continue;

            // camera → world（用上一帧的 pose）
            Eigen::Vector3d pw = R_prev.transpose() * (pc_prev - t_prev);

            // 对应的当前帧 2D 点
            const auto& kp_curr = kps_curr[i_curr];

            pts3d_pnp.emplace_back(static_cast<float>(pw.x()),
                                   static_cast<float>(pw.y()),
                                   static_cast<float>(pw.z()));
            pts2d_pnp.emplace_back(kp_curr.pt.x, kp_curr.pt.y);
        }

        // ---- M5: RobustPnP（复刻 original pipeline）----
        // init_pose = prev_pose（速度先验）
        if (pts3d_pnp.size() >= 4) {
            sfc_core::frontend::PnPResult pnp_result;
            auto optimized_pose = pnp_solver.optimize(pts3d_pnp, pts2d_pnp,
                                                       prev_pose, &pnp_result);

            d.pnp_success = pnp_result.success;
            d.pnp_t = optimized_pose.t;
            d.pnp_t_norm = optimized_pose.t.norm();
            d.pnp_inlier_ratio = pnp_result.inlier_ratio;

            // RobustPnP 的 quality：EXCELLENT/GOOD/ACCEPTABLE/POOR/FAILED
            if (pnp_result.quality == sfc_core::frontend::PnPResult::Quality::EXCELLENT)
                d.pnp_quality = "EXCELLENT";
            else if (pnp_result.quality == sfc_core::frontend::PnPResult::Quality::GOOD)
                d.pnp_quality = "GOOD";
            else if (pnp_result.quality == sfc_core::frontend::PnPResult::Quality::ACCEPTABLE)
                d.pnp_quality = "ACCEPTABLE";
            else if (pnp_result.quality == sfc_core::frontend::PnPResult::Quality::POOR)
                d.pnp_quality = "POOR";
            else
                d.pnp_quality = "FAILED";

            // 赋值：直接取 pose.t 作为世界位置（original tracking 逻辑）
            curr_pose = optimized_pose;
        }

        // ---- M7: cam_world = pose.t ----
        d.cam_world = curr_pose.t;

        // ---- GT 对比 ----
        TUMGroundtruth query = queryGT(F.timestamp, gt);
        if (query.timestamp > 0) {
            d.gt_cam = query.t;
            TUMGroundtruth qprev = queryGT(F_prev.timestamp, gt);
            if (qprev.timestamp > 0)
                d.gt_frame_dt = (query.t - qprev.t).norm();
        }
        d.slam_frame_dt = (d.cam_world - diags.back().cam_world).norm();
        if (d.gt_frame_dt > 1e-6) {
            d.scale_ratio = d.slam_frame_dt / d.gt_frame_dt;
            scale_ratios.push_back(d.scale_ratio);
        }

        cum_path_slam += d.slam_frame_dt;
        cum_path_gt += d.gt_frame_dt;

        // ---- 打印诊断 ----
        bool do_diag = (fi % args.diag_interval == 0);
        bool do_compare = (fi % args.compare_interval == 0);

        if (do_diag) {
            std::cout << "----------------------------------------------------------\n";
            std::cout << "[frame=" << std::setw(4) << fi
                      << " ts=" << std::fixed << std::setprecision(3) << d.ts << "s]\n";
            std::cout << "[M0] raw_center=" << std::fixed << std::setprecision(1) << d.raw_depth_center
                      << " dtype=" << F.depth.type() << "\n";
            std::cout << "[M1] depth_mean=" << std::fixed << std::setprecision(4) << d.depth_mean << "m"
                      << " (scale=" << args.depth_scale << ")\n";
            std::cout << "[M2] camera_pts=" << d.n_pts3d_cam << " z_mean=" << std::fixed << std::setprecision(3) << d.pt3d_z_mean << "m\n";
            std::cout << "[M2b] world_pts_for_PnP=" << pts3d_pnp.size() << "\n";
            std::cout << "[M3] features=" << d.n_features << "\n";
            std::cout << "[M4] matches raw=" << d.n_matches_raw
                      << " inlier=" << d.n_matches_inlier << "\n";
            std::cout << "[M5] RobustPnP success=" << (d.pnp_success?"Y":"N")
                      << " quality=" << d.pnp_quality
                      << " inlier=" << std::fixed << std::setprecision(1) << d.pnp_inlier_ratio*100 << "%\n";
            std::cout << "[M5]   pose.t=(" << std::fixed << std::setprecision(4)
                      << d.pnp_t.x() << "," << d.pnp_t.y() << "," << d.pnp_t.z() << ") |t|=" << d.pnp_t_norm << "m\n";
            std::cout << "[M7] cam_world=(" << std::fixed << std::setprecision(4)
                      << d.cam_world.x() << "," << d.cam_world.y() << "," << d.cam_world.z() << ")m\n";

            if (do_compare && query.timestamp > 0) {
                std::cout << "[M9] ★ GT=(" << std::fixed << std::setprecision(4)
                          << query.t.x() << "," << query.t.y() << "," << query.t.z() << ")m\n";
                Eigen::Vector3d err = d.cam_world - query.t;
                std::cout << "[M9]   frame_dt SLAM=" << std::fixed << std::setprecision(5)
                          << d.slam_frame_dt << "m"
                          << " GT=" << d.gt_frame_dt << "m"
                          << " ratio=" << std::fixed << std::setprecision(3) << d.scale_ratio << "\n";
                std::cout << "[M9]   pos_error=" << std::fixed << std::setprecision(4) << err.norm() << "m\n";
                if (d.scale_ratio > 5.0) {
                    std::cout << "[M9]   *** 警告: ratio = " << std::fixed << std::setprecision(1)
                              << d.scale_ratio << "x 异常！\n";
                } else if (d.scale_ratio > 0) {
                    std::cout << "[M9]   ✓ 正常范围\n";
                }
            }

            std::cout << "[M8] cum_path SLAM=" << std::fixed << std::setprecision(3)
                      << cum_path_slam << "m"
                      << " GT=" << cum_path_gt << "m";
            if (cum_path_gt > 0.01)
                std::cout << " overall=" << std::fixed << std::setprecision(3)
                          << cum_path_slam / cum_path_gt << "x";
            std::cout << "\n\n";
        }

        // 每 50 帧简洁输出
        if (fi % 50 == 0 && fi > 0 && !do_diag) {
            std::cout << "  [frame=" << std::setw(4) << fi << "] "
                      << "cum_path=" << std::fixed << std::setprecision(2) << cum_path_slam << "m"
                      << " |t|=" << std::fixed << std::setprecision(5) << d.slam_frame_dt << "m"
                      << " z_mean=" << std::fixed << std::setprecision(3) << d.pt3d_z_mean << "m"
                      << " matches=" << d.n_matches_inlier << "/" << d.n_matches_raw
                      << " PnP=" << d.pnp_quality
                      << "\n";
        }

        // ---- 累积更新 ----
        prev_kps_undist = kps_curr;
        prev_pts3d_cam = curr_pts3d_cam;
        prev_pose_history.push_back(curr_pose);

        // 如果 PnP 失败太多，用原地踏步
        if (!d.pnp_success || d.pnp_inlier_ratio < 0.05) {
            curr_pose = prev_pose_history[prev_pose_history.size() - 2];
        }

        diags.push_back(d);
    }

    // =========================================================================
    // 汇总报告
    // =========================================================================
    double cum_path_gt_val = 0;
    if (!gt.empty()) {
        cum_path_gt_val = (gt.back().t - gt[0].t).norm();
    }

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  Module Diagnostic Summary\n";
    std::cout << "============================================================\n";

    // 帧间位移比值
    if (!scale_ratios.empty()) {
        double sum_sr = 0, min_sr = 1e9, max_sr = 0;
        int cnt_valid = 0;
        for (double r : scale_ratios) {
            if (r > 0 && r < 1000) {  // 排除灾难失败帧
                sum_sr += r; cnt_valid++;
                if (r < min_sr) min_sr = r;
                if (r > max_sr) max_sr = r;
            }
        }
        double avg_valid = cnt_valid > 0 ? sum_sr / cnt_valid : 0;

        std::cout << "\n[诊断] ★★★ 帧间位移 SLAM/GT 比值（排除灾难帧）★★★\n";
        std::cout << "[诊断]   有效样本: " << cnt_valid << " / " << scale_ratios.size() << " 帧\n";
        if (cnt_valid > 0) {
            std::cout << "[诊断]   平均 ratio = " << std::fixed << std::setprecision(3) << avg_valid << "x\n";
            std::cout << "[诊断]   范围 [" << std::fixed << std::setprecision(3) << min_sr
                      << ", " << std::fixed << std::setprecision(3) << max_sr << "]\n";
        }

        if (avg_valid > 3.0) {
            std::cout << "[诊断] ⚠ 帧间位移偏大 " << std::fixed << std::setprecision(1) << avg_valid << "x\n";
            std::cout << "[诊断]   → 如果 z_mean 正常(0.5~2m)，则 depth_scale 可能已正确\n";
            std::cout << "[诊断]   → 问题可能在 RobustPnP 的 BA 优化过程中引入了额外尺度\n";
        } else if (avg_valid > 1.5) {
            std::cout << "[诊断] ✓ 帧间位移偏差较小\n";
        } else if (avg_valid > 0) {
            std::cout << "[诊断] ✓ 帧间位移正常\n";
        }
    }

    // 深度统计
    double z_sum = 0, z_cnt = 0;
    for (const auto& d : diags) { z_sum += d.pt3d_z_mean; z_cnt++; }
    double z_avg = z_cnt > 0 ? z_sum / z_cnt : 0;
    std::cout << "\n[诊断] 3D 点平均深度: " << std::fixed << std::setprecision(4) << z_avg << "m\n";
    std::cout << "[诊断] depth_scale = " << std::fixed << std::setprecision(6) << args.depth_scale << "\n";
    std::cout << "[诊断]   判定: z_avg=0.5~2.0m → depth_scale 正确 ✓\n";
    std::cout << "[诊断]        z_avg=5~12m    → depth_scale 偏大 10x ★\n";
    std::cout << "[诊断]        z_avg=0.05~0.2m → depth_scale 偏小 10x ★\n";

    // PnP 质量统计
    int cnt_excellent = 0, cnt_good = 0, cnt_acceptable = 0, cnt_poor = 0, cnt_failed = 0;
    for (const auto& d : diags) {
        if (d.id == 0) continue;
        if (d.pnp_quality == "EXCELLENT") cnt_excellent++;
        else if (d.pnp_quality == "GOOD") cnt_good++;
        else if (d.pnp_quality == "ACCEPTABLE") cnt_acceptable++;
        else if (d.pnp_quality == "POOR") cnt_poor++;
        else cnt_failed++;
    }
    int total_valid = N - 1;
    std::cout << "\n[诊断] PnP 质量分布 (n=" << total_valid << "):\n";
    std::cout << "[诊断]   EXCELLENT: " << cnt_excellent << " ("
              << std::fixed << std::setprecision(1) << (100.0*cnt_excellent/total_valid) << "%)\n";
    std::cout << "[诊断]   GOOD:      " << cnt_good << " ("
              << std::fixed << std::setprecision(1) << (100.0*cnt_good/total_valid) << "%)\n";
    std::cout << "[诊断]   ACCEPTABLE:" << cnt_acceptable << " ("
              << std::fixed << std::setprecision(1) << (100.0*cnt_acceptable/total_valid) << "%)\n";
    std::cout << "[诊断]   POOR:      " << cnt_poor << " ("
              << std::fixed << std::setprecision(1) << (100.0*cnt_poor/total_valid) << "%)\n";
    std::cout << "[诊断]   FAILED:    " << cnt_failed << " ("
              << std::fixed << std::setprecision(1) << (100.0*cnt_failed/total_valid) << "%)\n";

    // 轨迹统计
    std::cout << "\n[诊断] 轨迹累积:\n";
    std::cout << "[诊断]   SLAM: " << std::fixed << std::setprecision(3) << cum_path_slam << "m\n";
    std::cout << "[诊断]   GT:   " << std::fixed << std::setprecision(3) << cum_path_gt << "m\n";
    if (cum_path_gt > 0.01) {
        double overall = cum_path_slam / cum_path_gt;
        std::cout << "[诊断]   整体比例: SLAM/GT = " << std::fixed << std::setprecision(3) << overall << "x\n";
        if (overall > 5.0) {
            std::cout << "[诊断]   ★★★ 累积轨迹异常偏大 " << std::fixed << std::setprecision(1) << overall << "x\n";
            std::cout << "[诊断]   建议 depth_scale = " << std::fixed << std::setprecision(6)
                      << (args.depth_scale / overall) << " (当前 " << args.depth_scale << " / " << overall << ")\n";
        }
    }

    // 终点对比
    if (!gt.empty() && !diags.empty()) {
        const auto& last = diags.back();
        const auto& last_gt = gt.back();
        Eigen::Vector3d end_err = last.cam_world - last_gt.t;
        std::cout << "\n[诊断] 终点:\n";
        std::cout << "[诊断]   SLAM: (" << std::fixed << std::setprecision(4)
                  << last.cam_world.x() << "," << last.cam_world.y() << "," << last.cam_world.z() << ")m\n";
        std::cout << "[诊断]   GT:   (" << std::fixed << std::setprecision(4)
                  << last_gt.t.x() << "," << last_gt.t.y() << "," << last_gt.t.z() << ")m\n";
        std::cout << "[诊断]   终点误差: " << std::fixed << std::setprecision(4) << end_err.norm() << "m\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "  Done. frames=" << N << "\n";
    std::cout << "============================================================\n";

    return 0;
}
