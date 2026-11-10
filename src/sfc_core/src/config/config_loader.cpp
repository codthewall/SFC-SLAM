// ================================================================
// SFC-SLAM Config Loader Implementation
// 文件路径: src/sfc_core/src/config/config_loader.cpp
// ================================================================

#include "sfc_core/config/config_loader.hpp"
#include "sfc_core/loopdetection/geometric_verifier.hpp"
#include "sfc_core/loopdetection/loop_detector.hpp"
#include "sfc_core/loopdetection/pose_graph_optimizer.hpp"
#include "sfc_core/loopdetection/place_recognition.hpp"
#include "sfc_core/frontend/covis_manager.hpp"
#include "sfc_core/frontend/tracking.hpp"
#include "sfc_core/frontend/robust_pnp_solver.hpp"
#include "sfc_core/feature/feature_matcher.hpp"

#ifdef sfc_core
#undef sfc_core
#endif

#include <opencv2/core.hpp>
#include <sstream>
#include <cstdlib>
#include <vector>

namespace sfc_core {
namespace config {

// ================================================================
// Helper: read scalar values safely from OpenCV FileStorage
// ================================================================

static double getDouble(const cv::FileStorage& fs, const char* key, double default_val) {
    cv::FileNode node = fs[key];
    if (node.empty()) return default_val;
    return static_cast<double>(node);
}

static int getInt(const cv::FileStorage& fs, const char* key, int default_val) {
    cv::FileNode node = fs[key];
    if (node.empty()) return default_val;
    return static_cast<int>(node);
}

static bool getBool(const cv::FileStorage& fs, const char* key, bool default_val) {
    cv::FileNode node = fs[key];
    if (node.empty()) return default_val;
    return static_cast<int>(node) != 0;
}

static std::string getString(const cv::FileStorage& fs, const char* key, const std::string& default_val) {
    cv::FileNode node = fs[key];
    if (node.empty()) return default_val;
    return static_cast<std::string>(node);
}

// 同一个语义参数在 yaml 里存在历史别名时按顺序取第一个命中的键。
// 存在的理由：config/*.yaml 里长期混用 fast_threshold 与 ini_th_fast 两套名字，
// 只认其中一个会让另一半配置静默失效（R-019）。
static int getIntAny(const cv::FileStorage& fs,
                     const std::vector<const char*>& keys,
                     int default_val) {
    for (const char* key : keys) {
        cv::FileNode node = fs[key];
        if (!node.empty()) return static_cast<int>(node);
    }
    return default_val;
}

static double getDoubleAny(const cv::FileStorage& fs,
                           const std::vector<const char*>& keys,
                           double default_val) {
    for (const char* key : keys) {
        cv::FileNode node = fs[key];
        if (!node.empty()) return static_cast<double>(node);
    }
    return default_val;
}

static bool getBoolAny(const cv::FileStorage& fs,
                       const std::vector<const char*>& keys,
                       bool default_val) {
    for (const char* key : keys) {
        cv::FileNode node = fs[key];
        if (!node.empty()) return static_cast<int>(node) != 0;
    }
    return default_val;
}

// ================================================================
// GeometricVerifierConfig
// ================================================================

loopdetection::GeometricVerifierConfig
loadGeometricVerifierConfig(const std::string& yaml_path) {
    loopdetection::GeometricVerifierConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for GeometricVerifierConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for GeometricVerifierConfig" << std::endl;
        return cfg;
    }

    cfg.ransac_iterations = getInt(fs, "ransac_iterations", cfg.ransac_iterations);
    cfg.threshold_px = getDouble(fs, "threshold_px", cfg.threshold_px);
    cfg.max_icp_inlier_ratio = getDouble(fs, "max_icp_inlier_ratio", cfg.max_icp_inlier_ratio);
    cfg.min_inliers = getInt(fs, "min_inliers", cfg.min_inliers);
    cfg.min_inlier_ratio = getDouble(fs, "min_inlier_ratio", cfg.min_inlier_ratio);
    cfg.min_icp_inliers = getInt(fs, "min_icp_inliers", cfg.min_icp_inliers);
    cfg.min_icp_inlier_ratio = getDouble(fs, "min_icp_inlier_ratio", cfg.min_icp_inlier_ratio);
    cfg.icp_reproj_thresh_px = getDouble(fs, "icp_reproj_thresh_px", cfg.icp_reproj_thresh_px);
    cfg.min_essential_inliers = getInt(fs, "min_essential_inliers", cfg.min_essential_inliers);
    cfg.min_essential_inlier_ratio = getDouble(fs, "min_essential_inlier_ratio", cfg.min_essential_inlier_ratio);
    cfg.min_depth = getDouble(fs, "min_depth", cfg.min_depth);
    cfg.max_depth = getDouble(fs, "max_depth", cfg.max_depth);
    cfg.use_fundamental = getBool(fs, "use_fundamental", cfg.use_fundamental);

    std::cerr << "[ConfigLoader] Loaded GeometricVerifierConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// LoopDetectorConfig
// ================================================================

loopdetection::LoopDetectorConfig
loadLoopDetectorConfig(const std::string& yaml_path) {
    loopdetection::LoopDetectorConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for LoopDetectorConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for LoopDetectorConfig" << std::endl;
        return cfg;
    }

    cfg.min_loop_separation = getInt(fs, "min_loop_separation", cfg.min_loop_separation);
    cfg.max_match_distance = getInt(fs, "max_match_distance", cfg.max_match_distance);
    cfg.lowe_ratio_threshold = getDouble(fs, "lowe_ratio_threshold", cfg.lowe_ratio_threshold);
    cfg.min_common_words = getInt(fs, "min_common_words", cfg.min_common_words);
    // D3-c：ransac_iterations / ransac_threshold_px / max_icp_inlier_ratio 不在本结构里了，
    // 它们是 GeometricVerifierConfig 的字段，由 loadGeometricVerifierConfig() 从
    // config/geometric_verifier.yaml 读，再经 loadAllConfigs() 注入 loop_detector。
    cfg.min_geometric_inliers = getInt(fs, "min_geometric_inliers", cfg.min_geometric_inliers);
    cfg.candidate_pool_size = getInt(fs, "candidate_pool_size", cfg.candidate_pool_size);
    cfg.enable_multi_candidate_verification =
        getBool(fs, "enable_multi_candidate_verification", cfg.enable_multi_candidate_verification);
    cfg.max_candidates_to_verify = getInt(fs, "max_candidates_to_verify", cfg.max_candidates_to_verify);
    cfg.min_candidates_to_accept = getInt(fs, "min_candidates_to_accept", cfg.min_candidates_to_accept);
    cfg.best_candidate_ratio_threshold =
        getDouble(fs, "best_candidate_ratio_threshold", cfg.best_candidate_ratio_threshold);
    cfg.require_candidate_agreement =
        getBool(fs, "require_candidate_agreement", cfg.require_candidate_agreement);

    // 质量评分参数（默认值取结构体自身，避免 loader 里再放一份字面量造成两处默认值，R-019）
    cfg.geometric_inlier_weight =
        getDouble(fs, "geometric_inlier_weight", cfg.geometric_inlier_weight);
    cfg.inlier_ratio_weight = getDouble(fs, "inlier_ratio_weight", cfg.inlier_ratio_weight);
    cfg.bow_score_weight = getDouble(fs, "bow_score_weight", cfg.bow_score_weight);

    // Sim3 尺度估计
    cfg.frame_dist_min = getDouble(fs, "frame_dist_min", cfg.frame_dist_min);
    cfg.frame_dist_max = getDouble(fs, "frame_dist_max", cfg.frame_dist_max);

    // D5-e：DBoW2 查询与候选空间闸门（原来写死在 detect() 里）
    cfg.dbow_temporal_window = getInt(fs, "dbow_temporal_window", cfg.dbow_temporal_window);
    cfg.dbow_max_results = getInt(fs, "dbow_max_results", cfg.dbow_max_results);
    cfg.dbow_min_distance_m = getDouble(fs, "dbow_min_distance_m", cfg.dbow_min_distance_m);
    cfg.loop_max_spatial_dist_m =
        getDouble(fs, "loop_max_spatial_dist_m", cfg.loop_max_spatial_dist_m);
    cfg.min_frames_between_for_spatial_gate = getInt(
        fs, "min_frames_between_for_spatial_gate", cfg.min_frames_between_for_spatial_gate);
    // D4-d：回环检测节流（关键帧计）
    cfg.loop_check_interval_kf =
        getInt(fs, "loop_check_interval_kf", cfg.loop_check_interval_kf);

    // 候选一致性
    cfg.max_spatial_dist_for_consistency =
        getDouble(fs, "max_spatial_dist_for_consistency", cfg.max_spatial_dist_for_consistency);
    cfg.consistency_warning_threshold =
        getDouble(fs, "consistency_warning_threshold", cfg.consistency_warning_threshold);

    // 诊断输出开关：回环"没找到"是常态，逐帧刷屏会淹没真正的失败信息（R-017）
    cfg.verbose = getBool(fs, "verbose", cfg.verbose);

    std::cerr << "[ConfigLoader] Loaded LoopDetectorConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// PoseGraphOptimizerOptions
// ================================================================

loopdetection::PoseGraphOptimizerOptions
loadPoseGraphOptimizerConfig(const std::string& yaml_path) {
    loopdetection::PoseGraphOptimizerOptions opts;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for PoseGraphOptimizerOptions" << std::endl;
        return opts;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for PoseGraphOptimizerOptions" << std::endl;
        return opts;
    }

    opts.max_iterations = getInt(fs, "max_iterations", opts.max_iterations);
    opts.robust_kernel_delta = getDouble(fs, "robust_kernel_delta", opts.robust_kernel_delta);
    opts.verbose = getBool(fs, "verbose", opts.verbose);

    // P1-3 修复：补充所有信息矩阵参数的 YAML 加载
    // 之前只加载了 max_iterations / robust_kernel_delta / verbose 三个字段，
    // em_trans_info / em_rot_info / icp_*_info 等全部使用默认值（极小/不正确），
    // 导致 PGO 的 EM 回环边权重错误，轨迹膨胀到 180m。
    opts.odom_information_scale =
        getDouble(fs, "odom_information_scale", opts.odom_information_scale);
    opts.odom_min_information_scale =
        getDouble(fs, "odom_min_information_scale", opts.odom_min_information_scale);
    opts.icp_rot_info =
        getDouble(fs, "icp_rot_info", opts.icp_rot_info);
    opts.icp_trans_info =
        getDouble(fs, "icp_trans_info", opts.icp_trans_info);
    opts.em_rot_info =
        getDouble(fs, "em_rot_info", opts.em_rot_info);
    opts.em_trans_info =
        getDouble(fs, "em_trans_info", opts.em_trans_info);
    opts.em_min_rot_scale =
        getDouble(fs, "em_min_rot_scale", opts.em_min_rot_scale);
    opts.em_min_trans_info =
        getDouble(fs, "em_min_trans_info", opts.em_min_trans_info);
    opts.loop_information_scale =
        getDouble(fs, "loop_information_scale", opts.loop_information_scale);
    opts.loop_min_information_scale =
        getDouble(fs, "loop_min_information_scale", opts.loop_min_information_scale);
    // D4-a：回环边的"内点对数"加权与 D4-c 的里程计边缩放区间。
    // 四个键此前不存在（权重由代码里的常数决定），现在由 yaml 决定。
    opts.loop_inlier_count_ref =
        getDouble(fs, "loop_inlier_count_ref", opts.loop_inlier_count_ref);
    opts.loop_inlier_count_min_factor =
        getDouble(fs, "loop_inlier_count_min_factor", opts.loop_inlier_count_min_factor);
    opts.loop_inlier_count_max_factor =
        getDouble(fs, "loop_inlier_count_max_factor", opts.loop_inlier_count_max_factor);
    opts.odom_info_scale_min =
        getDouble(fs, "odom_info_scale_min", opts.odom_info_scale_min);
    opts.odom_info_scale_max =
        getDouble(fs, "odom_info_scale_max", opts.odom_info_scale_max);

    std::cerr << "[ConfigLoader] Loaded PoseGraphOptimizerOptions from " << yaml_path << std::endl;
    std::cerr << "  em_trans_info=" << opts.em_trans_info
              << " em_rot_info=" << opts.em_rot_info
              << " icp_trans_info=" << opts.icp_trans_info
              << " icp_rot_info=" << opts.icp_rot_info
              << " robust_kernel_delta=" << opts.robust_kernel_delta
              << std::endl;
    return opts;
}

// ================================================================
// CovisConfig
// ================================================================

frontend::CovisConfig
loadCovisConfig(const std::string& yaml_path) {
    frontend::CovisConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for CovisConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for CovisConfig" << std::endl;
        return cfg;
    }

    cfg.window_size = getInt(fs, "window_size", cfg.window_size);
    cfg.min_covis_frames = getInt(fs, "min_covis_frames", cfg.min_covis_frames);
    cfg.candidate_confirm_threshold = getInt(fs, "candidate_confirm_threshold", cfg.candidate_confirm_threshold);
    cfg.max_reproj_error = getDouble(fs, "max_reproj_error", cfg.max_reproj_error);
    cfg.max_candidates = getInt(fs, "max_candidates", cfg.max_candidates);
    cfg.max_confirmed = getInt(fs, "max_confirmed", cfg.max_confirmed);
    cfg.candidate_check_interval = getInt(fs, "candidate_check_interval", cfg.candidate_check_interval);
    cfg.staleness_thresh = getInt(fs, "staleness_thresh", cfg.staleness_thresh);
    cfg.quantize_scale = getDouble(fs, "quantize_scale", 20.0);
    cfg.same_point_distance_threshold = getDouble(fs, "same_point_distance_threshold", 0.10);
    cfg.reproj_search_radius = getDouble(fs, "reproj_search_radius", 12.0);
    cfg.icp_loop_max_distance_m = getDouble(fs, "icp_loop_max_distance_m", 6.0);
    cfg.ba_min_edges = getInt(fs, "ba_min_edges", 10);
    cfg.ba_max_iterations = getInt(fs, "ba_max_iterations", 20);
    cfg.ba_huber_delta = getDouble(fs, "ba_huber_delta", 5.0);
    cfg.pgo_max_allowed_trans_corr = getDouble(fs, "pgo_max_allowed_trans_corr", 2.0);
    cfg.pgo_max_allowed_rot_corr_deg = getDouble(fs, "pgo_max_allowed_rot_corr_deg", 180.0);
    cfg.pgo_max_allowed_rot_corr_deg_reject = getDouble(fs, "pgo_max_allowed_rot_corr_deg_reject", 50.0);
    cfg.pgo_warn_trans_corr = getDouble(fs, "pgo_warn_trans_corr", 2.0);
    cfg.pgo_warn_rot_corr_deg = getDouble(fs, "pgo_warn_rot_corr_deg", 45.0);
    cfg.loop_detection_interval = getInt(fs, "loop_detection_interval", cfg.loop_detection_interval);
    cfg.loop_min_inliers = getInt(fs, "loop_min_inliers", cfg.loop_min_inliers);
    cfg.enable_loop_detection = getBool(fs, "enable_loop_detection", cfg.enable_loop_detection);
    cfg.enable_pgo = getBool(fs, "enable_pgo", cfg.enable_pgo);
    cfg.enable_local_ba = getBool(fs, "enable_local_ba", cfg.enable_local_ba);
    cfg.verbose = getBool(fs, "verbose", cfg.verbose);
    // ★★★ 图像分辨率 ★★★
    cfg.img_width = getInt(fs, "img_width", 640);
    cfg.img_height = getInt(fs, "img_height", 480);
    // 相机内参（用于 BA 重投影）
    double fx = getDouble(fs, "camera_fx", 525.0);
    double fy = getDouble(fs, "camera_fy", 525.0);
    double cx = getDouble(fs, "camera_cx", 319.5);
    double cy = getDouble(fs, "camera_cy", 239.5);
    cfg.K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    std::cerr << "[ConfigLoader] Loaded CovisConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// TrackingConfig
// ================================================================

frontend::TrackingConfig
loadTrackingConfig(const std::string& yaml_path) {
    frontend::TrackingConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for TrackingConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for TrackingConfig" << std::endl;
        return cfg;
    }

    // 深度处理
    cfg.depth_scale = getDouble(fs, "depth_scale", cfg.depth_scale);
    cfg.min_depth = getDouble(fs, "min_depth", cfg.min_depth);
    cfg.max_depth = getDouble(fs, "max_depth", cfg.max_depth);
    cfg.depth_edge_thresh = getDouble(fs, "depth_edge_thresh", cfg.depth_edge_thresh);
    cfg.frame_jump_max_translation =
        getDouble(fs, "frame_jump_max_translation", cfg.frame_jump_max_translation);
    cfg.frame_jump_max_rotation =
        getDouble(fs, "frame_jump_max_rotation", cfg.frame_jump_max_rotation);

    // 注：extractor_config / matcher_config 故意不在这里读。
    // 它们的唯一来源分别是 feature_extractor.yaml 与 feature_matcher.yaml，
    // 由 loadAllConfigs() 在最后合并进来。历史上 tracking.yaml 也写了一份
    // 同名字段，两处取值不同时无法判断哪个生效（R-019）。

    // PnP 参数（默认值一律取自结构体，避免 loader 里再复制一份字面量 —— R-019）
    cfg.pnp_max_ransac_iterations =
        getInt(fs, "pnp_max_ransac_iterations", cfg.pnp_max_ransac_iterations);
    cfg.pnp_initial_thresh_px =
        getDouble(fs, "pnp_initial_thresh_px", cfg.pnp_initial_thresh_px);
    cfg.pnp_final_thresh_px =
        getDouble(fs, "pnp_final_thresh_px", cfg.pnp_final_thresh_px);
    cfg.pnp_expansion_rounds =
        getInt(fs, "pnp_expansion_rounds", cfg.pnp_expansion_rounds);
    cfg.pnp_expansion_factor =
        getDouble(fs, "pnp_expansion_factor", cfg.pnp_expansion_factor);
    cfg.pnp_ba_max_iterations =
        getInt(fs, "pnp_ba_max_iterations", cfg.pnp_ba_max_iterations);
    cfg.pnp_use_huber_kernel =
        getBool(fs, "pnp_use_huber_kernel", cfg.pnp_use_huber_kernel);
    cfg.pnp_huber_delta = getDouble(fs, "pnp_huber_delta", cfg.pnp_huber_delta);
    cfg.pnp_use_depth_weighting =
        getBool(fs, "pnp_use_depth_weighting", cfg.pnp_use_depth_weighting);
    // poseOptimization 的内点下限。此前只存在于 TrackingConfig 里、没有 YAML 入口，
    // 默认值 5 意味着 5 个点就敢接受 6 自由度解 —— 退化帧会被当成成功。
    cfg.pose_opt_min_inliers = getInt(fs, "pose_opt_min_inliers", cfg.pose_opt_min_inliers);

    // 关键帧判定（判定本体见 frontend/keyframe_policy.hpp）
    cfg.min_keyframe_translation = getDouble(fs, "min_keyframe_translation", cfg.min_keyframe_translation);
    cfg.min_keyframe_rotation = getDouble(fs, "min_keyframe_rotation", cfg.min_keyframe_rotation);
    cfg.min_features = getInt(fs, "min_features", cfg.min_features);
    cfg.forced_keyframe_interval = getInt(fs, "forced_keyframe_interval", cfg.forced_keyframe_interval);
    cfg.min_keyframe_gap_frames = getInt(fs, "min_keyframe_gap_frames", cfg.min_keyframe_gap_frames);
    cfg.keyframe_min_tracked_inliers =
        getInt(fs, "keyframe_min_tracked_inliers", cfg.keyframe_min_tracked_inliers);
    cfg.keyframe_weak_tracked_ratio =
        getDouble(fs, "keyframe_weak_tracked_ratio", cfg.keyframe_weak_tracked_ratio);
    cfg.keyframe_rescue_min_inliers =
        getInt(fs, "keyframe_rescue_min_inliers", cfg.keyframe_rescue_min_inliers);
    cfg.keyframe_rescue_max_gap =
        getInt(fs, "keyframe_rescue_max_gap", cfg.keyframe_rescue_max_gap);

    // 局部地图与建点配额
    cfg.local_map_keyframe_window =
        getInt(fs, "local_map_keyframe_window", cfg.local_map_keyframe_window);
    cfg.map_point_grace_frames =
        getInt(fs, "map_point_grace_frames", cfg.map_point_grace_frames);
    cfg.max_new_map_points_per_kf =
        getInt(fs, "max_new_map_points_per_kf", cfg.max_new_map_points_per_kf);

    // EMA 和死区
    cfg.relative_t_ema_alpha = getDouble(fs, "relative_t_ema_alpha", cfg.relative_t_ema_alpha);
    cfg.stationary_deadzone_enabled =
        getBool(fs, "stationary_deadzone_enabled", cfg.stationary_deadzone_enabled);
    cfg.stationary_trans_deadzone_m =
        getDouble(fs, "stationary_trans_deadzone_m", cfg.stationary_trans_deadzone_m);
    cfg.stationary_rot_deadzone_rad =
        getDouble(fs, "stationary_rot_deadzone_rad", cfg.stationary_rot_deadzone_rad);

    // 模块使能
    // enable_pgo / enable_local_ba 不再从这里加载：纯 VO 路径不消费这两个开关，
    // 真正的拥有者是 PoseGraphOptimizerOptions 与 CovisConfig（R-019 单一来源）。
    cfg.verbose_diag = getBool(fs, "verbose_diag", cfg.verbose_diag);
    cfg.disable_undistort = getBool(fs, "disable_undistort", cfg.disable_undistort);

    // 相机内参不在这里加载：TrackingConfig 的 K/D 由调用方从
    // config/camera_fr{1,2,3}.yaml（loadCameraIntrinsicsConfig）填入。
    // 曾经在这里再放一份 fr2 默认值，导致"yaml 里写着 fr2、实际跑的是 fr1"
    // 这种看不出问题的配置（R-019 反例）。

    std::cerr << "[ConfigLoader] Loaded TrackingConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// RobustPnPConfig
// ================================================================

frontend::RobustPnPConfig
loadRobustPnPConfig(const std::string& yaml_path) {
    frontend::RobustPnPConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for RobustPnPConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for RobustPnPConfig" << std::endl;
        return cfg;
    }

    cfg.max_ransac_iterations = getInt(fs, "max_ransac_iterations", cfg.max_ransac_iterations);
    cfg.initial_inlier_thresh_px = getDouble(fs, "initial_inlier_thresh_px", cfg.initial_inlier_thresh_px);
    cfg.final_inlier_thresh_px = getDouble(fs, "final_inlier_thresh_px", cfg.final_inlier_thresh_px);
    cfg.confidence = getDouble(fs, "confidence", cfg.confidence);
    cfg.use_adaptive_threshold = getBool(fs, "use_adaptive_threshold", cfg.use_adaptive_threshold);
    cfg.min_inlier_ratio = getDouble(fs, "min_inlier_ratio", cfg.min_inlier_ratio);
    cfg.expansion_rounds = getInt(fs, "expansion_rounds", cfg.expansion_rounds);
    cfg.expansion_factor = getDouble(fs, "expansion_factor", cfg.expansion_factor);
    cfg.ba_max_iterations = getInt(fs, "ba_max_iterations", cfg.ba_max_iterations);
    cfg.use_huber_kernel = getBool(fs, "use_huber_kernel", cfg.use_huber_kernel);
    cfg.huber_delta = getDouble(fs, "huber_delta", cfg.huber_delta);
    cfg.use_depth_weighting = getBool(fs, "use_depth_weighting", cfg.use_depth_weighting);
    cfg.joint_point_optimization = getBool(fs, "joint_point_optimization", cfg.joint_point_optimization);
    cfg.verbose = getBool(fs, "verbose", cfg.verbose);

    std::cerr << "[ConfigLoader] Loaded RobustPnPConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// OctreeORBConfig (ORB-SLAM3 风格)
// ================================================================

feature::OctreeORBConfig
loadOctreeORBConfig(const std::string& yaml_path) {
    feature::OctreeORBConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for OctreeORBConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for OctreeORBConfig" << std::endl;
        return cfg;
    }

    cfg.n_features = getInt(fs, "n_features", cfg.n_features);
    cfg.scale_factor = static_cast<float>(getDouble(fs, "scale_factor", cfg.scale_factor));
    cfg.n_levels = getInt(fs, "n_levels", cfg.n_levels);
    // 别名：config/feature_extractor.yaml 用的是 fast_threshold / edge_threshold
    cfg.ini_threshold_fast = getIntAny(
        fs, {"fast_threshold", "ini_th_fast"}, cfg.ini_threshold_fast);
    cfg.min_threshold_fast = getIntAny(
        fs, {"min_fast_threshold", "min_th_fast"}, cfg.min_threshold_fast);
    cfg.edge_threshold = getInt(fs, "edge_threshold", cfg.edge_threshold);
    cfg.patch_size = getInt(fs, "patch_size", cfg.patch_size);

    std::cerr << "[ConfigLoader] Loaded OctreeORBConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// MatcherConfig
// ================================================================

feature::MatcherConfig
loadFeatureMatcherConfig(const std::string& yaml_path) {
    feature::MatcherConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for MatcherConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for MatcherConfig" << std::endl;
        return cfg;
    }

    cfg.lowe_ratio = static_cast<float>(getDouble(fs, "lowe_ratio", cfg.lowe_ratio));
    // 别名：config/feature_matcher.yaml 里叫 max_match_dist
    cfg.max_hamming_dist = getIntAny(
        fs, {"max_hamming_dist", "max_match_dist"}, cfg.max_hamming_dist);
    cfg.cross_check = getBool(fs, "cross_check", cfg.cross_check);
    cfg.th_low = getInt(fs, "th_low", cfg.th_low);
    cfg.th_high = getInt(fs, "th_high", cfg.th_high);
    cfg.histo_length = getInt(fs, "histo_length", cfg.histo_length);
    cfg.check_orientation = getBool(fs, "check_orientation", cfg.check_orientation);

    std::cerr << "[ConfigLoader] Loaded MatcherConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// PlaceRecognitionConfig
// ================================================================

loopdetection::PlaceRecognitionConfig
loadPlaceRecognitionConfig(const std::string& yaml_path) {
    loopdetection::PlaceRecognitionConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), using defaults for PlaceRecognitionConfig" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", using defaults for PlaceRecognitionConfig" << std::endl;
        return cfg;
    }

    cfg.vocab_build_threshold = getInt(fs, "vocab_build_threshold", cfg.vocab_build_threshold);
    cfg.vocab_branching_factor = getInt(fs, "vocab_branching_factor", cfg.vocab_branching_factor);
    cfg.vocab_depth = getInt(fs, "vocab_depth", cfg.vocab_depth);
    cfg.kmeans_max_iterations = getInt(fs, "kmeans_max_iterations", cfg.kmeans_max_iterations);
    cfg.new_word_distance_threshold =
        getDouble(fs, "new_word_distance_threshold", cfg.new_word_distance_threshold);
    cfg.hard_example_window = getInt(fs, "hard_example_window", cfg.hard_example_window);
    cfg.hard_threshold = getDouble(fs, "hard_threshold", cfg.hard_threshold);
    cfg.temporal_window = getInt(fs, "temporal_window", cfg.temporal_window);
    cfg.enable_online_learning = getBool(fs, "enable_online_learning", cfg.enable_online_learning);
    cfg.idf_update_interval = getInt(fs, "idf_update_interval", cfg.idf_update_interval);
    cfg.hard_example_learn_interval =
        getInt(fs, "hard_example_learn_interval", cfg.hard_example_learn_interval);
    cfg.min_hard_examples_to_learn =
        getInt(fs, "min_hard_examples_to_learn", cfg.min_hard_examples_to_learn);
    cfg.max_new_words_per_learn = getInt(fs, "max_new_words_per_learn", cfg.max_new_words_per_learn);
    cfg.use_tfidf = getBool(fs, "use_tfidf", cfg.use_tfidf);
    cfg.normalize_l1 = getBool(fs, "normalize_l1", cfg.normalize_l1);
    cfg.consistency_bonus = getDouble(fs, "consistency_bonus", cfg.consistency_bonus);
    cfg.min_candidate_score = getDouble(fs, "min_candidate_score", cfg.min_candidate_score);

    std::cerr << "[ConfigLoader] Loaded PlaceRecognitionConfig from " << yaml_path << std::endl;
    return cfg;
}

// ================================================================
// CameraIntrinsicsConfig
// ================================================================

cv::Mat CameraIntrinsicsConfig::makeK() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    return K;
}

cv::Mat CameraIntrinsicsConfig::makeD() const {
    cv::Mat D = cv::Mat_<double>(1, 5);
    D.at<double>(0, 0) = k1;
    D.at<double>(0, 1) = k2;
    D.at<double>(0, 2) = p1;
    D.at<double>(0, 3) = p2;
    D.at<double>(0, 4) = k3;
    return D;
}

CameraIntrinsicsConfig loadCameraIntrinsicsConfig(const std::string& yaml_path) {
    CameraIntrinsicsConfig cfg;

    cv::FileStorage fs;
    try {
        fs = cv::FileStorage(yaml_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << " (" << e.what() << "), camera intrinsics unavailable" << std::endl;
        return cfg;
    }
    if (!fs.isOpened()) {
        std::cerr << "[ConfigLoader] WARNING: Cannot open " << yaml_path
                  << ", camera intrinsics unavailable" << std::endl;
        return cfg;
    }

    cfg.fx = getDouble(fs, "fx", cfg.fx);
    cfg.fy = getDouble(fs, "fy", cfg.fy);
    cfg.cx = getDouble(fs, "cx", cfg.cx);
    cfg.cy = getDouble(fs, "cy", cfg.cy);
    cfg.width = getInt(fs, "width", cfg.width);
    cfg.height = getInt(fs, "height", cfg.height);
    cfg.k1 = getDouble(fs, "k1", cfg.k1);
    cfg.k2 = getDouble(fs, "k2", cfg.k2);
    cfg.p1 = getDouble(fs, "p1", cfg.p1);
    cfg.p2 = getDouble(fs, "p2", cfg.p2);
    cfg.k3 = getDouble(fs, "k3", cfg.k3);

    if (!cfg.isValid()) {
        std::cerr << "[ConfigLoader] WARNING: " << yaml_path
                  << " 缺少有效 fx/fy，内参视为未配置" << std::endl;
    }
    return cfg;
}

// ================================================================
// loadAllConfigs
// ================================================================

AllConfigs loadAllConfigs(const std::string& config_dir) {
    AllConfigs all_cfg;

    all_cfg.geometric_verifier = loadGeometricVerifierConfig(
        makePath(config_dir, "geometric_verifier.yaml"));
    all_cfg.loop_detector = loadLoopDetectorConfig(
        makePath(config_dir, "loop_detector.yaml"));
    // D3-d：geometric_verifier.yaml 以前只读进 AllConfigs 就断了，没有任何消费者，
    // 几何验证器实际用的是 LoopDetector 构造函数里的字面量。这里把整份配置塞进
    // loop_detector，由构造函数原样交给 GeometricVerifier —— 一条链，不再抄字段。
    all_cfg.loop_detector.geometric_verifier = all_cfg.geometric_verifier;
    all_cfg.pose_graph_optimizer = loadPoseGraphOptimizerConfig(
        makePath(config_dir, "pose_graph_optimizer.yaml"));
    all_cfg.covis_manager = loadCovisConfig(
        makePath(config_dir, "covis_manager.yaml"));
    all_cfg.tracking = loadTrackingConfig(
        makePath(config_dir, "tracking.yaml"));
    all_cfg.robust_pnp_solver = loadRobustPnPConfig(
        makePath(config_dir, "robust_pnp_solver.yaml"));
    // 特征提取参数的实际文件是 feature_extractor.yaml。
    // 历史版本这里读 octree_orb.yaml，而仓库里从来没有这个文件，
    // 于是每次启动都会 WARNING 后静默使用结构体默认值（R-019 反例）。
    all_cfg.feature_extractor = loadOctreeORBConfig(
        makePath(config_dir, "feature_extractor.yaml"));
    all_cfg.feature_matcher = loadFeatureMatcherConfig(
        makePath(config_dir, "feature_matcher.yaml"));
    all_cfg.place_recognition = loadPlaceRecognitionConfig(
        makePath(config_dir, "place_recognition.yaml"));

    // 合并：TrackingConfig 内嵌的提取/匹配子配置，唯一来源是
    // feature_extractor.yaml 与 feature_matcher.yaml。
    // 在这里合并而不是在 loadTrackingConfig 里读，是为了只有一个合并点：
    // 否则"tracking.yaml 也写一份"的历史写法会回来，两处取值不同时无法判断谁生效（R-019）。
    all_cfg.tracking.extractor_config = all_cfg.feature_extractor;
    all_cfg.tracking.matcher_config = all_cfg.feature_matcher;

    std::cerr << "[ConfigLoader] All configs loaded from " << config_dir << std::endl;
    return all_cfg;
}

}  // namespace config
}  // namespace sfc_core
