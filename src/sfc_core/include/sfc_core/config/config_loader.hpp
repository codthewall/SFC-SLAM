// ================================================================
// SFC-SLAM Config Loader
// 文件路径: config/config_loader.hpp
// 模块: 通用 - YAML 配置文件加载器
// ================================================================
//
// 功能说明：
//   统一的 YAML 配置文件加载器，将 YAML 文件中的参数加载到
//   各个模块的 Config 结构体中。
//   如果 YAML 文件不存在或参数缺失，使用 Config 的默认值。
//
// 使用方法：
//   ```cpp
//   #include "sfc_core/config/config_loader.hpp"
//
//   // 加载单个模块配置
//   auto geo_cfg = sfc_core::config::loadGeometricVerifierConfig("config/geometric_verifier.yaml");
//
//   // 加载所有配置
//   auto all_cfg = sfc_core::config::loadAllConfigs("config");
//   TrackingConfig track_cfg = all_cfg.tracking;
//   ```
//
// YAML 文件列表（相对于 config/ 目录）：
//   - geometric_verifier.yaml    (GeometricVerifierConfig)
//   - loop_detector.yaml        (LoopDetectorConfig)
//   - covis_manager.yaml        (CovisConfig)
//   - tracking.yaml             (TrackingConfig)
//   - robust_pnp_solver.yaml     (RobustPnPConfig)
//   - feature_extractor.yaml    (ExtractorConfig)
//   - feature_matcher.yaml      (MatcherConfig)
//   - pose_graph_optimizer.yaml (PoseGraphOptimizerOptions)
//   - place_recognition.yaml    (PlaceRecognitionConfig)
//
// ================================================================

#ifndef SFC_CORE_CONFIG_CONFIG_LOADER_HPP_
#define SFC_CORE_CONFIG_CONFIG_LOADER_HPP_

#include <string>
#include <iostream>
#include <opencv2/core.hpp>

#include "sfc_core/loopdetection/geometric_verifier.hpp"
#include "sfc_core/loopdetection/loop_detector.hpp"
#include "sfc_core/loopdetection/pose_graph_optimizer.hpp"
#include "sfc_core/loopdetection/place_recognition.hpp"
#include "sfc_core/frontend/covis_manager.hpp"
#include "sfc_core/frontend/tracking.hpp"
#include "sfc_core/frontend/robust_pnp_solver.hpp"
#include "sfc_core/feature/octree_orb.hpp"
#include "sfc_core/feature/feature_matcher.hpp"

namespace sfc_core {
namespace config {

// ================================================================
// 相机内参（独立于 TrackingConfig，便于按数据集切换）
// ================================================================
struct CameraIntrinsicsConfig {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    int width = 640;
    int height = 480;
    // OpenCV 5 参数畸变顺序：k1 k2 p1 p2 k3
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double k3 = 0.0;

    bool isValid() const { return fx > 0.0 && fy > 0.0; }
    cv::Mat makeK() const;
    cv::Mat makeD() const;
};

/**
 * 从 YAML 文件加载 CameraIntrinsicsConfig。
 * @param yaml_path YAML 文件完整路径
 * @return CameraIntrinsicsConfig；文件缺失时 isValid()==false
 */
CameraIntrinsicsConfig loadCameraIntrinsicsConfig(const std::string& yaml_path);

// ================================================================
// 全局配置结构体（包含所有子配置）
// ================================================================

struct AllConfigs {
    loopdetection::GeometricVerifierConfig geometric_verifier;
    loopdetection::LoopDetectorConfig loop_detector;
    loopdetection::PoseGraphOptimizerOptions pose_graph_optimizer;
    frontend::CovisConfig covis_manager;
    frontend::TrackingConfig tracking;
    frontend::RobustPnPConfig robust_pnp_solver;
    feature::OctreeORBConfig feature_extractor;  // Octree ORB (ORB-SLAM3 风格)
    feature::MatcherConfig feature_matcher;
    loopdetection::PlaceRecognitionConfig place_recognition;
};

// ================================================================
// 辅助函数（内部使用）
// ================================================================

// ================================================================
// 主加载函数
// ================================================================

/**
 * 从 YAML 文件加载 GeometricVerifierConfig。
 * @param yaml_path YAML 文件完整路径
 * @return GeometricVerifierConfig
 */
loopdetection::GeometricVerifierConfig
loadGeometricVerifierConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 LoopDetectorConfig。
 * @param yaml_path YAML 文件完整路径
 * @return LoopDetectorConfig
 */
loopdetection::LoopDetectorConfig
loadLoopDetectorConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 PoseGraphOptimizerOptions。
 * @param yaml_path YAML 文件完整路径
 * @return PoseGraphOptimizerOptions
 */
loopdetection::PoseGraphOptimizerOptions
loadPoseGraphOptimizerConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 CovisConfig。
 * @param yaml_path YAML 文件完整路径
 * @return CovisConfig
 */
frontend::CovisConfig
loadCovisConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 TrackingConfig。
 * @param yaml_path YAML 文件完整路径
 * @return TrackingConfig
 */
frontend::TrackingConfig
loadTrackingConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 RobustPnPConfig。
 * @param yaml_path YAML 文件完整路径
 * @return RobustPnPConfig
 */
frontend::RobustPnPConfig
loadRobustPnPConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 OctreeORBConfig（ORB-SLAM3 风格）。
 * @param yaml_path YAML 文件完整路径
 * @return OctreeORBConfig
 */
feature::OctreeORBConfig
loadOctreeORBConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 MatcherConfig。
 * @param yaml_path YAML 文件完整路径
 * @return MatcherConfig
 */
feature::MatcherConfig
loadFeatureMatcherConfig(const std::string& yaml_path);

/**
 * 从 YAML 文件加载 PlaceRecognitionConfig。
 * @param yaml_path YAML 文件完整路径
 * @return PlaceRecognitionConfig
 */
loopdetection::PlaceRecognitionConfig
loadPlaceRecognitionConfig(const std::string& yaml_path);

/**
 * 从配置目录加载所有配置。
 * @param config_dir 配置文件目录路径（不含末尾斜杠）
 * @return AllConfigs
 */
AllConfigs loadAllConfigs(const std::string& config_dir);

/**
 * 辅助函数：构建 YAML 文件路径。
 * @param config_dir 配置目录
 * @param filename 文件名（含 .yaml 后缀）
 * @return 完整路径
 */
inline std::string makePath(const std::string& config_dir, const std::string& filename) {
    if (config_dir.empty()) return filename;
    if (config_dir.back() == '/' || config_dir.back() == '\\') {
        return config_dir + filename;
    }
    return config_dir + "/" + filename;
}

}  // namespace config
}  // namespace sfc_core

#endif  // SFC_CORE_CONFIG_CONFIG_LOADER_HPP_
