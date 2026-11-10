// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_MAPPING_OPTIMIZER_HPP_
#define SFC_CORE_MAPPING_OPTIMIZER_HPP_

#include <vector>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <Eigen/Dense>

// g2o 头文件（在自定义命名空间外部引入，引入后主动清除宏）
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_fixed_sized_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>
#include <g2o/types/sba/edge_project_xyz.h>

// 清除 g2o 注入的 sfc_core 宏，防止命名空间污染
#ifdef sfc_core
#undef sfc_core
#endif

namespace sfc_core {
namespace mapping {

// ================================================================
// SFCEdgeSE3ProjectXYZOnlyPose — 只优化相机位姿（用于 Tracking::poseOptimization）
// 使用 g2o 官方的 EdgeSE3ProjectXYZOnlyPose
// ================================================================
class SFCEdgeSE3ProjectXYZOnlyPose : public g2o::EdgeSE3ProjectXYZOnlyPose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ================================================================
// SFCEdgeSE3ProjectXYZ — 优化地图点和相机位姿（用于 LocalBA）
// 使用 g2o 官方的 EdgeSE3ProjectXYZ
// ================================================================
class SFCEdgeSE3ProjectXYZ : public g2o::EdgeSE3ProjectXYZ {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace mapping
} // namespace sfc_core

#endif // SFC_CORE_MAPPING_OPTIMIZER_HPP_
