#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/frontend/reprojection_association.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sfc_core {
namespace frontend {

namespace {

// 投影深度下限（米）：投影到相机后方或贴脸的点直接丢弃。
constexpr double kProjectionMinCamZ = 0.05;
// 世界系 z 下限（米）。这是旧实现的约束，保留以免同时改变这一路的行为；
// 它的物理含义（世界 z 低 = 点不可见）只在首帧相机大致水平时成立。
constexpr double kProjectionMinWorldZ = 0.05;
// 搜索半径的夹取范围（像素）。近处半径随深度放大，但太小的窗会漏掉投影误差
// 本身的抖动，太大的窗等于没有像素约束。
constexpr float kProjectionRadiusMinPx = 10.0f;
constexpr float kProjectionRadiusMaxPx = 30.0f;

}  // namespace

ProjectionMatch findKeypointNearProjection(const std::vector<cv::KeyPoint>& keypoints,
                                           const cv::Mat& descriptors,
                                           const cv::Mat& ref_descriptor,
                                           const cv::Point2f& projected_uv,
                                           const ProjectionSearchConfig& config,
                                           const std::vector<char>& blocked) {
    ProjectionMatch result;
    if (ref_descriptor.empty() || descriptors.empty()) return result;
    if (descriptors.type() != CV_8UC1 || ref_descriptor.type() != CV_8UC1) return result;
    if (ref_descriptor.cols != descriptors.cols) return result;

    const int n_keypoints =
        std::min(static_cast<int>(keypoints.size()), descriptors.rows);
    const float radius2 = config.radius_px * config.radius_px;
    const int n_blocked = static_cast<int>(blocked.size());

    int best_dist = 256;    // 最近邻汉明距离
    int second_dist = 256;  // 次近邻汉明距离；仍是 256 表示窗内只有一个候选
    int best_index = -1;

    for (int k = 0; k < n_keypoints; ++k) {
        if (k < n_blocked && blocked[k]) continue;
        const float du = keypoints[k].pt.x - projected_uv.x;
        const float dv = keypoints[k].pt.y - projected_uv.y;
        if (du * du + dv * dv > radius2) continue;

        const int dist = cv::norm(ref_descriptor, descriptors.row(k), cv::NORM_HAMMING);
        if (dist < best_dist) {
            second_dist = best_dist;
            best_dist = dist;
            best_index = k;
        } else if (dist < second_dist) {
            second_dist = dist;
        }
    }

    if (best_index < 0) return result;
    if (best_dist > config.hamming_threshold) return result;
    // Lowe 比值检验：窗内出现两个"看起来一样像"的特征时，像素上更近的那个未必
    // 就是这条 3D 点对应的特征，宁可丢掉这条关联也不要把错配喂给 PnP。
    if (second_dist < 256 &&
        static_cast<float>(best_dist) > config.lowe_ratio * static_cast<float>(second_dist)) {
        return result;
    }

    result.current_index = best_index;
    result.descriptor_distance = best_dist;
    return result;
}

std::vector<cv::DMatch> matchReferenceByProjection(
    const std::vector<cv::KeyPoint>& ref_kps,
    const cv::Mat& ref_descriptors,
    const std::vector<Eigen::Vector3d>& ref_pts_cam,
    const std::vector<bool>& ref_depth_valid,
    const Eigen::Matrix3d& R_ref_wc,
    const Eigen::Vector3d& t_ref_wc,
    const Eigen::Matrix3d& R_pred_wc,
    const Eigen::Vector3d& t_pred_wc,
    const std::vector<cv::KeyPoint>& current_kps,
    const cv::Mat& current_descriptors,
    int image_width,
    int image_height,
    const PinholeIntrinsics& intrinsics,
    const ProjectionSearchConfig& config,
    float search_radius_scale) {
    std::vector<cv::DMatch> matches;
    if (ref_kps.empty() || ref_descriptors.empty() || current_kps.empty()) return matches;
    if (ref_descriptors.type() != CV_8UC1 || current_descriptors.type() != CV_8UC1) {
        return matches;
    }

    const int n_ref = std::min({static_cast<int>(ref_kps.size()), ref_descriptors.rows,
                                static_cast<int>(ref_pts_cam.size())});
    // 占用标记用定长数组而不是 set：内层是"每个参考点扫一遍全帧特征"，
    // 用 set 查询会把这件事变成每个候选一次红黑树查找。
    std::vector<char> taken(current_kps.size(), 0);
    const Eigen::Matrix3d R_pred_cw = R_pred_wc.transpose();

    for (int i = 0; i < n_ref; ++i) {
        if (i < static_cast<int>(ref_depth_valid.size()) && !ref_depth_valid[i]) continue;
        const Eigen::Vector3d& p_cam = ref_pts_cam[i];
        if (!p_cam.allFinite() || p_cam.norm() < 1e-6) continue;

        const Eigen::Vector3d p_world = R_ref_wc * p_cam + t_ref_wc;
        if (p_world.z() < kProjectionMinWorldZ) continue;

        const Eigen::Vector3d p_cur = R_pred_cw * (p_world - t_pred_wc);
        const double z = p_cur.z();
        if (!std::isfinite(z) || z < kProjectionMinCamZ) continue;

        const double invz = 1.0 / z;
        const double u = intrinsics.fx * p_cur.x() * invz + intrinsics.cx;
        const double v = intrinsics.fy * p_cur.y() * invz + intrinsics.cy;
        if (u < 0.0 || u >= image_width || v < 0.0 || v >= image_height) continue;

        // 半径按投影深度缩放：近处同样的世界位移对应更多像素。
        ProjectionSearchConfig search = config;
        search.radius_px = std::min(
            std::max(static_cast<float>(z) * search_radius_scale, kProjectionRadiusMinPx),
            kProjectionRadiusMaxPx);

        const ProjectionMatch pm = findKeypointNearProjection(
            current_kps, current_descriptors, ref_descriptors.row(i),
            cv::Point2f(static_cast<float>(u), static_cast<float>(v)), search, taken);
        if (pm.current_index < 0) continue;

        matches.emplace_back(i, pm.current_index, static_cast<float>(pm.descriptor_distance));
        taken[pm.current_index] = 1;
    }
    return matches;
}

std::vector<AssociationCandidate> verifyAssociationsAgainstPose(
    const std::vector<cv::DMatch>& matches,
    const std::vector<Eigen::Vector3d>& pts_cam,
    const std::vector<bool>& depth_valid,
    const Eigen::Matrix3d& R_ref_wc,
    const Eigen::Vector3d& t_ref_wc,
    const std::vector<cv::Point2f>& current_uv,
    const Eigen::Matrix3d& R_cur_wc,
    const Eigen::Vector3d& t_cur_wc,
    const PinholeIntrinsics& intrinsics,
    float max_error_px,
    double min_depth_m) {
    std::vector<AssociationCandidate> accepted;
    const Eigen::Matrix3d R_cw = R_cur_wc.transpose();

    // 同一 keypoint 只保留重投影误差最小的那条关联：一个像素挂两个 3D 点会让
    // poseOptimization 对同一个观测重复计权，等价于给错点加了一次额外的话语权。
    std::vector<float> best_error(current_uv.size(), std::numeric_limits<float>::max());
    std::vector<int> best_slot(current_uv.size(), -1);

    for (const auto& m : matches) {
        const int ref_i = m.queryIdx;
        const int cur_i = m.trainIdx;
        if (ref_i < 0 || cur_i < 0) continue;
        if (ref_i >= static_cast<int>(pts_cam.size())) continue;
        if (ref_i < static_cast<int>(depth_valid.size()) && !depth_valid[ref_i]) continue;
        if (cur_i >= static_cast<int>(current_uv.size())) continue;

        const Eigen::Vector3d& p_cam_ref = pts_cam[ref_i];
        if (!p_cam_ref.allFinite() || p_cam_ref.norm() < 1e-6) continue;

        const Eigen::Vector3d p_world = R_ref_wc * p_cam_ref + t_ref_wc;
        const Eigen::Vector3d p_cam_cur = R_cw * (p_world - t_cur_wc);
        if (!p_cam_cur.allFinite() || p_cam_cur.z() < min_depth_m) continue;

        const double invz = 1.0 / p_cam_cur.z();
        const double u = intrinsics.fx * p_cam_cur.x() * invz + intrinsics.cx;
        const double v = intrinsics.fy * p_cam_cur.y() * invz + intrinsics.cy;
        const float err = static_cast<float>(
            std::hypot(u - current_uv[cur_i].x, v - current_uv[cur_i].y));
        if (!std::isfinite(err) || err > max_error_px) continue;
        if (err >= best_error[cur_i]) continue;

        best_error[cur_i] = err;
        if (best_slot[cur_i] < 0) {
            best_slot[cur_i] = static_cast<int>(accepted.size());
            accepted.push_back(AssociationCandidate{});
        }
        AssociationCandidate& slot = accepted[best_slot[cur_i]];
        slot.current_index = cur_i;
        slot.ref_index = ref_i;
        slot.world_point = p_world;
    }

    return accepted;
}

}  // namespace frontend
}  // namespace sfc_core
