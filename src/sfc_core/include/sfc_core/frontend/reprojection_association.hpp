#ifndef SFC_CORE_FRONTEND_REPROJECTION_ASSOCIATION_HPP
#define SFC_CORE_FRONTEND_REPROJECTION_ASSOCIATION_HPP

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <vector>

namespace sfc_core {
namespace frontend {

// ================================================================
// 投影关联（3D-2D correspondence）的公共判定
//
// 为什么单独成文件：这两条判据此前各自散落在 tracking.cpp 的重定位路径里，
// 一处只按"投影位置最近的像素"取点（不比描述子、不做比值检验），另一处把
// 未经几何复核的匹配直接当成"已验证观测"写进当前帧。两者都是"把弱证据当
// 强证据用"，此处把它们写成可单测的纯函数，判定标准只有一个来源（R-019）。
// ================================================================

/// 针孔内参。纯计算不需要 cv::Mat 形式的 K。
struct PinholeIntrinsics {
    double fx = 525.0;
    double fy = 525.0;
    double cx = 319.5;
    double cy = 239.5;
};

/// 投影搜索的判据参数。
struct ProjectionSearchConfig {
    float radius_px = 20.0f;        // 搜索半径（像素）
    int hamming_threshold = 60;     // 最近邻描述子距离上限
    float lowe_ratio = 0.75f;       // 最近邻/次近邻距离比上限
};

/// 一条通过描述子 + 比值检验的投影匹配。
struct ProjectionMatch {
    int current_index = -1;          // 当前帧 keypoint 下标，-1 表示无匹配
    int descriptor_distance = 256;   // 汉明距离
};

/// 一条与给定位姿自洽的 3D-2D 关联。
struct AssociationCandidate {
    int current_index = -1;                  // 当前帧 keypoint 下标
    int ref_index = -1;                      // 参考帧 keypoint 下标（用作观测记录）
    Eigen::Vector3d world_point = Eigen::Vector3d::Zero();  // 该关联对应的世界点
};

/// 在 projected_uv 的 radius_px 邻域内，为 ref_descriptor 找当前帧 keypoint。
///
/// 取点顺序是：先按像素窗筛候选，再按汉明距离选最近邻，最后做 Lowe 比值检验。
/// 只按像素距离取最近的 keypoint 会让"投影位置附近恰好有个别的特征点"变成一条
/// 3D-2D 对应，PnP 的输入里因此混进大量错配，解只能停在初值附近。
///
/// @param keypoints       当前帧 keypoint（理想像素）
/// @param descriptors     当前帧描述子，CV_8U，每行一个
/// @param ref_descriptor  参考侧描述子（单行）
/// @param projected_uv    参考点按待求位姿投影出的像素坐标
/// @param config          半径/汉明门限/比值门限
/// @param blocked         长度不小于 keypoints.size() 的占用标记，1 表示该 keypoint 已被占用
/// @return 匹配结果；current_index < 0 表示该投影点无可用对应
ProjectionMatch findKeypointNearProjection(const std::vector<cv::KeyPoint>& keypoints,
                                           const cv::Mat& descriptors,
                                           const cv::Mat& ref_descriptor,
                                           const cv::Point2f& projected_uv,
                                           const ProjectionSearchConfig& config,
                                           const std::vector<char>& blocked);

/// 用预测位姿把参考关键帧的 3D 点投影到当前帧，为每个投影点找描述子一致的 keypoint。
///
/// 这是重定位"策略 1（利用运动模型先验）"的全部几何：位姿先验只负责把 3D 点送
/// 到画面上的大致位置，是否正确由描述子 + 比值检验决定，像素窗只用来圈候选。
///
/// @param ref_kps            参考关键帧 keypoint（理想像素）
/// @param ref_descriptors    参考关键帧描述子，行号与 ref_kps 对齐
/// @param ref_pts_cam        参考关键帧相机系 3D 点
/// @param ref_depth_valid    参考关键帧深度有效性
/// @param R_ref_wc           参考关键帧 R_wc
/// @param t_ref_wc           参考关键帧 t_wc
/// @param R_pred_wc          预测位姿 R_wc
/// @param t_pred_wc          预测位姿 t_wc
/// @param current_kps        当前帧 keypoint（理想像素）
/// @param current_descriptors 当前帧描述子
/// @param image_width        图像宽（投影落到画面外则丢弃）
/// @param image_height       图像高
/// @param intrinsics         针孔内参
/// @param config             半径/汉明门限/比值门限
/// @param search_radius_scale 半径按投影深度缩放的系数（近处位移对应更多像素）
/// @return 匹配列表：queryIdx=参考点下标，trainIdx=当前帧 keypoint 下标，distance=汉明距离
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
    float search_radius_scale = 7.5f);

/// 用给定位姿复核一批 3D-2D 候选，只保留重投影误差在门限内的那些。
///
/// 参考侧的 pts_cam 先按 (R_ref_wc, t_ref_wc) 变到世界系，再用 (R_cur_wc, t_cur_wc)
/// 投影回当前帧像素，与 current_uv 比距离。current_index 互斥：同一 keypoint 只会
/// 保留误差最小的那条关联，否则 poseOptimization 会对同一个像素重复计权。
///
/// @param matches       参考帧→当前帧的匹配（queryIdx 索引 pts_cam，trainIdx 索引 current_uv）
/// @param pts_cam       参考帧相机系 3D 点
/// @param depth_valid   参考帧深度有效标记
/// @param R_ref_wc      参考帧 R_wc（相机→世界）
/// @param t_ref_wc      参考帧 t_wc
/// @param current_uv    当前帧 keypoint 像素坐标（按 keypoint 下标索引）
/// @param R_cur_wc      当前帧 R_wc
/// @param t_cur_wc      当前帧 t_wc
/// @param intrinsics    针孔内参
/// @param max_error_px  允许的重投影误差（像素）
/// @param min_depth_m   投影到相机后方或过近的点直接丢弃
/// @return 通过复核的关联列表，按 current_index 互斥
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
    double min_depth_m);

}  // namespace frontend
}  // namespace sfc_core

#endif  // SFC_CORE_FRONTEND_REPROJECTION_ASSOCIATION_HPP
