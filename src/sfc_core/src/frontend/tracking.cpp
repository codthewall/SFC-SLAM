// ================================================================
// 必须在命名空间外部的最开头清除 sfc_core 宏
// ================================================================
#ifdef sfc_core
#undef sfc_core
#endif

// 第三方库（g2o / Eigen / OpenCV）必须在命名空间外部引入
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <limits>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

// g2o 头文件
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include "sfc_core/mapping/optimizer.hpp"

// 项目自身头文件
#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/frontend/tracking.hpp"
#include "sfc_core/frontend/depth_filter.hpp"
#include "sfc_core/frontend/keyframe_policy.hpp"
#include "sfc_core/frontend/robust_pnp_solver.hpp"
#include "sfc_core/feature/octree_orb.hpp"
#include "sfc_core/feature/bow_adapter.hpp"
#include "sfc_core/feature/feature_matcher.hpp"
#include "sfc_core/config/config_loader.hpp"
#include "sfc_core/utils/camera_utils.hpp"

namespace sfc_core {
namespace frontend {

// 新建地图点的观察期（帧数）现在由 TrackingConfig::map_point_grace_frames 提供，
// 默认 8。这里不再保留同名常量 —— 同一参数两个来源时，改了一个另一个不生效（R-019）。

// E6：近邻查重的体素边长（米）与判定半径。体素只用来把候选集从全图缩到
// 一个立方体，最终仍按真实距离判定，所以体素本身略大于半径。
constexpr double kMapPointVoxelSize = 0.10;
constexpr double kMapPointDuplicateRadius = 0.05;

// R4/R-018：投影匹配的几何一致性容差 = max(搜索半径 × 缩放, 下限)。
// 原实现是 max(radius × 2.5, 20px)，20px 的下限比搜索窗本身（15–30px）还大，
// 等于让窗口内任何候选都通过几何验证、只靠描述子距离决定匹配。
// 8px 足够覆盖金字塔高层和少量运动模糊，同时挡得住"投影位置差出一个窗口"的错配。
constexpr float kGeoVerifyRadiusScale = 1.5f;
constexpr float kGeoVerifyMinPx = 8.0f;

// R4/R-018：参考关键帧 PnP 结果与上一帧位姿的一致性上限。
// 原值 5.0m / 1.5rad（≈86°）事实上等于"PnP 永远通过"，会把落进镜像解、
// 对应点错配的解当成成功。参考关键帧跟踪是帧间小位移，正常在厘米级，
// 收紧到 0.3m / 0.3rad（≈17°）：超限即回滚位姿并置 mbVelocityValid_=false，
// 让下一帧改走运动模型/重定位，而不是把一个错解一帧帧沿用下去。
constexpr double PNP_T_DIFF_THRESH_M = 0.3;
constexpr double PNP_ROT_DIFF_THRESH_RAD = 0.3;

// 世界坐标 → int64 体素键。三轴各 21 bit（±2^20 个体素 ≈ ±100 km），
// 对室内数据集足够，且不会因为坐标大范围平移而塌缩到同一个键。
inline int64_t voxelKey(const Eigen::Vector3d& p) {
    const int64_t ix = static_cast<int64_t>(std::floor(p.x() / kMapPointVoxelSize));
    const int64_t iy = static_cast<int64_t>(std::floor(p.y() / kMapPointVoxelSize));
    const int64_t iz = static_cast<int64_t>(std::floor(p.z() / kMapPointVoxelSize));
    const int64_t mask = (static_cast<int64_t>(1) << 21) - 1;
    return ((ix & mask) << 42) | ((iy & mask) << 21) | (iz & mask);
}

// ================================================================
// ORB 描述子 Hamming 距离
// ================================================================
inline int descriptorDistance(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty()) return 256;
    if (a.cols != b.cols || a.rows != b.rows) return 256;
    return cv::norm(a, b, cv::NORM_HAMMING);
}

// ================================================================
// 坐标系转换辅助函数
// 注意：SFC-SLAM 的 Pose {q, t} 表示 T_wc = [R_wc | t_wc]
// 即：世界坐标 = R_wc * 相机坐标 + t_wc
// 其中 R_wc = q.toRotationMatrix()，t_wc = t
// ================================================================

// 从 Pose (T_wc) 获取 world→camera 的变换 (T_cw = T_wc^{-1})
// T_cw = [R_cw | t_cw]，R_cw = R_wc^T，t_cw = -R_wc^T * t_wc
inline std::pair<Eigen::Matrix3d, Eigen::Vector3d> getCwFromPose(const Pose& pose) {
    Eigen::Matrix3d R_cw = pose.q.toRotationMatrix().transpose();
    Eigen::Vector3d t_cw = -R_cw * pose.t;
    return {R_cw, t_cw};
}

// 从相机坐标转换到世界坐标：P_world = R_wc * P_cam + t_wc
inline Eigen::Vector3d camToWorld(const Eigen::Vector3d& pt_cam, const Pose& pose) {
    Eigen::Matrix3d R_wc = pose.q.toRotationMatrix();
    Eigen::Vector3d t_wc = pose.t;
    return R_wc * pt_cam + t_wc;
}

// 从世界坐标转换到相机坐标：P_cam = R_cw * P_world + t_cw
inline Eigen::Vector3d worldToCam(const Eigen::Vector3d& pt_world, const Pose& pose) {
    auto [R_cw, t_cw] = getCwFromPose(pose);
    return R_cw * pt_world + t_cw;
}

// getCwFromPose 的逆运算：把 T_cw（g2o / OpenCV 的约定）还原成对外的 T_wc
// T_wc = T_cw^{-1} = [R_cw^T | -R_cw^T * t_cw]
// 只在必须与 g2o 顶点 / PnP 结果交换位姿的地方使用（R1）。
inline Pose poseFromCw(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw) {
    Pose out;
    out.q = Eigen::Quaterniond(R_cw.transpose());
    out.q.normalize();
    out.t = -(R_cw.transpose() * t_cw);
    return out;
}

// 检查位姿是否有效（无NaN、平移合理）
inline bool isValidPose(const Pose& pose, double max_trans = 1000.0) {
    if (!pose.t.allFinite()) return false;
    if (!pose.q.coeffs().allFinite()) return false;
    if (pose.t.norm() > max_trans) return false;
    if (pose.q.norm() < 0.1 || pose.q.norm() > 2.0) return false;
    return true;
}

// ★★★ Bug Fix: 双向一致性检查（forward-backward check）★★★
// 问题：描述子匹配只做单向匹配，不检查反向一致性
// 修复：匹配后做反向验证，确保 2D 特征点与 3D 地图点的双向重投影一致
inline bool checkBidirectionalConsistency(
    const Eigen::Vector3d& p3d_world,
    const cv::KeyPoint& kp_curr,
    const cv::Mat& desc_curr,
    const cv::Mat& desc_last,
    const Pose& curr_pose,
    const Pose& last_pose,
    double fx, double fy, double cx, double cy,
    float radius = 15.0f) {
    
    // Step 1: 用当前帧位姿将 3D 点投影到当前帧，检查与 kp_curr 的一致性
    Eigen::Vector3d pt_cam_curr = worldToCam(p3d_world, curr_pose);
    
    if (pt_cam_curr.z() < 0.05 || !pt_cam_curr.allFinite()) return false;
    
    double invz = 1.0 / pt_cam_curr.z();
    float u_proj = static_cast<float>(fx * pt_cam_curr.x() * invz + cx);
    float v_proj = static_cast<float>(fy * pt_cam_curr.y() * invz + cy);
    
    float dx = kp_curr.pt.x - u_proj;
    float dy = kp_curr.pt.y - v_proj;
    float reproj_dist = std::sqrt(dx * dx + dy * dy);
    
    // 重投影误差超过半径的一半，视为不一致
    if (reproj_dist > radius * 0.5f) return false;
    
    // Step 2: 用上一帧位姿将 3D 点投影到上一帧，检查描述子一致性
    Eigen::Vector3d pt_cam_last = worldToCam(p3d_world, last_pose);
    
    if (pt_cam_last.z() < 0.05 || !pt_cam_last.allFinite()) return true;  // 上一帧可能看不到该点
    
    // 检查描述子距离是否足够小（双向一致性）
    if (!desc_curr.empty() && !desc_last.empty()) {
        int dist = descriptorDistance(desc_curr, desc_last);
        if (dist > 100) return false;  // 描述子差异过大
    }
    
    return true;
}

// ================================================================
// 统一获取相机内参（避免硬编码525.0/319.5）
// ================================================================
struct CameraParams {
    double fx, fy, cx, cy;
};

inline CameraParams getCameraParams(const cv::Mat& K) {
    CameraParams cp;
    if (!K.empty() && K.cols == 3 && K.rows == 3) {
        cp.fx = K.at<double>(0, 0);
        cp.fy = K.at<double>(1, 1);
        cp.cx = K.at<double>(0, 2);
        cp.cy = K.at<double>(1, 2);
    } else {
        // 默认值 - 仅作为后备，实际使用时应确保K已正确配置
        cp.fx = cp.fy = 525.0;
        cp.cx = 319.5;
        cp.cy = 239.5;
    }
    return cp;
}

// ================================================================
// 四元数归一化辅助函数（防止数值误差累积）
// ================================================================
inline void normalizeQuaternion(Eigen::Quaterniond& q) {
    q.normalize();
    // 确保四元数 w 为正（最短弧表示）
    if (q.w() < 0) {
        q.coeffs() = -q.coeffs();
    }
}

// ================================================================
// 关键帧相机位姿常量（单位：米）
// ================================================================
static constexpr double KF_BASELINE = 0.0;   // 单目 RGB-D，基线 = 0（无立体匹配）
// 原先这里还有一个 KF_TH_DEPTH = 40.0、注释写"深度阈值 = max_depth * 4.0"，
// 但全文件没有任何引用，而语义本身也是把 ThDepth（基线倍数）当成了米。已删除。

// 投影前的最小相机系深度（米）。低于此值 1/z 放大误差，投影坐标不可信。
// 取 0.05m 与 isInFront() 默认门槛一致，对应深度相机近端有效量程。
constexpr double MIN_PROJECTION_Z = 0.05;

// 重定位单帧跳变上限（米）。超过该位移的重定位结果认为是匹配到了场景中的相似位置
// （perceptual aliasing），而非相机真实移动。室内手持序列相邻帧通常 <0.1m 量级。
constexpr double RELOC_MAX_DELTA_T = 3.0;

// 重定位输出位姿的相机高度下限（米）。TUM fr1 手持序列相机高度约 0.9~1.3m，
// 这里只做"明显钻到地下"的粗筛，不约束正常高度波动。
constexpr double MIN_RELOC_CAMERA_Z = -2.0;

// 帧间位姿跳变上限由 config_.frame_jump_max_translation / frame_jump_max_rotation
// 提供（默认 0.15 m / 0.26 rad）。原实现把 0.5 m / 0.524 rad 硬编码在这里，
// 对 TUM fr1（真实单帧位移 1~5 cm）等于没有守卫，实测 0.135 m 的瞬移被直接放行。

// 单次回环修正里，单个关键帧位姿允许被移动的最大距离（米）。
// PGO 是全局优化，修正几十厘米到几米都正常；超过这个量级说明回环边本身错了
// （尺度估计失败或误匹配），此时宁可整次丢弃，也不要让一条错边改写整条轨迹。
// 依据：TUM fr1 系列场景尺度 <10m，一条正确回环不会需要把关键帧搬过整个场景。
constexpr double LOOP_MAX_KF_CORRECTION_M = 10.0;

// ================================================================
// 统一投影/反投影辅助：pts_3d = 相机坐标系，pts_3d_world = 世界坐标系
//   camToWorld(pt_cam, pose)  = R_wc * pt_cam + t_wc   （pose = T_wc）
//   worldToCam(pt_world, pose) = R_cw * pt_world + t_cw  （t_cw = -R_cw * t_wc）
// ================================================================

// 检查 3D 点是否在相机前方且有效
inline bool isInFront(const Eigen::Vector3d& pt_cam, double min_depth = 0.05) {
    return pt_cam.z() >= min_depth && pt_cam.allFinite();
}

// 将世界坐标点投影到当前帧相机（用于 searchLocalPoints / searchByProjection）
// 返回是否投影成功（深度 > min_depth）
inline bool projectWorldToCam(
    const Eigen::Vector3d& pt_world,
    const Pose& frame_pose,        // frame_pose = T_wc（相机在世界中的位姿）
    double fx, double fy, double cx, double cy,
    double& u, double& v, double& z,
    double min_depth = 0.05) {

    Eigen::Vector3d pt_cam = worldToCam(pt_world, frame_pose);
    if (!isInFront(pt_cam, min_depth)) return false;

    z = pt_cam.z();
    double invz = 1.0 / z;
    u = fx * pt_cam.x() * invz + cx;
    v = fy * pt_cam.y() * invz + cy;
    return true;
}

// 从当前帧相机点转换到世界坐标（用于 createNewKeyFrame 深度点三角化）
// P_world = R_wc * P_cam + t_wc
inline Eigen::Vector3d camToWorldPoint(
    const Eigen::Vector3d& pt_cam,
    const Pose& frame_pose) {   // frame_pose = T_wc
    return camToWorld(pt_cam, frame_pose);
}

Tracking::Tracking(const TrackingConfig& config) : config_(config) {
    // ★★★ 使用 Octree ORB（ORB-SLAM3 风格）★★★
    extractor_ = std::make_unique<feature::OctreeORB>(config.extractor_config);
    // ★★★ BoW 词袋适配器（ORB-SLAM3 风格重定位）★★★
    bow_adapter_ = std::make_unique<feature::BoWAdapter>();
    matcher_ = std::make_unique<feature::FeatureMatcher>(config.matcher_config);

    RobustPnPConfig pnp_cfg;
    pnp_cfg.max_ransac_iterations = config.pnp_max_ransac_iterations;
    pnp_cfg.initial_inlier_thresh_px = config.pnp_initial_thresh_px;
    pnp_cfg.final_inlier_thresh_px = config.pnp_final_thresh_px;
    pnp_cfg.expansion_rounds = config.pnp_expansion_rounds;
    pnp_cfg.expansion_factor = config.pnp_expansion_factor;
    pnp_cfg.ba_max_iterations = config.pnp_ba_max_iterations;
    pnp_cfg.use_huber_kernel = config.pnp_use_huber_kernel;
    pnp_cfg.huber_delta = config.pnp_huber_delta;
    pnp_cfg.use_depth_weighting = config.pnp_use_depth_weighting;
    pnp_cfg.use_adaptive_threshold = true;
    pnp_cfg.verbose = false;  // 关闭调试输出，生产环境应设为 false
    robust_pnp_solver_ = std::make_unique<RobustPnPSolver>(config.K, pnp_cfg);
    // R3：畸变只在一个地方生效。全系统（含 PnP）统一消费 _理想像素_：
    // 关键点先经 undistortPoints(P=K) 变成理想像素，深度采样仍走畸变坐标
    // （深度图与畸变 RGB 对齐）。因此 PnP 必须收到空的 D —— 若这里传 config.D，
    // OpenCV 会把已经是理想坐标的点再"去畸变"一次，等于把畸变模型应用两遍
    // （计划 §四 的告警："不能同时做两件事，二者叠加会把误差方向翻倍"）。
    // 本系统选的是「全程理想像素」那一支，畸变在帧准备阶段一次处理完。
    robust_pnp_solver_->setCamera(config.K, cv::Mat());
    robust_pnp_solver_->setPnPMethod(RobustPnPSolver::PnPMethod::EPNP);
    // 构造期只打印一次求解器配置，受同一开关控制，避免库被嵌入时无条件刷屏。
    { if (config_.verbose_diag) std::cerr << "[Tracking] PnP method: EPnP (RANSAC)" << std::endl; }

    mbVelocityValid_ = false;
    mLastFrame_ = nullptr;
    mCurrentFrame_ = nullptr;
    mnLastRelocFrameId_ = 0;
    mbMapUpdated_ = false;
    mbVO_ = false;
    mTrackedFr_ = 0;
    mnMatchesInliers_ = 0;
    // 注意：std::deque 没有 reserve() 方法
    mvpCurrentMapPointIds_.reserve(500);
    mLastFrameMapPointIds_.reserve(500);

    // ================================================================
    // 初始化 ORB-SLAM3 风格后端
    // ================================================================
    // 创建地图管理器
    mpMap_ = std::make_shared<mapping::Map>();

    // 创建 Atlas（多地图管理器）
    mpAtlas_ = std::make_shared<mapping::Atlas>();
    // 前端直接使用 Atlas 管理的活动地图
    mpMap_ = mpAtlas_->getActiveMap();

    // 创建并启动 LocalMapping 线程
    mpLocalMapper_ = std::make_shared<mapping::LocalMapping>(mpMap_);
    // 设置地图点替换回调：当 LocalMapping 中发生替换时通知 Tracking
    mpLocalMapper_->setReplacementCallback([this](int old_id, int new_id) {
        mReplaceMap_[old_id] = new_id;
    });
    mpLocalMapper_->run();
}

// ================================================================
// 地图点管理（ID-based，避免指针失效问题）
// ================================================================
int Tracking::createMapPoint(const Eigen::Vector3d& world_pt, const cv::Mat& desc,
                            int frame_id, int kpt_idx, int obs_count, bool permanent) {
    // 地图点是整个前端的共享状态：一个 NaN 点进入 mAllMapPoints_ 之后，
    // 每帧的投影、BA、关键帧判定都会读到它，且没有清理路径能把毒扩散收回来
    // （日志里 depth=[1e10,-1e10] 的帧就是所有点都已翻到相机背后的表现）。
    // 因此在这里做唯一的准入校验，拒绝非有限值和零向量。
    if (!world_pt.allFinite() || world_pt.norm() < 1e-6) {
        return -1;
    }

    // E6-3：近邻查重。同一个物理点在相邻关键帧会被反复"新建"（深度点不共享 id），
    // 于是地图里堆出大量相距几毫米的重复点，既浪费匹配预算又会在 BA 里互相顶。
    // 只在建永久点时查重：临时点的语义就是"本帧一次性"，复用别人的 id 会把它
    // 绑进 mAllMapPoints_ 的生命周期里。
    if (permanent) {
        const int64_t vkey = voxelKey(world_pt);
        auto vit = mVoxelMapPoints_.find(vkey);
        if (vit != mVoxelMapPoints_.end()) {
            for (int cand_id : vit->second) {
                auto cit = mAllMapPoints_.find(cand_id);
                if (cit == mAllMapPoints_.end() || cit->second.is_bad) continue;
                if ((cit->second.world_point - world_pt).norm() > kMapPointDuplicateRadius) continue;
                // 命中重复：不新建 id，只把这次观测记到已有点上。
                // 复用 id 的代价必须一起处理 —— 调用方在 mvpCurrentMapPointIds_ 里
                // 拿到这个 id 后就是正常的地图点关联，没有额外分支。
                addObservationToMapPoint(cand_id, frame_id, kpt_idx, desc);
                return cand_id;
            }
        }
    }

    MapPointCandidate mp;
    mp.id = next_mp_id_++;
    mp.world_point = world_pt;
    if (!desc.empty()) mp.descriptor = desc.clone();
    mp.observations.emplace_back(frame_id, kpt_idx);
    mp.frontend_observation_count = obs_count;
    mp.confirmed = (obs_count >= 2);
    mp.is_bad = false;
    mp.created_frame_id = frame_id;
    mp.last_observed_frame_id = frame_id;

    if (permanent) {
        mAllMapPoints_[mp.id] = mp;
        mNewMapPointIds_.push_back(mp.id);
        addToVoxelIndex(mp.id, world_pt);
    } else {
        // E6-1：临时 VO 点进独立池。它们每帧重建数百个、生命周期只有一帧，
        // 以前和永久点混在 mAllMapPoints_ 里且永不回收，是地图无限增长的唯一来源。
        mTemporalMapPoints_[mp.id] = mp;
    }
    return mp.id;
}

MapPointCandidate* Tracking::getMapPoint(int id) {
    auto it = mAllMapPoints_.find(id);
    if (it != mAllMapPoints_.end()) return &it->second;
    auto tit = mTemporalMapPoints_.find(id);
    if (tit != mTemporalMapPoints_.end()) return &tit->second;
    return nullptr;
}

// ================================================================
// E6：体素索引维护
// 索引只为近邻查重服务，允许有误差（点移动后键会过期），因此只在
// 建点/退休时维护，不跟着每次 BA 更新坐标重建。
// ================================================================
void Tracking::addToVoxelIndex(int id, const Eigen::Vector3d& world_pt) {
    mVoxelMapPoints_[voxelKey(world_pt)].push_back(id);
}

void Tracking::removeFromVoxelIndex(int id, const Eigen::Vector3d& world_pt) {
    auto vit = mVoxelMapPoints_.find(voxelKey(world_pt));
    if (vit == mVoxelMapPoints_.end()) return;
    auto& ids = vit->second;
    ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
    if (ids.empty()) mVoxelMapPoints_.erase(vit);
}

// ================================================================
// E6：清空临时点池
// ================================================================
void Tracking::clearTemporalMapPoints() {
    mlpTemporalPoints_.clear();
    mTemporalMapPoints_.clear();
}

void Tracking::addObservationToMapPoint(int mp_id, int frame_id, int kpt_idx, const cv::Mat& desc) {
    // R5：前端计数与后端计数必须同时增长。
    // 后端 mapping::MapPoint::found_count_ 只是统计量；真正决定点位是否存活的是
    // 前端 MapPointCandidate::frontend_observation_count —— 若这里不 ++，
    // cleanOldMapPoints() 会把所有点判成"仅单次观测"，第一个关键帧就清空整张地图。
    if (MapPointCandidate* cand = getMapPoint(mp_id)) {
        cand->frontend_observation_count++;
        // 同一次匹配可能被重复记账，观测列表按 (frame, kpt) 去重
        const std::pair<int, int> obs(frame_id, kpt_idx);
        if (std::find(cand->observations.begin(), cand->observations.end(), obs) ==
            cand->observations.end()) {
            cand->observations.push_back(obs);
        }
        cand->confirmed = (cand->frontend_observation_count >= 2);
        cand->last_observed_frame_id = frame_id;
    }
    mapping::MapPoint* mp = mpMap_->getMapPoint(mp_id);
    if (!mp) return;
    mp->addObservation(frame_id, kpt_idx);
    mp->increaseFound();
    if (mp->getDescriptor().empty() && !desc.empty()) {
        mp->setDescriptor(desc);
    }
}

// ================================================================
// 从深度图获取 3D 点
// ================================================================
void Tracking::compute3DPoints(Frame::Ptr frame, const cv::Mat& imDepth) {
    const size_t n_kpts = frame->undistorted_keypoints.empty()
        ? frame->keypoints.size() : frame->undistorted_keypoints.size();
    frame->pts_3d.clear();
    frame->pts_3d.reserve(n_kpts);
    frame->depth_valid.clear();
    frame->depth_valid.reserve(n_kpts);

    const cv::Mat& K = config_.K;
    CameraParams cam = getCameraParams(K);
    double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    const int rows = imDepth.rows, cols = imDepth.cols;
    // 深度图在 track() 的 Step 2 已统一换算成 CV_32FC1（米），这里不再分整数/浮点。
    const bool depth_is_float = (!imDepth.empty() && imDepth.type() == CV_32FC1);

    // R3：采样深度与反投影必须用两套不同的坐标，这是本节的核心。
    //   采样：深度图与 _畸变_ RGB 对齐（TUM RGB-D 等），必须用畸变像素 kp.pt；
    //   反投影：相机模型是理想针孔，必须用 _理想_ 像素。
    // 之前两者都用同一套坐标，于是「畸变像素 + 理想针孔」在 3D 点与 2D 观测两侧
    // 完全自洽、残差接近 0，真实的方向偏差被这套"自洽"掩盖掉了。
    // 因此：索引始终对齐 keypoints（畸变），反投影优先取 undistorted_keypoints。
    const bool has_ideal = !frame->undistorted_keypoints.empty() &&
                           frame->undistorted_keypoints.size() == frame->keypoints.size();
    const auto& sample_kpts = frame->keypoints;
    for (size_t i = 0; i < sample_kpts.size(); ++i) {
        const auto& kp_sample = sample_kpts[i];
        const int u = cvRound(kp_sample.pt.x), v = cvRound(kp_sample.pt.y);
        const bool inside = (u >= 0 && u < cols && v >= 0 && v < rows);
        const double depth = (inside && depth_is_float) ? static_cast<double>(imDepth.at<float>(v, u)) : 0.0;
        // 一处判完：像素在图内、深度有限、落在 [min_depth, max_depth]、
        // 且不落在深度不连续处（depth_edge_thresh 的消费者，见 depth_filter.hpp）。
        if (!inside || !isValidDepthSample(imDepth, u, v, depth, config_.min_depth,
                                          config_.max_depth, config_.depth_edge_thresh)) {
            frame->depth_valid.push_back(false);
            frame->pts_3d.emplace_back(0, 0, 0);
            continue;
        }
        // 理想像素（无畸变时就是原始像素）
        const cv::Point2f px = has_ideal ? frame->undistorted_keypoints[i].pt : kp_sample.pt;
        double x = (px.x - cx) * depth / fx;
        double y = (px.y - cy) * depth / fy;
        frame->depth_valid.push_back(true);
        frame->pts_3d.emplace_back(x, y, depth);
    }
}

// ================================================================
// 更新帧的世界坐标 3D 点
// ================================================================
void Tracking::updateWorldPoints(Frame::Ptr frame) {
    if (!frame) return;
    frame->pts_3d_world.clear();
    frame->pts_3d_world.reserve(frame->pts_3d.size());
    for (size_t i = 0; i < frame->pts_3d.size(); ++i) {
        if (!frame->depth_valid[i] || frame->pts_3d[i].norm() < 1e-6) {
            frame->pts_3d_world.emplace_back(0, 0, 0);
        } else {
            frame->pts_3d_world.push_back(camToWorldPoint(frame->pts_3d[i], frame->pose));
        }
    }
}

// ================================================================
// 主跟踪函数
// ================================================================
Pose Tracking::track(const cv::Mat& rgb_img,
                    const cv::Mat& depth_img,
                    const cv::Mat& ir_img,
                    double timestamp) {
    // 一次性打印生效参数，便于确认 yaml 是否真的被消费（受 verbose_diag 控制）
    static int diag_printed = 0;
    if (diag_printed == 0) {
        diag_printed = 1;
        { if (config_.verbose_diag) std::cerr << "[TRACK-START] verbose_diag=" << config_.verbose_diag
                  << " disable_undistort=" << config_.disable_undistort
                  << " max_depth=" << config_.max_depth
                  << " n_features=" << config_.extractor_config.n_features
                  << std::endl; }
    }

    // Step 1: 创建当前帧
    mCurrentFrame_ = Frame::createFrame(rgb_img, depth_img, ir_img, timestamp);
    mCurrentFrame_->id = next_frame_id_++;
    mvpCurrentMapPointIds_.assign(mCurrentFrame_->keypoints.size(), -1);

    // Step 2: 深度图预处理
    // 策略：统一在此处做类型转换和单位统一（mm→m 或 cm→m）
    // - CV_32FC1: 假设已经是米，直接clone
    // - CV_16UC1/其他: 使用config_.depth_scale转换
    // 注意：depth_scale的默认值需要根据数据集设置：
    //   TUM PNG: 0.0002 (mm→m, pixel/5000, 官方标准)
    //   某些数据集: 0.001 (mm→m, pixel/1000)
    cv::Mat imDepth;
    if (depth_img.type() == CV_32FC1) {
        // 已经是float米，直接用
        imDepth = depth_img.clone();
    } else if (depth_img.type() == CV_16UC1) {
        // PNG格式毫米单位
        depth_img.convertTo(imDepth, CV_32FC1, config_.depth_scale);
    } else if (depth_img.type() == CV_16SC1) {
        // 带符号的short格式
        depth_img.convertTo(imDepth, CV_32FC1, config_.depth_scale);
    } else {
        // 其他格式，统一转换
        imDepth = depth_img.clone();
        imDepth.convertTo(imDepth, CV_32FC1, config_.depth_scale);
    }

    // Step 3: ORB 特征提取（在原始图像上提取）
    // ★★★ 重要：必须在原始图像上提取，否则坐标系统不匹配 ★★★
    extractor_->extract(mCurrentFrame_->gray_img, cv::Mat(), mCurrentFrame_->keypoints,
                       mCurrentFrame_->descriptors);

    // Step 4: 关键点去畸变 —— 这是全系统唯一决定"用哪套像素坐标"的地方。
    //   D 为空                          → undistorted_keypoints = keypoints（原样，没畸变可去）
    //   D 非空                          → undistortPoints(P=K) 重新像素化 → 理想像素
    //   D 非空 且 disable_undistort=true → 仍然去畸变（开关只对 D 为空的相机有意义），
    //                                     首次生效时打一条警告
    // 下游（compute3DPoints / poseOptimization / PnP）一律直接消费 undistorted_keypoints，
    // 谁也不再自己二次去畸变；深度采样单独固定用畸变像素。
    mCurrentFrame_->undistorted_keypoints.clear();
    if (!mCurrentFrame_->keypoints.empty() && !config_.K.empty()) {
        mCurrentFrame_->undistorted_keypoints.reserve(mCurrentFrame_->keypoints.size());
        if (config_.D.empty()) {
            // 没有畸变系数可去，原样复制。
            for (const auto& kp : mCurrentFrame_->keypoints)
                mCurrentFrame_->undistorted_keypoints.push_back(kp);
        } else {
            // D 非空时必须去畸变 —— disable_undistort=true 在这里不再有决定权。
            // 原因：配置里把 disable_undistort 打开时，2D 观测和 3D 反投影会同时用
            // 畸变像素，"自洽"地把真实的方向偏差掩盖掉，残差看起来很小、位姿却是错的。
            // freiburg1 的畸变在画面上能到中位 4.8 px、p90 10.5 px、边缘 38 px，
            // 而 poseOptimization 的 chi2=5.99 门限在 octave0 只相当于 3.5 px ——
            // 也就是说一张图的多数特征点都会被判成外点。这个开关只允许在 D 为空时生效，
            // 其余情况一律按"必须去畸变"处理，并且只提示一次，不逐帧刷屏。
            if (config_.disable_undistort) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    std::cerr << "[TRK] 警告: disable_undistort=true 但 D 非空，"
                                 "已忽略该开关并强制去畸变（否则 2D/3D 两侧同时使用畸变像素，"
                                 "重投影残差自洽但位姿错误）。" << std::endl;
                }
            }
            std::vector<cv::Point2f> input_pts;
            input_pts.reserve(mCurrentFrame_->keypoints.size());
            for (const auto& kp : mCurrentFrame_->keypoints)
                input_pts.emplace_back(kp.pt);

            std::vector<cv::Point2f> undist_pts(input_pts.size());
            CameraParams cam = getCameraParams(config_.K);
            double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;
            // R=noArray(): 输入是畸变像素；P=noArray(): 输出归一化坐标 (x/z, y/z)
            cv::undistortPoints(input_pts, undist_pts, config_.K, config_.D,
                                cv::noArray(), cv::noArray());

            for (size_t i = 0; i < mCurrentFrame_->keypoints.size(); ++i) {
                cv::KeyPoint kp = mCurrentFrame_->keypoints[i];
                // P=noArray() 给的是归一化坐标，必须自己乘回内参才是理想像素
                kp.pt.x = static_cast<float>(fx * undist_pts[i].x + cx);
                kp.pt.y = static_cast<float>(fy * undist_pts[i].y + cy);
                mCurrentFrame_->undistorted_keypoints.push_back(kp);
            }
        }
    }

    // Step 5: 从深度图获取 3D 点（使用去畸变后的关键点坐标）
    compute3DPoints(mCurrentFrame_, imDepth);

    // Step 6: 初始化
    mCurrentFrame_->pose = Pose();
    updateWorldPoints(mCurrentFrame_);

    if (!is_initialized_) {
        initialize();
    }
    mTrackedFr_++;

    // 消费 BA 更新
    consumeBAUpdates();

    // ORB-SLAM3: 每帧结束时清空临时地图点
    // E6：必须用 clearTemporalMapPoints() —— 只清 id 列表会让临时点尸体
    // 继续留在 mTemporalMapPoints_ 里，等于换个容器继续泄漏。
    clearTemporalMapPoints();

    mbMapUpdated_ = mbMapUpdatedByBA_;
    mbMapUpdatedByBA_ = false;

    // ★★★ Bug Fix: 调用 updateLastFrame() 以便创建临时地图点 ★★★
    // ORB-SLAM3 在每帧跟踪开始时调用 UpdateLastFrame()：
    //   1. 用运动模型预测更新上一帧的 pose（仅当 mbVelocityValid_ 时）
    //   2. 为上一帧中没有关联地图点的深度特征点创建临时 VO 地图点（mlpTemporalPoints）
    // SFC-SLAM 原问题：
    //   - updateLastFrame() 从未被调用 → 临时地图点从不创建
    //   - 导致 searchByProjection() 没有足够的 3D 点输入 → PnP inliers=0 → 跟踪失败
    // 注意：必须在此处调用（在 mlpTemporalPoints_ 清空之前），此时 mLastFrame_ 仍是上一帧。
    // 同时，在 mlpTemporalPoints_ 清空前用上一帧的校正 pose 重新创建临时地图点，
    // 使其在下一帧（当前帧）的 searchByProjection / searchLocalPoints 中可用。
    if (mLastFrame_) {
        updateLastFrame();
    }

    // 清理时机（本次修复的关键之一）：mlpTemporalPoints_ / mTemporalMapPoints_
    // 的生命周期是「一帧」——上一帧的临时点服务于当前帧的匹配。
    // 正确顺序：帧首 clearTemporalMapPoints() → updateLastFrame() 为 mLastFrame_
    // 重建临时点并写回 mLastFrameMapPointIds_ → 本帧用它做匹配与位姿优化。
    // 原先这里是「updateLastFrame() 之后紧接一次 clearTemporalMapPoints()」，
    // 恰好把刚建好的点全部销毁，id 立刻变悬空：searchByProjection 把悬空 id
    // 原样搬进 mvpCurrentMapPointIds_，poseOptimization 取到 nullptr，
    // 只剩极少数永久点能约束位姿（实测内点从 900 掉到 0~20）。
    //
    // ★★★ 在重置前保存上一帧的内点数 ★★★
    // mnMatchesInliers_ 将在跟踪开始时被重置为 0，
    // 但 trackWithMotionModel() 需要知道上一帧的跟踪质量来调整搜索阈值
    last_inlier_count_ = mnMatchesInliers_;
    mnMatchesInliers_ = 0;

    bool tracking_ok = false;
    std::string track_method = "NONE";
    bool motion_model_updated = false;
    bool skip_local_ba = false;
    int nGood = 0;

    // ★★★ 无条件调试：打印每帧跟踪入口信息 ★★★
    {
        int valid_mp = 0;
        for (int x : mvpCurrentMapPointIds_) if (x >= 0) valid_mp++;
        { if (config_.verbose_diag) std::cerr << "  [TRK-ENTER f=" << mCurrentFrame_->id << "] state=" << static_cast<int>(mState_)
                  << " vel_valid=" << mbVelocityValid_ << " last_nGood=" << mnMatchesInliers_
                  << " last_inliers=" << last_inlier_count_ << " valid_mp=" << valid_mp
                  << " nKFs=" << mvpLocalKeyFrames_.size()
                  << " is_init=" << is_initialized_
                  << std::endl; }
    }

    // 跟踪阶段
    if (mState_ == OK) {
        checkReplacedInLastFrame();

        if (!mbVelocityValid_ || mCurrentFrame_->id < mnLastRelocFrameId_ + 2) {
            if (!mvpLocalKeyFrames_.empty()) {
                const KeyFrameCacheEntry& ref_kf = mvpLocalKeyFrames_.back();
                mCurrentFrame_->pose = ref_kf.world_pose;
            } else {
                mCurrentFrame_->pose = Pose();
            }
            nGood = trackReferenceKeyFrame();
            if (nGood > 0) {
                tracking_ok = true;
                track_method = "REF_KF";
            }
        } else {
            // 正常跟踪：先用运动模型，失败则用参考关键帧
            nGood = trackWithMotionModel();
            if (nGood > 0) {
                tracking_ok = true;
                track_method = "MOTION_MODEL";
                mnRelocConsecutiveFailures_ = 0;
            } else {
                nGood = trackReferenceKeyFrame();
                if (nGood > 0) {
                    tracking_ok = true;
                    track_method = "REF_KF_FALLBACK";
                    mnRelocConsecutiveFailures_ = 0;
                } else {
                    // ★★★ Bug Fix: 旋转失败时添加 searchByProjection 中间回退 ★★★
                    // trackWithMotionModel 失败后直接用 trackReferenceKeyFrame，
                    // 但 BoW 匹配数量少（90-120）且 EPnP 对初始方向敏感，
                    // 导致 poseOptimization 收敛到错误局部极小。
                    // 修复：在 BoW fallback 之前，先尝试 searchByProjection——
                    // 它不依赖 BoW，对旋转场景更鲁棒，能提供更好的初始 pose。
                    if (mLastFrame_) {
                        mvpCurrentMapPointIds_.assign(mCurrentFrame_->keypoints.size(), -1);
                        int nmatches_sbp = searchByProjection(*mCurrentFrame_, *mLastFrame_, 30.0f, false);
                        if (nmatches_sbp >= 15) {
                            const int nGood_sbp = poseOptimization();
                            // poseOptimization() 只在 nInliers >= pose_opt_min_inliers 时返回正数，
                            // 因此这里只有"通过"一个分支；原先的 `else if (nGood_sbp > 0)`
                            // 分支会在内点不足时仍把本次跟踪判定为成功，等于绕过了内点门槛。
                            if (nGood_sbp > 0) {
                                nGood = nGood_sbp;
                                mnMatchesInliers_ = nGood;
                                updateWorldPoints(mCurrentFrame_);
                                updateMotionModel();
                                tracking_ok = true;
                                track_method = "SBP_FALLBACK";
                                mnRelocConsecutiveFailures_ = 0;
                            }
                        }
                    }
                }
            }
        }

        if (!tracking_ok) {
            // ★★★ 无条件调试：打印失败原因 ★★★
            { if (config_.verbose_diag) std::cerr << "  [TRK-FAIL f=" << mCurrentFrame_->id << "] method_used=" << track_method
                      << " nGood=" << nGood << " nKFs=" << mvpLocalKeyFrames_.size()
                      << " -> " << (static_cast<int>(mvpLocalKeyFrames_.size()) > 10 ? "RECENTLY_LOST" : "LOST")
                      << std::endl; }
            if (static_cast<int>(mvpLocalKeyFrames_.size()) > 10) {
                mState_ = RECENTLY_LOST;
                mTimeStampLost_ = timestamp;
            } else {
                mState_ = LOST;
            }
        } else {
            // ★★★ 无条件调试：打印成功跟踪方法 ★★★
            { if (config_.verbose_diag) std::cerr << "  [TRK-OK f=" << mCurrentFrame_->id << "] method=" << track_method
                      << " nGood=" << nGood << " valid_mp_after="
                      << [this]() {
                          int cnt = 0;
                          for (int x : mvpCurrentMapPointIds_) if (x >= 0) cnt++;
                          return cnt;
                      }()
                      << " pose=(" << std::fixed << std::setprecision(3)
                      << mCurrentFrame_->pose.t.x() << "," << mCurrentFrame_->pose.t.y() << "," << mCurrentFrame_->pose.t.z()
                      << ") |t|=" << mCurrentFrame_->pose.t.norm()
                      << std::endl; }
        }
    } else {
        // RECENTLY_LOST / LOST 状态：尝试重定位
        bool reloc_ok = relocalization();
        { if (config_.verbose_diag) std::cerr << "  [TRK-RELOC f=" << mCurrentFrame_->id << "] reloc_ok=" << reloc_ok
                  << " state_before=" << static_cast<int>(mState_)
                  << " time_since_lost=" << ((mState_ != OK) ? (timestamp - mTimeStampLost_) : 0.0)
                  << " mnRelocConsecutiveFailures=" << mnRelocConsecutiveFailures_
                  << std::endl; }
        if (reloc_ok) {
            tracking_ok = true;
            track_method = "RELOC";
            mState_ = OK;
            // 使用标志变量代替goto，跳转到normal_tracking_loop
            tracking_ok = true;

            // ★★★ Bug Fix: RELOC 成功后必须将结果 pose 应用到当前帧 ★★★
            // 原问题：relocalization() 函数内部计算了 best_pose 并返回 true，但没有
            // 将其写入 mCurrentFrame_->pose，导致 normal tracking loop 使用错误的 pose。
            // 修复：在 normal loop 入口处应用 best_pose（如果 reloc 刚成功）。
            // 注：best_pose 已在 relocalization() 结束时设置到 mCurrentFrame_->pose，
            // 此处需要确保它被正确传递到 normal loop。
            // （已经在 relocalization() 末尾做了 mCurrentFrame_->pose = best_pose）

            // ★★★ Bug Fix: 修复 RELOC 后无限重定位循环 ★★★
            // 根本原因：RELOC 成功后没有设置 mLastFrame_，导致：
            //   1. 下一帧进入 track() 时，mLastFrame_ == nullptr
            //   2. trackWithMotionModel() 检查 mbVelocityValid_ 为 false（reset 时设为 false）
            //   3. 进入 RECENTLY_LOST/LOST 分支 → 再次 relocalization()
            //   4. RELOC 成功但 mLastFrame_ 仍为 nullptr → 循环
            // 修复：RELOC 成功后立即将当前帧设为上一帧，解除无限循环。
            // 注意：RELOC 帧的 pts_3d 为空，下一帧 trackWithMotionModel() 会因无 3D 点
            // 而快速失败，然后回退到 trackReferenceKeyFrame()，这正是期望的行为。
            mLastFrame_ = mCurrentFrame_;
            last_inlier_count_ = mnMatchesInliers_;
            // 重置速度：RELOC 后的运动完全未知，必须跳过 motion model
            mbVelocityValid_ = false;
        }
        if (mState_ == RECENTLY_LOST && (timestamp - mTimeStampLost_) > 5.0) mState_ = LOST;
    }

    if (tracking_ok) {
        Pose current_pose = mCurrentFrame_->pose;

        // ★★★ Bug Fix #36: 始终在跟踪成功后调用 poseOptimization() 进行 BA 精化 ★★★
        // 根本原因：原代码在跟踪成功后设置 skip_local_ba=true，跳过了 poseOptimization()。
        // 导致：
        // 1. REF_KF_FALLBACK / REF_KF / MOTION_MODEL 成功后，直接使用 PnP 的粗位姿，不进行 BA 精化
        // 2. PnP 粗位姿本身有较大误差（通常 10-50px 重投影误差），未精化就直接用于下一帧预测
        // 3. RELOC 成功后，位姿未经验证就使用
        //
        // poseOptimization() 的作用（不应被跳过）：
        // 1. 使用鲁棒核 Huber 核容忍较大初始误差（chi2_th=100000）
        // 2. 4轮迭代 BA 精化位姿（通常收敛 1-10px）
        // 3. 验证位姿变化（t_diff < 1m, rot_diff < 1rad），防止发散
        // 4. 标记外点、更新 found_count（地图点质量维护）
        //
        // 修复：在所有跟踪成功路径之后，统一调用 poseOptimization()
        updateLocalMap(true);
        searchLocalPoints();
        nGood = poseOptimization();
        mnMatchesInliers_ = nGood;

        // 帧间跳变保护：poseOptimization 内部只校验「优化结果 vs 本帧初值」，
        // 初值本身错了它看不出来。这里再和上一帧的真实位姿比一次——室内手持
        // 30Hz 下单帧位移 3–7cm、转角 <10°，超出量级说明这一帧的解不可信。
        // RELOC 例外：重定位本来就是「跳回正确位置」，此时和上一帧比必然很大，
        // 由 relocalization() 自己的内点率/位移门槛负责（见 RELOC_MAX_DELTA_T）。
        if (nGood > 0 && track_method != "RELOC" && mLastFrame_) {
            const double frame_t =
                (mCurrentFrame_->pose.t - mLastFrame_->pose.t).norm();
            Eigen::Quaterniond q_c = mCurrentFrame_->pose.q;
            Eigen::Quaterniond q_l = mLastFrame_->pose.q;
            if (q_c.coeffs().dot(q_l.coeffs()) < 0.0) q_c.coeffs() = -q_c.coeffs();
            const double frame_rot = Eigen::AngleAxisd(q_c * q_l.inverse()).angle();
            if (frame_t > config_.frame_jump_max_translation ||
                frame_rot > config_.frame_jump_max_rotation) {
                if (config_.verbose_diag) {
                    std::cerr << "[Track] f=" << mCurrentFrame_->id
                              << " 帧间跳变过大 t=" << frame_t << "m rot=" << frame_rot
                              << "rad (上限 " << config_.frame_jump_max_translation << "m / "
                              << config_.frame_jump_max_rotation << "rad) → 判为跟踪失败"
                              << std::endl;
                }
                nGood = 0;
            }
        }

        if (nGood <= 0) {
            tracking_ok = false;
        }

        if (nGood > 0) {
            updateWorldPoints(mCurrentFrame_);
            mbVO_ = (nGood < 10);
            updateMotionModel();
            motion_model_updated = true;
        }

        // 删除外点
        for (size_t i = 0; i < mvpCurrentMapPointIds_.size(); ++i) {
            int mp_id = mvpCurrentMapPointIds_[i];
            if (mp_id < 0) continue;
            auto* mp = getMapPoint(mp_id);
            if (mp && mp->is_bad) {
                mvpCurrentMapPointIds_[i] = -1;
            }
        }
    }

    if (!tracking_ok) {
        mbVelocityValid_ = false;
        mCurrentFrame_->pose = mLastFrame_ ? mLastFrame_->pose : Pose();
    } else if (!motion_model_updated) {
        // 仅当跟踪成功且尚未更新过运动模型时才更新
        updateMotionModel();
    }

    bool is_keyframe = tracking_ok && needNewKeyFrame();

    // ★★★ Bug Fix: RELOC 成功后必须立即创建关键帧 ★★★
    // 原问题：RELOC 成功后不创建关键帧，导致 mvpLocalKeyFrames_ 始终是旧的关键帧，
    // 下一帧 trackReferenceKeyFrame() 用旧的关键帧做初始化，导致跟踪质量差，
    // 频繁触发新的 RELOC，形成恶性循环，关键帧间隔过大（平均 1.37m），
    // 导致轨迹累积误差（135m vs 实际 ~12m）。
    // 修复：RELOC 成功后立即创建关键帧，保证局部地图始终是最新的。
    if (tracking_ok && track_method == "RELOC") {
        is_keyframe = true;
        if (config_.verbose_diag) {
            std::cerr << "[Track] *** FORCE KF after RELOC: frame=" << mCurrentFrame_->id
                      << " inliers=" << mnMatchesInliers_ << " ***" << std::endl;
        }
    }

    if (is_keyframe) createNewKeyFrame();
    mbLastFrameIsKeyFrame_ = is_keyframe;

    // 回环检测 + PGO。只在刚建关键帧后做：关键帧本身已经稀疏
    // （fr1_desk2 全程十几个到几十个），每帧都查 BoW 是浪费。
    // 命中时 applyOptimizedPosesToFrontend() 会改写 mCurrentFrame_->pose，
    // 所以必须早于下面 mLastFrame_ = mCurrentFrame_ 的赋值——
    // 否则 mLastFrame_ 记下的是修正前的位姿，下一帧恒速预测会跟着错一次。
    if (is_keyframe && loop_detector_) {
        // D1-d：写回前的快照。PGO 一次性改四类状态（mAllKeyFrames_ 位姿、
        // mvpLocalKeyFrames_ 副本、后端 KeyFrame、回环库位姿），外加当前帧与
        // mLastFrame_。一旦写回后当前帧相对写回前跳变超过 frame_jump_max_*，
        // 说明这条回环边把本帧拽到了物理上不可能的位置——整次写回必须撤回：
        // 只退当前帧会让关键帧与当前帧位姿互相矛盾，比不回滚更糟。
        const Pose pose_before_loop = mCurrentFrame_->pose;
        const Pose last_pose_before_loop = mLastFrame_ ? mLastFrame_->pose : Pose();
        std::unordered_map<int, Pose> kf_poses_before_loop;
        kf_poses_before_loop.reserve(mAllKeyFrames_.size());
        for (const auto& kv : mAllKeyFrames_) {
            kf_poses_before_loop.emplace(kv.first, kv.second.world_pose);
        }

        if (detectAndCorrectLoop()) {
            Eigen::Quaterniond q_after = mCurrentFrame_->pose.q;
            Eigen::Quaterniond q_before = pose_before_loop.q;
            if (q_after.coeffs().dot(q_before.coeffs()) < 0.0) q_after.coeffs() = -q_after.coeffs();
            const double loop_t = (mCurrentFrame_->pose.t - pose_before_loop.t).norm();
            const double loop_rot = Eigen::AngleAxisd(q_after * q_before.inverse()).angle();
            if (loop_t > config_.frame_jump_max_translation ||
                loop_rot > config_.frame_jump_max_rotation) {
                // 错误日志不受 verbose_diag 控制：这是一次真实的"回环被判为不可信"，
                // 需要和 mnDetectedLoops_ 的计数一起出现在默认日志里。
                std::cerr << "[Loop] 写回后跳变过大 t=" << loop_t << "m rot=" << loop_rot
                          << "rad (上限 " << config_.frame_jump_max_translation << "m / "
                          << config_.frame_jump_max_rotation << "rad)，回滚本次回环写回"
                          << std::endl;
                for (const auto& kv : kf_poses_before_loop) {
                    auto all_it = mAllKeyFrames_.find(kv.first);
                    if (all_it != mAllKeyFrames_.end()) all_it->second.world_pose = kv.second;
                    for (auto& local_kf : mvpLocalKeyFrames_) {
                        if (local_kf.frame_id == kv.first) local_kf.world_pose = kv.second;
                    }
                    if (mpMap_) {
                        if (mapping::KeyFrame* mkf = mpMap_->getKeyFrame(kv.first)) {
                            mkf->setPose(kv.second);
                        }
                    }
                    if (loop_detector_) loop_detector_->updateKeyFramePose(kv.first, kv.second);
                }
                mCurrentFrame_->pose = pose_before_loop;
                updateWorldPoints(mCurrentFrame_);
                if (mLastFrame_) {
                    mLastFrame_->pose = last_pose_before_loop;
                    updateWorldPoints(mLastFrame_);
                }
                tracking_ok = false;   // 本帧位姿已不可信，按跟踪失败处理
            }
        }
    }

    // ★★★ 诊断日志：每帧跟踪结果汇总 ★★★
    // 输出 track_method、skip_local_ba、nGood（实际内点数）、mnMatchesInliers_（诊断内点数）、
    // is_keyframe，用于区分纯诊断 bug（inliers=0 但 track 实际成功）和真正的轨迹 bug。
    if (config_.verbose_diag) {
        static int last_diag_frame = -1;
        if (last_diag_frame != mCurrentFrame_->id) {
            last_diag_frame = mCurrentFrame_->id;
            { if (config_.verbose_diag) std::cerr << "[Track] f=" << mCurrentFrame_->id
                      << " method=" << (tracking_ok ? track_method : "NONE")
                      << " skip_ba=" << (tracking_ok ? (skip_local_ba ? "Y" : "N") : "-")
                      << " nGood=" << (tracking_ok ? nGood : -1)
                      << " mnMatchesInliers_=" << mnMatchesInliers_
                      << " kf=" << (is_keyframe ? "Y" : "N")
                      << std::endl; }
        }
    }

    // 保存当前帧地图点ID用于下一帧
    mLastFrameMapPointIds_ = mvpCurrentMapPointIds_;

    // E6：帧末统一回收坏点。这里是本帧流程的收口 —— 所有原本持有
    // MapPointCandidate* 的局部变量都已离开作用域，erase 不会留下悬空指针。
    retireBadMapPoints();

    mLastFrame_ = mCurrentFrame_;
    last_inlier_count_ = mnMatchesInliers_;

    // 帧末不再重建临时地图点。这里原有一段"用当前帧 pose 重建 mlpTemporalPoints_"的代码，
    // 但它只把 id 塞进 mlpTemporalPoints_，没有写回 mLastFrameMapPointIds_，
    // 而 searchByProjection 是通过 mLastFrameMapPointIds_ 找上一帧地图点的 ——
    // 于是这批点谁也找不到，纯属每帧白建一批、再被下一次 clearTemporalMapPoints() 销毁。
    // 现在统一由下一帧开头的 updateLastFrame() 建（那条路径会正确写回 id）。

    return mCurrentFrame_->pose;
}

// ================================================================
// 初始化
// ================================================================
Pose Tracking::initialize() {
    const size_t n_features = mCurrentFrame_->keypoints.size();
    int valid_depth_count = 0;
    for (size_t i = 0; i < mCurrentFrame_->pts_3d.size(); ++i) {
        if (mCurrentFrame_->depth_valid[i]) valid_depth_count++;
    }
    
    if (n_features < 100 || valid_depth_count < 50) {
        return Pose();
    }
    
    mCurrentFrame_->pose = Pose();
    mCurrentFrame_->is_keyframe = true;
    is_initialized_ = true;

    mvpLocalKeyFrames_.clear();
    mvpCurrentMapPointIds_.assign(mCurrentFrame_->keypoints.size(), -1);

    KeyFrameCacheEntry init_kf;
    init_kf.global_id = mCurrentFrame_->id;
    init_kf.frame_id = mCurrentFrame_->id;
    init_kf.keypoints = mCurrentFrame_->keypoints;
    init_kf.undistorted_keypoints = mCurrentFrame_->undistorted_keypoints;
    init_kf.descriptors = mCurrentFrame_->descriptors.clone();
    init_kf.pts_3d = mCurrentFrame_->pts_3d;
    init_kf.pts_3d_world = mCurrentFrame_->pts_3d_world;
    init_kf.depth_valid = mCurrentFrame_->depth_valid;
    init_kf.world_pose = mCurrentFrame_->pose;
    mvpLocalKeyFrames_.push_back(init_kf);
    mAllKeyFrames_[init_kf.frame_id] = init_kf;
    mpReferenceKF_ = init_kf.global_id;

    if (bow_adapter_ && bow_adapter_->isLoaded() && !mCurrentFrame_->descriptors.empty()) {
        bow_adapter_->addKeyFrame(mCurrentFrame_->id, mCurrentFrame_->descriptors);
    }

    const Pose& Tc = mCurrentFrame_->pose;

    for (size_t i = 0; i < mCurrentFrame_->pts_3d.size(); ++i) {
        if (!mCurrentFrame_->depth_valid[i]) continue;
        const auto& p3d_cam = mCurrentFrame_->pts_3d[i];
        if (p3d_cam.norm() < 1e-6) continue;
        Eigen::Vector3d pt_world = p3d_cam;
        cv::Mat desc;
        if (i < static_cast<size_t>(mCurrentFrame_->descriptors.rows))
            desc = mCurrentFrame_->descriptors.row(static_cast<int>(i)).clone();
        int mp_id = createMapPoint(pt_world, desc, mCurrentFrame_->id, static_cast<int>(i), 1);
        mvpCurrentMapPointIds_[i] = mp_id;  // <0 表示该特征点无地图点，后续按 -1 跳过
    }
    
    mVelocity_.q = Eigen::Quaterniond::Identity();
    mVelocity_.t = Eigen::Vector3d::Zero();
    mbVelocityValid_ = false;

    mnLastKeyFrameId_ = mCurrentFrame_->id;

    mLastFrame_ = mCurrentFrame_;
    mLastFrameMapPointIds_ = mvpCurrentMapPointIds_;
    mState_ = OK;
    return mCurrentFrame_->pose;
}

// ================================================================
// 更新上一帧（ORB-SLAM3 UpdateLastFrame）
// 1. 用参考关键帧更新上一帧的 pose
// 2. 为上一帧中没有关联地图点的深度特征点创建临时 VO 地图点（mlpTemporalPoints）
//    这些点不参与局部建图，仅用于当前帧的纯跟踪
//
// 建点配额：原实现在循环里用 `th_depth = max_depth * 4.0`（=48m）做提前 break，
// 那是把 ORB-SLAM3 的 ThDepth（基线的倍数）当成"米"来用；而 compute3DPoints()
// 早就把 > max_depth 的点过滤掉了，所以那条 break 永远不触发，等于没有配额。
// 现在改用 config_.max_new_map_points_per_kf 做真正的配额。
//
// R2：不再用速度预测覆盖上一帧的位姿。早先这里把 mLastFrame_->pose 换成
// mVelocity_ * mLastFrame_->pose 再 updateWorldPoints() 重算世界点。上一帧的 pose
// 是 poseOptimization 优化过的结果，是系统里最可信的位姿之一；用恒速模型外推去覆盖它，
// 等于每帧都把优化结果丢掉一次，误差按帧累积。ORB-SLAM3 里被替换的是"参考关键帧位姿 ×
// Tlr"算出的临时值，主体仍是已优化位姿；这里没有跨关键帧的相对量，直接保留即可。
// ================================================================
void Tracking::updateLastFrame() {
    if (!mLastFrame_) return;

    // R2：不再用速度预测覆盖上一帧的位姿（理由见函数头注释）。
    // 下面只保留临时 VO 地图点的创建。

    // Step 2: 为上一帧中没有关联地图点的深度特征点创建临时 VO 地图点
    // 仅当上一帧有有效深度点时才处理
    if (mLastFrame_->pts_3d.empty() || mLastFrame_->depth_valid.empty()) return;

    const Pose& last_frame_pose = mLastFrame_->pose;

    const cv::Mat& K = config_.K;
    CameraParams cam = getCameraParams(K);
    double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    const int max_points = config_.max_new_map_points_per_kf;

    int n_points = 0;
    const size_t n_kpts = std::min(mLastFrame_->pts_3d.size(),
                                    mLastFrame_->depth_valid.size());
    for (size_t i = 0; i < n_kpts; ++i) {
        if (!mLastFrame_->depth_valid[i]) continue;

        // 检查是否已有地图点关联。命中坏点时要允许重建：坏点仍然占着
        // mLastFrameMapPointIds_[i] 的槽位，若这里无条件跳过，这个槽位就永远
        // 不会再生成新点（R5 的第二个后果）。
        if (i < mLastFrameMapPointIds_.size() && mLastFrameMapPointIds_[i] >= 0) {
            const MapPointCandidate* existing = getMapPoint(mLastFrameMapPointIds_[i]);
            if (existing && !existing->is_bad) {
                n_points++;
                continue;
            }
            mLastFrameMapPointIds_[i] = -1;  // 坏点占位，清掉后重建
        }

        const auto& p3d_cam = mLastFrame_->pts_3d[i];
        if (p3d_cam.norm() < 1e-6) continue;

        if (n_points >= max_points) break;

        // 相机坐标系 → 世界坐标系
        Eigen::Vector3d p3d_world = camToWorldPoint(p3d_cam, last_frame_pose);

        // 用上一帧的描述子
        cv::Mat desc;
        if (i < static_cast<size_t>(mLastFrame_->descriptors.rows))
            desc = mLastFrame_->descriptors.row(static_cast<int>(i));

        int mp_id = createMapPoint(p3d_world, desc, mLastFrame_->id, static_cast<int>(i), 1,
                                   /*permanent=*/false);
        if (mp_id < 0) continue;  // 3D 点非法，createMapPoint 已拒绝
        if (i < mLastFrameMapPointIds_.size())
            mLastFrameMapPointIds_[i] = mp_id;

        mlpTemporalPoints_.push_back(mp_id);
        n_points++;
    }
}

// ================================================================
// 运动模型跟踪
// ================================================================
// 运动模型跟踪（返回 int = nGood，0 = 失败）
int Tracking::trackWithMotionModel() {
    if (!mLastFrame_ || !mbVelocityValid_) {
        return false;
    }

    // 恒速模型预测
    const Pose pose_before_tracking = mCurrentFrame_->pose;  // 失败回滚点
    Eigen::Matrix3d R_vel = mVelocity_.q.toRotationMatrix();
    Pose predicted_pose;
    predicted_pose.q = mVelocity_.q * mLastFrame_->pose.q;
    normalizeQuaternion(predicted_pose.q);
    predicted_pose.t = R_vel * mLastFrame_->pose.t + mVelocity_.t;
    mCurrentFrame_->pose = predicted_pose;

    mvpCurrentMapPointIds_.assign(mCurrentFrame_->keypoints.size(), -1);

    const auto& curr_kpts = (!mCurrentFrame_->undistorted_keypoints.empty())
        ? mCurrentFrame_->undistorted_keypoints : mCurrentFrame_->keypoints;

    int th = 20;
    int nmatches = searchByProjection(*mCurrentFrame_, *mLastFrame_,
                                      static_cast<float>(th), false);

    if (nmatches < 15) {
        th = 30;
        nmatches = searchByProjection(*mCurrentFrame_, *mLastFrame_,
                                       static_cast<float>(th), false);
    }

    if (nmatches < 15) {
        { if (config_.verbose_diag) std::cerr << "  [TMM f=" << mCurrentFrame_->id << "] FAILED: nmatches=" << nmatches << " (<15)" << std::endl; }
        // R4：失败必须把位姿退回本帧起点，不能把恒速外推的预测值留在帧上。
        // 调用方会在失败后改走 trackReferenceKeyFrame / RELOC，那些路径用
        // mCurrentFrame_->pose 当初值；留下一个错误预测等于给它们喂了坏初值。
        mCurrentFrame_->pose = pose_before_tracking;
        return 0;
    }

    // R4：poseOptimization 会就地改写 mCurrentFrame_->pose，失败时退回预测值
    // 之外的本帧起点，避免把"优化了一半"的位姿留给下一条跟踪路径。
    int nGood = poseOptimization();
    mnMatchesInliers_ = nGood;

    if (nGood < config_.pose_opt_min_inliers) {
        { if (config_.verbose_diag) std::cerr << "  [TMM f=" << mCurrentFrame_->id << "] FAILED: nmatches=" << nmatches << " but poseOpt nGood=" << nGood << " (<" << config_.pose_opt_min_inliers << ")" << std::endl; }
        mCurrentFrame_->pose = pose_before_tracking;
        return 0;
    }

    updateWorldPoints(mCurrentFrame_);
    updateMotionModel();

    return nGood;
}

// ================================================================
// 参考关键帧跟踪（使用 SearchByBoW 几何匹配，ORB-SLAM3 风格）
// 返回 int = nGood，0 = 失败
// ================================================================
int Tracking::trackReferenceKeyFrame() {
    if (mvpLocalKeyFrames_.empty()) {
        if (config_.verbose_diag) std::cerr << "[TRKF] f=" << mCurrentFrame_->id << " FAILED: no local KFs" << std::endl;
        return 0;
    }

    const KeyFrameCacheEntry* ref_kf_ptr = nullptr;
    if (mbBestRelocKFValid_) {
        ref_kf_ptr = &mBestRelocKF_;
    } else {
        ref_kf_ptr = &mvpLocalKeyFrames_.back();
    }
    const KeyFrameCacheEntry& ref_kf = *ref_kf_ptr;

    if (mLastFrame_) {
        mCurrentFrame_->pose = mLastFrame_->pose;
    } else {
        mCurrentFrame_->pose = ref_kf.world_pose;
    }

    std::vector<int> bow_matches;
    int n_bow_matches = 0;
    if (bow_adapter_ && bow_adapter_->isLoaded()) {
        n_bow_matches = bow_adapter_->searchByBoW(
            mCurrentFrame_->descriptors, ref_kf.frame_id, bow_matches);
    }

    if (n_bow_matches < 5) {
        { if (config_.verbose_diag) std::cerr << "  [TRKF f=" << mCurrentFrame_->id << "] FAILED: n_bow_matches=" << n_bow_matches << " (<5)" << std::endl; }
        return 0;
    }

    mvpCurrentMapPointIds_.assign(mCurrentFrame_->keypoints.size(), -1);

    const auto& curr_kps = (!mCurrentFrame_->undistorted_keypoints.empty())
        ? mCurrentFrame_->undistorted_keypoints : mCurrentFrame_->keypoints;

    // R3：不再在这里做 undistortPoints。
    // `curr_kps` 已经由 trackFrame() 的 Step 4 统一整理成「相机模型要的那一套坐标」：
    //   disable_undistort=false → undistorted_keypoints 是理想像素（畸变已去除）
    //   disable_undistort=true  → undistorted_keypoints 是原始像素（D 为空时才允许）
    // 在这里再做一次 undistortPoints，等于把畸变模型对理想像素又施加一遍，
    // 与 compute3DPoints() 用的理想像素反投影方向相反，残差会被系统性放大。

    std::vector<Eigen::Vector3d> pts3d;
    std::vector<Eigen::Vector2d> pts2d;

    for (int i_f = 0; i_f < (int)bow_matches.size(); ++i_f) {
        int i_ref = bow_matches[i_f];
        if (i_ref < 0) continue;

        // R4：bow_matches 的长度由描述子行数决定，而 curr_kps / mvpCurrentMapPointIds_
        // 由关键点数决定；两者不保证相等。之前直接取 curr_kps[i_f] 会越界读到
        // 相邻内存（UB），越界索引正好是 R4 列出的那一处。这里统一加边界闸。
        if (i_f >= static_cast<int>(curr_kps.size())) continue;
        if (i_f >= static_cast<int>(mvpCurrentMapPointIds_.size())) continue;
        if (i_ref >= static_cast<int>(ref_kf.depth_valid.size())) continue;
        if (!ref_kf.depth_valid[i_ref]) continue;
        if (i_ref >= static_cast<int>(ref_kf.pts_3d.size())) continue;
        if (i_ref >= static_cast<int>(ref_kf.descriptors.rows)) continue;

        const auto& p3d_cam = ref_kf.pts_3d[i_ref];
        if (p3d_cam.norm() < 1e-6) continue;

        Eigen::Vector3d p3d_world = camToWorldPoint(p3d_cam, ref_kf.world_pose);

        pts3d.push_back(p3d_world);

        // 2D 索引修正：`bow_matches[i_f]` 是**参考帧**关键点索引，
        // 当前帧的对应点必须用循环索引 i_f 直接取，否则 2D-3D 对应关系几乎全错。
        // 坐标本身就是理想像素（R3，见上），PnP 收到的 D 为空，不再二次去畸变。
        pts2d.emplace_back(curr_kps[i_f].pt.x, curr_kps[i_f].pt.y);

        if (mvpCurrentMapPointIds_[i_f] < 0) {
            cv::Mat desc;
            if (i_f < mCurrentFrame_->descriptors.rows)
                desc = mCurrentFrame_->descriptors.row(i_f);
            int mp_id = createMapPoint(p3d_world, desc, mCurrentFrame_->id, i_f, 1);
            if (mp_id < 0) continue;  // 3D 点非法，createMapPoint 已拒绝

            mapping::MapPoint* mp = mpMap_->getMapPoint(mp_id);
            if (mp) {
                mp->addObservation(ref_kf.frame_id, i_ref);
                mp->increaseFound();
                if (mp->getDescriptor().empty() && i_ref < ref_kf.descriptors.rows)
                    mp->setDescriptor(ref_kf.descriptors.row(i_ref));
            }
            mvpCurrentMapPointIds_[i_f] = mp_id;
        }
    }

    if (pts3d.size() < 4) {
        { if (config_.verbose_diag) std::cerr << "  [TRKF f=" << mCurrentFrame_->id << "] FAILED: pts3d=" << pts3d.size() << " (<4)" << std::endl; }
        return 0;
    }

    // ★★★ 诊断：检查 PnP 输入数据 ★★★
    if (config_.verbose_diag) {
        double x_sum = 0, y_sum = 0, z_sum = 0;
        double u_sum = 0, v_sum = 0;
        for (size_t i = 0; i < pts3d.size() && i < 10; ++i) {
            x_sum += pts3d[i].x();
            y_sum += pts3d[i].y();
            z_sum += pts3d[i].z();
            u_sum += pts2d[i].x();
            v_sum += pts2d[i].y();
        }
        int n = std::min(pts3d.size(), size_t(10));
        { if (config_.verbose_diag) std::cerr << "[TRKF-PnP-INPUT f=" << mCurrentFrame_->id << "] n=" << pts3d.size()
                  << " avg_3d=(" << x_sum/n << "," << y_sum/n << "," << z_sum/n << ")"
                  << " avg_2d=(" << u_sum/n << "," << v_sum/n << ")"
                  << std::endl; }
    }

    sfc_core::frontend::PnPResult pnp_result;
    Pose initial_pose = mCurrentFrame_->pose;
    mCurrentFrame_->pose = robust_pnp_solver_->optimize(
        pts3d, pts2d, mCurrentFrame_->pose, &pnp_result, true);
    if (!pnp_result.success) {
        { if (config_.verbose_diag) std::cerr << "  [TRKF f=" << mCurrentFrame_->id << "] FAILED: PnP failed" << std::endl; }
        return 0;
    }

    double pnp_t_diff = (mCurrentFrame_->pose.t - initial_pose.t).norm();
    Eigen::Quaterniond pnp_q_diff = mCurrentFrame_->pose.q * initial_pose.q.inverse();
    if (initial_pose.q.coeffs().dot(mCurrentFrame_->pose.q.coeffs()) < 0) {
        pnp_q_diff.coeffs() = -pnp_q_diff.coeffs();
    }
    Eigen::AngleAxisd pnp_angle_axis(pnp_q_diff);
    double pnp_rot_diff = pnp_angle_axis.angle();

    // R4：这两个阈值原先是 5.0m / 1.5rad（约 86°），事实上等于「PnP 结果永远通过」。
    // 参考关键帧跟踪是帧间小位移，正常应在厘米级；放宽到 5m/86° 只会把
    // PnP 解崩（对应点错配、EPnP 落到镜像解）的帧当成成功，然后把垃圾位姿
    // 交给 poseOptimization 去"救"，误差就这样一帧帧累积成漂移。
    // 收紧到 0.3m / 0.3rad（约 17°），失败即回滚并置 mbVelocityValid_=false，
    // 让下一帧改走运动模型/重定位，而不是硬把一个错解沿用下去。
    if (pnp_t_diff > PNP_T_DIFF_THRESH_M || pnp_rot_diff > PNP_ROT_DIFF_THRESH_RAD) {
        { if (config_.verbose_diag) std::cerr << "  [TRKF f=" << mCurrentFrame_->id << "] FAILED: t_diff=" << pnp_t_diff << " rot_diff=" << pnp_rot_diff << " (thresh: t=" << PNP_T_DIFF_THRESH_M << " rot=" << PNP_ROT_DIFF_THRESH_RAD << ")" << std::endl; }
        mCurrentFrame_->pose = initial_pose;
        mbVelocityValid_ = false;
        return 0;
    }

    int nGood = poseOptimization();

    if (nGood >= config_.pose_opt_min_inliers && mLastFrame_) {
        const Pose& curr = mCurrentFrame_->pose;
        const Pose& last_real = mLastFrame_->pose;

        // ★★★ Bug Fix: 正确的帧间速度计算 ★★★
        // 原问题（Fix 14）：mLastFrame_->pose 被第 1019 行覆盖为 ref_kf.world_pose，
        // 导致计算出的 Δ = T(curr) ⊕ T(ref_kf) 是错误的大旋转（如 0.589 rad），
        // 下一帧用此错误速度预测 → nmatches=0 → 持续 REF_KF。
        //
        // 修复策略：
        // 1. 在第 1019 行覆盖之前，记录原始的真正上一帧 pose
        // 2. 用原始 pose 计算正确的帧间运动 Δ = T(curr) ⊕ T(last_real)
        // 3. 合理性检查通过后才更新 mVelocity_
        {
            Eigen::Quaterniond q_curr = curr.q;
            Eigen::Quaterniond q_last = last_real.q;
            if (q_curr.coeffs().dot(q_last.coeffs()) < 0.0)
                q_curr.coeffs() = -q_curr.coeffs();
            mVelocity_.q = q_curr * q_last.inverse();
            normalizeQuaternion(mVelocity_.q);
            Eigen::Matrix3d R_delta = mVelocity_.q.toRotationMatrix();
            mVelocity_.t = curr.t - R_delta * last_real.t;

            // 合理性检查：平移不超过 1m，旋转不超过 1rad
            double trans_norm = mVelocity_.t.norm();
            Eigen::AngleAxisd aa(mVelocity_.q);
            double rot_angle = aa.angle();
            const double max_trans = 1.0;
            const double max_rot = 1.0;
            if (trans_norm <= max_trans && rot_angle <= max_rot) {
                mbVelocityValid_ = true;
            } else {
                mbVelocityValid_ = false;
            }
        }
        // 这里原有一行 `mLastFrame_->pose = mCurrentFrame_->pose;`，已删除。
        // 它的后果是：下一帧 updateLastFrame() 用"当前帧的 pose"给"上一帧的深度图"建
        // 临时地图点，等于把上一帧的点云整体挪了一个帧间位移；updateMotionModel() 用
        // (curr - last) 求速度时 last 已经被改成 curr，速度恒为 0，运动模型彻底失效；
        // 帧间跳变守卫比较的也是同一个位姿，自比自永远不超限。
        // 上一帧位姿只允许由 track() 末尾的 mLastFrame_ = mCurrentFrame_ 推进，别处不得改写。
    } else {
        mbVelocityValid_ = false;
    }

    if (nGood < config_.pose_opt_min_inliers) {
        { if (config_.verbose_diag) std::cerr << "  [TRKF f=" << mCurrentFrame_->id << "] FAILED: poseOpt nGood=" << nGood << " (<" << config_.pose_opt_min_inliers << ")" << std::endl; }
        return 0;
    }

    mnMatchesInliers_ = nGood;

    updateWorldPoints(mCurrentFrame_);

    // 关于运动模型：上面那段 mVelocity_ 用的是真实的帧间差 (curr ⊖ mLastFrame_)，语义正确，
    // 保留。不要在这里调用 updateMotionModel() —— 那个函数假定"上一帧位姿未被改写"，
    // 并且会在位移超过 0.15m/帧时直接放弃更新。也不必在此改写 mLastFrame_->pose：
    // 上一帧位姿由 track() 末尾统一推进。

    return nGood;
}

// ================================================================
// SearchByProjection（ORB_SLAM3 风格）
// ================================================================
int Tracking::searchByProjection(Frame& CurrentFrame, const Frame& LastFrame,
                                  float th, bool bMono) {
    int nmatches = 0;

    // ★★★ Bug Fix: 详细诊断日志 ★★★
    if (config_.verbose_diag) {
        static int last_frame_id = -1;
        if (last_frame_id != CurrentFrame.id) {
            last_frame_id = CurrentFrame.id;
            // 统计有效深度点数量
            int valid_depth = 0;
            for (size_t i = 0; i < LastFrame.pts_3d.size(); ++i) {
                if (LastFrame.depth_valid[i]) valid_depth++;
            }
            { if (config_.verbose_diag) std::cerr << "[SearchByProjection] frame=" << CurrentFrame.id
                      << " last_pts3d=" << LastFrame.pts_3d.size()
                      << " last_valid_depth=" << valid_depth
                      << " last_pose=(" << std::fixed << std::setprecision(3)
                      << LastFrame.pose.t.transpose() << ")"
                      << " curr_pose=(" << CurrentFrame.pose.t.transpose() << ")"
                      << " th=" << th << std::endl; }
        }
    }

    // 判断相机运动方向（前进/后退），用于 octave 过滤
    const Eigen::Vector3d& t_curr = CurrentFrame.pose.t;
    const Eigen::Vector3d& t_last = LastFrame.pose.t;
    double tlc_z = t_curr.z() - t_last.z();
    const bool bForward = (tlc_z > 0.5) && !bMono;
    const bool bBackward = (tlc_z < -0.5) && !bMono;

    const auto& last_kpts = (!LastFrame.undistorted_keypoints.empty())
        ? LastFrame.undistorted_keypoints : LastFrame.keypoints;
    const auto& curr_kpts = (!CurrentFrame.undistorted_keypoints.empty())
        ? CurrentFrame.undistorted_keypoints : CurrentFrame.keypoints;

    const cv::Mat& K = config_.K;
    CameraParams cam = getCameraParams(K);
    double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    std::vector<char> kpt_matched(curr_kpts.size(), 0);

    // ★★★ Bug Fix B11: 添加旋转一致性过滤（ORB-SLAM3 histogram）★★★
    // ORB-SLAM3 SearchByProjection(): 在最佳匹配后统计旋转角分布，
    // 剔除旋转角度异常的匹配（如误匹配在 histogram 的角落里）。
    // 这样可以过滤掉由于外观相似但几何不一致的误匹配。
    static constexpr int HISTO_LENGTH = 30;
    std::vector<int> rotHist[HISTO_LENGTH];
    for (int h = 0; h < HISTO_LENGTH; ++h) rotHist[h].reserve(500);
    const float rotFactor = 1.0f / HISTO_LENGTH;

    const size_t last_n = LastFrame.pts_3d.size();
    for (size_t i = 0; i < last_n; ++i) {
        if (!LastFrame.depth_valid[i]) continue;

        // 从上一帧相机系转换到世界系
        Eigen::Vector3d p3d_world = camToWorldPoint(LastFrame.pts_3d[i], LastFrame.pose);
        if (p3d_world.norm() < 1e-6) continue;

        // 投影到当前帧
        Eigen::Vector3d pt_cam = worldToCam(p3d_world, CurrentFrame.pose);
        // ★★★ Bug Fix: NaN检查防止无效深度 ★★★
        // 原bug: 只检查z>0.05，但如果z=NaN则不触发continue，导致1.0/NaN=NaN
        double z = pt_cam.z();
        if (z < 0.05 || !std::isfinite(z)) continue;

        double invz = 1.0 / z;
        float u_f = static_cast<float>(fx * pt_cam.x() * invz + cx);
        float v_f = static_cast<float>(fy * pt_cam.y() * invz + cy);
        if (u_f < 0 || u_f >= CurrentFrame.gray_img.cols ||
            v_f < 0 || v_f >= CurrentFrame.gray_img.rows) continue;

        int nLastOctave = (i < last_kpts.size()) ? last_kpts[i].octave : 0;
        float radius = th;
        if (config_.extractor_config.n_levels > 1) {
            float scale = std::pow(config_.extractor_config.scale_factor,
                                  static_cast<float>(nLastOctave));
            radius *= scale;
        }

        int octave_min = bBackward ? 0 : std::max(0, nLastOctave - 1);
        int octave_max = bForward ? (config_.extractor_config.n_levels - 1) : nLastOctave + 1;

        std::vector<int> candidates;
        for (size_t k = 0; k < curr_kpts.size(); ++k) {
            if (kpt_matched[k]) continue;
            if (curr_kpts[k].octave < octave_min || curr_kpts[k].octave > octave_max) continue;
            float du = curr_kpts[k].pt.x - u_f;
            float dv = curr_kpts[k].pt.y - v_f;
            if (du * du + dv * dv <= radius * radius)
                candidates.push_back(static_cast<int>(k));
        }
        if (candidates.empty()) continue;

        int best_k = -1, best_dist = 256;

        // 用 LastFrame 的描述子做匹配
        if (i < static_cast<size_t>(LastFrame.descriptors.rows)) {
            const cv::Mat& last_desc = LastFrame.descriptors.row(static_cast<int>(i));
            for (int cand_k : candidates) {
                if (cand_k >= CurrentFrame.descriptors.rows) continue;
                int d = descriptorDistance(last_desc, CurrentFrame.descriptors.row(cand_k));
                if (d < best_dist) { best_dist = d; best_k = cand_k; }
            }
        }

        if (best_k >= 0) {
            // ★★★ Bug Fix: 几何一致性验证 ★★★
            // 问题：描述子匹配只检查外观相似性，不检查几何一致性
            // 修复：匹配后检查重投影误差，剔除几何不一致的误匹配
            const auto& curr_kp = curr_kpts[best_k];
            double reproj_u = curr_kp.pt.x;
            double reproj_v = curr_kp.pt.y;
            double dx = reproj_u - u_f;
            double dy = reproj_v - v_f;
            double reproj_dist = std::sqrt(dx * dx + dy * dy);
            
            // R4：几何容差原先是 max(radius*2.5, 20px)，20px 下限让"几何验证"
            // 在任何近景/宽搜索上都失去意义 —— 搜索窗本身才 15-30px，
            // 上下限一放宽，等于只靠描述子距离决定匹配。
            // 收到 max(radius*1.5, 8px)：8px 已经覆盖金字塔高层 + 小运动模糊，
            // 又确实能挡住投影位置差出一个窗口的错配。
            float geo_thresh = std::max(radius * kGeoVerifyRadiusScale, kGeoVerifyMinPx);
            if (reproj_dist > geo_thresh) {
                // 几何一致性差，跳过此匹配
                best_k = -1;
            }
        }
        
        if (best_k >= 0) {
            // ★★★ Bug Fix: 使用配置中的分层阈值替代硬编码100 ★★★
            // Motion Model跟踪使用更严格的阈值(TH_LOW=50)，参考KF跟踪使用宽松阈值(TH_HIGH=100)
            // 这与ORB_SLAM3 ORBmatcher.cc中的策略一致
            int hamming_th = (th > 15.0f) 
                ? config_.matcher_config.th_high   // 宽松阈值用于初始搜索
                : config_.matcher_config.th_low; // 严格阈值用于精细匹配
            if (best_dist <= hamming_th) {
                kpt_matched[best_k] = 1;
                nmatches++;
            }

            // ★★★ Bug Fix B11: 收集旋转角到 histogram ★★★
            // ORB-SLAM3: float rot = kpLF.angle - kpCF.angle; bin = round(rot*factor);
            float rot = 0.0f;
            if (i < last_kpts.size())
                rot = last_kpts[i].angle - curr_kpts[best_k].angle;
            if (rot < 0.0f) rot += 360.0f;
            int bin = static_cast<int>(round(rot * rotFactor));
            if (bin == HISTO_LENGTH) bin = 0;
            if (bin >= 0 && bin < HISTO_LENGTH)
                rotHist[bin].push_back(best_k);

            // ★★★ Bug Fix B9: 正确复用上一帧已创建的地图点 ★★★
            // ORB-SLAM3 UpdateLastFrame() 创建临时地图点（mlpTemporalPoints），
            // SearchByProjection 将 CurrentFrame 的 keypoint 与 LastFrame 的 keypoint 匹配，
            // 然后隐式地通过 mLastFrame.mvpMapPoints[i] 复用 LastFrame 的地图点。
            //
            // SFC-SLAM 没有 mlpTemporalPoints 机制，需要显式处理：
            // 1. 如果当前帧 keypoint 已有地图点关联（TrackRefKF 等建立）→ 跳过（不覆盖）
            // 2. 如果上一帧 keypoint 有已有地图点（mLastFrameMapPointIds_[i] >= 0）→ 复用
            // 3. 否则 → 创建新地图点
            if (best_k < static_cast<int>(mvpCurrentMapPointIds_.size())) {
                if (mvpCurrentMapPointIds_[best_k] >= 0) {
                    // 已有地图点（如 TrackRefKF 建立），不覆盖
                } else {
                    const int reused_id =
                        (i < mLastFrameMapPointIds_.size()) ? mLastFrameMapPointIds_[i] : -1;
                    // 复用前必须确认 id 还活着：临时地图点只活一帧，悬空 id 一旦被搬进
                    // mvpCurrentMapPointIds_，poseOptimization 取到 nullptr，那帧就只能靠
                    // 极少数永久点求位姿（内点 900→0~20 的直接原因之一）。
                    if (reused_id >= 0 && getMapPoint(reused_id) != nullptr) {
                        // 上一帧 keypoint i 的地图点仍然有效，复用它（对应 ORB-SLAM3 的隐式复用）
                        mvpCurrentMapPointIds_[best_k] = reused_id;
                    } else {
                        // 创建新地图点
                        cv::Mat desc;
                        if (best_k < CurrentFrame.descriptors.rows)
                            desc = CurrentFrame.descriptors.row(best_k);
                        int mp_id = createMapPoint(p3d_world, desc, CurrentFrame.id, best_k, 1);
                        if (mp_id < 0) continue;  // 3D 点非法，createMapPoint 已拒绝
                        mvpCurrentMapPointIds_[best_k] = mp_id;
                    }
                }
            }
        }
    }

    // ★★★ Bug Fix B11: 旋转一致性过滤 ★★★
    // ORB-SLAM3 SearchByProjection(): 匹配后统计旋转角分布，
    // 只保留旋转角在 histogram 主峰区域的匹配，剔除异常旋转的误匹配。
    int nMatches1 = 0, nMatches2 = 0;
    for (int h = 0; h < HISTO_LENGTH; ++h) {
        if (rotHist[h].size() > nMatches1) {
            nMatches2 = nMatches1;
            nMatches1 = static_cast<int>(rotHist[h].size());
        } else if (rotHist[h].size() > nMatches2) {
            nMatches2 = static_cast<int>(rotHist[h].size());
        }
    }
    if (nMatches1 < 0.4f * nmatches) {
        int ind1 = -1, ind2 = -1;
        int max1 = 0, max2 = 0;
        for (int h = 0; h < HISTO_LENGTH; ++h) {
            int sz = static_cast<int>(rotHist[h].size());
            if (sz > max1) { max2 = max1; ind2 = ind1; max1 = sz; ind1 = h; }
            else if (sz > max2) { max2 = sz; ind2 = h; }
        }
        std::set<int> keep_indices;
        for (int h = 0; h < HISTO_LENGTH; ++h) {
            if (rotHist[h].size() > 0 && (h == ind1 || h == ind2 || h == (ind1 + 1) % HISTO_LENGTH || h == (ind1 + HISTO_LENGTH - 1) % HISTO_LENGTH)) {
                for (int idx : rotHist[h]) keep_indices.insert(idx);
            }
        }
        std::fill(kpt_matched.begin(), kpt_matched.end(), 0);
        nmatches = 0;
        for (int idx : keep_indices) {
            if (idx >= 0 && idx < static_cast<int>(kpt_matched.size())) {
                kpt_matched[idx] = 1;
                nmatches++;
            }
        }
        // ★★★ Bug Fix: 旋转过滤后清除被过滤 keypoint 的地图点关联 ★★★
        // 地图点分配（第 1394-1408 行）在过滤前执行，但过滤后这些 keypoint 的匹配被撤销。
        // 若不清理 mvpCurrentMapPointIds_，被过滤的 keypoint 仍关联着错误的地图点，
        // 导致 poseOptimization 等后续函数使用错误数据。
        // 修复：遍历所有 keypoint，将被过滤（不在 keep_indices 中）但有地图点关联的条目清零。
        for (size_t k = 0; k < kpt_matched.size(); ++k) {
            if (kpt_matched[k]) continue;  // 已通过过滤
            if (k >= mvpCurrentMapPointIds_.size()) continue;
            if (mvpCurrentMapPointIds_[k] >= 0) {
                // 此 keypoint 被过滤，撤销其地图点关联
                mvpCurrentMapPointIds_[k] = -1;
            }
        }
    }

    // ★★★ Bug Fix: 详细诊断日志 ★★★
    if (config_.verbose_diag) {
        static int last_logged_frame = -1;
        if (last_logged_frame != CurrentFrame.id) {
            last_logged_frame = CurrentFrame.id;
            // 统计旋转直方图主峰比例
            int nMatches1 = 0;
            for (int h = 0; h < HISTO_LENGTH; ++h) {
                if ((int)rotHist[h].size() > nMatches1) nMatches1 = rotHist[h].size();
            }
            float rot_ratio = (nmatches > 0) ? (float)nMatches1 / nmatches : 0.0f;
            { if (config_.verbose_diag) std::cerr << "[SearchByProjection-END] frame=" << CurrentFrame.id
                      << " nmatches=" << nmatches
                      << " rot_peak_ratio=" << std::fixed << std::setprecision(2) << rot_ratio
                      << " (good if > 0.4)" << std::endl; }
        }
    }

    return nmatches;
}

// ================================================================
// 重定位
// ================================================================
bool Tracking::relocalization() {
    if (mvpLocalKeyFrames_.empty()) {
        return false;
    }

    // ================================================================
    // ★★★ 新增：运动模型先验预测当前帧 pose ★★★
    // 思路：RELOC 时相邻帧运动很小，利用上一帧 pose + 速度预测当前帧位置
    // 然后用预测 pose 做投影匹配，几何一致性高，PnP 内点率会大幅提升
    // ================================================================
    Pose predicted_pose;
    bool has_predicted_pose = false;
    
    if (mLastFrame_ && mbVelocityValid_) {
        Eigen::Matrix3d R_pred = mVelocity_.q.toRotationMatrix() * mLastFrame_->pose.q.toRotationMatrix();
        Eigen::Vector3d t_pred = mLastFrame_->pose.t + mVelocity_.t;
        predicted_pose.q = Eigen::Quaterniond(R_pred);
        normalizeQuaternion(predicted_pose.q);  // ★★★ 防止四元数数值误差累积 ★★★
        predicted_pose.t = t_pred;
        has_predicted_pose = true;
    } else if (mLastFrame_) {
        predicted_pose = mLastFrame_->pose;
        has_predicted_pose = true;
    }
    
    // ★★★ Bug Fix: Reloc 多次失败后渐进放宽搜索阈值 ★★★
    // 问题：Reloc 连续失败时，所有候选关键帧都以相同标准评判，
    // 导致"坏候选优先匹配"后，好的候选反而被跳过。
    // 修复：连续失败越多，搜索窗口越大，PnP 阈值越宽松。
    // 注意：放宽后的阈值仍然足够严格（至少 5 个内点 + 5% 内点率），
    // 防止极低质量的 RELOC 通过导致灾难性路径漂移。
    int n_fail = mnRelocConsecutiveFailures_;
    // 搜索半径随失败次数扩大（base=10，每失败 5 次 +5px）
    int reloc_search_radius = std::min(10 + (n_fail / 5) * 5, 30);
    // 最低内点数：15 起步，连续失败时放宽，但硬下限 20。
    // 为什么下限是 20 而不是 10：日志里 f=610 之后接受过 inliers=8~14 的 reloc，
    // 这些解的 6-DoF 位姿由不到 15 个点支撑，残差自由度不足以约束位姿，
    // 一旦写回 pose 就会被强制建 KF 固化进地图，形成 LOST→RELOC 死循环。
    constexpr int RELOC_MIN_INLIERS_FLOOR = 20;
    int reloc_min_inliers = std::max(30 - (n_fail / 5), RELOC_MIN_INLIERS_FLOOR);
    // 最低内点比例：0.25 起步，失败时降到硬下限 0.15。
    // 为什么下限是 0.15 而不是 0.05：5% 内点率等价于 95% 的输入是误匹配，
    // 在这种信噪比下 RANSAC 只能拟合到噪声，位姿没有意义。
    constexpr double RELOC_MIN_RATIO_FLOOR = 0.15;
    double reloc_min_ratio = std::max(0.25 - (n_fail / 10) * 0.02, RELOC_MIN_RATIO_FLOOR);
    if (config_.verbose_diag) {
        std::cerr << "[Reloc] f=" << mCurrentFrame_->id
                  << " progressive: search_r=" << reloc_search_radius
                  << " min_inliers=" << reloc_min_inliers
                  << " min_ratio=" << reloc_min_ratio
                  << " (consecutive_fail=" << n_fail << ")" << std::endl;
    }

    // BoW 全局搜索候选关键帧（扩大候选数：20→50，随着失败次数增加）
    int n_bow_candidates = std::min(20 + (n_fail / 5) * 10, 80);
    std::vector<KeyFrameCacheEntry> bow_candidates = getBoWCandidateKeyFrames(n_bow_candidates);

    const auto& curr_kpts = (!mCurrentFrame_->undistorted_keypoints.empty())
        ? mCurrentFrame_->undistorted_keypoints : mCurrentFrame_->keypoints;

    bool reloc_ok = false;
    Pose best_pose;
    int best_inliers = 0, best_matches = 0;
    int best_kf_id = -1;  // ★★★ 用 frame_id 代替指针（避免 dangling pointer）★★★

    // ★★★ ORB-SLAM3 策略：遍历局部关键帧 + BoW 全局候选关键帧 ★★★
    // 合并两个候选池，统一处理
    std::vector<const KeyFrameCacheEntry*> all_candidate_ptrs;
    all_candidate_ptrs.reserve(mvpLocalKeyFrames_.size() + bow_candidates.size());
    for (const auto& kf : mvpLocalKeyFrames_)
        all_candidate_ptrs.push_back(&kf);
    for (const auto& kf : bow_candidates)
        all_candidate_ptrs.push_back(&kf);

    for (const KeyFrameCacheEntry* p_ref_kf : all_candidate_ptrs) {
        const auto& ref_kf = *p_ref_kf;
        std::vector<cv::DMatch> raw, inliers;
        
        // ================================================================
        // ★★★ 两种匹配策略：投影匹配（优先）vs BoW 匹配（备用）★★★
        // ================================================================
        std::vector<cv::DMatch> proj_matches;  // 投影匹配结果
        
        if (has_predicted_pose) {
            // ================================================================
            // 策略 1：投影匹配（利用运动模型先验）
            // 用预测 pose 将关键帧的 3D 点投影到当前帧，在投影位置附近搜索匹配
            // 几何一致性高，PnP 内点率大幅提升
            // ================================================================
            const cv::Mat& K = config_.K;
            CameraParams cam = getCameraParams(K);
            double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

            const auto& ref_kps = (!ref_kf.undistorted_keypoints.empty())
                ? ref_kf.undistorted_keypoints : ref_kf.keypoints;
            std::set<int> matched_curr_indices;

            for (int i = 0; i < static_cast<int>(ref_kps.size()); ++i) {
                if (i >= static_cast<int>(ref_kf.depth_valid.size()) ||
                    !ref_kf.depth_valid[i]) continue;
                if (i >= static_cast<int>(ref_kf.pts_3d.size())) continue;
                const auto& p3d_cam = ref_kf.pts_3d[i];
                if (p3d_cam.norm() < 1e-6) continue;

                // 参考关键帧：相机点 → 世界点
                Eigen::Vector3d p3d_world = camToWorldPoint(p3d_cam, ref_kf.world_pose);
                if (p3d_world.z() < 0.05) continue;

                // 当前帧：用预测位姿投影到相机
                Eigen::Vector3d pt_cam_curr = worldToCam(p3d_world, predicted_pose);
                double z_curr = pt_cam_curr.z();
                if (z_curr < 0.05 || !std::isfinite(z_curr)) continue;

                double invz = 1.0 / z_curr;
                float u_f = static_cast<float>(fx * pt_cam_curr.x() * invz + cx);
                float v_f = static_cast<float>(fy * pt_cam_curr.y() * invz + cy);
                if (u_f < 0 || u_f >= mCurrentFrame_->gray_img.cols ||
                    v_f < 0 || v_f >= mCurrentFrame_->gray_img.rows) continue;

                float radius = static_cast<float>(15.0 * pt_cam_curr.z() / 2.0);
                radius = std::min(std::max(radius, 10.0f), 30.0f);

                float best_dist = 256;
                int best_idx = -1;
                for (size_t k = 0; k < curr_kpts.size(); ++k) {
                    if (matched_curr_indices.count(static_cast<int>(k))) continue;
                    float du = curr_kpts[k].pt.x - u_f;
                    float dv = curr_kpts[k].pt.y - v_f;
                    float dist = std::sqrt(du*du + dv*dv);
                    if (dist < radius && dist < best_dist) {
                        best_dist = dist;
                        best_idx = static_cast<int>(k);
                    }
                }

                if (best_idx >= 0) {
                    cv::DMatch m;
                    m.queryIdx = i;
                    m.trainIdx = best_idx;
                    m.distance = best_dist;
                    proj_matches.push_back(m);
                    matched_curr_indices.insert(best_idx);
                }
            }
        }
        
        // 使用投影匹配或 BoW 匹配
        if (has_predicted_pose && proj_matches.size() >= 10) {
            raw = proj_matches;
        } else {
            matcher_->match(ref_kf.descriptors, mCurrentFrame_->descriptors, raw);
        }
        
        const auto& ref_kps = (!ref_kf.undistorted_keypoints.empty())
            ? ref_kf.undistorted_keypoints : ref_kf.keypoints;
        if (raw.size() < 15) continue;

        // ★★★ 几何过滤：用 5 点法 RANSAC 估计本质矩阵，筛选极线一致性匹配 ★★★
        // ★★★ 放宽到 10 个匹配（更激进地尝试重定位）★★★
        std::vector<cv::Point2f> pts1_norm, pts2_norm;
        std::vector<int> raw_idx_for_norm;
        CameraParams cam = getCameraParams(config_.K);
        double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;
        for (int raw_i = 0; raw_i < static_cast<int>(raw.size()); ++raw_i) {
            const auto& m = raw[raw_i];
            if (m.queryIdx >= static_cast<int>(ref_kps.size())) continue;
            if (m.trainIdx >= static_cast<int>(curr_kpts.size())) continue;
            double u1 = ref_kps[m.queryIdx].pt.x, v1 = ref_kps[m.queryIdx].pt.y;
            double u2 = curr_kpts[m.trainIdx].pt.x, v2 = curr_kpts[m.trainIdx].pt.y;
            // 归一化坐标（除以焦距）
            pts1_norm.emplace_back((u1 - cx) / fx, (v1 - cy) / fy);
            pts2_norm.emplace_back((u2 - cx) / fx, (v2 - cy) / fy);
            raw_idx_for_norm.push_back(raw_i);
        }
        if (pts1_norm.size() < 10) continue;  // ★★★ 降低到 10 个匹配（更激进）★★★

        // 5 点法 + RANSAC 估计本质矩阵
        // 注意：pts1_norm/pts2_norm 已经是归一化坐标（除以了 fx, fy），
        // 不应该再传入 K_mat。应该使用 findEssentialMat 的无 K 重载版本。
        // ★★★ Bug Fix: 添加输入验证防止 OpenCV assertion 崩溃 ★★★
        // OpenCV findEssentialMat 在内部会调用 cv::undistortPointsInternal，
        // 如果输入数据无效（NaN, Inf, 或超出合理范围）会触发 assertion 崩溃。
        // 验证：确保所有点都是有限值且在合理范围内（归一化坐标通常在 [-2, 2] 范围内）
        bool valid_input = true;
        for (size_t i = 0; i < pts1_norm.size() && valid_input; ++i) {
            if (!std::isfinite(pts1_norm[i].x) || !std::isfinite(pts1_norm[i].y) ||
                !std::isfinite(pts2_norm[i].x) || !std::isfinite(pts2_norm[i].y)) {
                valid_input = false;
            }
            if (std::abs(pts1_norm[i].x) > 10 || std::abs(pts1_norm[i].y) > 10 ||
                std::abs(pts2_norm[i].x) > 10 || std::abs(pts2_norm[i].y) > 10) {
                valid_input = false;
            }
        }
        if (!valid_input) {
            inliers = raw;
        } else {
            std::vector<uchar> E_inlier_mask;
            try {
                cv::findEssentialMat(pts1_norm, pts2_norm,
                                     cv::FM_RANSAC, 0.99, 0.8, E_inlier_mask);

                inliers.clear();
                for (size_t n = 0; n < E_inlier_mask.size(); ++n) {
                    if (E_inlier_mask[n]) {
                        inliers.push_back(raw[raw_idx_for_norm[n]]);
                    }
                }
            } catch (const cv::Exception& e) {
                inliers = raw;
            }
        }

        if (inliers.size() < 10) continue;  // ★★★ 降低到 10 个匹配（更激进重定位）★★★

        std::vector<Eigen::Vector3d> pts3d;
        std::vector<Eigen::Vector2d> pts2d;
        for (const auto& m : inliers) {
            int idx_ref = m.queryIdx, idx_curr = m.trainIdx;
            if (idx_ref >= static_cast<int>(ref_kf.depth_valid.size()) ||
                !ref_kf.depth_valid[idx_ref]) continue;
            if (idx_ref >= static_cast<int>(ref_kf.pts_3d.size())) continue;
            const auto& p3d_cam = ref_kf.pts_3d[idx_ref];
            if (p3d_cam.norm() < 1e-6) continue;

            Eigen::Vector3d p3d_world = camToWorldPoint(p3d_cam, ref_kf.world_pose);
            pts3d.push_back(p3d_world);
            pts2d.emplace_back(curr_kpts[idx_curr].pt.x, curr_kpts[idx_curr].pt.y);
        }
        if (pts3d.size() < 4) continue;

        // ★★★ 关键修复：PnP 初始位姿应该接近真实相机位姿 ★★★
        // 原问题：
        //   - predicted_pose = mLastFrame_->pose + velocity，而第一帧 pose=(0,0,0)
        //   - Reloc 失败时 mLastFrame_ 始终是第一帧（ID=0）
        //   - 导致 predicted_pose ≈ (0,0,0)，PnP 初始值完全错误
        //   - PnP 无法从错误初值收敛到正确解
        // 修复策略：
        //   1. 如果参考关键帧是第一帧（pose≈0），尝试用相对运动估计
        //   2. 否则用参考关键帧的位姿作为初始估计
        Pose pnp_init_pose;
        if (has_predicted_pose) {
            // 检查预测位姿是否合理（不能太接近原点）
            double pred_norm = predicted_pose.t.norm();
            double ref_norm = ref_kf.world_pose.t.norm();
            
            if (pred_norm < 0.01 && ref_norm < 0.01) {
                if (mLastFrame_ && mLastFrame_->id > 0) {
                    pnp_init_pose = predicted_pose;
                } else {
                    pnp_init_pose = ref_kf.world_pose;
                }
            } else {
                pnp_init_pose = predicted_pose;
            }
        } else {
            pnp_init_pose = ref_kf.world_pose;
        }
        
        PnPResult result;
        Pose pose = robust_pnp_solver_->optimize(pts3d, pts2d, pnp_init_pose, &result, true);

        const double inlier_ratio = (pts3d.size() > 0)
            ? static_cast<double>(result.inliers) / pts3d.size() : 0.0;
        const bool catastrophic = (pose.t.norm() > 100.0);
        // 相机高度粗筛：TUM 手持序列高度约 1m，PnP 在退化情形下会给出
        // 钻到地面以下的解，这里只拦明显不可能的负值。
        const bool valid_z = pose.t.z() > MIN_RELOC_CAMERA_Z;

        // ★★★ Bug Fix: 详细诊断日志 ★★★
        if (config_.verbose_diag) {
            std::cerr << "[Reloc-PnP] ref_kf=" << ref_kf.frame_id
                      << " pts3d=" << pts3d.size() << " init_t=(" << std::fixed << std::setprecision(3)
                      << pnp_init_pose.t.transpose() << ")"
                      << " result: inliers=" << result.inliers
                      << " ratio=" << std::fixed << std::setprecision(2) << inlier_ratio
                      << " final_t=(" << pose.t.transpose() << ")"
                      << " |t|=" << std::setprecision(3) << pose.t.norm()
                      << " z=" << pose.t.z()
                      << " valid_z=" << valid_z
                      << " quality=" << static_cast<int>(result.quality) << std::endl;
        }

        bool this_is_better = (result.inliers > best_inliers);

        // 帧间位移合理性只是必要条件，不能替代内点率检验。
        // 原实现 `(inlier_ratio >= min_ratio || reasonable_delta)` 里
        // reasonable_delta 默认 true，于是只要位移 <3m 就绕过内点率，
        // 日志中 ratio=0.07 的 reloc 正是从这条路径进了地图。
        bool delta_reasonable = true;
        if (mLastFrame_) {
            const double delta_t = (pose.t - mLastFrame_->pose.t).norm();
            delta_reasonable = (delta_t < RELOC_MAX_DELTA_T);
        }

        if (result.success && result.inliers >= reloc_min_inliers && !catastrophic && valid_z &&
            inlier_ratio >= reloc_min_ratio && delta_reasonable && this_is_better) {
            best_inliers = result.inliers;
            best_matches = static_cast<int>(pts3d.size());
            best_pose = pose;
            reloc_ok = true;
            best_kf_id = ref_kf.frame_id;
        }

        // ★★★ ORB-SLAM3 渐进式搜索：当第一次 PnP 结果不够好时 ★★★
        // ORB-SLAM3 Relocalization():
        //   if(nGood<50) { 宽窗口搜索 + PoseOptimization }
        //   if(nGood>30 && nGood<50) { 窄窗口搜索 + PoseOptimization }
        // 作用：当 BoW 匹配不够好时，通过投影搜索找到更多匹配来验证位姿
        // 渐进式复核的触发门槛：首轮 PnP 至少给出若干个几何一致的内点，
        // 才值得扩大搜索窗口补匹配。注意这只是"是否再试一次"的触发条件，
        // 真正的接受门槛在下方的 result_wide 判定里，与主路径同源（reloc_min_*）。
        constexpr int PROGRESSIVE_RETRY_MIN_INLIERS = 10;
        if (!reloc_ok && result.success && result.inliers >= PROGRESSIVE_RETRY_MIN_INLIERS &&
            result.inliers < 50 && valid_z) {
            // Step 1: 用更宽的搜索窗口（th=20）重新搜索匹配
            Pose pose_to_test = pose;
            std::set<int> sFound;
            std::vector<std::pair<int, Eigen::Vector3d>> additional_pts;

            // 收集当前已确认的地图点
            for (const auto& m : inliers) {
                int idx_ref = m.queryIdx, idx_curr = m.trainIdx;
                if (idx_ref >= static_cast<int>(ref_kf.depth_valid.size()) ||
                    !ref_kf.depth_valid[idx_ref]) continue;
                if (idx_ref >= static_cast<int>(ref_kf.pts_3d.size())) continue;
                const auto& p3d_cam = ref_kf.pts_3d[idx_ref];
                if (p3d_cam.norm() < 1e-6) continue;
                Eigen::Vector3d p3d_world = camToWorldPoint(p3d_cam, ref_kf.world_pose);
                additional_pts.emplace_back(idx_curr, p3d_world);
                sFound.insert(idx_curr);
            }

            // 宽窗口搜索：把 additional_pts 里的地图点按 pose_to_test 投影，
            // 在投影邻域内找描述子匹配的特征点，命中则并入 sFound。
            const cv::Mat& K = config_.K;
            CameraParams cam = getCameraParams(K);
            const double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;
            constexpr float WIDE_SEARCH_RADIUS_PX = 15.0f;   // 宽窗口半径，比常规 th=10 大一档
            constexpr int WIDE_SEARCH_MAX_HAMMING = 100;     // 256bit BRIEF 的距离上限，超过视为不同点
            const auto& curr_kpts = (!mCurrentFrame_->undistorted_keypoints.empty())
                ? mCurrentFrame_->undistorted_keypoints : mCurrentFrame_->keypoints;
            for (const auto& [idx_curr, p3d_world] : additional_pts) {
                const Eigen::Vector3d pt_cam = worldToCam(p3d_world, pose_to_test);
                const double z = pt_cam.z();
                if (z < MIN_PROJECTION_Z || !std::isfinite(z)) continue;

                const double invz = 1.0 / z;
                const float u_f = static_cast<float>(fx * pt_cam.x() * invz + cx);
                const float v_f = static_cast<float>(fy * pt_cam.y() * invz + cy);
                if (u_f < 0 || u_f >= mCurrentFrame_->gray_img.cols ||
                    v_f < 0 || v_f >= mCurrentFrame_->gray_img.rows) continue;

                for (size_t k = 0; k < curr_kpts.size(); ++k) {
                    if (sFound.count(static_cast<int>(k))) continue;
                    const float du = curr_kpts[k].pt.x - u_f;
                    const float dv = curr_kpts[k].pt.y - v_f;
                    if (du * du + dv * dv > WIDE_SEARCH_RADIUS_PX * WIDE_SEARCH_RADIUS_PX) continue;
                    if (idx_curr < mCurrentFrame_->descriptors.rows &&
                        k < static_cast<size_t>(mCurrentFrame_->descriptors.rows)) {
                        const int dist = descriptorDistance(mCurrentFrame_->descriptors.row(idx_curr),
                                                            mCurrentFrame_->descriptors.row(static_cast<int>(k)));
                        if (dist <= WIDE_SEARCH_MAX_HAMMING) {
                            sFound.insert(static_cast<int>(k));
                        }
                    }
                }
            }

            // 重新构建 3D-2D 对应并 PoseOptimization
            std::vector<Eigen::Vector3d> pts3d_wide;
            std::vector<Eigen::Vector2d> pts2d_wide;
            std::vector<std::pair<int, int>> idx_map;  // (ref_idx, curr_idx)

            // ★★★ Bug Fix: 将当前帧相机坐标的 3D 点正确变换到世界坐标 ★★★
            // 原问题：mCurrentFrame_->pts_3d 是当前帧相机坐标系的点，
            // 但下面的代码用 R_cw/t_cw（来自 pose_to_test = 参考关键帧的位姿）
            // 将当前帧相机坐标的点错误地转换成了世界坐标。
            // 例如：p3d_cam=(0.5, 0.2, 2.0)（相机前方 2m），R_cw/ref_kf.world_pose=(0,0,0)，
            // 结果：pt_world ≈ (0.5, 0.2, 2.0) — 但参考关键帧在原点，这意味着
            // 这个 3D 点被放在了相机正前方 2m 的位置，而实际上当前帧已经移动了。
            // 正确做法：用 pose_to_test（第一次 PnP 的结果）的逆变换，
            // 将当前帧相机坐标点转换到世界坐标。
            // pose_to_test = T_wc（世界→相机），其 rotation 部分就是 R_wc
            // 直接将当前帧相机点转换到世界坐标：P_world = R_wc * P_cam + t_wc
            // 其中 t_wc = pose_to_test.t 就是相机在世界坐标系中的位置
            Eigen::Matrix3d R_wc = pose_to_test.q.toRotationMatrix();
            const Eigen::Vector3d& t_wc = pose_to_test.t;  // 相机中心的世界坐标
            // curr_kpts 已在上面宽窗口搜索处取过（同一个作用域），这里直接复用，
            // 重复声明会编译失败（conflicting declaration）。
            for (size_t i = 0; i < mCurrentFrame_->pts_3d.size(); ++i) {
                if (!mCurrentFrame_->depth_valid[i]) continue;
                if (!sFound.count(static_cast<int>(i))) continue;
                const auto& p3d_cam = mCurrentFrame_->pts_3d[i];
                if (p3d_cam.norm() < 1e-6) continue;
                Eigen::Vector3d pt_world = R_wc * p3d_cam + t_wc;
                pts3d_wide.push_back(pt_world);
                pts2d_wide.emplace_back(curr_kpts[i].pt.x, curr_kpts[i].pt.y);
            }

            if (!pts3d_wide.empty()) {
                PnPResult result_wide;
                Pose pose_wide = robust_pnp_solver_->optimize(pts3d_wide, pts2d_wide, pose_to_test, &result_wide, true);

                // 宽窗口复核是"补匹配后再解一次"的路径，门槛必须与主路径同源，
                // 否则只是把主路径拒绝掉的低质量解从旁路放进来。
                const double wide_ratio = static_cast<double>(result_wide.inliers) / pts3d_wide.size();
                bool wide_delta_reasonable = true;
                if (mLastFrame_) {
                    wide_delta_reasonable =
                        ((pose_wide.t - mLastFrame_->pose.t).norm() < RELOC_MAX_DELTA_T);
                }
                if (result_wide.success && result_wide.inliers > best_inliers &&
                    result_wide.inliers >= reloc_min_inliers &&
                    wide_ratio >= reloc_min_ratio && wide_delta_reasonable &&
                    pose_wide.t.norm() < 100.0 && pose_wide.t.z() > MIN_RELOC_CAMERA_Z) {
                    best_inliers = result_wide.inliers;
                    best_matches = static_cast<int>(pts3d_wide.size());
                    best_pose = pose_wide;
                    reloc_ok = true;
                    best_kf_id = ref_kf.frame_id;
                }
            }
        }
    }

    if (!reloc_ok) {
        mnRelocConsecutiveFailures_++;
        if (config_.verbose_diag) {
            std::cerr << "[Reloc] f=" << mCurrentFrame_->id
                      << " FAILED: best_inliers=" << best_inliers
                      << " (need >= reloc_min_inliers=" << reloc_min_inliers << ")"
                      << " total_consecutive_fail=" << mnRelocConsecutiveFailures_ << std::endl;
        }
        return false;
    }

    // Reloc 成功：重置失败计数器
    mnRelocConsecutiveFailures_ = 0;

    // ★★★ Bug Fix: 查找最佳关键帧而不是使用悬空的best_kf指针 ★★★
    // 从mAllKeyFrames_中查找best_kf_id对应的关键帧
    const KeyFrameCacheEntry* p_best_kf = nullptr;
    auto it_best_kf = mAllKeyFrames_.find(best_kf_id);
    if (it_best_kf != mAllKeyFrames_.end()) {
        p_best_kf = &it_best_kf->second;
    } else {
        // 如果找不到，从候选列表中查找
        for (const auto* p_cand : all_candidate_ptrs) {
            if (p_cand->frame_id == best_kf_id) {
                p_best_kf = p_cand;
                break;
            }
        }
    }
    
    if (!p_best_kf) {
        { if (config_.verbose_diag) std::cerr << "[Reloc] ERROR: best_kf not found for id=" << best_kf_id << std::endl; }
        return false;
    }
    const auto& best_kf = *p_best_kf;

    // 重新匹配并创建地图点
    std::vector<cv::DMatch> raw2, inliers2;
    matcher_->match(best_kf.descriptors, mCurrentFrame_->descriptors, raw2);
    const auto& best_kps = (!best_kf.undistorted_keypoints.empty())
        ? best_kf.undistorted_keypoints : best_kf.keypoints;
    matcher_->removeOutliers(raw2, best_kps, curr_kpts, inliers2);

    mvpCurrentMapPointIds_.assign(mCurrentFrame_->keypoints.size(), -1);
    for (const auto& m : inliers2) {
        int idx_ref = m.queryIdx, idx_curr = m.trainIdx;
        if (idx_ref >= static_cast<int>(best_kf.depth_valid.size()) ||
            !best_kf.depth_valid[idx_ref]) continue;
        if (idx_ref >= static_cast<int>(best_kf.pts_3d.size())) continue;
        const auto& p3d_cam = best_kf.pts_3d[idx_ref];
        if (p3d_cam.norm() < 1e-6) continue;

        Eigen::Vector3d p3d_world = camToWorldPoint(p3d_cam, best_kf.world_pose);

        cv::Mat desc;
        if (idx_curr < mCurrentFrame_->descriptors.rows)
            desc = mCurrentFrame_->descriptors.row(idx_curr);
        int mp_id = createMapPoint(p3d_world, desc, mCurrentFrame_->id, idx_curr, 1);
        if (mp_id < 0) continue;  // 3D 点非法，createMapPoint 已拒绝

        // 补充参考关键帧的观测
        mapping::MapPoint* mp = mpMap_->getMapPoint(mp_id);
        if (mp) {
            mp->addObservation(best_kf.frame_id, idx_ref);
            mp->increaseFound();
            if (mp->getDescriptor().empty() && idx_ref < best_kf.descriptors.rows)
                mp->setDescriptor(best_kf.descriptors.row(idx_ref));
        }

        if (idx_curr < static_cast<int>(mvpCurrentMapPointIds_.size()))
            mvpCurrentMapPointIds_[idx_curr] = mp_id;
    }

    // ★★★ Bug Fix: RELOC 后必须将 best_pose 应用到当前帧 ★★★
    // 原问题：relocalization() 计算了 best_pose 但忘记赋值给 mCurrentFrame_->pose。
    // 导致：(1) updateWorldPoints() 用错误的 pose 计算世界坐标点；(2) 保存的
    // keyframe entry.world_pose 是错误的；(3) 下一帧的 trackReferenceKeyFrame()
    // 用错误的 pose 初始化；(4) 累积误差导致轨迹爆炸式漂移。
    // 这就是为什么路径总长 135m 而实际场景只有 ~12m 的根本原因！
    mCurrentFrame_->pose = best_pose;

    updateWorldPoints(mCurrentFrame_);

    mnMatchesInliers_ = best_inliers;
    last_inlier_count_ = mnMatchesInliers_;

    {
        KeyFrameCacheEntry entry;
        entry.global_id = mCurrentFrame_->id;
        entry.frame_id = mCurrentFrame_->id;
        entry.keypoints = mCurrentFrame_->keypoints;
        entry.undistorted_keypoints = mCurrentFrame_->undistorted_keypoints;
        entry.descriptors = mCurrentFrame_->descriptors.clone();
        entry.pts_3d = mCurrentFrame_->pts_3d;
        entry.pts_3d_world = mCurrentFrame_->pts_3d_world;
        entry.depth_valid = mCurrentFrame_->depth_valid;
        entry.world_pose = mCurrentFrame_->pose;

        for (int mp_id : mvpCurrentMapPointIds_) {
            if (mp_id >= 0) {
                entry.map_point_ids.insert(mp_id);
            }
        }

        mAllKeyFrames_[entry.frame_id] = entry;
        // 与 initialize() 和 updateLocalMap() 保持一致：检查 isLoaded()
        if (bow_adapter_ && bow_adapter_->isLoaded() && !mCurrentFrame_->descriptors.empty()) {
            bow_adapter_->addKeyFrame(mCurrentFrame_->id, mCurrentFrame_->descriptors);
        }
    }
    mnLastRelocFrameId_ = mCurrentFrame_->id;

    if (config_.verbose_diag)
        std::cerr << "[Reloc] f=" << mCurrentFrame_->id << " OK" << std::endl;
    return true;
}

// ================================================================
// 更新局部地图
// add_current_frame: RELOC 成功后为 false，避免将坏帧加入 mvpLocalKeyFrames_
// ================================================================
void Tracking::updateLocalMap(bool add_current_frame) {
    // R4：mvpLocalKeyFrames_ 顾名思义只放**关键帧**。
    // 之前这里把普通帧也 push_back 进去（只要它不是"刚创建的关键帧"），
    // 结果 .back() 经常指向普通帧，trackReferenceKeyFrame() 拿它当参考帧
    // 去解 PnP —— 普通帧没有经过关键帧的观测筛选，位姿还是本帧 poseOptimization
    // 的结果，用它当参考等于自我循环，误差锁死无法被关键帧纠正。
    // 现在：关键帧进 mvpLocalKeyFrames_（由 createNewKeyFrame 负责），
    // 普通帧只更新参考帧指针与 BoW 索引，不再混进关键帧窗口。
    const bool is_current_kf = (mCurrentFrame_ && mCurrentFrame_->id == mnLastKeyFrameId_);
    if (add_current_frame && !is_current_kf) {
        // R4：普通帧不再进入 mvpLocalKeyFrames_，也不注册进 BoW 索引。
        // 之前这里 push_back 普通帧，close 掉的是「参考帧 = 最新一帧」这个假设；
        // 现在只保证参考帧指针指向当前帧（供诊断/可视化读取），关键帧窗口
        // 由 createNewKeyFrame() 独占维护。
        mpReferenceKF_ = mCurrentFrame_->id;

        if (config_.verbose_diag && mvpLocalKeyFrames_.empty()) {
            std::cerr << "[updateLocalMap] 警告: 关键帧窗口为空，本轮无参考关键帧可用"
                      << std::endl;
        }
    }

    // ★★★ 更新 Atlas 参考地图点（ORB-SLAM3 mpAtlas->SetReferenceMapPoints）★★★
    // 将当前帧的地图点设为参考地图点（用于可视化）
    if (mpAtlas_ && mpMap_) {
        std::vector<mapping::MapPoint*> ref_pts;
        for (int mp_id : mvpCurrentMapPointIds_) {
            if (mp_id < 0) continue;
            // E6：临时 VO 点现在住在独立池里，必须走 getMapPoint() 两池合并查找，
            // 直接 find(mAllMapPoints_) 会让临时点一律当成"不存在"。
            const MapPointCandidate* cand = getMapPoint(mp_id);
            if (cand && !cand->is_bad) {
                if (auto* pMP = mpMap_->getMapPoint(mp_id)) {
                    ref_pts.push_back(pMP);
                }
            }
        }
        mpMap_->setReferenceMapPoints(ref_pts);
    }
}

// ================================================================
// 搜索局部地图点
// ================================================================
void Tracking::searchLocalPoints() {
    // ★★★ ORB-SLAM3 风格：只搜索局部关键帧关联的地图点 ★★★
    // 构建局部关键帧关联的地图点集合，仅对这些点做投影匹配

    // Step 1: 收集已匹配的地图点
    std::set<int> sAlreadyFound;
    for (int mp_id : mvpCurrentMapPointIds_) {
        if (mp_id >= 0) sAlreadyFound.insert(mp_id);
    }

    // ★★★ 从局部关键帧的 map_point_ids 收集候选地图点 ★★★
    std::vector<int> candidate_ids;
    candidate_ids.reserve(5000);
    for (const KeyFrameCacheEntry& kf : mvpLocalKeyFrames_) {
        for (int mp_id : kf.map_point_ids) {
            if (sAlreadyFound.count(mp_id) == 0) {
                candidate_ids.push_back(mp_id);
            }
        }
    }

    // 如果局部关键帧没有足够地图点，补充全局搜索（备用）
    // ★★★ Bug Fix: 提高全局搜索触发阈值 ★★★
    // 原阈值 100 太低，导致局部地图点稍少时就回退到全图搜索。
    // 提高到 200（与 ORB-SLAM3 一致），确保只有在局部关键帧真正不充足时才触发全局搜索。
    bool use_local_only = (candidate_ids.size() >= 200);
    if (!use_local_only) {
        // 局部关键帧的点不够用，从全图补充。
        // 原实现直接遍历 mAllMapPoints_（unordered_map，哈希序）取前 500 个，
        // 注释写着"只接受合理范围内的点"，实际是"随机抽 500 个点" ——
        // 哈希序和"离相机近"没有任何关系。现在先按距离收集，再取最近的 500 个。
        const Eigen::Vector3d cam_pos = mCurrentFrame_->pose.t;
        // max_depth 本身就是"这个深度传感器还相信的极限"，不需要再乘 4。
        const double max_dist = config_.max_depth;
        struct NearbyCandidate {
            double dist;
            int id;
        };
        std::vector<NearbyCandidate> nearby;
        for (const auto& kv : mAllMapPoints_) {
            if (kv.second.is_bad) continue;
            if (sAlreadyFound.count(kv.first) != 0) continue;
            const double dist = (kv.second.world_point - cam_pos).norm();
            if (dist > max_dist) continue;
            nearby.push_back(NearbyCandidate{dist, kv.first});
        }
        const size_t max_global_add = 500;
        if (nearby.size() > max_global_add) {
            std::partial_sort(nearby.begin(), nearby.begin() + max_global_add, nearby.end(),
                              [](const NearbyCandidate& a, const NearbyCandidate& b) {
                                  return a.dist < b.dist;
                              });
            nearby.resize(max_global_add);
        }
        for (const auto& c : nearby) candidate_ids.push_back(c.id);
    }

    const auto& curr_kpts = (!mCurrentFrame_->undistorted_keypoints.empty())
        ? mCurrentFrame_->undistorted_keypoints : mCurrentFrame_->keypoints;

    const cv::Mat& K = config_.K;
    CameraParams cam = getCameraParams(K);
    double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    // ★ 关键帧永久点优先于上一帧的临时点 ★
    // searchByProjection(当前帧, 上一帧) 会先把上一帧的临时点贴到当前帧的关键点上。
    // 那些点是用"上一帧的位姿 + 上一帧的深度"现建的，拿它们做位姿优化等于每一步都
    // 站在上一步的误差上：上一帧偏 1.5 cm，这一帧就再偏 1.5 cm，误差按帧串成链。
    // 实测正是这个链条让每帧平移比真值快约 1.5 倍（整段路径 15.9 m 对真值 9.9 m）。
    // 所以这里把"已匹配到临时点"的关键点留成候选，让关键帧的永久点有机会把它顶掉
    // —— 位姿重新锚到更早、视差更大、更可信的观测上，链条就断了。
    std::vector<char> kpt_matched(curr_kpts.size(), 0);
    const size_t n_slots = std::min(mvpCurrentMapPointIds_.size(), curr_kpts.size());
    for (size_t i = 0; i < n_slots; ++i) {
        const int mp_id = mvpCurrentMapPointIds_[i];
        if (mp_id < 0) continue;
        if (mTemporalMapPoints_.count(mp_id) == 0) kpt_matched[i] = 1;
    }

    int search_count = 0;
    const int max_search = 500;

    for (int mp_id : candidate_ids) {
        if (search_count >= max_search) break;
        if (sAlreadyFound.count(mp_id)) continue;

        // E6：候选里可能混着临时 VO 点 id（searchByProjection 会把它们写进
        // mvpCurrentMapPointIds_），必须两池合并查找。
        MapPointCandidate* mp_ptr = getMapPoint(mp_id);
        if (!mp_ptr) continue;

        MapPointCandidate& mp = *mp_ptr;
        if (mp.is_bad) continue;

        const auto& pt_world = mp.world_point;
        if (pt_world.norm() < 1e-6) continue;

        Eigen::Vector3d pt_cam = worldToCam(pt_world, mCurrentFrame_->pose);
        double z = pt_cam.z();
        if (z < 0.05 || !std::isfinite(z)) continue;

        double d = z;
        if (d < config_.min_depth || d > config_.max_depth) continue;

        double invz = 1.0 / z;
        float u_f = static_cast<float>(fx * pt_cam.x() * invz + cx);
        float v_f = static_cast<float>(fy * pt_cam.y() * invz + cy);
        if (u_f < 0 || u_f >= mCurrentFrame_->gray_img.cols ||
            v_f < 0 || v_f >= mCurrentFrame_->gray_img.rows) continue;

        float radius = 15.0f;
        if (d > 7.0) radius = 30.0f;
        else if (d > 4.0) radius = 20.0f;
        float radius2 = radius * radius;

        int best_k = -1, best_dist = 256;
        for (size_t k = 0; k < curr_kpts.size(); ++k) {
            if (kpt_matched[k]) continue;
            float du = curr_kpts[k].pt.x - u_f;
            float dv = curr_kpts[k].pt.y - v_f;
            if (du * du + dv * dv > radius2) continue;
            if (k >= static_cast<size_t>(mCurrentFrame_->descriptors.rows)) continue;
            if (mp.descriptor.empty()) continue;
            int dist = descriptorDistance(mp.descriptor, mCurrentFrame_->descriptors.row(static_cast<int>(k)));
            if (dist < best_dist) { best_dist = dist; best_k = static_cast<int>(k); }
        }

        if (best_k >= 0 && best_dist <= 100) {
            kpt_matched[best_k] = 1;
            sAlreadyFound.insert(mp_id);
            cv::Mat desc;
            if (best_k < static_cast<int>(mCurrentFrame_->descriptors.rows))
                desc = mCurrentFrame_->descriptors.row(best_k).clone();
            addObservationToMapPoint(mp_id, mCurrentFrame_->id, best_k, desc);
            if (best_k < static_cast<int>(mvpCurrentMapPointIds_.size()))
                mvpCurrentMapPointIds_[best_k] = mp_id;
            search_count++;
        }
    }
}

// ================================================================
// int Tracking::poseOptimization()
//
// 使用 g2o Levenberg-Marquardt BA 优化相机位姿（模仿 ORB-SLAM3 Optimizer::PoseOptimization）。
//
// 与 EPnP 的根本区别：
// - EPnP：使用虚拟控制点近似，重投影误差的 Jacobian 近似，与 pose 流形不完全匹配
// - g2o BA：真正的 LM 优化，最小化 Σ||e_i(R,t)||²，Jacobian 解析推导，与 se3 流形完全匹配
// - EPnP 旋转估计每帧微调，累积误差导致后续帧发散；g2o BA 4轮迭代+outlier剔除，稳定收敛
// ================================================================
int Tracking::poseOptimization() {
    // NaN 输入保护
    if (!mCurrentFrame_->pose.t.hasNaN() && !mCurrentFrame_->pose.q.coeffs().hasNaN() &&
        mCurrentFrame_->pose.q.norm() > 0.1) {
        // proceed
    } else {
        if (mLastFrame_) mCurrentFrame_->pose = mLastFrame_->pose;
        else mCurrentFrame_->pose = Pose();
    }

    // R3：坐标约定只在一个地方决定 —— trackFrame() 的 Step 4。
    //   disable_undistort=false → undistorted_keypoints = 理想像素（undistortPoints 后重新像素化）
    //   disable_undistort=true  → undistorted_keypoints = 原始像素（仅当 D 为空时成立）
    // 本函数与 compute3DPoints() 都直接消费这套坐标，谁也不再自己做去畸变：
    //   (A) 3D 世界点：畸变像素采样深度 + 理想像素反投影（见 compute3DPoints）
    //   (B) 2D 观测：理想像素
    //   (C) g2o 边：标准针孔投影
    // 三者都在同一个理想针孔坐标系里，残差才可比。
    // 之前这里对已经是理想像素的坐标又 undistortPoints 一次，等于把畸变模型
    // 反向再施加一遍，和 (A) 的方向相反 —— 这正是 R3 要消除的系统性偏差。
    const auto& curr_kpts = (!mCurrentFrame_->undistorted_keypoints.empty())
        ? mCurrentFrame_->undistorted_keypoints : mCurrentFrame_->keypoints;
    const Pose& p = mCurrentFrame_->pose;

    // 获取相机内参
    CameraParams cam = getCameraParams(config_.K);
    double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;


    // 收集有效的 3D-2D 对应（从地图点）
    std::vector<std::tuple<int, Eigen::Vector3d, Eigen::Vector2d>> valid;
    for (size_t i = 0; i < mvpCurrentMapPointIds_.size(); ++i) {
        int mp_id = mvpCurrentMapPointIds_[i];
        if (mp_id < 0) continue;
        if (i >= curr_kpts.size()) continue;
        auto* mp = getMapPoint(mp_id);
        if (!mp || mp->is_bad) continue;
        const Eigen::Vector2d obs(curr_kpts[i].pt.x, curr_kpts[i].pt.y);
        valid.emplace_back(static_cast<int>(i), mp->world_point, obs);
    }

    if (valid.size() < 10) return 0;

    // ★★★ Bug Fix: 诊断特征点分布和深度值 ★★★
    if (config_.verbose_diag) {
        static int last_diag_frame = -1;
        if (last_diag_frame != mCurrentFrame_->id) {
            last_diag_frame = mCurrentFrame_->id;
            double u_min = 1e10, u_max = -1e10, v_min = 1e10, v_max = -1e10;
            double depth_min = 1e10, depth_max = -1e10, depth_sum = 0;
            int depth_count = 0;

            // ★★★ 诊断：计算重投影误差分布 ★★★
            double reproj_error_sum = 0;
            double reproj_error_max = 0;
            int reproj_count = 0;

            CameraParams cam_diag = getCameraParams(config_.K);
            double fx_d = cam_diag.fx, fy_d = cam_diag.fy, cx_d = cam_diag.cx, cy_d = cam_diag.cy;

            for (const auto& t : valid) {
                int kpt_idx; Eigen::Vector3d Xw; Eigen::Vector2d obs;
                std::tie(kpt_idx, Xw, obs) = t;
                u_min = std::min(u_min, obs(0));
                u_max = std::max(u_max, obs(0));
                v_min = std::min(v_min, obs(1));
                v_max = std::max(v_max, obs(1));

                // 计算深度（从世界点到相机）
                Eigen::Vector3d pt_cam = worldToCam(Xw, p);
                double z = pt_cam.z();
                if (z > 0 && z < 10) {
                    depth_min = std::min(depth_min, z);
                    depth_max = std::max(depth_max, z);
                    depth_sum += z;
                    depth_count++;
                }

                // 计算重投影误差（像素）
                if (z > 0.1) {
                    double invz = 1.0 / z;
                    double proj_x = fx_d * pt_cam.x() * invz + cx_d;
                    double proj_y = fy_d * pt_cam.y() * invz + cy_d;
                    double dx = obs(0) - proj_x;
                    double dy = obs(1) - proj_y;
                    double err = std::sqrt(dx * dx + dy * dy);
                    reproj_error_sum += err;
                    reproj_error_max = std::max(reproj_error_max, err);
                    reproj_count++;
                }
            }
            double avg_depth = depth_count > 0 ? depth_sum / depth_count : 0;
            double avg_reproj = reproj_count > 0 ? reproj_error_sum / reproj_count : 0;
            Eigen::Matrix3d R_wc = p.q.toRotationMatrix();
            { if (config_.verbose_diag) std::cerr << "[poseOpt-DIST f=" << mCurrentFrame_->id << "] "
                      << "n=" << valid.size()
                      << " u=[" << u_min << "," << u_max << "] v=[" << v_min << "," << v_max << "]"
                      << " depth=[" << depth_min << "," << depth_max << "] avg=" << avg_depth
                      << " img_size=" << mCurrentFrame_->gray_img.cols << "x" << mCurrentFrame_->gray_img.rows
                      << " PnP_pose=(" << std::fixed << std::setprecision(3)
                      << p.t.x() << "," << p.t.y() << "," << p.t.z() << ")"
                      << " R_wc_det=" << std::fixed << std::setprecision(4) << R_wc.determinant()
                      << " init_reproj_err_avg=" << avg_reproj << " max=" << reproj_error_max
                      << std::endl; }
        }
    }

    // ================================================================
    // 设置 g2o 优化器（与 ORB-SLAM3 Optimizer::PoseOptimization 完全一致）
    // ================================================================
    using BlockSolver = g2o::BlockSolver_6_3;
    using LinearSolver = g2o::LinearSolverDense<BlockSolver::PoseMatrixType>;

    std::unique_ptr<LinearSolver> linearSolver(new LinearSolver());
    std::unique_ptr<BlockSolver> blockSolver(new BlockSolver(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    // 添加单个位姿顶点（当前帧的 T_cw）
    // R1：p 是对外约定的 T_wc，g2o::VertexSE3Expmap 要的是 T_cw，
    // 必须在这里显式换算，不能再把 R_wc/t_wc 直接塞进顶点。
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    auto [R_cw_init, t_cw_init] = getCwFromPose(p);
    vSE3->setEstimate(g2o::SE3Quat(R_cw_init, t_cw_init));
    vSE3->setId(0);
    vSE3->setFixed(false);
    // ★★★ Bug Fix: 确保顶点不是 marginalizable，否则 g2o 可能跳过优化 ★★★
    vSE3->setMarginalized(false);
    optimizer.addVertex(vSE3);

    // Huber 核的拐点。取 config_.pnp_huber_delta（默认 2.45，= sqrt(5.991)），
    // 与 ORB_SLAM3 Optimizer.cc 的 deltaMono 一致。此前这里硬编码 5.0，
    // 于是 tracking.yaml 里的 pnp_huber_delta 谁都改不动 —— 又一个"配了不生效"（R-019）。
    const float deltaMono = static_cast<float>(config_.pnp_huber_delta);
    const int nIters[4] = {10, 10, 10, 10};  // 4 轮，每轮 10 次迭代

    // 添加所有边（地图点对应）
    std::vector<mapping::SFCEdgeSE3ProjectXYZOnlyPose*> edges;
    edges.reserve(valid.size());

    for (const auto& t : valid) {
        int kpt_idx; Eigen::Vector3d Xw; Eigen::Vector2d obs;
        std::tie(kpt_idx, Xw, obs) = t;

        mapping::SFCEdgeSE3ProjectXYZOnlyPose* e = new mapping::SFCEdgeSE3ProjectXYZOnlyPose();
        e->setMeasurement(obs);
        // 观测权重按金字塔层（ORB-SLAM3 的 mvInvLevelSigma2 做法）：
        // 高层特征点的像素定位精度按 scale^level 变差，权重应按 scale^(2*level) 衰减。
        // 之前所有层都用 octave 0 的固定值，等于把最模糊的高层观测与最锐利的
        // octave 0 观测同等对待 —— 优化被少量高层点主导，这是 R4 的要害之一。
        int level = 0;
        if (kpt_idx < static_cast<int>(curr_kpts.size())) {
            level = curr_kpts[kpt_idx].octave;
        }
        const int n_levels = std::max(1, config_.extractor_config.n_levels);
        level = std::clamp(level, 0, n_levels - 1);
        const float level_scale =
            std::pow(std::max(1.0f, config_.extractor_config.scale_factor),
                     static_cast<float>(level));
        const float invSigma2 = 1.0f / (1.2f * 1.2f * level_scale * level_scale);
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

        // Huber 鲁棒核（与 ORB-SLAM3 完全一致）
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
        e->setRobustKernel(rk);
        rk->setDelta(deltaMono);

        // g2o::EdgeSE3ProjectXYZOnlyPose 的 public 成员
        e->Xw = Xw;
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        e->setVertex(0, vSE3);
        optimizer.addEdge(e);
        edges.push_back(e);
    }

    int nInitialCorrespondences = static_cast<int>(valid.size());

    // ================================================================
    // 4 轮迭代优化（与 ORB-SLAM3 完全一致）
    // 每轮结束后标记 outlier，下一轮不再优化 outlier
    // ================================================================
    // 保存每轮优化后的结果，作为下一轮的初始值
    g2o::SE3Quat current_estimate = g2o::SE3Quat(R_cw_init, t_cw_init);
    int nBad = 0;
    // χ²(0.95, 2dof) = 5.99，来源：ORB-SLAM3 Optimizer::PoseOptimization 的 thHuber2D。
    // 为什么四轮都用同一个值：前两轮放宽到 1e4 相当于两轮不做外点剔除，
    // 外点足以把位姿拖到错误盆地；后两轮再收紧只是把一个已经失败的初值当真相。
    constexpr float CHI2_THRESHOLD_2DOF_95 = 5.99f;

    for (size_t it = 0; it < 4; it++) {
        const float chi2_th_this_round = CHI2_THRESHOLD_2DOF_95;

        vSE3->setEstimate(current_estimate);
        // ★★★ Bug Fix: 检查 initializeOptimization 返回值 ★★★
        // 如果返回 false，说明没有可优化的顶点（可能所有边都被标记为 level=1）
        int n_level0 = 0;
        for (auto* e : edges) {
            if (e->level() == 0) n_level0++;
        }
        if (n_level0 == 0) {
            // 诊断日志
            if (config_.verbose_diag) {
                std::cerr << "[poseOptimization] Round " << it << ": no level-0 edges to optimize!" << std::endl;
            }
            break;
        }
        bool init_ok = optimizer.initializeOptimization(0);
        if (!init_ok) {
            if (config_.verbose_diag) {
                std::cerr << "[poseOptimization] Round " << it << ": initializeOptimization failed!" << std::endl;
            }
            break;
        }
        if (config_.verbose_diag) {
            std::cerr << "[poseOpt-Round" << it << "] chi2_th=" << chi2_th_this_round << std::endl;
        }
        optimizer.optimize(nIters[it]);

        // ★★★ Bug Fix: 诊断优化是否真的改变了 pose ★★★
        g2o::SE3Quat after_opt = vSE3->estimate();
        if (config_.verbose_diag && it == 0) {
            static int last_diag = -1;
            if (last_diag != mCurrentFrame_->id) {
                last_diag = mCurrentFrame_->id;
                // 检查边的状态（chi2 在 computeError 后才有效）
                int edge_errors_ok = 0;
                double chi2_sum_before = 0;
                double pixel_error_sum = 0;
                double pixel_error_max = 0;
                for (size_t ei = 0; ei < valid.size(); ++ei) {
                    mapping::SFCEdgeSE3ProjectXYZOnlyPose* e_check = new mapping::SFCEdgeSE3ProjectXYZOnlyPose();
                    int kpt_idx; Eigen::Vector3d Xw; Eigen::Vector2d obs;
                    std::tie(kpt_idx, Xw, obs) = valid[ei];
                    e_check->setMeasurement(obs);
                    e_check->Xw = Xw;
                    e_check->fx = fx;
                    e_check->fy = fy;
                    e_check->cx = cx;
                    e_check->cy = cy;
                    e_check->setVertex(0, vSE3);
                    e_check->computeError();
                    double chi2 = e_check->chi2();
                    chi2_sum_before += chi2;
                    if (chi2 < 1e6) edge_errors_ok++;
                    // 计算实际像素误差
                    double px_err = std::sqrt(e_check->error()(0) * e_check->error()(0) +
                                              e_check->error()(1) * e_check->error()(1));
                    pixel_error_sum += px_err;
                    pixel_error_max = std::max(pixel_error_max, px_err);
                    delete e_check;
                }
                double avg_px_err = pixel_error_sum / std::max(1.0, static_cast<double>(valid.size()));
                { if (config_.verbose_diag) std::cerr << "[poseOpt-OPT f=" << mCurrentFrame_->id << "] "
                          << "before_opt=(t=" << current_estimate.translation().transpose()
                          << ") after_opt=(t=" << after_opt.translation().transpose()
                          << ") delta_t=" << (after_opt.translation() - current_estimate.translation()).norm()
                          << " edge_errors_ok=" << edge_errors_ok << "/" << valid.size()
                          << " chi2_avg=" << (chi2_sum_before / std::max(1.0, static_cast<double>(valid.size())))
                          << " avg_px_err=" << avg_px_err << " max_px_err=" << pixel_error_max
                          << std::endl; }
            }
        }
        current_estimate = after_opt;

        for (size_t i = 0; i < edges.size(); i++) {
            mapping::SFCEdgeSE3ProjectXYZOnlyPose* e = edges[i];

            e->computeError();

            // 为什么移除鲁棒核是错的：Huber 核把外点的代价从二次压成线性，
            // 正是它阻止了 chi2=2e4 级别的错误匹配主导位姿。去掉核后这些外点
            // 以二次代价参与 LM，位姿会被单向拉偏——日志里 init_reproj_err_avg
            // 从 10px 涨到 3022px 就是这个机制。
            // 因此保留 Huber 核（见上方 rk->setDelta），只在每轮末按 chi2 剔除。

            // 每一轮（含第 0 轮）都按 chi2 剔除外点，避免外点参与下一次迭代。
            // 这里刻意不做"回填 level 0"：与 ORB-SLAM3 Optimizer::PoseOptimization 保持一致，
            // 一旦被判为外点就在本次调用内保持外点身份。回填会让同一个边在几轮之间反复
            // 进出优化集，LM 的代价面在轮次之间跳动，反而更难复现。若后续实测发现
            // 内点被系统性低估（nInliers 明显小于肉眼可见的正确匹配数），再单独试验回填。
            if (e->chi2() > chi2_th_this_round) {
                e->setLevel(1);
            }
        }
        
        // ★★★ Bug Fix: 添加 chi2 诊断日志 ★★★
        if (config_.verbose_diag && edges.size() > 0) {
            int n_level0_after = 0;
            double chi2_min = 1e10, chi2_max = 0, chi2_sum = 0;
            for (auto* e : edges) {
                if (e->level() == 0) n_level0_after++;
                double chi2 = e->chi2();
                chi2_min = std::min(chi2_min, chi2);
                chi2_max = std::max(chi2_max, chi2);
                chi2_sum += chi2;
            }
            { if (config_.verbose_diag) std::cerr << "[poseOpt-Round" << it << "] edges=" << edges.size() 
                      << " level0=" << n_level0_after 
                      << " chi2_min=" << chi2_min << " chi2_max=" << chi2_max 
                      << " chi2_avg=" << (chi2_sum / edges.size())
                      << " th=" << chi2_th_this_round << std::endl; }
        }
        current_estimate = vSE3->estimate();
    }

    // 统计所有被标记为 outlier 的边（nBad 已在循环中累加）
    for (size_t i = 0; i < edges.size(); i++) {
        if (edges[i]->level() == 1) {
            nBad++;
        }
    }

    // 恢复最终优化结果
    vSE3->setEstimate(current_estimate);

    // 恢复优化后的 pose
    // R1：g2o 顶点里是 T_cw，必须换回对外的 T_wc 才能写回 mCurrentFrame_->pose
    g2o::SE3Quat se3_recov = vSE3->estimate();
    Pose new_pose = poseFromCw(se3_recov.rotation().toRotationMatrix(), se3_recov.translation());

    if (!new_pose.t.allFinite() || !new_pose.q.coeffs().allFinite() ||
        new_pose.t.norm() > 1e6 || new_pose.q.norm() < 0.1) {
        return 0;
    }

    // 防止 poseOptimization 收敛到错误的局部最小值
    double t_diff = (new_pose.t - p.t).norm();
    Eigen::Quaterniond q_diff = new_pose.q * p.q.inverse();
    if (p.q.coeffs().dot(new_pose.q.coeffs()) < 0) {
        q_diff.coeffs() = -q_diff.coeffs();
    }
    Eigen::AngleAxisd angle_axis(q_diff);
    double angle_val = angle_axis.angle();
    double rot_diff = std::min(angle_val, 2 * M_PI - angle_val);

    // ★★★ Bug Fix: 添加 pose 变化诊断日志 ★★★
    if (config_.verbose_diag) {
        static int last_diag_frame = -1;
        if (last_diag_frame != mCurrentFrame_->id) {
            last_diag_frame = mCurrentFrame_->id;
            { if (config_.verbose_diag) std::cerr << "[poseOpt-DBG f=" << mCurrentFrame_->id << "] "
                      << "before=(" << p.t.x() << "," << p.t.y() << "," << p.t.z() << ") "
                      << "after=(" << new_pose.t.x() << "," << new_pose.t.y() << "," << new_pose.t.z() << ") "
                      << "t_diff=" << t_diff << " rot_diff=" << rot_diff << std::endl; }
        }
    }

    if (t_diff > 1.0 || rot_diff > 1.0) {
        if (config_.verbose_diag) {
            std::cerr << "  [poseOpt f=" << mCurrentFrame_->id << "] REJECTED: t_diff=" << t_diff
                      << " rot_diff=" << rot_diff << " (thresh: 1.0m, 1.0rad)" << std::endl;
        }
        return 0;
    }

    // ★★★ Bug Fix: 运动一致性检查——防止 BA 收敛到错误的局部极小 ★★★
    // 根本原因：即使预过滤和 chi2_th 收紧，BA 仍可能因残存的 outliers 收敛到错误解。
    // 诊断证据：帧 1 优化后 t_diff=0.111m（相对于初始 PnP pose），但这个 PnP pose 本身就是错的。
    // TUM fr1_desk2 帧间位移约 3-7cm，但优化后跳变到 16cm。
    // 修复：用运动模型预测 pose，与 BA 结果对比——如果差异过大，说明 BA 失败了。
    if (mbVelocityValid_ && mLastFrame_ && t_diff < 1.0 && rot_diff < 1.0) {
        // 运动模型预测的 pose：T_pred = mVelocity_ * T_last
        Eigen::Matrix3d R_vel = mVelocity_.q.toRotationMatrix();
        Pose pred_pose;
        pred_pose.q = mVelocity_.q * mLastFrame_->pose.q;
        normalizeQuaternion(pred_pose.q);
        pred_pose.t = R_vel * mLastFrame_->pose.t + mVelocity_.t;

        double t_diff_from_motion = (new_pose.t - pred_pose.t).norm();
        Eigen::Quaterniond q_motion = pred_pose.q;
        if (q_motion.coeffs().dot(new_pose.q.coeffs()) < 0)
            q_motion.coeffs() = -q_motion.coeffs();
        Eigen::Quaterniond q_diff_motion = new_pose.q * q_motion.inverse();
        Eigen::AngleAxisd aa_motion(q_diff_motion);
        double rot_diff_from_motion = aa_motion.angle();

        // 室内手持场景：单帧预测误差应 < 0.3m，< 30°
        const double MOTION_T_DIFF_TH = 0.50;  // 50cm
        const double MOTION_ROT_DIFF_TH = 0.524; // 30° in radians

        if (t_diff_from_motion > MOTION_T_DIFF_TH || rot_diff_from_motion > MOTION_ROT_DIFF_TH) {
            if (config_.verbose_diag) {
                static int last_motion_diag = -1;
                if (last_motion_diag != mCurrentFrame_->id) {
                    last_motion_diag = mCurrentFrame_->id;
                    { if (config_.verbose_diag) std::cerr << "  [poseOpt-MOTION f=" << mCurrentFrame_->id << "] "
                              << "BA pose inconsistent with motion model: "
                              << "t_diff_from_motion=" << t_diff_from_motion
                              << " rot_diff_from_motion=" << rot_diff_from_motion
                              << " (thresh: " << MOTION_T_DIFF_TH << "m, " << MOTION_ROT_DIFF_TH << "rad)"
                              << " → REJECT (force fallback to motion model)"
                              << std::endl; }
                }
            }
            // ★★★ Bug Fix: 强制拒绝——BA 结果与运动模型不一致，说明 BA 收敛到错误局部极小 ★★★
            return 0;
        }
    }

    // 标记外点（对应到 mvpCurrentMapPointIds_）
    std::vector<bool> is_outlier(valid.size(), false);
    int nOutliersThisRun = 0;
    for (size_t i = 0; i < edges.size(); i++) {
        if (edges[i]->level() == 1) {
            is_outlier[i] = true;
            nOutliersThisRun++;
        }
    }
    for (size_t i = 0; i < valid.size(); i++) {
        if (is_outlier[i]) {
            int kpt_idx; Eigen::Vector3d Xw; Eigen::Vector2d obs;
            std::tie(kpt_idx, Xw, obs) = valid[i];
            if (kpt_idx >= 0 && kpt_idx < static_cast<int>(mvpCurrentMapPointIds_.size()))
                mvpCurrentMapPointIds_[kpt_idx] = -1;
        }
    }

    // 更新地图点追踪计数
    for (size_t i = 0; i < valid.size(); i++) {
        if (is_outlier[i]) continue;
        int kpt_idx; Eigen::Vector3d Xw; Eigen::Vector2d obs;
        std::tie(kpt_idx, Xw, obs) = valid[i];
        int mp_id = mvpCurrentMapPointIds_[kpt_idx];
        if (mp_id < 0) continue;
        // 同时更新前端 MapPointCandidate 和后端 mapping::MapPoint 的 found 计数
        MapPointCandidate* mp = getMapPoint(mp_id);
        if (!mp || mp->is_bad) continue;
        mp->found_count++;
        if (auto* mp_backend = mpMap_->getMapPoint(mp_id))
            mp_backend->increaseFound();
    }

    const int nInliers = nInitialCorrespondences - nBad;

    // 内点不足时拒绝本次位姿：低内点率下的解没有统计支撑，
    // 一旦写回 pose，下一帧的恒速预测和局部地图投影都会继承这个错误，
    // 错误会自激放大（日志里 f=610 之后的 LOST→RELOC 死循环即此机制）。
    // 返回 0 交给调用方按“跟踪失败”处理，不污染当前帧位姿。
    if (nInliers < config_.pose_opt_min_inliers) {
        if (config_.verbose_diag) {
            std::cerr << "[poseOpt] f=" << mCurrentFrame_->id
                      << " nInliers=" << nInliers << " < " << config_.pose_opt_min_inliers
                      << " → REJECT pose" << std::endl;
        }
        return 0;
    }

    mCurrentFrame_->pose = new_pose;

    return nInliers;
}
// ================================================================
// 检查替换的地图点
// ================================================================
// 检查并替换上一帧的地图点（ORB-SLAM3 CheckReplacedInLastFrame）
// LocalMapping 可能会将低质量地图点替换为高质量点
// ================================================================
void Tracking::checkReplacedInLastFrame() {
    if (!mLastFrame_) return;

    // 遍历上一帧的所有地图点
    for (size_t i = 0; i < mLastFrameMapPointIds_.size(); ++i) {
        int mp_id = mLastFrameMapPointIds_[i];
        if (mp_id < 0) continue;

        auto it_replace = mReplaceMap_.find(mp_id);
        if (it_replace != mReplaceMap_.end()) {
            // 被替换为新点
            int new_mp_id = it_replace->second;
            mLastFrameMapPointIds_[i] = new_mp_id;
            // 同步更新 mAllMapPoints_ 的观测记录
            auto* new_mp = getMapPoint(new_mp_id);
            if (new_mp) {
                new_mp->observations.emplace_back(mLastFrame_->id, static_cast<int>(i));
                new_mp->frontend_observation_count++;
                new_mp->confirmed = (new_mp->frontend_observation_count >= 2);
                new_mp->last_observed_frame_id = mLastFrame_->id;
            }
        }
    }
}

// ================================================================
// 更新运动模型
// ================================================================
void Tracking::updateMotionModel() {
    if (!mCurrentFrame_ || !mLastFrame_) return;
    const Pose& curr = mCurrentFrame_->pose;
    const Pose& last = mLastFrame_->pose;

    Eigen::Quaterniond q_curr = curr.q;
    Eigen::Quaterniond q_last = last.q;
    if (q_curr.coeffs().dot(q_last.coeffs()) < 0.0)
        q_curr.coeffs() = -q_curr.coeffs();

    // 计算相对运动
    mVelocity_.q = q_curr * q_last.inverse();
    normalizeQuaternion(mVelocity_.q);  // ★★★ 防止四元数数值误差累积 ★★★
    Eigen::Matrix3d R_delta = mVelocity_.q.toRotationMatrix();
    mVelocity_.t = curr.t - R_delta * last.t;
    
    // ★★★ Bug Fix: 添加运动模型合理性检查 ★★★
    // 检查平移和旋转是否在合理范围内
    double trans_norm = mVelocity_.t.norm();
    Eigen::AngleAxisd aa(mVelocity_.q);
    double rot_angle = aa.angle();  // 弧度
    
    // R4：室内 30 Hz 手持每帧真实位移是 3–7 cm，原阈值 1.0 m/帧 只是"别爆炸"的
    // 兜底，等于对恒速模型没有任何约束：位姿一旦跳 0.9 m 仍会被当成有效速度
    // 沿用到下一帧，误差就这么被写进预测里。收到 0.15 m/帧（约真值上限的 2 倍），
    // 超限时不更新运动模型，让下一帧退回去用参考关键帧。
    const double max_trans_per_frame = 0.15;  // 米
    const double max_rot_per_frame = 30.0 * M_PI / 180.0;  // 30 度
    
    if (trans_norm > max_trans_per_frame || rot_angle > max_rot_per_frame) {
        // 运动过大，可能是跟踪失败，不更新运动模型
        if (config_.verbose_diag) {
            std::cerr << "[MotionModel] REJECT: trans=" << trans_norm << "m rot=" << rot_angle*180/M_PI << "deg" << std::endl;
        }
        return;
    }
    
    mbVelocityValid_ = true;
}

// ================================================================
// 关键帧判定（薄适配层）
// ================================================================
// 判定本身的唯一事实来源是 frontend/keyframe_policy.hpp 的 decideKeyframe()。
// 这个函数只负责两件事：把 Tracking 的内部状态整理成 KeyframeEvidence，
// 再把 TrackingConfig 里的阈值搬进 KeyframePolicyParams。
//
// 原先这里是一段 116 行的条件堆砌（c1a/c1b/c1c/c2/c1d/c1e 用 || 串起来），
// 其中三个毛病直接导致了"640 帧插 562 个关键帧"：
//   * c1b = (帧间隔 >= 10)，无条件的 —— 每 10 帧必插，与跟踪质量无关；
//   * c1c = (内点数 < max(10, 参考点*0.25) || 近距离点太少)，没有任何帧间隔下限，
//     而近距离判据又基于 `th_depth = max_depth*4.0`（=48m）—— 全部点都算"近"，
//     bNeedToInsertClose 于是长期为真，c1c 几乎每帧成立；
//   * local_mapping_busy 恒为 false（isBusy 调用被注释掉），那段"忙碌时只允许
//     紧急插帧"的分支是死代码，等于没有任何限流。
namespace {

// 两帧位姿之间的旋转夹角（rad）。
double rotationAngleBetween(const Eigen::Quaterniond& q_from, const Eigen::Quaterniond& q_to) {
    Eigen::Quaterniond dq = q_to * q_from.inverse();
    dq.normalize();
    return 2.0 * std::atan2(dq.vec().norm(), std::abs(dq.w()));
}

// 判定结果的可读标签，只用于诊断日志。
const char* keyframeDecisionReason(const KeyframeDecision& d) {
    if (d.forced) return "forced";
    if (d.rescue) return "rescue";
    if (d.by_tracking_quality) return "quality";
    if (d.by_motion) return "motion";
    return "none";
}

}  // namespace

// 证据 2 的取法有讲究：必须遍历参考关键帧**自己的** map_point_ids。
// 原实现用 ref_kf.pts_3d 的长度当循环上界，却拿同一个下标去索引
// mLastFrameMapPointIds_ —— 只有"参考帧恰好就是上一帧"时两者才是同一批特征点，
// 一旦走 REF_KF / RELOC 路径，nRefMatches 就是个随机数。
// 证据 3 用"相对关键帧"而不是"相对上一帧"：min_keyframe_translation 的语义
// 就是关键帧之间的最小基线，拿帧间位移会让抖动本身也被算成"走了很远"。
bool Tracking::needNewKeyFrame() {
    if (!mCurrentFrame_) return false;

    const int frame_gap = mCurrentFrame_->id - mnLastKeyFrameId_;
    const KeyFrameCacheEntry* ref_kf =
        mvpLocalKeyFrames_.empty() ? nullptr : &mvpLocalKeyFrames_.back();

    int ref_matches = 0;
    double trans_since_kf = 0.0;
    double rot_since_kf = 0.0;
    if (ref_kf != nullptr) {
        for (const int mp_id : ref_kf->map_point_ids) {
            const MapPointCandidate* mp = (mp_id < 0) ? nullptr : getMapPoint(mp_id);
            if (mp != nullptr && !mp->is_bad) ++ref_matches;
        }
        trans_since_kf = (mCurrentFrame_->pose.t - ref_kf->world_pose.t).norm();
        rot_since_kf = rotationAngleBetween(ref_kf->world_pose.q, mCurrentFrame_->pose.q);
    }

    KeyframePolicyParams params;
    params.min_gap_frames = config_.min_keyframe_gap_frames;
    params.forced_interval = config_.forced_keyframe_interval;
    params.min_tracked_inliers = config_.keyframe_min_tracked_inliers;
    params.weak_tracked_ratio = config_.keyframe_weak_tracked_ratio;
    params.keyframe_translation = config_.min_keyframe_translation;
    params.keyframe_rotation = config_.min_keyframe_rotation;
    params.rescue_min_inliers = config_.keyframe_rescue_min_inliers;
    params.rescue_max_gap = config_.keyframe_rescue_max_gap;

    KeyframeEvidence evidence;
    evidence.frame_gap = frame_gap;
    evidence.inliers = mnMatchesInliers_;
    evidence.ref_matches = ref_matches;
    evidence.velocity_valid = mbVelocityValid_;
    evidence.translation_since_kf = trans_since_kf;
    evidence.rotation_since_kf = rot_since_kf;

    const KeyframeDecision decision = decideKeyframe(params, evidence);

    if (config_.verbose_diag && decision.need_keyframe) {
        std::cerr << "[NeedKF] f=" << mCurrentFrame_->id << " gap=" << evidence.frame_gap
                  << " inliers=" << evidence.inliers << "/" << evidence.ref_matches
                  << " d_t=" << trans_since_kf << " d_r=" << rot_since_kf
                  << " why=" << keyframeDecisionReason(decision) << std::endl;
    }

    return decision.need_keyframe;
}

// ================================================================
// 创建新关键帧（ORB-SLAM3 风格）
// ================================================================
void Tracking::createNewKeyFrame() {
    mnLastKeyFrameId_ = mCurrentFrame_->id;

    // ★★★ 关键修复：必须在创建关键帧前更新世界坐标点 ★★★
    // 若 pts_3d_world 为空或不完整，后续 searchByProjection 会计算出错误的投影位置。
    // updateWorldPoints 会用当前帧 pose 将 pts_3d 变换到世界系并填入 pts_3d_world。
    updateWorldPoints(mCurrentFrame_);

    // ================================================================
    // Step 1: 保存到本地缓存（用于前端跟踪）
    // ================================================================
    KeyFrameCacheEntry new_kf;
    new_kf.global_id = mCurrentFrame_->id;
    new_kf.frame_id = mCurrentFrame_->id;
    new_kf.keypoints = mCurrentFrame_->keypoints;
    new_kf.undistorted_keypoints = mCurrentFrame_->undistorted_keypoints;
    new_kf.descriptors = mCurrentFrame_->descriptors.clone();
    new_kf.pts_3d = mCurrentFrame_->pts_3d;
    new_kf.pts_3d_world = mCurrentFrame_->pts_3d_world;
    new_kf.depth_valid = mCurrentFrame_->depth_valid;
    new_kf.world_pose = mCurrentFrame_->pose;

    // ★★★ 填充关键帧关联的地图点 ID 集合（ORB-SLAM3 风格）★★★
    for (int mp_id : mvpCurrentMapPointIds_) {
        if (mp_id >= 0) {
            new_kf.map_point_ids.insert(mp_id);
        }
    }

    mvpLocalKeyFrames_.push_back(new_kf);
    // 局部关键帧窗口（可配置，默认 40）。原值 20 太小：关键帧一密，窗口里只剩下
    // 时间上相邻、几乎没有视差的帧，searchLocalPoints 拿不到有横向约束的点。
    const int window =
        std::max(1, config_.local_map_keyframe_window);
    while (mvpLocalKeyFrames_.size() > static_cast<size_t>(window))
        mvpLocalKeyFrames_.pop_front();

    // ★★★ 同时保存到全局关键帧缓存（用于 BoW 重定位候选搜索）★★★
    mAllKeyFrames_[new_kf.frame_id] = new_kf;

    // ================================================================
    // Step 2: 创建 ORB-SLAM3 风格 KeyFrame 并发送给 LocalMapping
    // ================================================================
    auto* pKF = new mapping::KeyFrame(mCurrentFrame_->id);
    pKF->setPose(mCurrentFrame_->pose);
    pKF->setKeys(mCurrentFrame_->keypoints);
    pKF->setUndistortedKeys(mCurrentFrame_->undistorted_keypoints);
    pKF->setDescriptors(mCurrentFrame_->descriptors);
    pKF->setStereoPoints(mCurrentFrame_->pts_3d);
    pKF->setDepthValid(mCurrentFrame_->depth_valid);

    // 设置相机内参
    double fx = config_.K.at<double>(0, 0);
    double fy = config_.K.at<double>(1, 1);
    double cx = config_.K.at<double>(0, 2);
    double cy = config_.K.at<double>(1, 2);
    pKF->setCameraParams(fx, fy, cx, cy,
                        1.0 / fx, 1.0 / fy,
                        0.0, config_.max_depth * 2.0);

    // ================================================================
    // Step 2: 关联已有地图点到关键帧（复用 mpMap_ 中的唯一 MapPoint 实例）
    // ================================================================
    for (size_t i = 0; i < mvpCurrentMapPointIds_.size(); ++i) {
        int mp_id = mvpCurrentMapPointIds_[i];
        if (mp_id < 0) continue;
        mapping::MapPoint* pMP = mpMap_->getMapPoint(mp_id);
        if (pMP && !pMP->isBad()) {
            pMP->addObservation(mCurrentFrame_->id, static_cast<int>(i));
            pKF->addMapPoint(static_cast<int>(i), pMP);
        }
    }

    // ================================================================
    // Step 3: 用 RGB-D 深度为没有关联地图点的特征点创建新地图点（ORB-SLAM3 CreateNewKeyFrame 核心逻辑）
    // ORB-SLAM3: 对 RGB-D / Stereo，遍历所有特征点，
    // 对深度有效且没有关联地图点的点，用深度直接三角化（UnprojectStereo）创建新地图点。
    // ================================================================
    {
        const cv::Mat& K = config_.K;
        CameraParams cam = getCameraParams(K);
        double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

        // 按深度排序（浅的点优先，确保近处有足够多的地图点）
        std::vector<std::pair<float, size_t>> vDepthIdx;
        const size_t n_kpts = std::min(mCurrentFrame_->pts_3d.size(),
                                        mCurrentFrame_->depth_valid.size());
        vDepthIdx.reserve(n_kpts);
        for (size_t i = 0; i < n_kpts; ++i) {
            if (!mCurrentFrame_->depth_valid[i]) continue;
            const auto& p3d = mCurrentFrame_->pts_3d[i];
            if (p3d.norm() < 1e-6) continue;
            vDepthIdx.emplace_back(static_cast<float>(p3d.norm()), i);
        }
        if (!vDepthIdx.empty()) {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int n_points = 0;
            // 每个关键帧最多新建的地图点数（可配置，默认 300）。
            // 原值硬编码 100：vDepthIdx 已按深度升序排好，配额一满就直接截断，
            // 远处的强结构点（墙面/桌沿）永远拿不到名额，视野边缘长期没有 3D 点。
            // 上限本身是必要的（否则每帧能建 ~2000 个点，地图膨胀到百万级）。
            const int max_points = std::max(1, config_.max_new_map_points_per_kf);
            for (const auto& dp : vDepthIdx) {
                if (n_points >= max_points) break;
                size_t i = dp.second;

                // 已有地图点的特征点跳过；命中坏点则清槽位重建（否则坏点永久占位）
                if (i < mvpCurrentMapPointIds_.size() && mvpCurrentMapPointIds_[i] >= 0) {
                    const MapPointCandidate* existing = getMapPoint(mvpCurrentMapPointIds_[i]);
                    if (existing && !existing->is_bad) {
                        n_points++;
                        continue;
                    }
                    mvpCurrentMapPointIds_[i] = -1;
                }

                // 深度已经按升序遍历，剩下的点只会更远；配额在循环开头已经检查过。

                // 用深度图反投影得到相机坐标系 3D 点
                const auto& p3d_cam = mCurrentFrame_->pts_3d[i];
                Eigen::Vector3d p3d_world = mCurrentFrame_->pts_3d_world.empty()
                    ? Eigen::Vector3d(0, 0, 0)  // fallback（正常应该有值）
                    : mCurrentFrame_->pts_3d_world[i];
                if (p3d_world.norm() < 1e-6) {
                    p3d_world = camToWorldPoint(p3d_cam, mCurrentFrame_->pose);
                }

                // 获取描述子
                cv::Mat desc;
                if (i < static_cast<size_t>(mCurrentFrame_->descriptors.rows))
                    desc = mCurrentFrame_->descriptors.row(static_cast<int>(i));

                // 创建前端地图点
                int mp_id = createMapPoint(p3d_world, desc, mCurrentFrame_->id, static_cast<int>(i), 1);
                if (mp_id < 0) continue;  // 3D 点非法，createMapPoint 已拒绝
                mvpCurrentMapPointIds_[i] = mp_id;

                // 创建或复用唯一后端地图点并关联到关键帧
                mapping::MapPoint* pNewMP = mpMap_->getMapPoint(mp_id);
                if (!pNewMP) {
                    pNewMP = new mapping::MapPoint(mp_id, p3d_world);
                    if (!desc.empty())
                        pNewMP->setDescriptor(desc);
                    pNewMP->setReferenceKeyFrame(mCurrentFrame_->id);
                    pNewMP->setFirstKFid(mCurrentFrame_->id);
                    mpMap_->addMapPoint(pNewMP);
                }
                pNewMP->setWorldPos(p3d_world);
                pNewMP->addObservation(mCurrentFrame_->id, static_cast<int>(i));
                pKF->addMapPoint(static_cast<int>(i), pNewMP);

                n_points++;
            }

            if (config_.verbose_diag && n_points > 0) {
                std::cerr << "[CreateKF] f=" << mCurrentFrame_->id
                          << " new_depth_pts=" << n_points << std::endl;
            }
        }
    }

    // 添加到地图
    mpMap_->addKeyFrame(pKF);

    // ★★★ 注册到 BoW 数据库（用于重定位）★★★
    if (bow_adapter_ && bow_adapter_->isLoaded()) {
        bow_adapter_->addKeyFrame(mCurrentFrame_->id, mCurrentFrame_->descriptors);
    }

    // 发送给 LocalMapping 线程处理
    mpLocalMapper_->insertKeyFrame(pKF);

    // ================================================================
    // Step 3: 前端三角化和清理（临时保留，后续由 LocalMapping 接管）
    // ================================================================
    triangulateNewMapPoints();
    cleanOldMapPoints();
    // E6：把上一步登记的坏点真正从地图里删掉。放在 registerKeyFrameForLoopDetection
    // 之前 —— 回环登记会读 map_point_ids，先退役可以让它看到干净的 id 集合，
    // 而不是一批已经判坏、马上要消失的点。
    retireBadMapPoints();

    // ================================================================
    // Step 4: 登记到回环检测库 + 位姿图
    // ================================================================
    // 必须放在三角化之后：三角化会往 mAllMapPoints_ 里补点，登记时点上更全。
    // 未启用回环（loop_detector_ == nullptr，即纯 VO）时这是个空调用。
    registerKeyFrameForLoopDetection(new_kf);
}

// ================================================================
// 三角化新地图点
// ================================================================
void Tracking::triangulateNewMapPoints() {
    if (!mLastFrame_) return;

    const auto& last_kpts = (!mLastFrame_->undistorted_keypoints.empty())
        ? mLastFrame_->undistorted_keypoints : mLastFrame_->keypoints;
    const auto& curr_kpts = (!mCurrentFrame_->undistorted_keypoints.empty())
        ? mCurrentFrame_->undistorted_keypoints : mCurrentFrame_->keypoints;

    const Pose& Tcw = mCurrentFrame_->pose;
    const Pose& Tlw = mLastFrame_->pose;

    const cv::Mat& K = config_.K;
    CameraParams cam = getCameraParams(K);
    double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    for (size_t i = 0; i < mLastFrame_->pts_3d.size(); ++i) {
        if (!mLastFrame_->depth_valid[i]) continue;
        if (i >= mLastFrameMapPointIds_.size()) continue;
        if (mLastFrameMapPointIds_[i] >= 0) continue;  // 已有地图点

        const auto& p3d_last_cam = mLastFrame_->pts_3d[i];
        if (p3d_last_cam.norm() < 1e-6) continue;
        if (i >= last_kpts.size()) continue;

        Eigen::Vector3d p3d_world = camToWorldPoint(p3d_last_cam, Tlw);

        // 投影到当前帧找对应点
        Eigen::Vector3d pt_cam_curr = worldToCam(p3d_world, Tcw);
        // ★★★ Bug Fix: NaN检查防止无效深度 ★★★
        double z_curr = pt_cam_curr.z();
        if (z_curr < 0.05 || !std::isfinite(z_curr)) continue;

        double invz = 1.0 / z_curr;
        float u_proj = static_cast<float>(fx * pt_cam_curr.x() * invz + cx);
        float v_proj = static_cast<float>(fy * pt_cam_curr.y() * invz + cy);

        float best_dist = 50.0f;
        int best_k = -1;
        for (size_t k = 0; k < curr_kpts.size(); ++k) {
            if (mvpCurrentMapPointIds_[k] >= 0) continue;  // 已有地图点
            float du = curr_kpts[k].pt.x - u_proj;
            float dv = curr_kpts[k].pt.y - v_proj;
            float d = du * du + dv * dv;
            if (d < best_dist * best_dist) { best_dist = std::sqrt(d); best_k = static_cast<int>(k); }
        }

            if (best_k >= 0 && best_dist < 50.0f) {
                // 描述子验证
                if (i < static_cast<size_t>(mLastFrame_->descriptors.rows) &&
                    best_k < static_cast<int>(mvpCurrentMapPointIds_.size())) {
                    if (!mLastFrame_->descriptors.empty() && !mCurrentFrame_->descriptors.empty()) {
                        int d = descriptorDistance(mLastFrame_->descriptors.row(static_cast<int>(i)),
                            mCurrentFrame_->descriptors.row(best_k));
                        if (d > 100) continue;
                    }
                }
                cv::Mat desc;
                if (best_k < mCurrentFrame_->descriptors.rows)
                    desc = mCurrentFrame_->descriptors.row(best_k);
                int mp_id = createMapPoint(p3d_world, desc, mCurrentFrame_->id, best_k, 2);
                if (mp_id < 0) continue;  // 3D 点非法，createMapPoint 已拒绝
                if (best_k < static_cast<int>(mvpCurrentMapPointIds_.size()))
                    mvpCurrentMapPointIds_[best_k] = mp_id;
            }
    }
}

// ================================================================
// 清理过期/低质量地图点
//
// R5 重写：只处理"观察期内的新点"（mNewMapPointIds_），不再每建一个关键帧
// 就 O(N) 扫全图。判定规则：
//   1. 已被别的帧匹配到（frontend_observation_count >= 2）→ 确认，移出观察期
//   2. 观察期已过（建点帧到当前帧 >= config_.map_point_grace_frames）且仍只有
//      1 次观测 → 标 is_bad（这就是"单次观测质量过低"的原意）
//   3. 观察期未过 → 继续留在观察期，下一轮再看
// 旧实现的问题在于新建点在同一轮 createNewKeyFrame 里就被判坏：它们当然
// 还没机会被匹配，于是第一个关键帧就把整张前端地图清空。
// ================================================================
void Tracking::cleanOldMapPoints() {
    if (mNewMapPointIds_.empty()) return;
    const int current_frame_id = mCurrentFrame_ ? mCurrentFrame_->id
                                               : (mLastFrame_ ? mLastFrame_->id : -1);
    size_t keep = 0;
    for (size_t i = 0; i < mNewMapPointIds_.size(); ++i) {
        const int id = mNewMapPointIds_[i];
        auto it = mAllMapPoints_.find(id);
        if (it == mAllMapPoints_.end()) continue;  // 点已不存在，直接丢弃
        MapPointCandidate& cand = it->second;
        if (cand.is_bad) continue;                 // 已被判坏，不再占用观察期
        if (cand.frontend_observation_count >= 2) continue;  // 已确认，移出观察期
        const int age = (cand.created_frame_id >= 0 && current_frame_id >= 0)
                      ? (current_frame_id - cand.created_frame_id) : 0;
        // 观察期长度来自 config_.map_point_grace_frames（默认 8 帧）。
        // 原来是硬编码常量 3：关键帧建点时 obs_count=1，只有 3 帧时间去等第二次观测，
        // 而 3 帧内 searchLocalPoints 往往还没轮到它 —— 于是整批新点刚建好就被判死，
        // 地图永远长不起来，跟踪只能靠越来越少的旧点。
        const int grace = std::max(1, config_.map_point_grace_frames);
        if (age < grace) {
            mNewMapPointIds_[keep++] = id;  // 仍在观察期
            continue;
        }
        cand.is_bad = true;
        // E6-2：登记下来交给 retireBadMapPoints() 真正 erase。
        // 不能就地 erase —— 上面刚拿到的 cand 引用会失效，而且本帧流程里
        // 可能还有别处持有该点的 MapPointCandidate*。
        mBadMapPointIds_.push_back(id);
    }
    mNewMapPointIds_.resize(keep);
}

// ================================================================
// E6-2：真正回收坏点
//
// 只标记 is_bad 等于把地图变成只增不减的日志：fr1_desk2 跑 640 帧能堆到
// 六位数个点，投影匹配与全局搜索的预算全被死点吃掉。
// 这里把 cleanOldMapPoints() 登记的坏点从 mAllMapPoints_ 删掉，并把所有
// 引用它的 id 槽位一起清掉（否则下一帧会拿着悬空 id 去 getMapPoint）。
//
// 调用时机约束：必须在单帧流程末尾、没有任何 MapPointCandidate* 还被持有
// 的时点调用，否则那些裸指针会指向已 erase 的节点。
// ================================================================
void Tracking::retireBadMapPoints() {
    if (mBadMapPointIds_.empty()) return;

    for (int id : mBadMapPointIds_) {
        auto it = mAllMapPoints_.find(id);
        if (it != mAllMapPoints_.end()) {
            removeFromVoxelIndex(id, it->second.world_point);
            mAllMapPoints_.erase(it);
        }

        // 清掉所有指向这个 id 的槽位
        for (int& slot : mvpCurrentMapPointIds_) {
            if (slot == id) slot = -1;
        }
        for (int& slot : mLastFrameMapPointIds_) {
            if (slot == id) slot = -1;
        }
        for (KeyFrameCacheEntry& kf : mvpLocalKeyFrames_) {
            kf.map_point_ids.erase(id);
        }
        for (auto& kv : mAllKeyFrames_) {
            kv.second.map_point_ids.erase(id);
        }
        // 后端 mapping::MapPoint 是另一套容器（mpMap_ 持有所有权），
        // 这里不动它：前端 id 与后端 id 取值相同但生命周期独立，
        // 后端由 LocalMapper 自己决定何时剔除。
    }
    mBadMapPointIds_.clear();
}

// ================================================================
// 获取确认的地图点
// ================================================================
std::vector<Eigen::Vector3d> Tracking::getConfirmedMapPoints() const {
    std::vector<Eigen::Vector3d> result;
    for (const auto& kv : mAllMapPoints_) {
        if (kv.second.is_bad) continue;
        if (kv.second.confirmed && kv.second.world_point.norm() > 1e-6)
            result.push_back(kv.second.world_point);
    }
    return result;
}

// ================================================================
// 接口实现
// ================================================================
std::vector<cv::KeyPoint> Tracking::getCurrentKeypoints() const {
    return mCurrentFrame_ ? mCurrentFrame_->keypoints : std::vector<cv::KeyPoint>{};
}

std::vector<cv::KeyPoint> Tracking::getCurrentUndistortedKeypoints() const {
    if (!mCurrentFrame_) return {};
    return mCurrentFrame_->undistorted_keypoints.empty()
        ? mCurrentFrame_->keypoints
        : mCurrentFrame_->undistorted_keypoints;
}

void Tracking::initLoopDetection(
        const loopdetection::LoopDetectorConfig& loop_cfg,
        const loopdetection::PoseGraphOptimizerOptions& pgo_cfg) {
    // 内参必须显式塞进 LoopDetector：config_loader 的 loadLoopDetectorConfig
    // 不读 camera_K / camera_D（它们不是 yaml 字段），漏了这一步几何校验会退化成
    // 用单位阵当内参，PnP 解出来的相对位姿没有物理意义。
    loopdetection::LoopDetectorConfig cfg = loop_cfg;
    cfg.camera_K = config_.K;
    cfg.camera_D = config_.D;
    // D4-d：检测节流来自配置，默认 1（每个关键帧）
    mnLoopCheckIntervalKF_ = std::max(1, cfg.loop_check_interval_kf);

    loop_detector_ = std::make_unique<loopdetection::LoopDetector>(cfg);
    pose_graph_optimizer_ = std::make_unique<loopdetection::PoseGraphOptimizer>(pgo_cfg);

    // 之前建过的关键帧要补登记，否则开启 SLAM 的头几十帧里回环库是空的，
    // min_loop_separation 一直不满足，等于白跑。
    // 必须按 id 升序登记：LoopDetector::detect() 用 keyframe_ids_.back() 当作"当前帧"，
    // 乱序登记会让它把某个旧帧当成当前帧去比，min_loop_separation 也跟着算错。
    std::vector<int> kf_ids;
    kf_ids.reserve(mAllKeyFrames_.size());
    for (const auto& kv : mAllKeyFrames_) kf_ids.push_back(kv.first);
    std::sort(kf_ids.begin(), kf_ids.end());

    mnLastPGOKeyFrameId_ = -1;
    for (int kf_id : kf_ids) {
        const KeyFrameCacheEntry& kf = mAllKeyFrames_.at(kf_id);
        loop_detector_->addKeyFrame(kf.frame_id, kf.keypoints, kf.pts_3d,
                                    kf.descriptors, kf.world_pose);
        pose_graph_optimizer_->addOrUpdateVertex(kf.frame_id, kf.world_pose);
        mnLastPGOKeyFrameId_ = kf.frame_id;
    }
    // 补登记的关键帧之间补里程计边：位姿图上相邻顶点之间没有约束时，
    // PGO 会把它们当作相互独立的量，回环一拉整段就散架。
    for (size_t i = 1; i < kf_ids.size(); ++i) {
        const Pose& pa = mAllKeyFrames_.at(kf_ids[i - 1]).world_pose;
        const Pose& pb = mAllKeyFrames_.at(kf_ids[i]).world_pose;
        pose_graph_optimizer_->addOdometryEdge(kf_ids[i - 1], kf_ids[i],
                                               pa.toMatrix().inverse() * pb.toMatrix());
    }

    mnLastLoopCheckKFId_ = -1;
    mnDetectedLoops_ = 0;

    if (config_.verbose_diag) {
        std::cerr << "[LoopDetection] 已启用：已登记关键帧=" << mAllKeyFrames_.size()
                  << " min_loop_separation=" << cfg.min_loop_separation
                  << " min_geometric_inliers=" << cfg.min_geometric_inliers
                  << " pgo_max_iterations=" << pgo_cfg.max_iterations << std::endl;
    }
}

int Tracking::getNumLoopEdges() const {
    return pose_graph_optimizer_ ? pose_graph_optimizer_->numLoopEdges() : 0;
}

void Tracking::registerKeyFrameForLoopDetection(const KeyFrameCacheEntry& kf) {
    if (!loop_detector_ || !pose_graph_optimizer_) return;

    // pts_3d 是相机系 3D 点（LoopDetector 内部会用自己的 pose 转世界），
    // 这里必须传 kf.pts_3d，不能传 kf.pts_3d_world，否则同一批点会被变换两次。
    loop_detector_->addKeyFrame(kf.frame_id, kf.keypoints, kf.pts_3d,
                                kf.descriptors, kf.world_pose);

    pose_graph_optimizer_->addOrUpdateVertex(kf.frame_id, kf.world_pose);

    // 与上一个关键帧连里程计边。addOdometryEdge 不会自动建顶点，
    // 顶点缺失时它只打一行拒绝日志就返回（见 pose_graph_optimizer.cpp:620），
    // 所以上面的 addOrUpdateVertex 必须在前。
    if (mnLastPGOKeyFrameId_ >= 0 && mnLastPGOKeyFrameId_ != kf.frame_id) {
        auto prev_it = mAllKeyFrames_.find(mnLastPGOKeyFrameId_);
        if (prev_it != mAllKeyFrames_.end()) {
            Eigen::Matrix4d T_prev = prev_it->second.world_pose.toMatrix();
            Eigen::Matrix4d T_cur = kf.world_pose.toMatrix();
            // 与 CovisManager::computeRelativePose 同一约定：T_prev→cur = T_w_prev^-1 * T_w_cur
            pose_graph_optimizer_->addOdometryEdge(mnLastPGOKeyFrameId_, kf.frame_id,
                                                   T_prev.inverse() * T_cur);
        }
    }
    mnLastPGOKeyFrameId_ = kf.frame_id;
}

bool Tracking::detectAndCorrectLoop() {
    if (!loop_detector_ || !pose_graph_optimizer_ || !mCurrentFrame_) return false;

    auto kf_it = mAllKeyFrames_.find(static_cast<int>(mCurrentFrame_->id));
    if (kf_it == mAllKeyFrames_.end()) return false;
    const KeyFrameCacheEntry& kf = kf_it->second;

    // 节流：每个关键帧只查一次。同一 KF 重复进来（例如外部重复调用）直接跳过，
    // 否则同一帧会被反复加回环边，位姿图里堆出一串重复约束。
    if (mnLastLoopCheckKFId_ == kf.frame_id) return false;
    // D4-d：按关键帧间隔节流（默认 1 = 不节流，与改动前行为一致）。
    // 回环检测比建关键帧贵得多：一次 detect() 要做 BoW 查询 + 几何验证 + Sim3/ICP，
    // 而 PGO 是全局优化。此前唯一的闸门是"同一 KF 不重复进来"，等于每个关键帧都全跑一遍。
    if (mnLoopCheckIntervalKF_ > 1 && mnLastLoopCheckKFId_ >= 0 &&
        kf.frame_id - mnLastLoopCheckKFId_ < mnLoopCheckIntervalKF_) {
        return false;
    }
    mnLastLoopCheckKFId_ = kf.frame_id;

    // L2：这里传 nullptr 是**有意为之**，不再是"忘记传"。
    // 传 nullptr 意味着不用 CovisManager 的 confirmed map point 做阶段 1，
    // 但 LoopDetector::geometricVerification() 的阶段 2（描述子 matches +
    // keyframe_db_ 里自己存的世界坐标点）已经提到守卫外，所以 covis_matches
    // 依然能构造出对应点。原实现把阶段 2 也写在 if (covis_mgr != nullptr) 里，
    // 于是这条路径上 covis_matches 恒空、Sim3 永不进入。
    const auto candidate = loop_detector_->detect(kf.keypoints, kf.pts_3d,
                                                  kf.descriptors, nullptr);
    if (!candidate.valid) return false;

    if (config_.verbose_diag) {
        std::cerr << "[Loop] 候选 KF=" << candidate.current_kf_id
                  << " ↔ " << candidate.matched_kf_id
                  << " inliers=" << candidate.geometric_inliers
                  << " ratio=" << candidate.inlier_ratio
                  << " raw=" << candidate.raw_matches
                  << (candidate.is_icp ? " [ICP]" : " [EM]") << std::endl;
    }

    // L4：addLoopEdge 现在返回 bool —— false 表示这条约束被拒绝或去重（顶点缺失、
    // 尺度/正交性校验失败、同一对关键帧已有边、addEdge 失败）。此时图结构没变，
    // 继续 optimize() 只是把旧图再解一遍，还会让 mnDetectedLoops_ 与 loop_edge_data_
    // 脱节（日志显示"检测到回环"，图里根本没有这条边）。
    const bool edge_added = pose_graph_optimizer_->addLoopEdge(candidate.current_kf_id,
                                                               candidate.matched_kf_id,
                                                               candidate.relative_pose,
                                                               1.0,
                                                               candidate.geometric_inliers,
                                                               candidate.raw_matches,
                                                               candidate.is_icp);
    if (!edge_added) {
        if (config_.verbose_diag) {
            std::cerr << "[Loop] 回环边未被采纳（重复或校验失败），跳过本次 PGO：KF="
                      << candidate.current_kf_id << " ↔ " << candidate.matched_kf_id << std::endl;
        }
        return false;
    }

    int iter = 0;
    double pgo_chi2 = 0.0;
    try {
        // D4-b：chi2 是判断"约束是否真的参与求解"的唯一直接证据。
        // 曾经出现过 odom 边被 setLevel(1) 整类移出优化、图上只剩 1 条回环边、
        // chi2 恒 0 却报优化成功的情况；只看迭代次数看不出来。
        iter = pose_graph_optimizer_->optimize(&pgo_chi2, nullptr);
    } catch (const std::exception& e) {
        // 错误日志不受 verbose_diag 控制（R-021）：PGO 抛异常说明约束或解算出了问题
        std::cerr << "[Loop] PGO 异常：" << e.what() << "，放弃本次回环" << std::endl;
        return false;
    }
    if (iter <= 0) {
        std::cerr << "[Loop] PGO 未收敛（iterations=" << iter << "），放弃本次回环" << std::endl;
        return false;
    }

    const int n_corrected = applyOptimizedPosesToFrontend();
    if (n_corrected <= 0) return false;

    mnDetectedLoops_++;
    std::cerr << "[Loop] 回环修正完成：KF=" << candidate.current_kf_id
              << " ↔ " << candidate.matched_kf_id
              << " iterations=" << iter
              << " chi2=" << pgo_chi2
              << " 修正关键帧=" << n_corrected
              << " 累计回环=" << mnDetectedLoops_ << std::endl;
    return true;
}

int Tracking::applyOptimizedPosesToFrontend() {
    if (!pose_graph_optimizer_) return -1;

    std::unordered_map<int, Pose> optimized;
    pose_graph_optimizer_->getAllOptimizedPoses(&optimized);
    if (optimized.empty()) return 0;

    // 先算位移上限。PGO 是全局优化，几十厘米到几米的修正都正常；
    // 但把某个关键帧搬过整个场景（> LOOP_MAX_KF_CORRECTION_M）说明回环边本身错了
    // ——宁可整次丢弃，也不要让一条错边改写整条轨迹。
    double max_shift = 0.0;
    int max_shift_kf = -1;
    for (const auto& kv : optimized) {
        auto it = mAllKeyFrames_.find(kv.first);
        if (it == mAllKeyFrames_.end()) continue;
        if (!kv.second.t.allFinite() || !kv.second.q.coeffs().allFinite()) {
            std::cerr << "[Loop] PGO 输出含 NaN/Inf：KF=" << kv.first << "，放弃本次修正" << std::endl;
            return -1;
        }
        const double shift = (kv.second.t - it->second.world_pose.t).norm();
        if (shift > max_shift) { max_shift = shift; max_shift_kf = kv.first; }
    }
    if (max_shift > LOOP_MAX_KF_CORRECTION_M) {
        std::cerr << "[Loop] 修正量异常：KF=" << max_shift_kf << " 位移 " << max_shift
                  << "m > 上限 " << LOOP_MAX_KF_CORRECTION_M
                  << "m，放弃本次修正（回环边可能错误）" << std::endl;
        return -1;
    }

    // D1-c 之后不再需要 delta_by_kf（原用途只有一个：搬地图点）。
    // D5-d：先把局部窗口建成 frame_id → 下标，避免对每个优化过的关键帧都线性扫一遍 deque
    // （原写法是 K 个关键帧 × deque 20 项，规模小但每次写回都白扫，且随窗口增大而变差）。
    std::unordered_map<int, size_t> local_kf_index;
    local_kf_index.reserve(mvpLocalKeyFrames_.size());
    for (size_t i = 0; i < mvpLocalKeyFrames_.size(); ++i) {
        local_kf_index.emplace(mvpLocalKeyFrames_[i].frame_id, i);
    }

    int n_corrected = 0;
    for (const auto& kv : optimized) {
        const int kf_id = kv.first;
        const Pose& new_pose = kv.second;

        auto all_it = mAllKeyFrames_.find(kf_id);
        if (all_it == mAllKeyFrames_.end()) continue;

        all_it->second.world_pose = new_pose;
        n_corrected++;

        // 局部窗口里的副本（deque）也要改，否则 searchLocalPoints 还在用旧位姿投影
        auto local_it = local_kf_index.find(kf_id);
        if (local_it != local_kf_index.end()) {
            mvpLocalKeyFrames_[local_it->second].world_pose = new_pose;
        }
        // 后端地图里的 KeyFrame 也要改，getPose() 是下游唯一读取点
        if (mpMap_) {
            if (mapping::KeyFrame* mkf = mpMap_->getKeyFrame(kf_id)) {
                mkf->setPose(new_pose);
            }
        }
        // 回环库里的历史位姿同样要同步，否则下一次回环用的还是旧基准
        if (loop_detector_) loop_detector_->updateKeyFramePose(kf_id, new_pose);
    }

    // D1-c：不搬地图点（路线 A）。
    // 原实现在这里按"点的第一个观测关键帧"取一份 delta，把地图点世界坐标批量变换一次。
    // 一个点通常被 K 个关键帧共同观测，而这些关键帧在本次 PGO 里拿到的 delta 并不相同；
    // 用其中一个的 delta 搬点，等于宣布另外 K-1 个观测从此指向错误的世界位置，
    // 地图里长出重复结构，下一轮位姿优化再被这些错点拖回去。
    // 关键帧位姿是图优化的输出，可信；地图点世界坐标只是一堆三角化结果，不可信。
    // 所以回环只写回位姿，地图点留给后续 poseOptimization / BA 在新位姿下重新收敛。
    // 短期重投影误差会升高，判断标准是 ATE（PLAN 风险节已声明）。

    // 当前帧本身就是刚建的关键帧，位姿要跟优化结果对齐；
    // mLastFrame_ 施加同一增量，保证下一帧的恒速预测不会因为坐标系平移而跳一下。
    if (mCurrentFrame_) {
        auto cur_it = optimized.find(static_cast<int>(mCurrentFrame_->id));
        if (cur_it != optimized.end()) {
            const Pose new_cur = cur_it->second;
            const Eigen::Matrix4d d = new_cur.toMatrix() * mCurrentFrame_->pose.toMatrix().inverse();
            mCurrentFrame_->pose = new_cur;
            // pts_3d_world 是 createNewKeyFrame() 里用旧位姿算的，位姿一改它就过期了。
            // 后面 searchLocalPoints / searchByProjection 都直接读它，必须重算。
            updateWorldPoints(mCurrentFrame_);
            if (mLastFrame_) {
                mLastFrame_->pose.q = Eigen::Quaterniond(d.block<3, 3>(0, 0)) * mLastFrame_->pose.q;
                mLastFrame_->pose.t = d.block<3, 3>(0, 0) * mLastFrame_->pose.t + d.block<3, 1>(0, 3);
                updateWorldPoints(mLastFrame_);
            }
        }
    }

    std::cerr << "[Loop] 位姿图写回：" << n_corrected << " 个关键帧（D1-c：不搬地图点，"
              << "留给后续 poseOptimization 重新收敛），最大修正 "
              << max_shift << "m" << std::endl;
    return n_corrected;
}

const std::unordered_map<int, Eigen::Vector3d>&
Tracking::getLastBAMapPointUpdates() const {
    return last_ba_map_point_updates_;
}

bool Tracking::checkKeyFrame(Frame::Ptr frame, const PnPResult* pnp_result, const Pose* smoothed_pose) {
    // 这个接口原先三个入参全部忽略、直接转发 needNewKeyFrame()，等于同一个判断有两个入口。
    // 这里把入参纳入判断：PnP 内点太少说明本帧几何支撑不足，即使运动量够也不值得建帧。
    if (!frame) return false;

    const bool motion_says_yes = needNewKeyFrame();
    if (!motion_says_yes) return false;

    if (pnp_result && pnp_result->inliers < config_.min_features) {
        return false;  // 内点不足以支撑新帧的三角化
    }
    if (smoothed_pose && (smoothed_pose->t - frame->pose.t).norm() >
                             config_.min_keyframe_translation * 2.0) {
        // 平滑位姿与原始位姿差得太多，说明本帧位姿本身不可信，先不建帧
        return false;
    }
    return true;
}

bool Tracking::checkKeyFrameSimple(Frame::Ptr frame, const PnPResult* pnp_result, const Pose* pose) {
    // 简单版只要求：位姿有限 + 内点够用。给"宁可少建帧也别建坏帧"的场景用。
    if (!frame) return false;
    if (!frame->pose.t.allFinite() || !frame->pose.q.coeffs().allFinite()) return false;
    if (pnp_result && pnp_result->inliers < config_.min_features) return false;
    if (pose && !pose->t.allFinite()) return false;
    return needNewKeyFrame();
}

std::vector<std::pair<Eigen::Vector3d, cv::Point2f>>
Tracking::searchLocalMap(const Frame::Ptr& curr_frame, const Pose& curr_world_pose) {
    std::vector<std::pair<Eigen::Vector3d, cv::Point2f>> result;
    for (const auto& kv : mAllMapPoints_) {
        if (kv.second.is_bad) continue;
        const auto& pt = kv.second.world_point;
        if (pt.norm() < 1e-6) continue;

        const cv::Mat& K = config_.K;
        CameraParams cam = getCameraParams(K);
        double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

        Eigen::Vector3d pt_cam = worldToCam(pt, curr_world_pose);
        // ★★★ Bug Fix: NaN检查防止无效深度 ★★★
        double z = pt_cam.z();
        if (z < 0.05 || !std::isfinite(z)) continue;

        double invz = 1.0 / z;
        float u_f = static_cast<float>(fx * pt_cam.x() * invz + cx);
        float v_f = static_cast<float>(fy * pt_cam.y() * invz + cy);
        result.emplace_back(pt, cv::Point2f(u_f, v_f));
    }
    return result;
}

Pose Tracking::applyGravityConstraint(const Pose& pose) const {
    // NaN 安全：若输入 pose 含 NaN 或四元数退化，直接返回单位四元数
    if (!pose.t.allFinite() || !pose.q.coeffs().allFinite() ||
        std::abs(pose.q.norm() - 1.0) > 0.1) {
        Pose safe;
        safe.q = Eigen::Quaterniond(1, 0, 0, 0);
        safe.t = pose.t;
        return safe;
    }

    Pose constrained = pose;
    Eigen::Vector3d euler = pose.q.toRotationMatrix().eulerAngles(2, 1, 0);
    double yaw = euler[0];
    constrained.q = Eigen::Quaterniond(
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()));
    return constrained;
}

// ================================================================
// 消费 LocalMapping 的 BA 结果（ORB-SLAM3 mbMapUpdated 机制）
// 在每帧跟踪开始时调用，同步 BA 更新的地图点到前端
// ================================================================
void Tracking::consumeBAUpdates() {
    if (!mpLocalMapper_) return;

    // 获取 BA 更新的地图点
    auto updated_pts = mpLocalMapper_->getLastUpdatedPoints();

    // 语义是"本轮 BA 改了哪些点"，不是"历史上改过的所有点"：
    // 每轮先清空，否则长序列跑下来这张表只增不减，既占内存又让调用方
    // 分不清哪些是本轮更新（R-022：字段必须名副其实）。
    last_ba_map_point_updates_.clear();

    int n_updated = 0;
    for (const auto& [mp_id, world_pt] : updated_pts) {
        auto it = mAllMapPoints_.find(mp_id);
        // NaN 安全：丢弃 BA 返回的 NaN 地图点
        if (it != mAllMapPoints_.end() && !it->second.is_bad &&
            world_pt.hasNaN() == false && world_pt.norm() > 1e-6) {
            it->second.world_point = world_pt;
            // 同时留一份副本给外部（getLastBAMapPointUpdates）：调用方要的是"这一轮
            // BA 改了哪些点"，直接暴露 mAllMapPoints_ 会让调用方看到全量地图。
            last_ba_map_point_updates_[mp_id] = world_pt;
            n_updated++;
        }
    }

    // 获取 BA 优化的关键帧 ID（用于更新局部地图）
    auto opt_kfs = mpLocalMapper_->getLastOptimizedKFIds();
    if (!opt_kfs.empty()) {
        mbMapUpdatedByBA_ = true;
    }

    if (config_.verbose_diag && n_updated > 0)
        std::cerr << "[BA Update] " << n_updated << " points updated by Local BA" << std::endl;
}

// ================================================================
// 析构函数：关闭 LocalMapping 线程
// ================================================================
Tracking::~Tracking() {
    if (mpLocalMapper_) {
        mpLocalMapper_->shutDown();
    }
}

// ================================================================
// 设置 BoW Vocabulary（ORB-SLAM3 风格）
// ================================================================
void Tracking::setBoWVocabulary(const std::string& voc_path) {
    if (bow_adapter_) {
        feature::BoWConfig config;
        config.vocabulary_path = voc_path;
        bow_adapter_ = std::make_unique<feature::BoWAdapter>(config);
        std::cout << "[Tracking] BoW vocabulary loaded: " << (bow_adapter_->isLoaded() ? "success" : "failed") << std::endl;
    }
}

// ================================================================
// ORB-SLAM3 BoW 候选关键帧搜索
// 用当前帧的 BoW 描述向量查询数据库，返回相似度最高的 max_candidates 个关键帧
// ================================================================
std::vector<KeyFrameCacheEntry> Tracking::getBoWCandidateKeyFrames(int max_candidates) const {
    std::vector<KeyFrameCacheEntry> candidates;
    if (!bow_adapter_ || !bow_adapter_->isLoaded()) return candidates;
    if (!mCurrentFrame_ || mCurrentFrame_->descriptors.empty()) return candidates;

    // BoW 查询相似关键帧
    auto bow_candidates = bow_adapter_->queryCandidates(mCurrentFrame_->descriptors, 0.03);
    if (bow_candidates.empty()) return candidates;

    // 构建 local KFs 的 frame_id 集合（跳过已在 local KFs 中的候选）
    std::unordered_set<int> local_kf_ids;
    for (const auto& kf : mvpLocalKeyFrames_) {
        local_kf_ids.insert(kf.frame_id);
    }

    // 按相似度排序
    std::sort(bow_candidates.begin(), bow_candidates.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& cand : bow_candidates) {
        int cand_id = cand.first;
        // 跳过已经在 local KFs 中的（避免重复）
        if (local_kf_ids.count(cand_id)) continue;
        // 从全局关键帧缓存中查找
        auto it = mAllKeyFrames_.find(cand_id);
        if (it != mAllKeyFrames_.end()) {
            candidates.push_back(it->second);
            if ((int)candidates.size() >= max_candidates) break;
        }
    }
    return candidates;
}

} // namespace frontend
} // namespace sfc_core