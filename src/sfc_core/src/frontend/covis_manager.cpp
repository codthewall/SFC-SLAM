// ================================================================
//  第三方库（g2o / Eigen / OpenCV）必须在命名空间外部引入
//  原因：g2o CMake 注入 -Dsfc_core=1 宏，导致
//  sfc_core::std 被编译器误解析为 ::std 以外的东西。
//  解决方案：在自定义命名空间外部包含所有第三方头文件。
// ================================================================

// 强制在最前引入标准库（确保 std::* 在 g2o 污染前已解析）
#include <algorithm>
#include <climits>
#include <cmath>
#include <exception>
#include <iostream>
#include <iomanip>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// g2o 头文件（在自定义命名空间外部引入，引入后主动清除宏）
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

// SFC-SLAM 自定义边类（绕过 JacobianWorkspace 的直接雅可比计算）
#include "sfc_core/mapping/optimizer.hpp"

// 清除 g2o 注入的 sfc_core 宏，防止命名空间污染
#ifdef sfc_core
#undef sfc_core
#endif

// 项目自身头文件（g2o 已被干净引入，现在可以安全地包含项目头文件）
#include "sfc_core/frontend/covis_manager.hpp"
#include "sfc_core/loopdetection/loop_detector.hpp"
#include "sfc_core/loopdetection/pose_graph_optimizer.hpp"

#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

namespace sfc_core {
namespace frontend {

// 量化常数：1m / 5cm = 20.0（与 CovisManager::quantizeWorldPointKey 保持一致）
constexpr double QUANTIZE_SCALE = 20.0;

// ★★★ 优化 1: 提取量化格网哈希计算为公共静态方法 ★★★
// 消除多处重复的量化代码，提升可维护性
// Local helper for world-point quantization (takes scale parameter)
// NOTE: Renamed to avoid shadowing CovisManager::quantizeWorldPointKey
inline uint64_t quantizeWorldPointKey(const Eigen::Vector3d& pt_world, double quantize_scale) {
    Eigen::Vector3i q = (pt_world.array() * quantize_scale).round().cast<int>();
    return (static_cast<uint64_t>(q.x() + 1048576) & 0x1FFFFFFULL) |
           ((static_cast<uint64_t>(q.y() + 1048576) & 0x1FFFFFFULL) << 21) |
           ((static_cast<uint64_t>(q.z() + 1048576) & 0x1FFFFFFULL) << 42);
}

// Helper to count distinct keyframe IDs from observations.
// Uses sort+unique for O(N log N) with no dynamic memory allocation.
inline int countDistinctKeyFrames(const std::vector<MapPointObservation>& observations,
                                  std::vector<int>& distinct_ids) {
    // 先排序再唯一化：O(N log N) 但无内存分配，比每次 insert O(log N) 快
    distinct_ids.clear();
    distinct_ids.reserve(observations.size());
    for (const auto& obs : observations) {
        distinct_ids.push_back(obs.frame_id);
    }
    std::sort(distinct_ids.begin(), distinct_ids.end());
    auto it = std::unique(distinct_ids.begin(), distinct_ids.end());
    return static_cast<int>(it - distinct_ids.begin());
}

// ================================================================
// 位姿约定（R1）
// Pose{q, t} 在全系统表示 T_wc（P_world = R_wc * P_cam + t_wc）。
// g2o 的 VertexSE3Expmap + EdgeSE3ProjectXYZ 用的是 T_cw。
// 这里提供一对换算函数，局部 BA 在顶点进出两个方向各用一次。
// ================================================================
inline g2o::SE3Quat poseWcToSe3Quat(const Pose& t_wc) {
    Eigen::Matrix3d R_wc = t_wc.q.toRotationMatrix();
    Eigen::Matrix3d R_cw = R_wc.transpose();
    Eigen::Vector3d t_cw = -(R_cw * t_wc.t);
    return g2o::SE3Quat(R_cw, t_cw);
}

inline Pose se3QuatToPoseWc(const g2o::SE3Quat& t_cw) {
    Eigen::Matrix3d R_cw = t_cw.rotation().toRotationMatrix();
    Eigen::Matrix3d R_wc = R_cw.transpose();
    Eigen::Vector3d t_wc = -(R_wc * t_cw.translation());
    Pose out;
    out.q = Eigen::Quaterniond(R_wc);
    out.q.normalize();
    out.t = t_wc;
    return out;
}

CovisManager::CovisManager(const CovisConfig& config) : config_(config) {}

bool CovisManager::isSamePhysicalPoint(
        const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
    return (a - b).norm() < config_.same_point_distance_threshold;
}

std::vector<std::pair<Eigen::Vector3d, cv::Point2f>>
CovisManager::getConfirmedMapPointsForPnP(
        const std::vector<cv::KeyPoint>& curr_kps,
        const std::vector<cv::DMatch>& inlier_matches) const {

    std::vector<std::pair<Eigen::Vector3d, cv::Point2f>> result;
    if (keyframe_window_.empty()) return result;

    int last_kf_id = keyframe_window_.back()->frame_id;
    for (const auto& m : inlier_matches) {
        int query_idx = m.queryIdx;
        int train_idx = m.trainIdx;
        auto it = obs_key_to_confirmed_mp_.find({last_kf_id, query_idx});
        if (it == obs_key_to_confirmed_mp_.end()) continue;

        int mp_id = it->second;
        auto cit = confirmed_mappoints_.find(mp_id);
        if (cit == confirmed_mappoints_.end()) continue;
        if (cit->second->worldPoint().norm() < 1e-6) continue;

        if (train_idx < 0 || train_idx >= static_cast<int>(curr_kps.size())) continue;
        result.emplace_back(cit->second->worldPoint(), curr_kps[train_idx].pt);
    }
    return result;
}

void CovisManager::addKeyFrame(
        const std::vector<cv::KeyPoint>& kps,
        const std::vector<cv::KeyPoint>& undist_kps,
        const cv::Mat& desc,
        const std::vector<Eigen::Vector3d>& pts3d_curr,
        const std::vector<bool>& depth_valid_curr,
        const std::vector<cv::DMatch>& matches_to_prev,
        const std::vector<cv::KeyPoint>& prev_kps,
        const std::vector<cv::KeyPoint>& prev_undist_kps,
        const std::vector<Eigen::Vector3d>& pts3d_prev,
        const std::vector<bool>& depth_valid_prev,
        const cv::Mat& prev_desc,                    // ★★★ Bug Fix 新增：前一帧描述子 ★★★
        int prev_frame_id,                           // ★★★ Bug Fix 新增：前一帧 ID ★★★
        const Pose& prev_frame_pose_world,
        const Pose& kf_pose,
        int inlier_match_count,
        const std::vector<KeyFrameCacheEntry>& kf_cache_ref,
        const MultiKFMatchesByCache& matches_from_cache,
        int tracking_frame_id) {

    running_frames_++;

    // ★★★ Bug Fix #6: 使用 Tracking 传入的 frame_id（而非内部 running_frames_）★★★
    // 这样 kf_data_cache_ 的 global_id 与 getAllKeyFramePosesSnapshot() 返回的 key 一致，
    // BA 同步时可以正确匹配。
    int current_kf_id;
    if (tracking_frame_id >= 0) {
        current_kf_id = tracking_frame_id;
    } else {
        current_kf_id = running_frames_;
    }
    if (loop_protection_frames_ > 0) {
        loop_protection_frames_--;
        if (loop_protection_frames_ == 0) {
            protected_candidate_ids_.clear();
            protected_confirmed_ids_.clear();
        }
    }

    auto kf = KeyFrame::create(current_kf_id, 0.0, kps, undist_kps, desc,
                              pts3d_curr, depth_valid_curr, kf_pose);
    kf->buildSpatialIndex();  // ★★★ Bug Fix B2: 新 KeyFrame 必须构建空间格网索引，否则 reprojection 匹配中的二分搜索失效 ★★★
    // ★★★ 新关键帧加入时立即缓存其初始位姿（PNP 结果）
    // pnp_original_poses_cache_ 永远存原始 PnP，不被 PGO 覆盖
    if (!kf_pose.q.coeffs().hasNaN()) {
        pnp_original_poses_cache_[current_kf_id] = kf_pose;
    }
    
    // ★★★ Bug Fix: 缓存大小限制，防止无限增长 ★★★
    if (config_.enable_cache_cleanup) {
        // 清理过大的缓存（保留最近的max_cache_size个）
        auto cleanCache = [](auto& cache, int max_size) {
            if (static_cast<int>(cache.size()) > max_size) {
                // 保留最新的max_size个，删除最旧的
                // 对于unordered_map，我们保留最近插入的max_size个
                // 由于key是递增的frame_id，删除最小的key
                int to_remove = cache.size() - max_size;
                for (int i = 0; i < to_remove; ++i) {
                    int min_key = INT_MAX;
                    for (const auto& kv : cache) {
                        if (kv.first < min_key) min_key = kv.first;
                    }
                    if (min_key != INT_MAX) cache.erase(min_key);
                }
            }
        };
        cleanCache(pnp_original_poses_cache_, config_.max_cache_size);
        cleanCache(pgo_optimized_poses_cache_, config_.max_cache_size);
    }
    
    keyframe_window_.push_back(kf);
    kf_id_to_window_idx_[current_kf_id] = keyframe_window_.size() - 1;
    while (static_cast<int>(keyframe_window_.size()) > config_.window_size) {
        // ★★★ 淘汰前先缓存旧关键帧的最终位姿（PGO 顶点仍存在于图中，但 window 已丢失）
        // 用于 getAllOptimizedKeyFramePoses() 返回所有历史关键帧的修正后位姿
        const auto& old_kf = keyframe_window_.front();
        if (pose_graph_optimizer_ && old_kf) {
            // ★★★ 关键修复：只有 enable_pgo=true 时才用 PGO 位姿覆盖原始 PnP 位姿
            // enable_pgo=false 时，pgo_optimized_poses_cache_ 在 addKeyFrame 时写入的是
            // 原始 PnP 位姿，淘汰时不应再被 PGO 位姿覆盖（否则缓存被污染，回写轨迹时尺度爆炸）
            if (config_.enable_pgo) {
                Pose p;
                if (pose_graph_optimizer_->getOptimizedPose(old_kf->frame_id, &p)) {
                    pgo_optimized_poses_cache_[old_kf->frame_id] = p;
                }
            }
        }
        // 新增：淘汰前保存旧关键帧的确认点关联（用于回环时回填共视确认点）
        for (const auto& kv : old_kf->kpt_to_mappoint) {
            int kpt_idx = kv.first;
            int mp_id = kv.second;
            if (confirmed_mappoints_.count(mp_id)) {
                // 编码 {kf_id, kpt_idx} 为 uint64_t：frame_id(20bits) + kpt_idx(22bits)
                uint64_t kpt_key = (static_cast<uint64_t>(kpt_idx) << 20) |
                                   (static_cast<uint64_t>(old_kf->frame_id) & 0xFFFFFULL);
                kpt_key_to_confirmed_mp_[kpt_key] = mp_id;
                old_kf_confirmed_obs_[old_kf->frame_id].push_back({mp_id, kpt_idx});
            }
        }
        keyframe_window_.pop_front();
        kf_id_to_window_idx_.erase(old_kf->frame_id);
        // ★★★ Bug Fix: 出队后重建索引映射 ★★★
        // std::deque 的 pop_front() 会使所有后续元素索引左移 1 位，
        // 但 kf_id_to_window_idx_ 中剩余元素的映射值未更新，导致 getCovisibleMatches()
        // 读取到错误的 deque 位置（例如 frame B 应在位置 0 但 map 仍记录 1）。
        // 修复：出队后重建整个 map，保证索引与当前 deque 位置严格一致。
        kf_id_to_window_idx_.clear();
        for (size_t idx = 0; idx < keyframe_window_.size(); ++idx) {
            if (keyframe_window_[idx])
                kf_id_to_window_idx_[keyframe_window_[idx]->frame_id] = static_cast<int>(idx);
        }
    }

    // PGO：注册新顶点（用 BA 优化后的位姿）
    // ★★★ 纯 VO 模式：跳过所有 PGO 操作，不注册顶点和里程计边 ★★★
    if (pose_graph_optimizer_ && !kf_pose.q.coeffs().hasNaN() && config_.enable_pgo) {
        pose_graph_optimizer_->addOrUpdateVertex(current_kf_id, kf_pose);

        // PGO：添加里程计边（当前帧 ← 上一帧）
        // 信息矩阵基于内点率动态计算（高内点率 → 高权重）
        if (keyframe_window_.size() >= 2) {
            auto& prev_kf_ptr = keyframe_window_[keyframe_window_.size() - 2];
            if (!prev_kf_ptr) { /* null keyframe, skip PGO edge */ }
            else {
                int prev_kf_id = prev_kf_ptr->frame_id;
                Eigen::Matrix4d rel_T = computeRelativePose(prev_kf_ptr->pose, kf_pose);
                int raw_matches_count = static_cast<int>(matches_to_prev.size());
                int odom_inliers = (raw_matches_count > 0)
                                   ? static_cast<int>(inlier_match_count)
                                   : raw_matches_count;
                pose_graph_optimizer_->addOdometryEdge(
                    prev_kf_id, current_kf_id, rel_T,
                    1.0, odom_inliers, raw_matches_count);
            }
        }
    }

    // ★★★ 修复：按关键帧数而非帧数触发检查 ★★★
    // 原问题：forced_keyframe_interval=100 导致每 100 帧才检查一次，候选点没有机会晋升
    // 新策略：每 5 个关键帧（约等于 forced_keyframe_interval=20 的配套值）检查一次
    // 以关键帧数而非帧数计数，保证无论运动快慢，检查频率都与关键帧率匹配
    check_counter_++;
    if (check_counter_ >= 5 ||
        static_cast<int>(candidate_mappoints_.size()) > 2 * config_.max_candidates) {
        check_counter_ = 0;
        checkCandidates();  // 会调用 promoteCandidates() + cullCandidates() + cullConfirmedPoints()
    }

    // 回环检测（每隔 loop_detection_interval 帧检测一次）
    if (config_.enable_loop_detection && loop_detector_) {
        loop_detect_counter_++;
        if (loop_detect_counter_ >= config_.loop_detection_interval) {
            loop_detect_counter_ = 0;
            detectAndCorrectLoop();
        }
    }

    // 量化格网尺寸（从配置读取，默认为 20.0 = 5cm 格网）
    const double quantize_scale = config_.quantize_scale;
    // 处理匹配：用空间量化后的世界坐标作为 key（同一物理 3D 点在不同关键帧中有不同的 kpt_idx，但世界坐标相同）
    for (const auto& m : matches_to_prev) {
        int prev_kpt = m.queryIdx;
        int curr_kpt = m.trainIdx;
        if (prev_kpt < 0 || curr_kpt < 0 ||
            prev_kpt >= static_cast<int>(prev_kps.size()) ||
            prev_kpt >= static_cast<int>(pts3d_prev.size()) ||
            prev_kpt >= static_cast<int>(depth_valid_prev.size()) ||
            curr_kpt >= static_cast<int>(kps.size()) ||
            curr_kpt >= static_cast<int>(pts3d_curr.size()) ||
            curr_kpt >= static_cast<int>(depth_valid_curr.size()))
            continue;
        if (!depth_valid_prev[prev_kpt]) continue;
        // pts3d_prev 是「上一跟踪帧」相机系下的深度反投点，必须先变到世界系再建图/量化
        Eigen::Vector3d pt_cam = pts3d_prev[prev_kpt];
        if (pt_cam.norm() < 1e-6) continue;
        // ★★★ 关键修复：世界坐标变换必须与 trackWithMotionModel 保持一致 ★★★
        // 正确公式：P_world = R_cw^T * (P_cam - t_cw)
        // 其中 Pose.q = R_cw（世界→相机旋转），Pose.t = t_cw（世界→相机平移）
        // 原错误公式：P_world = q * P_cam + t，导致 Z 轴漂移 + 轨迹方向错误
        Eigen::Matrix3d R_cw = prev_frame_pose_world.q.toRotationMatrix();
        Eigen::Vector3d t_cw = prev_frame_pose_world.t;
        Eigen::Vector3d pt_world = R_cw.transpose() * (pt_cam - t_cw);

        // 量化到 5cm 格网格心，序列化为 uint64_t 哈希作为 map key
        // 优化：用整数哈希替代字符串序列化，省去 to_string 开销，O(1) 查找
        uint64_t pt_key = quantizeWorldPointKey(pt_world, quantize_scale);

        // 用量化世界坐标哈希做候选点去重：同一物理点 → pt_key 相同
        if (obs_key_to_mp_.count(pt_key)) {
            // ★★★ 命中已有候选/确认点：添加当前帧观测（含描述子）★★★
            int mp_id = obs_key_to_mp_[pt_key];
            if (candidate_mappoints_.count(mp_id)) {
                candidate_mappoints_[mp_id]->addObservation(
                    {current_kf_id, curr_kpt, undist_kps[curr_kpt].pt, pt_world, -1.0,
                     (!desc.empty() && curr_kpt < desc.rows) ? desc.row(curr_kpt) : cv::Mat()});
            } else if (confirmed_mappoints_.count(mp_id)) {
                confirmed_mappoints_[mp_id]->addObservation(
                    {current_kf_id, curr_kpt, undist_kps[curr_kpt].pt, pt_world, -1.0,
                     (!desc.empty() && curr_kpt < desc.rows) ? desc.row(curr_kpt) : cv::Mat()});
                obs_key_to_confirmed_mp_[{current_kf_id, curr_kpt}] = mp_id;
            }
        } else {
            // ★★★ 新建候选点：同时添加当前帧和前一帧的观测（含描述子）★★★
            auto mp = std::make_shared<MapPoint>(next_mappoint_id_++, pt_world);
            // 当前帧观测
            mp->addObservation({current_kf_id, curr_kpt,
                                undist_kps[curr_kpt].pt, pt_world, -1.0,
                                (!desc.empty() && curr_kpt < desc.rows) ? desc.row(curr_kpt) : cv::Mat()});
            // ★★★ 前一帧观测（增加 distinct_kf 用于确认）★★★
            if (!prev_desc.empty() && prev_kpt < prev_desc.rows &&
                prev_kpt < static_cast<int>(pts3d_prev.size()) &&
                prev_kpt < static_cast<int>(depth_valid_prev.size()) &&
                depth_valid_prev[prev_kpt]) {
                // 前一帧相机坐标系 3D 点（与 matches_to_prev 中的 prev_kpt 一致）
                Eigen::Vector3d prev_pt_cam = pts3d_prev[prev_kpt];
                Eigen::Matrix3d R_cw_prev = prev_frame_pose_world.q.toRotationMatrix();
                Eigen::Vector3d t_cw_prev = prev_frame_pose_world.t;
                Eigen::Vector3d prev_pt_world = R_cw_prev.transpose() * (prev_pt_cam - t_cw_prev);
                // 前一帧像素坐标（使用去畸变坐标，与世界坐标量化一致）
                cv::Point2f prev_pixel = (!prev_undist_kps.empty() &&
                                          prev_kpt < static_cast<int>(prev_undist_kps.size()))
                                         ? prev_undist_kps[prev_kpt].pt
                                         : prev_kps[prev_kpt].pt;
                // ★★★ frame_id 用 prev_frame_id（前一帧自己的 ID）★★★
                mp->addObservation({prev_frame_id, prev_kpt,  // ★★★ 修正：使用前一帧 ID ★★★
                                    prev_pixel, prev_pt_world, -1.0,
                                    prev_desc.row(prev_kpt)});
            }
            candidate_mappoints_[mp->id()] = mp;
            obs_key_to_mp_[pt_key] = mp->id();
            kf->kpt_to_mappoint[curr_kpt] = mp->id();
            // ★★★ Bug Fix: 同时关联到kpt_to_candidate用于Sim3回环修正 ★★★
            kf->kpt_to_candidate[curr_kpt] = mp->id();
        }
    }

    // ★★★ 关键修复：当 matches_to_prev 为空时（初始化帧或跟踪失败后的强制关键帧），
    // ★★★ 必须直接从当前帧的深度点创建初始地图点。
    // ★★★ 否则没有任何地图点被创建，导致后续所有帧的跟踪都会失败。
    if (matches_to_prev.empty()) {
        for (int curr_kpt = 0; curr_kpt < static_cast<int>(kps.size()); ++curr_kpt) {
            if (curr_kpt >= static_cast<int>(pts3d_curr.size()) ||
                curr_kpt >= static_cast<int>(depth_valid_curr.size()) ||
                curr_kpt >= static_cast<int>(undist_kps.size()))
                continue;
            if (!depth_valid_curr[curr_kpt]) continue;
            Eigen::Vector3d pt_cam = pts3d_curr[curr_kpt];
            if (pt_cam.norm() < 1e-6) continue;
            // 初始化帧的位姿是 identity：R_cw = I, t_cw = 0
            // P_world = R_cw^T * (P_cam - t_cw) = P_cam
            Eigen::Vector3d pt_world = pt_cam;
            uint64_t pt_key = quantizeWorldPointKey(pt_world, quantize_scale);
            if (!obs_key_to_mp_.count(pt_key)) {
                auto mp = std::make_shared<MapPoint>(next_mappoint_id_++, pt_world);
                mp->addObservation({current_kf_id, curr_kpt,
                                    undist_kps[curr_kpt].pt, pt_world, -1.0,
                                    (!desc.empty() && curr_kpt < desc.rows) ? desc.row(curr_kpt) : cv::Mat()});
                candidate_mappoints_[mp->id()] = mp;
                obs_key_to_mp_[pt_key] = mp->id();
                kf->kpt_to_mappoint[curr_kpt] = mp->id();
                // ★★★ Bug Fix: 同时关联到kpt_to_candidate用于Sim3回环修正 ★★★
                kf->kpt_to_candidate[curr_kpt] = mp->id();
            }
        }
    }

    // ==========================================================================
    // 多关键帧描述子匹配：为候选点添加来自历史关键帧的观测
    // 每个 match: queryIdx = 历史帧特征索引，trainIdx = 当前帧特征索引
    // 观测的 frame_id = 历史关键帧的 global_id（而非 current_kf_id）
    // 这样候选点就能获得来自多个不同关键帧的观测，满足 distinct_kf>=2 的确认条件
    // ==========================================================================
    if (!matches_from_cache.empty()) {
        int cache_obs_added = 0;

        for (const auto& kv : matches_from_cache) {
            int cache_idx = kv.first;
            const auto& matches = kv.second;

            if (cache_idx < 0 || cache_idx >= static_cast<int>(kf_cache_ref.size()))
                continue;
            const KeyFrameCacheEntry& cached = kf_cache_ref[cache_idx];

            for (const auto& m : matches) {
                int hist_kpt = m.queryIdx;   // 历史帧中的特征索引
                int curr_kpt = m.trainIdx;   // 当前帧中的特征索引

                // 边界检查（负数 + 所有向量大小）
                if (hist_kpt < 0 ||
                    hist_kpt >= static_cast<int>(cached.pts_3d.size()) ||
                    hist_kpt >= static_cast<int>(cached.depth_valid.size()) ||
                    hist_kpt >= static_cast<int>(cached.keypoints.size()) ||
                    curr_kpt < 0 ||
                    curr_kpt >= static_cast<int>(kps.size()) ||
                    curr_kpt >= static_cast<int>(pts3d_curr.size()) ||
                    curr_kpt >= static_cast<int>(depth_valid_curr.size()))
                    continue;

                // 历史帧该特征点必须有有效深度
                if (!cached.depth_valid[hist_kpt]) continue;

                // 当前帧该特征点也必须有有效深度
                if (!depth_valid_curr[curr_kpt]) continue;

                // 获取历史帧相机坐标系下的 3D 点
                Eigen::Vector3d pt_cam = cached.pts_3d[hist_kpt];
                if (pt_cam.norm() < 1e-6) continue;
                // ★★★ 关键修复：与 matches_to_prev 保持一致 ★★★
                // 正确公式：P_world = R_cw^T * (P_cam - t_cw)
                Eigen::Matrix3d R_cw = cached.world_pose.q.toRotationMatrix();
                Eigen::Vector3d t_cw = cached.world_pose.t;
                Eigen::Vector3d pt_world = R_cw.transpose() * (pt_cam - t_cw);

                // 5cm 格网量化（使用公共 helper，与上方 matches_to_prev 处理保持一致）
                uint64_t pt_key = quantizeWorldPointKey(pt_world, quantize_scale);

                // 如果该 3D 点已有候选点关联，为其添加来自历史关键帧的观测
                if (obs_key_to_mp_.count(pt_key)) {
                    int mp_id = obs_key_to_mp_[pt_key];
                    if (candidate_mappoints_.count(mp_id)) {
                        auto& mp = candidate_mappoints_[mp_id];
                        // ★★★ 使用去畸变后的关键点坐标 ★★★
                        if (!mp->hasObservation(cached.global_id)) {
                            cv::Point2f obs_pixel =
                                (!cached.undistorted_keypoints.empty() &&
                                 hist_kpt < static_cast<int>(cached.undistorted_keypoints.size()))
                                    ? cached.undistorted_keypoints[hist_kpt].pt
                                    : cached.keypoints[hist_kpt].pt;
                            // ★★★ 添加历史关键帧的描述子 ★★★
                            cv::Mat hist_desc = (!cached.descriptors.empty() &&
                                                hist_kpt < cached.descriptors.rows)
                                               ? cached.descriptors.row(hist_kpt) : cv::Mat();
                            mp->addObservation({
                                cached.global_id,   // 历史关键帧 ID（非 current_kf_id）
                                hist_kpt,           // 历史帧中的特征索引
                                obs_pixel,
                                pt_world,
                                -1.0,
                                hist_desc
                            });
                            cache_obs_added++;
                        }
                    }
                    // ★★★ 新增：confirmed 分支 ★★★
                    // 如果该 3D 点命中已确认地图点，也必须添加历史关键帧的观测
                    // 否则 getMapPointsObservedByKeyFrame(matched_kf_id) 找不到任何地图点
                    // 导致 Sim3Solver 预填充（共视匹配）永远失败
                    else if (confirmed_mappoints_.count(mp_id)) {
                        auto& mp = confirmed_mappoints_[mp_id];
                        if (!mp->hasObservation(cached.global_id)) {
                            // ★★★ 使用去畸变后的关键点坐标 ★★★
                            cv::Point2f confirmed_obs_pixel =
                                (!cached.undistorted_keypoints.empty() &&
                                 hist_kpt < static_cast<int>(cached.undistorted_keypoints.size()))
                                    ? cached.undistorted_keypoints[hist_kpt].pt
                                    : cached.keypoints[hist_kpt].pt;
                            // ★★★ 添加历史关键帧的描述子 ★★★
                            cv::Mat hist_desc = (!cached.descriptors.empty() &&
                                                hist_kpt < cached.descriptors.rows)
                                               ? cached.descriptors.row(hist_kpt) : cv::Mat();
                            mp->addObservation({
                                cached.global_id,   // 历史关键帧 ID（非 current_kf_id）
                                hist_kpt,           // 历史帧中的特征索引
                                confirmed_obs_pixel,
                                pt_world,
                                -1.0,
                                hist_desc
                            });
                            // 更新反向索引，使 getCovisibleMatchesWithCache 能快速定位
                            obs_key_to_confirmed_mp_[{cached.global_id, hist_kpt}] = mp_id;
                            cache_obs_added++;
                        }
                    }
                }
            }
        }

        if (cache_obs_added > 0) {
            // std::cerr << "[Covis] 多关键帧匹配：为候选点添加了 "
            //           << cache_obs_added << " 个来自历史关键帧的观测" << std::endl;
        }
    }

    // ==========================================================================
    // 几何重投影匹配：将窗口中所有历史关键帧的确认点重投影到当前关键帧，
    // 在重投影位置附近（REPROJ_SEARCH_RADIUS px）查找匹配的关键点。
    // 这让候选点能获得来自多个不同关键帧的观测，满足 distinct_kf>=2 条件。
    // ==========================================================================
    if (keyframe_window_.size() >= 2) {
        const double reproj_search_radius = config_.reproj_search_radius;  // 从配置读取（默认 12.0 px）
        int geo_match_count = 0;

        double fx = 525.0, fy = 525.0, cx = 319.5, cy = 239.5;
        if (!config_.K.empty() && config_.K.rows == 3 && config_.K.cols == 3) {
            fx = config_.K.at<double>(0, 0);
            fy = config_.K.at<double>(1, 1);
            cx = config_.K.at<double>(0, 2);
            cy = config_.K.at<double>(1, 2);
        }
        int img_w = config_.img_width;
        int img_h = config_.img_height;
        const cv::Size img_size(img_w, img_h);

        // 遍历窗口中除最新关键帧外的所有历史关键帧
        for (size_t hist_idx = 0; hist_idx + 1 < keyframe_window_.size(); ++hist_idx) {
            auto& hist_kf = keyframe_window_[hist_idx];
            int hist_kf_id = hist_kf->frame_id;

            // 遍历历史关键帧的所有确认点
            for (const auto& kv : hist_kf->kpt_to_mappoint) {
                int hist_kpt_idx = kv.first;
                int mp_id = kv.second;

                // 只处理确认点
                if (!confirmed_mappoints_.count(mp_id)) continue;
                const auto& mp = confirmed_mappoints_.at(mp_id);

                const Eigen::Vector3d& P_w = mp->worldPoint();

                // T_cw = [R_cw | t_cw]，相机位姿（从世界到相机）
                // ★★★ Bug Fix A2: R_cached 可能未初始化（buildSpatialIndex 在 addKeyFrame 中未被调用）
                // 每次 reprojection 匹配前确保缓存同步 ★★★
                if (!hist_kf->R_valid_) hist_kf->syncRotationCache();
                const Eigen::Matrix3d& R_cw = hist_kf->R_cached;
                const Eigen::Vector3d& t_cw = hist_kf->pose.t;

                // P_c = R_cw * P_w + t_cw（3D点从世界坐标变换到历史关键帧相机坐标）
                Eigen::Vector3d P_c = R_cw * P_w + t_cw;
                if (P_c.z() <= 1e-6) continue;  // 深度必须为正

                // 重投影到历史关键帧像素坐标（用于判断该点在该帧中可见）
                double u_hist = fx * P_c.x() / P_c.z() + cx;
                double v_hist = fy * P_c.y() / P_c.z() + cy;
                if (u_hist < 0 || u_hist >= img_size.width || v_hist < 0 || v_hist >= img_size.height) continue;

                // T_wc = [R_wc | t_wc]，当前关键帧位姿（从世界到相机）
                const Eigen::Matrix3d R_wc = kf_pose.q.toRotationMatrix();
                const Eigen::Vector3d& t_wc = kf_pose.t;

                // P_c_cur = R_wc * P_w + t_wc（3D点从世界坐标变换到当前关键帧相机坐标）
                Eigen::Vector3d P_c_cur = R_wc * P_w + t_wc;
                if (P_c_cur.z() <= 1e-6) continue;

                // 重投影到当前关键帧像素坐标
                double u_cur = fx * P_c_cur.x() / P_c_cur.z() + cx;
                double v_cur = fy * P_c_cur.y() / P_c_cur.z() + cy;
                if (u_cur < 0 || u_cur >= img_size.width || v_cur < 0 || v_cur >= img_size.height) continue;

                // 在当前关键帧特征点中查找距离重投影位置最近的点
                // 优化：使用O(log n)空间格网二分查找替代原来的O(n)线性扫描
                double best_dist2 = reproj_search_radius * reproj_search_radius;
                int best_curr_kpt = -1;

                // x坐标搜索窗口（reproj_search_radius 在 u 方向上的最大范围）
                float u_min = static_cast<float>(u_cur) - static_cast<float>(reproj_search_radius);
                float u_max = static_cast<float>(u_cur) + static_cast<float>(reproj_search_radius);

                // 二分查找lower_bound：找到第一个u >= u_min的位置
                // kf->kpt_grid_sorted按u坐标升序排列（buildSpatialIndex已预处理）
                int left = 0;
                int right = static_cast<int>(kf->kpt_grid_sorted.size());
                while (left < right) {
                    int mid = (left + right) / 2;
                    if (kf->kpt_grid_sorted[mid].u < u_min) left = mid + 1;
                    else right = mid;
                }

                // 从lower_bound位置向右扫描，只检查u在[u_min, u_max]范围内的点
                // ★★★ Bug #3: kpt_grid_sorted 只能从 undistorted_keypoints 构建，
                //    fallback 到 distorted keypoints 会导致格网索引错位（1~n px 误差）。
                //    TUM 畸变 k1=0.26/k2=-0.95 下不可忽略。
                if (kf->undistorted_keypoints.empty() ||
                    kf->undistorted_keypoints.size() != kf->keypoints.size()) {
                    if (config_.verbose) {
                        std::cerr << "[searchLocalMap] WARNING: undistorted_keypoints unavailable,"
                                  << " skipping frame_id=" << kf->frame_id << std::endl;
                    }
                    continue;  // 跳过该帧，避免几何错配
                }
                const auto& curr_kpts = kf->undistorted_keypoints;
                for (int ci = left; ci < static_cast<int>(kf->kpt_grid_sorted.size()); ++ci) {
                    float pu = kf->kpt_grid_sorted[ci].u;
                    if (pu > u_max) break;
                    int kpt_idx = kf->kpt_grid_sorted[ci].kpt_idx;
                    if (kpt_idx >= static_cast<int>(depth_valid_curr.size()) || !depth_valid_curr[kpt_idx]) continue;
                    if (kpt_idx >= static_cast<int>(curr_kpts.size())) continue;
                    // 使用去畸变后的坐标进行比较
                    const float kpt_x = curr_kpts[kpt_idx].pt.x;
                    const float kpt_y = curr_kpts[kpt_idx].pt.y;
                    double dv = kpt_y - v_cur;
                    if (std::abs(dv) > reproj_search_radius) continue;
                    double du = kpt_x - u_cur;
                    double dist2 = du * du + dv * dv;
                    if (dist2 < best_dist2) {
                        best_dist2 = dist2;
                        best_curr_kpt = kpt_idx;
                    }
                }

                if (best_curr_kpt < 0) continue;  // 当前帧中没有找到附近的特征点

                // 获取当前关键帧中该特征点对应的候选点
                auto it_curr_cand = kf->kpt_to_mappoint.find(best_curr_kpt);
                if (it_curr_cand == kf->kpt_to_mappoint.end()) continue;  // 该特征点未关联候选点

                int curr_mp_id = it_curr_cand->second;
                if (candidate_mappoints_.count(curr_mp_id)) {
                    auto& curr_mp = candidate_mappoints_.at(curr_mp_id);

                    // 检查量化世界坐标是否一致（使用公共 helper）
                    uint64_t pt_key2 = quantizeWorldPointKey(P_w, config_.quantize_scale);
                    auto it_orig = obs_key_to_mp_.find(pt_key2);
                    if (it_orig == obs_key_to_mp_.end() || it_orig->second != curr_mp_id) continue;

                    // ★★★ O(1) 检查：使用 hasObservation() 替代 O(n) 线性扫描 ★★★
                    if (curr_mp->hasObservation(hist_kf_id)) continue;

                    // 添加来自历史关键帧的观测（增加 distinct_kf 和 obs）
                    curr_mp->addObservation({hist_kf_id, hist_kpt_idx,
                                             cv::Point2f(static_cast<float>(u_hist),
                                                         static_cast<float>(v_hist)),
                                             P_w, -1.0,
                                             cv::Mat()});  // geo_match 无描述子，置空
                    geo_match_count++;
                } else if (confirmed_mappoints_.count(curr_mp_id)) {
                    // ★★★ 关键修复：confirmed 地图点也要接收当前帧的观测 ★★★
                    // 否则 confirmed 的 observations 里永远没有 hist_kf_id，
                    // 导致 getCovisibleMatchesWithCache(curr_kf, matched_kf) 找不到该共视点。
                    auto& conf_mp = confirmed_mappoints_.at(curr_mp_id);
                    // ★★★ O(1) 检查：使用 hasObservation() 替代 O(n) 线性扫描 ★★★
                    if (!conf_mp->hasObservation(current_kf_id)) {
                        // ★★★ 添加当前帧描述子 ★★★
                        cv::Mat curr_desc = (!desc.empty() && best_curr_kpt < desc.rows)
                                         ? desc.row(best_curr_kpt) : cv::Mat();
                        conf_mp->addObservation({current_kf_id, best_curr_kpt,
                                                 kps[best_curr_kpt].pt, P_w, -1.0,
                                                 curr_desc});
                        obs_key_to_confirmed_mp_[{current_kf_id, best_curr_kpt}] = curr_mp_id;
                        geo_match_count++;
                    }
                }
            }
        }
        if (geo_match_count > 0 && config_.verbose) {
            std::cerr << "[Covis] 几何重投影匹配：找到 " << geo_match_count
                      << " 对新观测 (窗口KF=" << keyframe_window_.size() << ")" << std::endl;
        }
    }

    // 立即触发一次候选点检查：关键帧加入后立即尝试将候选点提升为确认点
    // 注意：checkCandidates() 已在上面的 10 帧计数器触发中执行过了，
    // 此处只单独执行 promoteCandidates（将候选点晋升为确认点，不删除）。
    // 避免 checkCandidates() 重复调用导致 cullCandidates/cullConfirmedPoints 被执行两次。
    promoteCandidates();

    // 新增：回填已确认点的观测（使回环时能找到 curr_kf 和 matched_kf 的共视确认点）
    backfillConfirmedObservations(kf);

    // 回环校正后跳过若干帧 runLocalBA，避免 PGO + BA 双重内存峰值
    if (skip_local_ba_frames_ > 0) {
        skip_local_ba_frames_--;
    } else if (running_frames_ >= 30 && running_frames_ % 10 == 0) {
        // ★★★ 初始化保护：前 30 帧跳过 BA，避免不稳定初始地图点被固化 ★★★
        if (config_.enable_local_ba) {
            runLocalBA(true);
        } else {
            // 非 verbose 时也打印此行，确认 BA 被禁用
            std::cerr << "[Covis BA] SKIPPED: enable_local_ba=false（已禁用）" << std::endl;
        }
    }

    if (running_frames_ % 30 == 0 && config_.verbose) {
        std::cout << "[Covis] 帧=" << running_frames_
                  << " 关键帧=" << keyframe_window_.size()
                  << " 候选=" << candidate_mappoints_.size()
                  << " 确认=" << confirmed_mappoints_.size()
                  << std::endl;
    }

    // ★ BA 后更新最新关键帧的 pts_3d（用 BA 优化后的世界坐标反投影到相机系）
    // 这样当前帧的 pts_3d 始终与 BA 优化后的 world_pose 保持一致，
    // 后续几何验证时 world→camera 转换不会出错
    // ★★★ Bug Fix B1: 仅在至少有一个关键帧的位姿被成功写回时才执行 pts_3d 变换 ★★★
    // 原代码：BA 可能拒绝所有位姿修正（correction rejected），但 pts_3d 仍被错误地变换，
    // 导致 pts_3d 与 world_pose 不一致（pts_3d 被变换但 pose 未变）。
    // 新逻辑：BA 写回循环内部检查 kf_updated > 0，确保只有在校正生效时才变换。
    if (recent_ba_pose_update_ && !keyframe_window_.empty()) {
        bool any_pose_updated = last_ba_actual_pose_update_;
        if (any_pose_updated) {
            // ★★★ Bug Fix: 变换窗口内所有关键帧的 pts_3d（不仅仅是最新关键帧）★★★
            // 原问题：runLocalBA 优化了窗口内所有关键帧的 pose，但只有最新关键帧的 pts_3d 被变换。
            // 窗口内其他关键帧的 pts_3d 仍然是旧的相机坐标，与 BA 优化后的 pose 不一致。
            // 后续 SearchByProjection 投影这些关键帧的地图点时，几何计算全部错误。
            // 修复：遍历窗口内所有关键帧，用各自的 BA 优化后 pose 变换对应的 pts_3d。
            for (auto& kf_ptr : keyframe_window_) {
                if (!kf_ptr) continue;
                const Pose& world_pose = kf_ptr->pose;
                Eigen::Matrix3d R_wc = world_pose.q.toRotationMatrix();
                Eigen::Matrix3d R_cw = R_wc.transpose();
                Eigen::Vector3d t_cw = -R_cw * world_pose.t;
                for (size_t i = 0; i < kf_ptr->pts_3d.size(); ++i) {
                    if (!kf_ptr->depth_valid[i]) continue;
                    if (kf_ptr->pts_3d[i].norm() < 1e-6) continue;
                    kf_ptr->pts_3d[i] = R_cw * kf_ptr->pts_3d[i] + t_cw;
                }
            }
        }
        last_ba_actual_pose_update_ = false;
        recent_ba_pose_update_ = false;
    }
}

void CovisManager::checkCandidates() {
    if (config_.verbose) {
        std::cerr << "[Covis checkCandidates] START candidates=" << candidate_mappoints_.size()
                  << " confirmed=" << confirmed_mappoints_.size() << std::endl;
    }
    promoteCandidates();
    cullCandidates();
    cullConfirmedPoints();
    if (config_.verbose) {
        std::cerr << "[Covis checkCandidates] END candidates=" << candidate_mappoints_.size()
                  << " confirmed=" << confirmed_mappoints_.size() << std::endl;
    }
}

void CovisManager::promoteCandidates() {
    if (config_.verbose) {
        std::cerr << "[Covis promote] START size=" << candidate_mappoints_.size() << std::endl;
    }
    int promoted = 0;
    std::vector<int> distinct_ids;
    for (auto& kv : candidate_mappoints_) {
        auto& mp = kv.second;
        if (mp->status() != PointStatus::CANDIDATE) continue;

        // ★★★ 优化 1: 使用 unordered_map + vector 替代 std::set ★★★
        // std::set 每次 insert 都是 O(log N) 红黑树操作，改用先收集再排序唯一化
        int distinct_kf_count = countDistinctKeyFrames(mp->observations(), distinct_ids);
        int total_obs = mp->obsCount();

        if (distinct_kf_count >= config_.min_covis_frames &&
            total_obs >= config_.candidate_confirm_threshold) {
            mp->promote();
            // ★★★ ORB_SLAM3 风格：晋升时计算最佳描述子 ★★★
            mp->computeDescriptor();
            confirmed_mappoints_[mp->id()] = mp;
            for (const auto& obs : mp->observations())
                obs_key_to_confirmed_mp_[{obs.frame_id, obs.kpt_idx}] = mp->id();
            // ★★★ Bug Fix: 晋升时从 obs_key_to_mp_ 删除，防止旧候选 ID 残留导致错误查找 ★★★
            // obs_key_to_mp_ 使用量化世界坐标哈希 (uint64_t) 作为 key，不是 frame_id/kpt_idx 对
            for (const auto& obs : mp->observations()) {
                // 重新量化观测点的世界坐标（与插入时一致）
                Eigen::Vector3d wp = mp->worldPoint();
                uint64_t pt_key = quantizeWorldPointKey(wp, config_.quantize_scale);
                obs_key_to_mp_.erase(pt_key);
            }
            promoted++;
        }
    }
    // ★★★ 汇总打印：每帧只打印一次晋升统计，避免 1000+ 行日志导致程序卡顿 ★★★
    if (promoted > 0 && config_.verbose) {
        std::cout << "[Covis] 本次晋升 " << promoted << " 个候选点，确认点总数="
                  << confirmed_mappoints_.size() << std::endl;
    }
    if (config_.verbose) {
        std::cerr << "[Covis promote] END promoted=" << promoted << std::endl;
    }
}

void CovisManager::promoteCandidatesExternal() const {
    // 通过 const_cast 突破 const 限制，仅此一处使用
    const_cast<CovisManager*>(this)->promoteCandidates();
}

void CovisManager::cullCandidates() {
    // ★★★ 修复：无须清理时直接返回，避免每次遍历 1700+ 候选点造成卡顿 ★★★
    // 原逻辑：每帧都遍历全部候选点做分类，即使候选点数量在软限制以内（< 2×max）也白跑一次。
    // 新策略：只有候选点数量超过 max_candidates 时才执行清理逻辑。
    if (static_cast<int>(candidate_mappoints_.size()) <= config_.max_candidates) return;

    if (config_.verbose) {
        std::cerr << "[Covis cullCand] START candidates=" << candidate_mappoints_.size()
                  << " (max=" << config_.max_candidates << ")，执行清理" << std::endl;
    }

    // 一次性按观测次数升序删除最少的点，只遍历一次 map
    std::vector<std::pair<int, int>> id_obs_pairs;
    id_obs_pairs.reserve(candidate_mappoints_.size());
    for (const auto& kv : candidate_mappoints_) {
        id_obs_pairs.emplace_back(kv.first, kv.second->obsCount());
    }
    std::sort(id_obs_pairs.begin(), id_obs_pairs.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    int target = config_.max_candidates;
    int removed = 0;
    int to_remove = static_cast<int>(candidate_mappoints_.size()) - target;
    for (const auto& p : id_obs_pairs) {
        if (removed >= to_remove) break;
        int id = p.first;
        if (loop_protection_frames_ > 0 && protected_candidate_ids_.count(id)) continue;
        candidate_mappoints_.erase(id);
        for (auto it = obs_key_to_mp_.begin(); it != obs_key_to_mp_.end(); ) {
            if (it->second == id) it = obs_key_to_mp_.erase(it);
            else ++it;
        }
        ++removed;
    }
    if (config_.verbose) {
        std::cerr << "[Covis cullCand] END 删除了 " << removed << " 个，剩余 "
                  << candidate_mappoints_.size() << " 个候选点" << std::endl;
    }
}

void CovisManager::cullConfirmedPoints() {
    // 诊断：记录本次删除的数量（按原因分类）
    int removed_stale = 0, removed_reproj = 0, removed_bulk = 0;
    for (auto it = confirmed_mappoints_.begin();
         it != confirmed_mappoints_.end(); ) {
        // ★★★ 方案A：跳过受保护的确认点（回环共视点）★★★
        if (loop_protection_frames_ > 0 && protected_confirmed_ids_.count(it->first)) {
            ++it;
            continue;
        }
        const auto& mp = it->second;
        bool remove = false;
        if (mp->avgReprojError() > config_.max_reproj_error && mp->obsCount() > 5) {
            remove = true;
            removed_reproj++;
        }
        if (running_frames_ - mp->lastObsFrameId() > config_.staleness_thresh) {
            remove = true;
            removed_stale++;
        }
        if (remove) {
            // ★★★ Bug Fix: 删除确认点时同步清理 obs_key_to_mp_ ★★★
            // obs_key_to_mp_ 映射量化世界坐标哈希→mappoint ID。
            // 当确认点被删除后，obs_key_to_mp_ 中的对应条目仍保留（悬空哈希），
            // 后续帧命中这些悬空键时会读到不存在的 ID（已晋升的候选点用同一 key）。
            // 修复：遍历 obs_key_to_mp_，删除所有 value 为此 mappoint ID 的条目。
            int del_id = it->first;
            for (auto it_key = obs_key_to_mp_.begin(); it_key != obs_key_to_mp_.end(); ) {
                if (it_key->second == del_id) it_key = obs_key_to_mp_.erase(it_key);
                else ++it_key;
            }
            it = confirmed_mappoints_.erase(it);
        } else
            ++it;
    }

    // 确认点数量检查：只有在严重超限（> 2× max_confirmed）时才触发裁剪
    // 给予确认点足够的缓冲空间，避免频繁触发导致点被错误删除
    int max_confirmed = config_.max_confirmed;
    if (static_cast<int>(confirmed_mappoints_.size()) > 2 * max_confirmed) {
        if (config_.verbose) {
            std::cerr << "[Covis] 警告: cullConfirmedPoints 软截断 " << confirmed_mappoints_.size()
                      << " → " << 2 * max_confirmed << "（2×max）" << std::endl;
        }
        int to_remove = static_cast<int>(confirmed_mappoints_.size()) - 2 * max_confirmed;
        int removed = 0;
        // 按观测次数升序删除（优先删除观测少的点）
        std::vector<std::pair<int, int>> id_obs_pairs;
        for (const auto& kv : confirmed_mappoints_) {
            id_obs_pairs.emplace_back(kv.first, kv.second->obsCount());
        }
        std::sort(id_obs_pairs.begin(), id_obs_pairs.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& p : id_obs_pairs) {
            if (removed >= to_remove) break;
            int id = p.first;
            if (loop_protection_frames_ > 0 && protected_confirmed_ids_.count(id)) continue;
            for (const auto& obs : confirmed_mappoints_[id]->observations())
                obs_key_to_confirmed_mp_.erase({obs.frame_id, obs.kpt_idx});
            // ★★★ Bug Fix: bulk 删除也需清理 obs_key_to_mp_ ★★★
            for (auto it_key = obs_key_to_mp_.begin(); it_key != obs_key_to_mp_.end(); ) {
                if (it_key->second == id) it_key = obs_key_to_mp_.erase(it_key);
                else ++it_key;
            }
            confirmed_mappoints_.erase(id);
            ++removed;
        }
    }
    int total_removed = removed_stale + removed_reproj + removed_bulk;
    if (total_removed > 0 && config_.verbose) {
        std::cerr << "[Covis cullConf] 删除: stale=" << removed_stale
                  << " reproj=" << removed_reproj
                  << " bulk=" << removed_bulk
                  << " → END confirmed=" << confirmed_mappoints_.size() << std::endl;
    } else if (config_.verbose) {
        std::cerr << "[Covis cullConf] 无删除 END confirmed=" << confirmed_mappoints_.size() << std::endl;
    }
}

bool CovisManager::runLocalBA(bool) {
    if (!config_.enable_local_ba) return false;
    if (keyframe_window_.size() < 2) return false;
    if (confirmed_mappoints_.empty()) return false;

    double fx, fy, cx, cy;
    if (!config_.K.empty()) {
        fx = config_.K.at<double>(0, 0);
        fy = config_.K.at<double>(1, 1);
        cx = config_.K.at<double>(0, 2);
        cy = config_.K.at<double>(1, 2);
    } else {
        fx = 525.0; fy = 525.0; cx = 319.5; cy = 239.5;
    }

    // ★★★ EdgeSE3ProjectXYZ + 位姿+地图点联合优化 ★★★
    // 最新关键帧固定（世界锚点），其他位姿允许优化修正漂移
    // JacobianWorkspace: 18 = 2×6(位姿雅可比) + 2×3(地图点雅可比)
    auto linearSolver =
        std::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto blockSolver =
        std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    blockSolver->setSchur(false);  // 无边缘化变量，禁用 Schur
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // g2o OptimizableGraph 内置了 JacobianWorkspace，initializeOptimization() 会自动分配
    // ============================================================
    // 关键帧顶点：固定最新帧作为世界锚点，其他自由优化
    // ★★★ 关键修复 1：固定最新帧而非第一帧 ★★★
    // 原策略（固定第一帧）：
    //   - 第一帧在原点，BA 把所有中间帧拉向原点
    //   - 最新帧有漂移但被自由优化，漂移被锁入轨迹
    //   - 原点固定 → 所有后续帧共享漂移误差
    // 正确策略（固定最新帧，ORB_SLAM3 风格）：
    //   - 最新帧固定为世界锚点，保证当前跟踪尺度
    //   - 中间帧在最新帧的约束下优化，漂移被分散到中间帧
    //   - 后续新帧继承最新帧的正确尺度，抑制漂移累积
    // ★★★ 关键修复 2：BA 写回关键帧位姿 ★★★
    // 原代码：kf_updated++ 但什么都不写回，关键帧位姿完全不变
    // 问题：BA 优化后的位姿被丢弃，漂移无法通过 BA 修正
    // 修复：BA 结束后，将 g2o 优化后的位姿写回 keyframe_window_
    // ============================================================
    std::map<int, g2o::VertexSE3Expmap*> kf_vertex_map;
    int latest_kf_window_idx = static_cast<int>(keyframe_window_.size()) - 1;
    for (size_t i = 0; i < keyframe_window_.size(); ++i) {
        auto* v_se3 = new g2o::VertexSE3Expmap();
        v_se3->setId(static_cast<int>(i));
        if (static_cast<int>(i) == latest_kf_window_idx) {
            v_se3->setFixed(true);   // 最新帧固定，世界锚点
        } else {
            v_se3->setFixed(false);  // 其他帧自由优化
        }
        const Pose& kf_pose = keyframe_window_[i]->pose;
        // R1：kf_pose 是 T_wc，g2o 顶点要 T_cw
        v_se3->setEstimate(poseWcToSe3Quat(kf_pose));
        optimizer.addVertex(v_se3);
        kf_vertex_map[keyframe_window_[i]->frame_id] = v_se3;
    }

    // 地图点顶点 + 双向映射（使用连续局部ID，避免JacobianWorkspace越界）
    std::map<int, g2o::VertexPointXYZ*> mpid_to_vpoint;  // mappoint id -> vertex
    std::map<int, int> g2o_vid_to_mpid;                   // g2o vertex id -> mappoint id
    int point_idx = 0;
    for (const auto& kv : confirmed_mappoints_) {
        const auto& mp = kv.second;
        Eigen::Vector3d wp = mp->worldPoint();
        if (wp.hasNaN() || wp.norm() < 1e-6) continue;

        auto* v_point = new g2o::VertexPointXYZ();
        int point_id = static_cast<int>(keyframe_window_.size() + point_idx++);
        v_point->setId(point_id);  // 连续局部ID，防止JacobianWorkspace越界
        v_point->setEstimate(wp);
        v_point->setFixed(false);  // 地图点允许优化
        optimizer.addVertex(v_point);
        mpid_to_vpoint[kv.first] = v_point;
        g2o_vid_to_mpid[point_id] = kv.first;
    }

    // ★★★ Bug Fix Q2: 移除死代码 edge->setInformation(I) ★★★
    // edge 在 for 循环内每次新建，此处 edge 未定义（来自前面的错误重构）
    // BA 边的 information 在 EdgeSE3ProjectXYZDirect::setInformation() 中已设为 Identity
    // 无需额外设置
    int edge_count = 0;
    for (const auto& kv : confirmed_mappoints_) {
        const auto& mp = kv.second;
        if (!mpid_to_vpoint.count(kv.first)) continue;
        g2o::VertexPointXYZ* v_pt = mpid_to_vpoint[kv.first];

        for (const auto& obs : mp->observations()) {
            if (!kf_vertex_map.count(obs.frame_id)) continue;
            auto* edge = new sfc_core::mapping::SFCEdgeSE3ProjectXYZ();
            edge->setVertex(0, v_pt);                           // 点（VertexPointXYZ）
            edge->setVertex(1, kf_vertex_map[obs.frame_id]);   // 位姿（VertexSE3Expmap）
            Eigen::Vector2d meas(obs.pixel.x, obs.pixel.y);
            if (meas.hasNaN() || meas.norm() > 10000) continue;
            edge->setMeasurement(meas);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;
            auto* huber = new g2o::RobustKernelHuber();
            huber->setDelta(config_.ba_huber_delta > 0 ? config_.ba_huber_delta : 2.45);
            edge->setRobustKernel(huber);
            optimizer.addEdge(edge);
            edge_count++;
        }
    }

    if (edge_count < config_.ba_min_edges) return false;

    // ★★★ Bug Fix: 如果没有顶点（关键帧或地图点），跳过 BA，防止 0 vertices 警告 ★★★
    int num_kf = static_cast<int>(keyframe_window_.size());
    int num_pts = static_cast<int>(mpid_to_vpoint.size());
    if (num_kf < 1 || num_pts < 1) {
        if (config_.verbose) {
            std::cerr << "[Covis BA] 跳过：num_kf=" << num_kf << " num_pts=" << num_pts << std::endl;
        }
        return false;
    }

    // ★★★ 调试：打印 BA 图规模（verbose 时）★★★
    if (config_.verbose) {
        std::cerr << "[Covis BA] 图规模: 关键帧=" << keyframe_window_.size()
                  << " 地图点=" << mpid_to_vpoint.size()
                  << " 边=" << edge_count
                  << " 即将 optimize(20)..." << std::endl;
    }

    // g2o initializeOptimization() 内部自动分配 JacobianWorkspace
    // 但如果 g2o 版本不一致，allocate() 可能静默失败。
    // 为保险起见，显式预分配 workspace，防止 SIGSEGV。
    // 协方差 BA：最多 num_kf 个顶点，每个顶点 6 维，误差 3 维（地图点坐标）
    // JacobianWorkspace 维度：max(2×6 for pose vertex, 2×3 for point vertex) = 12
    // 第一个参数：最大顶点ID + 1，覆盖 keyframe顶点(0..num_kf-1)和点顶点(num_kf..num_kf+num_pts-1)
    optimizer.jacobianWorkspace().updateSize(num_kf + num_pts, 12);
    optimizer.jacobianWorkspace().allocate();

    // ★★★ 防御性检查：确保 optimizer 有顶点和边，防止 g2o 0 vertices 警告 ★★★
    // 即使 num_kf>=1 && num_pts>=1，addVertex 可能因 NaN 而全部失败
    if (optimizer.vertices().empty() || optimizer.edges().empty()) {
        if (config_.verbose) {
            std::cerr << "[Covis BA] 跳过：optimizer 无顶点(" << optimizer.vertices().size()
                      << ")或无边(" << optimizer.edges().size() << ")" << std::endl;
        }
        return false;
    }

    optimizer.initializeOptimization();
    if (config_.verbose) {
        std::cerr << "[Covis BA] optimizer.initializeOptimization() 完成，开始 optimize(20)..." << std::endl;
    }
    int iter = optimizer.optimize(20);
    if (config_.verbose) {
        std::cerr << "[Covis BA] optimizer.optimize() 完成，iter=" << iter << std::endl;
    }

    // 写回优化的位姿和地图点
    // ★★★ BA 写回关键帧位姿（不仅是地图点）★★★
    // 修复：原代码只写回地图点，关键帧位姿保持 PnP 结果不变
    // 现在：同时写回关键帧位姿，让 BA 的局部优化能修正漂移
    last_ba_actual_pose_update_ = false;  // ★★★ Bug Fix B1: 每次 BA 前重置 ★★★
    int kf_updated = 0;
    for (size_t i = 0; i < keyframe_window_.size(); ++i) {
        if (i == static_cast<size_t>(latest_kf_window_idx)) continue;  // 最新帧（锚点）固定，跳过
        int window_idx = static_cast<int>(i);
        int kf_id = keyframe_window_[i]->frame_id;
        if (!kf_vertex_map.count(kf_id)) continue;
        g2o::VertexSE3Expmap* v_se3 = kf_vertex_map[kf_id];
        if (!v_se3) continue;
        // R1：顶点里是 T_cw，先换回对外的 T_wc 再与 old_pose 比较和写回
        const Pose opt_pose = se3QuatToPoseWc(v_se3->estimate());
        Eigen::Quaterniond q_opt = opt_pose.q;
        Eigen::Vector3d t_opt = opt_pose.t;
        // NaN 安全检查
        if (!t_opt.hasNaN() && !q_opt.coeffs().hasNaN()) {
            const Pose& old_pose = keyframe_window_[i]->pose;
            double dz = std::abs(t_opt.z() - old_pose.t.z());
            // ★★★ Fix Issue C: BA 修正幅度过大时跳过写回，避免误差扩散 ★★★
            double delta_t = (t_opt - old_pose.t).norm();
            Eigen::Matrix3d delta_R = q_opt.toRotationMatrix() * old_pose.q.toRotationMatrix().transpose();
            double cos_angle = 0.5 * (delta_R.trace() - 1.0);
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
            double delta_angle = std::acos(cos_angle) * 180.0 / M_PI;
            
            // ★★★ 使用配置的阈值替代硬编码值 ★★★
            // 当启用完整BA修正时，使用较严格的阈值防止异常修正
            double max_trans = config_.enable_full_ba_correction 
                ? config_.ba_max_trans_correction_m : 0.20;
            double max_rot = config_.enable_full_ba_correction 
                ? config_.ba_max_rot_correction_deg : 15.0;
            if (delta_t > max_trans || delta_angle > max_rot) {
                if (config_.verbose) {
                    std::cerr << "[Covis-BA] WARNING: Large pose correction rejected"
                              << " id=" << kf_id << " dt=" << std::fixed << std::setprecision(4)
                              << delta_t << "m dR=" << std::setprecision(2) << delta_angle << "deg" << std::endl;
                }
                continue;  // Skip this keyframe, keep old pose
            }
            // ★★★ [Z轴调试] 打印 Covis BA 写回的关键帧 Z 变化 ★★★
            if (config_.verbose) {
                std::cerr << "[Covis-BA-WRITE] KF=" << std::setw(4) << kf_id
                          << " old_z=" << std::fixed << std::setprecision(4) << old_pose.t.z()
                          << " new_z=" << std::setprecision(4) << t_opt.z()
                          << " |dz|=" << std::setprecision(4) << dz << "m"
                          << " |dt|=" << (t_opt - old_pose.t).norm() << "m"
                          << std::endl;
            }
            // ★★★ 关键修复：配置可选的 BA 修正（平移+旋转）★★★
            // 当 enable_full_ba_correction=true 且有重力约束时，启用完整 BA 修正
            // 否则只让 BA 修正平移漂移，不引入旋转误差（防止旋转累积）
            if (config_.enable_full_ba_correction) {
                // 完整修正：使用 BA 优化的旋转和平移
                Pose new_pose(q_opt, t_opt);
                keyframe_window_[i]->pose = new_pose;
                keyframe_window_[i]->syncRotationCache();
                last_ba_actual_pose_update_ = true;
                kf_updated++;
            } else {
                // 部分修正：只使用 BA 的平移，保留原始旋转
                Pose new_pose(old_pose.q, t_opt);
                keyframe_window_[i]->pose = new_pose;
                keyframe_window_[i]->syncRotationCache();  // ★★★ Bug Fix: pose 修改后同步更新缓存旋转矩阵 ★★★
                last_ba_actual_pose_update_ = true;  // ★★★ Bug Fix B1: 记录实际写回了位姿 ★★★
                kf_updated++;
            }
        }
    }

    // 写回优化的地图点
    last_ba_mappoint_updates_.clear();
    for (const auto& kv : g2o_vid_to_mpid) {
        auto* v_point = dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(kv.first));
        if (v_point) {
            Eigen::Vector3d opt_wp = v_point->estimate();
            if (!opt_wp.hasNaN() && opt_wp.norm() > 1e-6) {
                last_ba_mappoint_updates_[kv.second] = opt_wp;
            }
        }
    }

    if (config_.verbose) {
        // 汇总所有被修正关键帧的 XY 修正量
        double max_dxy = 0.0, sum_dxy = 0.0;
        int count_dxy = 0;
        for (size_t i = 0; i < keyframe_window_.size(); ++i) {
            if (i == static_cast<size_t>(latest_kf_window_idx)) continue;
            int kf_id = keyframe_window_[i]->frame_id;
            if (!kf_vertex_map.count(kf_id)) continue;
            g2o::VertexSE3Expmap* v_se3 = kf_vertex_map[kf_id];
            if (!v_se3) continue;
            // R1：诊断量同样要换回 T_wc，否则拿到的是 t_cw（−R_wc^T·t_wc），
            // 与 old_pose.t 不可比，dXY 会变成随机数。
            Eigen::Vector3d t_opt = se3QuatToPoseWc(v_se3->estimate()).t;
            if (t_opt.hasNaN()) continue;
            const Pose& old_pose = keyframe_window_[i]->pose;
            if (old_pose.t.hasNaN()) continue;
            double dxy = std::sqrt(std::pow(t_opt.x() - old_pose.t.x(), 2) +
                                   std::pow(t_opt.y() - old_pose.t.y(), 2));
            sum_dxy += dxy;
            if (dxy > max_dxy) max_dxy = dxy;
            count_dxy++;
        }
        double avg_dxy = count_dxy > 0 ? sum_dxy / count_dxy : 0.0;

        std::cout << "[Covis BA] 关键帧=" << keyframe_window_.size()
                  << " 锚点固定=1"
                  << " 位姿更新=" << kf_updated
                  << " 地图点更新=" << last_ba_mappoint_updates_.size()
                  << "/" << confirmed_mappoints_.size()
                  << " 边=" << edge_count
                  << " 迭代=" << iter
                  << " max_dXY=" << std::fixed << std::setprecision(4) << max_dxy << "m"
                  << " avg_dXY=" << avg_dxy << "m"
                  << std::endl;
    }

    recent_ba_pose_update_ = true;

    // ★★★ 将 BA 优化后的地图点世界坐标写回 confirmed_mappoints_ ★★★
    // 原代码：last_ba_mappoint_updates_ 填充后从未被消费，confirmed_mappoints_ 永远是三角化坐标
    // 修复：BA 完成后用优化后的世界坐标覆盖三角化坐标，后续重投影匹配精度更高
    for (const auto& kv : last_ba_mappoint_updates_) {
        if (confirmed_mappoints_.count(kv.first)) {
            confirmed_mappoints_[kv.first]->updateWorldPoint(kv.second);
        }
        // 候选点也可能被 BA 优化（观测它的关键帧被 BA 修正）
        if (candidate_mappoints_.count(kv.first)) {
            candidate_mappoints_[kv.first]->updateWorldPoint(kv.second);
        }
    }

    return true;
}

Pose CovisManager::getLatestOptimizedPose() const {
    return getLatestKeyframePose();
}

Pose CovisManager::getLatestKeyframePose() const {
    if (keyframe_window_.empty())
        return Pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    return keyframe_window_.back()->pose;
}

int CovisManager::getLatestKeyframeId() const {
    if (keyframe_window_.empty()) return -1;
    return keyframe_window_.back()->frame_id;
}

std::map<int, Pose> CovisManager::getAllKeyFramePosesSnapshot() const {
    std::map<int, Pose> result;
    for (const auto& kf : keyframe_window_) {
        if (kf) result[kf->frame_id] = kf->pose;
    }
    return result;
}

std::vector<std::pair<Eigen::Vector3d, int>> CovisManager::getCovisibleMapPoints(
    int kf_id, int n_best) const {
    std::vector<std::pair<Eigen::Vector3d, int>> result;
    for (const auto& kv : confirmed_mappoints_) {
        const auto& mp = kv.second;
        if (mp->worldPoint().norm() < 1e-6) continue;
        bool observed_by_kf = false;
        int best_obs_count = mp->obsCount();
        for (const auto& obs : mp->observations()) {
            if (obs.frame_id == kf_id) {
                observed_by_kf = true;
                break;
            }
        }
        if (!observed_by_kf) continue;
        result.emplace_back(mp->worldPoint(), best_obs_count);
    }
    if (n_best > 0 && static_cast<int>(result.size()) > n_best) {
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        result.resize(n_best);
    }
    return result;
}

std::vector<CovisMatch> CovisManager::getCovisibleMatches(
    int curr_kf_id, int matched_kf_id) const {
    std::vector<CovisMatch> result;

    // ★★★ O(1) 查找：使用 kf_id_to_window_idx_ 直接索引 ★★★
    auto it_curr = kf_id_to_window_idx_.find(curr_kf_id);
    auto it_matched = kf_id_to_window_idx_.find(matched_kf_id);
    if (it_curr == kf_id_to_window_idx_.end() || it_matched == kf_id_to_window_idx_.end()) return result;

    const auto& curr_kf = keyframe_window_[it_curr->second];
    const auto& matched_kf = keyframe_window_[it_matched->second];

    // 遍历当前帧的确认点，检查是否也被匹配帧观测到
    for (const auto& kv : curr_kf->kpt_to_mappoint) {
        int curr_kpt = kv.first;
        int mp_id = kv.second;
        if (!confirmed_mappoints_.count(mp_id)) continue;
        const auto& mp = confirmed_mappoints_.at(mp_id);

        // 检查该地图点是否也被匹配帧观测到
        int matched_kpt = -1;
        double reproj_error = -1.0;
        for (const auto& obs : mp->observations()) {
            if (obs.frame_id == matched_kf_id) {
                matched_kpt = obs.kpt_idx;
                reproj_error = obs.reproj_error;
                break;
            }
        }
        if (matched_kpt < 0) continue;

        CovisMatch m;
        m.pt_world = mp->worldPoint();
        m.curr_kpt_idx = curr_kpt;
        m.matched_kpt_idx = matched_kpt;
        m.reproj_error = reproj_error;
        result.push_back(m);
    }
    return result;
}

std::vector<CovisMatch> CovisManager::getCovisibleMatchesWithCache(
    int curr_kf_id, int matched_kf_id) const {
    std::vector<CovisMatch> result;

    // 遍历所有确认地图点，对每个地图点检查它的 observations
    // 中是否有 curr_kf_id 和 matched_kf_id。
    // 这与 getCovisibleMatches 的区别在于：
    //   - getCovisibleMatches 依赖 curr_kf 和 matched_kf 都在 keyframe_window_ 中
    //   - 该方法只依赖 observations 中的 frame_id，适用于历史帧（已滑出窗口）的情况
    for (const auto& kv : confirmed_mappoints_) {
        const auto& mp = kv.second;
        int curr_kpt_idx = -1, matched_kpt_idx = -1;
        double reproj_error = -1.0;

        for (const auto& obs : mp->observations()) {
            if (obs.frame_id == curr_kf_id) curr_kpt_idx = obs.kpt_idx;
            if (obs.frame_id == matched_kf_id) {
                matched_kpt_idx = obs.kpt_idx;
                reproj_error = obs.reproj_error;
            }
        }
        if (curr_kpt_idx < 0 || matched_kpt_idx < 0) continue;

        CovisMatch m;
        m.pt_world = mp->worldPoint();
        m.curr_kpt_idx = curr_kpt_idx;
        m.matched_kpt_idx = matched_kpt_idx;
        m.reproj_error = reproj_error;
        result.push_back(m);
    }
    return result;
}

std::vector<int> CovisManager::getMapPointsObservedByKeyFrame(int kf_id) const {
    std::vector<int> result;
    for (const auto& kv : confirmed_mappoints_) {
        for (const auto& o : kv.second->observations()) {
            if (o.frame_id == kf_id) {
                result.push_back(kv.first);
                break;
            }
        }
    }
    return result;
}

void CovisManager::syncKeyFrameCache(std::vector<KeyFrameCacheEntry>& kf_cache) {
    // ★★★ Bug Fix #8: 实现空的 syncKeyFrameCache 函数体 ★★★
    // KeyFrameCacheEntry::pts_3d 按设计保持在相机坐标系（covis_core.hpp:202 注释）。
    // CovisManager 中 backfillConfirmedObservations 会将最新关键帧的 pts_3d
    // 从世界坐标变换到相机坐标（covis_manager.cpp:612）。
    // 此函数在回环检测完成后被调用（Tracking.cpp 当前未使用，但 API 存在）。
    // 将 kf_cache 中各关键帧的 pts_3d 重新从世界坐标变换到相机坐标，
    // 使 kf_cache 的相机坐标系与 CovisManager 的最新 BA pose 一致。
    for (auto& entry : kf_cache) {
        auto it = pgo_optimized_poses_cache_.find(entry.global_id);
        if (it == pgo_optimized_poses_cache_.end()) continue;
        const Pose& new_world_pose = it->second;
        Eigen::Matrix3d R_wc = new_world_pose.q.toRotationMatrix();
        Eigen::Matrix3d R_cw = R_wc.transpose();
        Eigen::Vector3d t_cw = -R_cw * new_world_pose.t;
        for (size_t pi = 0; pi < entry.pts_3d.size(); ++pi) {
            if (!entry.depth_valid[pi]) continue;
            if (entry.pts_3d[pi].norm() < 1e-6) continue;
            // 世界 → 相机
            entry.pts_3d[pi] = R_cw * entry.pts_3d[pi] + t_cw;
        }
        entry.world_pose = new_world_pose;
    }
}

std::vector<KeyFrame::Ptr> CovisManager::getAllKeyFrames() const {
    std::vector<KeyFrame::Ptr> result;
    if (pose_graph_optimizer_) {
        for (int kf_id : pose_graph_optimizer_->getVertexIds()) {
            for (const auto& kf : keyframe_window_) {
                if (kf->frame_id == kf_id) {
                    result.push_back(kf);
                    break;
                }
            }
        }
    }
    if (result.empty()) {
        result.insert(result.end(), keyframe_window_.begin(), keyframe_window_.end());
    }
    return result;
}

// ==================== 回环检测 + PGO ====================

void CovisManager::initLoopDetection(const loopdetection::LoopDetectorConfig& loop_cfg,
                                      const loopdetection::PoseGraphOptimizerOptions& pgo_cfg) {
    loopdetection::LoopDetectorConfig merged = loop_cfg;
    if ((merged.camera_K.empty() || merged.camera_K.rows != 3 ||
          merged.camera_K.cols != 3) &&
        !config_.K.empty() && config_.K.rows == 3 && config_.K.cols == 3) {
        merged.camera_K = config_.K.clone();
    }
    if (merged.camera_D.empty() && !config_.D.empty()) {
        merged.camera_D = config_.D.clone();
    }
    loop_detector_ = std::make_shared<loopdetection::LoopDetector>(merged);
    // Sim3Solver 共视地图点验证通过 detect() 的 covis_mgr 参数传入，无需 setCovisManager
    loopdetection::PoseGraphOptimizerOptions pgo_opts = pgo_cfg;
    // ★★★ 将 CovisConfig.verbose 传递给 PGOopts.verbose ★★★
    // 这样 PGO 内部的 verbose 日志也会被 config_.verbose 控制
    pgo_opts.verbose = config_.verbose;
    pose_graph_optimizer_ = std::make_shared<loopdetection::PoseGraphOptimizer>(pgo_opts);
    loop_detect_counter_ = 0;
    last_correction_T_ = Eigen::Matrix4d::Identity();
    std::cout << "[Covis] 回环检测已初始化（PGO）" << std::endl;
}

bool CovisManager::detectAndCorrectLoop() {
    if (!loop_detector_ || !pose_graph_optimizer_ || keyframe_window_.empty())
        return false;

    const auto& curr_kf = keyframe_window_.back();

    // ★★★ 修改 2：在回环验证之前，先回填 matched_kf 的观测 ★★★
    // 使下一次 geometricVerification 能通过 getCovisibleMatchesWithCache 找到共视匹配对
    // 注意：第一次回环时 last_loop_matched_kf_id_=-1（无历史），prefill 会跳过
    // 从第二次回环开始，prefill 会用上一次 matched_kf 的数据填充当前帧的观测
    prefillMatchedKFObservations(curr_kf);

    // 跳过没有有效 3D 点的帧
    if (curr_kf->pts_3d.empty()) return false;

    // 把当前帧加入检测数据库
    loop_detector_->addKeyFrame(curr_kf->frame_id,
                                 curr_kf->keypoints,
                                 curr_kf->pts_3d,
                                 curr_kf->descriptors,
                                 curr_kf->pose);

    // 执行回环检测（传入 this 以启用 Sim3Solver 共视地图点几何验证）
    auto candidate = loop_detector_->detect(curr_kf->keypoints,
                                              curr_kf->pts_3d,
                                              curr_kf->descriptors,
                                              this);

    if (!candidate.valid) {
        return false;
    }

    std::cout << "[Covis Loop] ★★★ 检测到回环！当前 KF=" << candidate.current_kf_id
              << " ↔ 历史 KF=" << candidate.matched_kf_id
              << " 内点=" << candidate.geometric_inliers
              << " (" << std::fixed << std::setprecision(1)
              << candidate.inlier_ratio * 100 << "%)"
              << (candidate.is_icp ? " [ICP]" : " [EM]") << std::endl;

    // 回环边使用 ICP 验证的相对变换（有真实尺度）或 Essential Matrix（旋转可靠，尺度归一化）
    double t_norm = candidate.relative_pose.block<3,1>(0,3).norm();
    std::cout << "[Covis Loop] 回环 |t|=" << t_norm << " m ("
              << (candidate.is_icp ? "ICP，带真实尺度" : "Essential Matrix，尺度归一化")
              << ")" << std::endl;

    // ★★★ 诊断：打印 Sim3 的旋转矩阵对角线，帮助定位 scale 爆炸的来源 ★★★
    {
        Eigen::Matrix3d R = candidate.relative_pose.block<3, 3>(0, 0);
        double scale_hint = R.norm() / std::sqrt(3.0);
        std::cerr << "[Covis Loop DEBUG] Sim3 R_diag=[" << R(0,0)
                  << ", " << R(1,1) << ", " << R(2,2)
                  << "] scale_hint=" << scale_hint << std::endl;
    }

    // ★★★ 防御性检查 1：平移向量尺度 ★★★
    // ICP 模式下 |t| 应该在合理范围内。
    // 对于 fr1_desk2 这类室内场景，相邻关键帧间距约 0.06m（median），
    // ★★★ 使用可配置阈值替代硬编码 3.0m ★★★
    // 回环检测到的历史帧与当前帧距离应在合理范围。
    // ICP 模式有真实尺度，阈值可设大；EM 模式 |t| 应该很小。
    double t_thresh = candidate.is_icp ? config_.icp_loop_max_distance_m : 2.0;
    if (t_norm > t_thresh || !candidate.relative_pose.allFinite()) {
        std::cerr << "[Covis Loop] ★ 拒绝回环：Sim3 |t|=" << t_norm
                  << "m > " << t_thresh << "m 或矩阵含 NaN（室内场景回环距离应 < " << t_thresh << "m）" << std::endl;
        return false;
    }

    // ★★★ 防御性检查 2：旋转矩阵尺度 ★★★
    // Sim3 输出后，旋转矩阵对角线应接近 1（允许 ±1% 误差）
    {
        Eigen::Matrix3d R = candidate.relative_pose.block<3, 3>(0, 0);
        double scale_hint = R.norm() / std::sqrt(3.0);
        if (!std::isfinite(scale_hint) || std::abs(scale_hint - 1.0) > 0.01) {
            std::cerr << "[Covis Loop] ★ 拒绝回环：旋转矩阵尺度异常 scale="
                      << scale_hint << "（对角线=[" << R(0,0) << ", "
                      << R(1,1) << ", " << R(2,2) << "]）" << std::endl;
            return false;
        }
    }

    // PGO：添加回环边
    // L4：返回 false 表示这条约束没有进入图（重复、顶点缺失或校验失败）。
    // 图结构没变，再调 optimize() 只是把旧图解一遍，却会打印"PGO 优化完成"，
    // 让日志看起来像回环生效了。
    if (!pose_graph_optimizer_->addLoopEdge(candidate.current_kf_id,
                                            candidate.matched_kf_id,
                                            candidate.relative_pose,
                                            1.0,
                                            candidate.geometric_inliers,
                                            candidate.raw_matches,
                                            candidate.is_icp)) {
        std::cerr << "[Covis Loop] 回环边未被采纳（重复或校验失败），跳过 PGO：KF="
                  << candidate.current_kf_id << " ↔ " << candidate.matched_kf_id << std::endl;
        return false;
    }

    // PGO：执行 SE3 位姿图优化
    int iter = 0;
    double pgo_chi2 = 0.0;
    try {
        // D4-b：chi2 是"约束是否真的进了 level 0"的直接证据，与迭代数一起看才有意义。
        iter = pose_graph_optimizer_->optimize(&pgo_chi2, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "[Covis Loop] ★ PGO 异常：" << e.what() << "，跳过修正" << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[Covis Loop] ★ PGO 未知异常，跳过修正" << std::endl;
        return false;
    }
    if (iter <= 0) {
        std::cerr << "[Covis Loop] PGO 优化失败，跳过修正" << std::endl;
        return false;
    }

    std::cout << "[Covis Loop] PGO 优化完成，迭代=" << iter
              << " chi2=" << pgo_chi2
              << " 回环边=" << pose_graph_optimizer_->numLoopEdges()
              << std::endl;

    // 把 PGO 优化后的位姿写回关键帧滑动窗口
    if (config_.enable_pgo) {
        // ★★★ 修改 1：EM 模式检测尺度膨胀 ★★★
        // EM 的 |t|=1.0m 归一化平移不携带真实尺度，PGO 可能产生 10+× 的尺度膨胀。
        // 在写入关键帧之前，先用 PGO 优化后的位姿检查回环两端关键帧之间的实际距离。
        if (!candidate.is_icp) {
            Pose pgo_curr, pgo_matched;
            if (pose_graph_optimizer_->getOptimizedPose(candidate.current_kf_id, &pgo_curr) &&
                pose_graph_optimizer_->getOptimizedPose(candidate.matched_kf_id, &pgo_matched)) {
                double loop_dist = (pgo_curr.t - pgo_matched.t).norm();
                double em_scale = loop_dist / std::max(t_norm, 0.01);
                // ★★★ 使用配置的阈值替代硬编码值 ★★★
                if (em_scale > config_.em_scale_explosion_threshold) {
                    std::cout << "[Covis Loop] 警告: EM 模式修正量过大 loop_dist=" << loop_dist
                              << "m（EM |t|=" << t_norm << "m，膨胀 " << em_scale
                              << "×），跳过回环修正（极可能是 EM 尺度归一化导致的假修正）" << std::endl;
                    return true;
                }
            }
        }

        // ★ 合理性检查：比较 PGO 优化前后每个关键帧的位姿差异
        // 从配置读取 PGO 修正阈值
        const double kMaxAllowedTransCorr = config_.pgo_max_allowed_trans_corr;
        const double kMaxAllowedRotCorrDeg = config_.pgo_max_allowed_rot_corr_deg;
        const double kMaxAllowedRotCorrDegReject = config_.pgo_max_allowed_rot_corr_deg_reject;
        const double kWarnTransCorr = config_.pgo_warn_trans_corr;
        const double kWarnRotCorrDeg = config_.pgo_warn_rot_corr_deg;
        double max_dtrans = 0.0, max_drot = 0.0;
        bool too_large_correction = false;
        if (config_.verbose) {
            std::cout << "[Covis Loop] 合理性检查：遍历 keyframe_window_（" << keyframe_window_.size()
                      << " 个关键帧）" << std::endl;
        }
        int skipped = 0;
        for (const auto& kf : keyframe_window_) {
            Pose pgo_pose;
            bool ok = pose_graph_optimizer_->getOptimizedPose(kf->frame_id, &pgo_pose);
            if (!ok) { skipped++; continue; }
            double dtrans = (pgo_pose.t - kf->pose.t).norm();
            Eigen::Quaterniond dq = pgo_pose.q * kf->pose.q.inverse();
            dq.normalize();
            double dang_deg = 2.0 * std::acos(std::max(-1.0, std::min(1.0, std::abs(dq.w())))) * 180.0 / M_PI;
            max_dtrans = std::max(max_dtrans, dtrans);
            max_drot = std::max(max_drot, dang_deg);
            if (config_.verbose) {
                std::cout << "[Covis Loop]   KF=" << kf->frame_id
                          << " getOptimizedPose=OK dtrans=" << std::fixed << std::setprecision(3)
                          << dtrans << "m dang=" << std::fixed << std::setprecision(1)
                          << dang_deg << "°" << std::endl;
            }
            if (dtrans > kMaxAllowedTransCorr || dang_deg > kMaxAllowedRotCorrDegReject) {
                too_large_correction = true;
                std::cerr << "[Covis Loop] 错误: KF=" << kf->frame_id
                          << " 修正过大 |t|=" << dtrans << "m rot=" << dang_deg
                          << "° > " << kMaxAllowedRotCorrDegReject << "°，拒绝回环修正（旋转修正异常大，极可能是误检）" << std::endl;
                break;
            }
            if (dang_deg > kWarnRotCorrDeg && dang_deg <= kMaxAllowedRotCorrDegReject) {
                if (config_.verbose) {
                    std::cout << "[Covis Loop]   警告: KF=" << kf->frame_id
                              << " 旋转修正 " << dang_deg << "° > " << kWarnRotCorrDeg << "°（但 < "
                              << kMaxAllowedRotCorrDegReject << "°，通过）" << std::endl;
                }
            }
        }
        if (skipped > 0) {
            std::cout << "[Covis Loop]   警告：" << skipped << " 个关键帧不在 g2o 图中（已跳过）" << std::endl;
        }

        // ★★★ 替换插值为直接 PGO 修正（Essential Graph 风格）★★★
        // 与 ORB_SLAM3 OptimizeEssentialGraph 一致：从 g2o 图中读取所有顶点的优化后位姿，
        // 一次性应用修正，替代不稳定的距离加权插值。
        std::unordered_map<int, Pose> all_pgo_poses;
        pose_graph_optimizer_->getAllOptimizedPoses(&all_pgo_poses);

        // 对窗口内每个关键帧，使用 PGO 优化后的直接位姿
        for (const auto& kf : keyframe_window_) {
            auto it_pgo = all_pgo_poses.find(kf->frame_id);
            if (it_pgo == all_pgo_poses.end()) continue;

            const Pose& pgo_pose = it_pgo->second;
            double dtrans = (pgo_pose.t - kf->pose.t).norm();
            Eigen::Quaterniond dq = pgo_pose.q * kf->pose.q.inverse();
            dq.normalize();
            double dang_deg = 2.0 * std::acos(std::max(-1.0, std::min(1.0, std::abs(dq.w())))) * 180.0 / M_PI;
            if (dtrans > max_dtrans) max_dtrans = dtrans;
            if (dang_deg > max_drot) max_drot = dang_deg;

            kf->pose = pgo_pose;
            kf->syncRotationCache();  // ★★★ Bug Fix: PGO pose 修改后同步更新缓存旋转矩阵 ★★★
            pgo_optimized_poses_cache_[kf->frame_id] = pgo_pose;
            if (config_.verbose) {
                std::cout << "[Covis Loop]   [PGO直写] KF=" << kf->frame_id
                          << " dtrans=" << std::fixed << std::setprecision(3) << dtrans << "m"
                          << " dang=" << std::fixed << std::setprecision(1) << dang_deg << "deg" << std::endl;
            }
        }

        if (!too_large_correction && keyframe_window_.size() > 0 && max_dtrans < 0.001 && max_drot < 0.1) {
            std::cout << "[Covis Loop]   警告：所有窗口内关键帧修正量极小（<0.001m/<0.1°），"
                      << "修正实际发生在滑动窗口外的历史关键帧上！" << std::endl;
        }

        // ★★★ 关键帧间距检查：检测回环修正导致的异常大幅跳变 ★★★
        // 当 PGO 把滑动窗口外的历史关键帧拉偏时，相邻关键帧之间会产生大跳变
        // 检查 PGO 修正后相邻关键帧的间距是否异常
        // 修复：使用中位数替代均值，使用 15x 阈值替代 5x（允许更长距离的回环）
        if (!too_large_correction && keyframe_window_.size() >= 2) {
            std::vector<double> all_dists;
            for (size_t i = 1; i < keyframe_window_.size(); ++i) {
                Pose p0, p1;
                if (pose_graph_optimizer_->getOptimizedPose(keyframe_window_[i - 1]->frame_id, &p0) &&
                    pose_graph_optimizer_->getOptimizedPose(keyframe_window_[i]->frame_id, &p1)) {
                    all_dists.push_back((p1.t - p0.t).norm());
                }
            }
            if (all_dists.size() >= 2) {
                std::sort(all_dists.begin(), all_dists.end());
                double median_interval = all_dists[all_dists.size() / 2];
                // 使用 15x 中位数阈值（允许跨越 210 帧的远距离回环，容忍较大局部跳变）
                double spacing_threshold = 15.0 * std::max(median_interval, 0.1);
                for (size_t i = 0; i < all_dists.size(); ++i) {
                    double dist = all_dists[i];
                    if (dist > spacing_threshold) {
                        int kf_a = keyframe_window_[i]->frame_id;
                        int kf_b = keyframe_window_[i + 1]->frame_id;
                        std::cerr << "[Covis Loop] 警告: 关键帧间距异常 "
                                  << "KF=" << kf_a << " → KF=" << kf_b
                                  << " 间距=" << std::fixed << std::setprecision(2) << dist
                                  << "m（中位数=" << std::fixed << std::setprecision(2)
                                  << median_interval << "m，阈值=" << std::fixed << std::setprecision(2)
                                  << spacing_threshold << "m），可能是正常回环跳变，降低阈值而非拒绝"
                                  << std::endl;
                    }
                }
            }
        }

        if (max_dtrans > kWarnTransCorr || max_drot > kWarnRotCorrDeg) {
            std::cout << "[Covis Loop] 警告: 最大修正 |t|=" << max_dtrans
                      << "m rot=" << max_drot << "°，可能包含噪声" << std::endl;
        }

        if (too_large_correction) {
            std::cout << "[Covis Loop] ★ PGO 修正过大，拒绝回写（保留原始 PnP 轨迹）" << std::endl;
        } else {
            // 将 PGO 图中所有顶点的优化结果写入缓存（包括不在窗口内的历史关键帧）
            // 窗口内帧已在上面直接更新，这里只处理历史帧
            std::unordered_map<int, Pose> all_pgo_poses;
            pose_graph_optimizer_->getAllOptimizedPoses(&all_pgo_poses);
            int hist_updated = 0;
            for (const auto& kv : all_pgo_poses) {
                int vid = kv.first;
                bool in_window = false;
                for (const auto& kf : keyframe_window_) {
                    if (kf->frame_id == vid) { in_window = true; break; }
                }
                if (in_window) continue;
                pgo_optimized_poses_cache_[vid] = kv.second;
                loop_detector_->updateKeyFramePose(vid, kv.second);
                ++hist_updated;
            }
            int window_updated = static_cast<int>(keyframe_window_.size());
            std::cout << "[Covis Loop] ★ PGO 修正已回写：max|t|=" << max_dtrans
                      << "m max|rot|=" << max_drot << "°"
                      << " 窗口内=" << window_updated << " 历史=" << hist_updated
                      << " 总缓存=" << pgo_optimized_poses_cache_.size() << std::endl;
            // ★★★ PGO 修正关键帧后，同步修正所有地图点的世界坐标 ★★★
            Eigen::Matrix4d identity_correction = Eigen::Matrix4d::Identity();
            applyPGOCorrection(identity_correction);
        }
    } else {
        std::cout << "[Covis Loop] PGO 已禁用，跳过位姿回写" << std::endl;
    }

    // ★★★ 瓶颈2核心修复：扩展 matched_kf 共视点收集范围 ★★★
    // 原逻辑只搜索 confirmed_mappoints_，但 matched_kf 的观测记录全部存在于候选点中，
    // 导致 last_matched_kf_covis_mp_ids_ 永远为空，后续 backfill 和 Sim3Solver 全部失效。
    //
    // 新逻辑：
    //   1. 搜索 confirmed_mappoints_（原有逻辑，处理已确认的点）
    //   2. 搜索 candidate_mappoints_（新增，对被 matched_kf 观测到的候选点执行即时晋升）
    //   3. 晋升后，候选点进入 confirmed_mappoints_，同时建立 matched_kf 的观测记录
    //   → 下一次 detectAndCorrectLoop 的 prefillMatchedKFObservations 能正确回填
    last_matched_kf_covis_mp_ids_.clear();
    last_loop_matched_kf_id_ = candidate.matched_kf_id;

    // ① 从已确认点中收集
    for (const auto& kv : confirmed_mappoints_) {
        const auto& mp = kv.second;
        for (const auto& obs : mp->observations()) {
            if (obs.frame_id == candidate.matched_kf_id) {
                last_matched_kf_covis_mp_ids_.push_back(kv.first);
                break;
            }
        }
    }

    // ② 从候选点中收集被 matched_kf 观测到的点，执行即时晋升
    if (!candidate_mappoints_.empty()) {
        std::vector<int> candidate_ids_promoted;
        for (const auto& kv : candidate_mappoints_) {
            const auto& mp = kv.second;
            if (mp->status() != PointStatus::CANDIDATE) continue;
            bool seen_by_matched = false;
            for (const auto& obs : mp->observations()) {
                if (obs.frame_id == candidate.matched_kf_id) {
                    seen_by_matched = true;
                    break;
                }
            }
            if (!seen_by_matched) continue;

            // 统计该候选点被多少个不同关键帧观测（包含 matched_kf）
            std::set<int> obs_kf_ids;
            for (const auto& obs : mp->observations())
                obs_kf_ids.insert(obs.frame_id);
            int distinct_kf_count = static_cast<int>(obs_kf_ids.size());
            int total_obs = mp->obsCount();

            if (distinct_kf_count >= config_.min_covis_frames &&
                total_obs >= config_.candidate_confirm_threshold) {
                // 晋升
                mp->promote();
                confirmed_mappoints_[mp->id()] = mp;
                for (const auto& obs : mp->observations())
                    obs_key_to_confirmed_mp_[{obs.frame_id, obs.kpt_idx}] = mp->id();
                candidate_ids_promoted.push_back(mp->id());

                // 将该晋升点加入 last_matched_kf_covis_mp_ids_（已有 matched_kf 观测）
                last_matched_kf_covis_mp_ids_.push_back(mp->id());

                if (static_cast<int>(confirmed_mappoints_.size()) > config_.max_confirmed) {
                    int min_obs = INT_MAX, worst_id = -1;
                    for (const auto& ckv : confirmed_mappoints_) {
                        if (ckv.second->obsCount() < min_obs) {
                            min_obs = ckv.second->obsCount();
                            worst_id = ckv.second->id();
                        }
                    }
                    if (worst_id >= 0) {
                        for (const auto& obs : confirmed_mappoints_[worst_id]->observations())
                            obs_key_to_confirmed_mp_.erase({obs.frame_id, obs.kpt_idx});
                        confirmed_mappoints_.erase(worst_id);
                    }
                }

                std::cout << "[Covis] ★ 候选点 " << mp->id()
                          << " 晋升为确认点（matched_kf=" << candidate.matched_kf_id
                          << "，distinct_kf=" << distinct_kf_count
                          << " obs=" << total_obs << "）" << std::endl;
            }
        }
        if (!candidate_ids_promoted.empty()) {
            std::cout << "[Covis] 共晋升 " << candidate_ids_promoted.size()
                      << " 个候选点为确认点（来自 matched_kf=" << candidate.matched_kf_id
                      << " 的共视匹配）" << std::endl;
        }
    }

    // ★★★ 方案A：填充回环共视点保护列表 ★★★
    // 收集所有被 matched_kf 观测过的候选点和确认点，加入保护列表
    // 这样在下一次回环检测发生前，这些点不会被 cullCandidates/cullConfirmedPoints 淘汰
    protected_candidate_ids_.clear();
    protected_confirmed_ids_.clear();
    for (const auto& kv : candidate_mappoints_) {
        const auto& mp = kv.second;
        if (mp->status() == PointStatus::CANDIDATE &&
            mp->hasObservation(candidate.matched_kf_id)) {
            protected_candidate_ids_.insert(mp->id());
        }
    }
    for (const auto& kv : confirmed_mappoints_) {
        const auto& mp = kv.second;
        if (mp->hasObservation(candidate.matched_kf_id)) {
            protected_confirmed_ids_.insert(mp->id());
        }
    }
    loop_protection_frames_ = config_.candidate_check_interval * 3;  // 保护 15 帧（≈3 次 cullCandidates 触发间隔）
    std::cout << "[Covis] ★ 方案A保护：matched_kf=" << candidate.matched_kf_id
              << " 候选保护=" << protected_candidate_ids_.size()
              << " 确认保护=" << protected_confirmed_ids_.size()
              << " 保护期=" << loop_protection_frames_ << " 帧" << std::endl;

    if (last_matched_kf_covis_mp_ids_.empty()) {
        // 诊断：打印详细信息，帮助定位问题根因
        int confirmed_obs_by_matched = 0;
        int candidate_obs_by_matched = 0;
        int confirmed_total = confirmed_mappoints_.size();
        int candidate_total = candidate_mappoints_.size();
        // 统计 matched_kf_id 在确认点和候选点的观测中各出现了多少次
        for (const auto& kv : confirmed_mappoints_) {
            for (const auto& obs : kv.second->observations()) {
                if (obs.frame_id == candidate.matched_kf_id) {
                    confirmed_obs_by_matched++;
                    break;
                }
            }
        }
        for (const auto& kv : candidate_mappoints_) {
            for (const auto& obs : kv.second->observations()) {
                if (obs.frame_id == candidate.matched_kf_id) {
                    candidate_obs_by_matched++;
                    break;
                }
            }
        }
        std::cerr << "[Covis] *** 警告: matched_kf=" << candidate.matched_kf_id
                  << " 在候选点和确认点中均无共视地图点，回环修正可能退化为无尺度模式"
                  << std::endl;
        std::cerr << "[Covis]   诊断: 确认点总数=" << confirmed_total
                  << " 候选点总数=" << candidate_total
                  << " matched_kf 出现在确认点中次数=" << confirmed_obs_by_matched
                  << " matched_kf 出现在候选点中次数=" << candidate_obs_by_matched
                  << std::endl;
        std::cerr << "[Covis]   提示: 这是因为候选点频繁被 cullCandidates 淘汰（每帧都触发）,"
                  << std::endl;
        std::cerr << "[Covis]          确认点频繁被 promoteCandidates 内的 max_confirmed 裁剪（每次晋升后都触发）。"
                  << std::endl;
        std::cerr << "[Covis]          候选点无法稳定增长到能被确认的数量，matched_kf 帧的地图点被全部删除了。"
                  << std::endl;
        // ★★★ 修复：无共视地图点时跳过回环，避免 EM 降级假回环污染 PGO ★★★
        std::cerr << "[Covis] ★ 跳过此次回环（matched_kf 无共视地图点，无法建立有效约束）" << std::endl;
        return false;
    } else {
        std::cout << "[Covis] 已缓存 matched_kf=" << candidate.matched_kf_id
                  << " 的 " << last_matched_kf_covis_mp_ids_.size()
                  << " 个共视确认地图点" << std::endl;
    }

    // 回环校正后跳过接下来 10 帧 runLocalBA，避免 PGO + BA 双重内存峰值导致 OOM
    skip_local_ba_frames_ = 10;

    return true;
}

// ★★★ P2-3: Sim3 修正后同步修正所有地图点坐标 ★★★
// ★★★ Bug #5: 参数保留用于 API 兼容性，实际修正量由内部 Sim3_delta 计算 ★★★
// @param correction_base_T 未使用参数（已匿名化）。
//    原始设计意图：用 correction_base_T 做链式变换，但实际由
//    T_new * T_old.inverse()（line 1689）计算修正量。
//    此参数保留以避免破坏现有调用方（如 line 1479）。
void CovisManager::applyPGOCorrection(const Eigen::Matrix4d&) {
    if (!pose_graph_optimizer_) return;

    // 使用已有的 getVertexIds() + getOptimizedPose() API
    auto vertex_ids = pose_graph_optimizer_->getVertexIds();
    if (vertex_ids.empty()) return;

    int n_corrected_kf = 0;
    int n_corrected_mp = 0;

    for (int kf_id : vertex_ids) {
        // 获取 PGO 优化后的位姿
        sfc_core::frontend::Pose pgo_pose;
        if (!pose_graph_optimizer_->getOptimizedPose(kf_id, &pgo_pose)) continue;

        // 找到该 keyframe（可能在 keyframe_window_ 中）
        KeyFrame::Ptr kf;
        for (const auto& kw : keyframe_window_) {
            if (kw->frame_id == kf_id) { kf = kw; break; }
        }
        if (!kf) continue;

        // 检查该 keyframe 是否被修正了（与原始 PnP 位姿差异 > 1mm）
        const auto it_orig = pnp_original_poses_cache_.find(kf_id);
        if (it_orig == pnp_original_poses_cache_.end()) continue;
        const sfc_core::frontend::Pose& orig_pose = it_orig->second;
        double pos_delta = (pgo_pose.t - orig_pose.t).norm();
        double angle_delta = orig_pose.q.angularDistance(pgo_pose.q);
        if (pos_delta < 0.001 && angle_delta < 0.001) continue;

        // 计算 SE3 修正量
        // T_corr_new = correction_base_T * T_corr_old
        // 对地图点 P_w：P_w_new = T_corr_new * T_corr_old.inv() * P_w_old = correction_base_T * P_w_old
        Eigen::Matrix4d T_old = Eigen::Matrix4d::Identity();
        T_old.block<3,3>(0,0) = orig_pose.q.toRotationMatrix();
        T_old.block<3,1>(0,3) = orig_pose.t;
        Eigen::Matrix4d T_new = Eigen::Matrix4d::Identity();
        T_new.block<3,3>(0,0) = pgo_pose.q.toRotationMatrix();
        T_new.block<3,1>(0,3) = pgo_pose.t;
        Eigen::Matrix4d Sim3_delta = T_new * T_old.inverse();

        // 遍历该关键帧观测的所有确认地图点，修正其世界坐标
        for (const auto& kmp_kv : kf->kpt_to_mappoint) {
            int mp_id = kmp_kv.second;
            if (!confirmed_mappoints_.count(mp_id)) continue;
            auto& mp = confirmed_mappoints_[mp_id];
            Eigen::Vector3d P_w_old = mp->worldPoint();
            if (P_w_old.norm() < 1e-6) continue;
            Eigen::Vector3d P_w_new = Sim3_delta.block<3,3>(0,0) * P_w_old
                                   + Sim3_delta.block<3,1>(0,3);
            if (P_w_new.array().allFinite()) {
                mp->updateWorldPoint(P_w_new);
                n_corrected_mp++;
            }
        }
        
        // ★★★ Bug Fix: 同时修正候选点的世界坐标 ★★★
        // 原bug: kpt_to_mappoint只包含确认点，候选点在Sim3修正时被遗漏
        // 导致候选点升级为确认点后仍持有旧的世界坐标，造成重投影误差
        for (const auto& cand_kv : kf->kpt_to_candidate) {
            int cand_id = cand_kv.second;
            if (!candidate_mappoints_.count(cand_id)) continue;
            auto& cand_mp = candidate_mappoints_[cand_id];
            Eigen::Vector3d P_w_old = cand_mp->worldPoint();
            if (P_w_old.norm() < 1e-6) continue;
            Eigen::Vector3d P_w_new = Sim3_delta.block<3,3>(0,0) * P_w_old
                                   + Sim3_delta.block<3,1>(0,3);
            if (P_w_new.array().allFinite()) {
                cand_mp->updateWorldPoint(P_w_new);
                n_corrected_mp++;
            }
        }

        n_corrected_kf++;
    }

    if (n_corrected_kf > 0 || n_corrected_mp > 0) {
        std::cerr << "[Covis Loop] ★ Sim3 地图点修正："
                  << n_corrected_kf << " 个关键帧, "
                  << n_corrected_mp << " 个地图点" << std::endl;
    }
}

bool CovisManager::hasNewLoop() const {
    return loop_detector_ && loop_detector_->hasNewLoop();
}

void CovisManager::clearNewLoopFlag() {
    if (loop_detector_) loop_detector_->clearNewLoopFlag();
}

std::vector<std::pair<int, Pose>> CovisManager::getAllOptimizedKeyFramePoses() const {
    std::vector<std::pair<int, Pose>> result;

    // ★★★ 优先读取 PGO 优化后的位姿（回环修正后的轨迹）★★★
    // 只有当 PGO 缓存为空（PGO 未运行或禁用）时才降级到原始 PnP
    bool has_pgo_poses = false;
    for (const auto& kv : pgo_optimized_poses_cache_) {
        has_pgo_poses = true;
        break;
    }

    if (has_pgo_poses) {
        // PGO 优化轨迹：包含所有历史关键帧的修正后位姿
        // 构建当前窗口帧 ID 集合（用于过滤，不返回窗口内帧的位姿）
        std::set<int> window_kf_ids;
        for (const auto& kf : keyframe_window_)
            if (kf) window_kf_ids.insert(kf->frame_id);

        // 只返回已被窗口淘汰的历史帧（不在窗口中的帧）
        for (const auto& kv : pgo_optimized_poses_cache_) {
            if (!window_kf_ids.count(kv.first))
                result.emplace_back(kv.first, kv.second);
        }
        // 补充：pnp_original_poses_cache_ 中有但 pgo_optimized_poses_cache_ 中没有的历史帧
        for (const auto& kv : pnp_original_poses_cache_) {
            if (!window_kf_ids.count(kv.first) &&
                pgo_optimized_poses_cache_.find(kv.first) == pgo_optimized_poses_cache_.end()) {
                result.emplace_back(kv.first, kv.second);
            }
        }
    } else {
        // PGO 未运行：降级到原始 PnP 轨迹
        for (const auto& kv : pnp_original_poses_cache_) {
            result.emplace_back(kv.first, kv.second);
        }
    }
    return result;
}

std::vector<std::pair<int, Pose>> CovisManager::getWindowKeyFramePoses() const {
    std::vector<std::pair<int, Pose>> result;
    for (const auto& kf : keyframe_window_) {
        if (kf) result.emplace_back(kf->frame_id, kf->pose);
    }
    return result;
}

Eigen::Matrix4d CovisManager::computeRelativePose(const Pose& pose_from,
                                                   const Pose& pose_to) {
    // ★★★ Bug Fix #6: 添加注释说明返回值约定 ★★★
    // 返回: T_from→to = T_wf.inv() * T_wt
    //   将 pose_from 坐标系中的点 p_from 变换到 pose_to 坐标系: p_to = T_from→to * p_from
    //   本质上: p_to = T_cw_pose_to * T_wc_pose_from * p_from = (T_cw_to * T_wc_from) * p_from
    // 调用约定: computeRelativePose(prev_kf.pose, curr_kf.pose) → T_prev→curr
    Eigen::Matrix4d T_wf = Eigen::Matrix4d::Identity();
    T_wf.block<3, 3>(0, 0) = pose_from.q.toRotationMatrix();
    T_wf.block<3, 1>(0, 3) = pose_from.t;

    Eigen::Matrix4d T_wt = Eigen::Matrix4d::Identity();
    T_wt.block<3, 3>(0, 0) = pose_to.q.toRotationMatrix();
    T_wt.block<3, 1>(0, 3) = pose_to.t;

    return T_wf.inverse() * T_wt;
}

// ★★★ 修改 2：提前回填 matched_kf 的观测 ★★★
// 在 detectAndCorrectLoop 开头调用，在 geometricVerification 之前运行
// 将 last_loop_matched_kf_id_ 的共视确认点回填到当前帧，
// 使 Sim3Solver 能够找到两帧之间的共视匹配对（即使 matched_kf 不在滑动窗口内）
void CovisManager::prefillMatchedKFObservations(const KeyFrame::Ptr& curr_kf) {
    if (!config_.enable_loop_detection) return;
    if (last_loop_matched_kf_id_ < 0 || last_matched_kf_covis_mp_ids_.empty()) return;

    const Pose& kf_pose = curr_kf->pose;
    const auto& kps = (!curr_kf->undistorted_keypoints.empty() && curr_kf->undistorted_keypoints.size() == curr_kf->keypoints.size())
        ? curr_kf->undistorted_keypoints : curr_kf->keypoints;
    int current_kf_id = curr_kf->frame_id;

    const Eigen::Matrix3d R_wc = kf_pose.q.toRotationMatrix();
    const Eigen::Vector3d& t_wc = kf_pose.t;
    const double fx = config_.K.at<double>(0, 0);
    const double fy = config_.K.at<double>(1, 1);
    const double cx = config_.K.at<double>(0, 2);
    const double cy = config_.K.at<double>(1, 2);
    int img_w = config_.img_width;
    int img_h = config_.img_height;
    const cv::Size img_size(img_w, img_h);
    constexpr double reproj_search_radius = 25.0;
    const double best_dist2_max = reproj_search_radius * reproj_search_radius;

    // 获取上一次回环 matched_kf 的位姿（优先 PGO 缓存）
    const Pose* mPose_ptr = nullptr;
    Pose mPose_local;
    if (pose_graph_optimizer_ &&
        pose_graph_optimizer_->getOptimizedPose(last_loop_matched_kf_id_, &mPose_local)) {
        mPose_ptr = &mPose_local;
    } else if (pnp_original_poses_cache_.count(last_loop_matched_kf_id_)) {
        mPose_ptr = &pnp_original_poses_cache_.at(last_loop_matched_kf_id_);
    }
    if (!mPose_ptr) return;

    const Eigen::Matrix3d R_m = mPose_ptr->q.toRotationMatrix();
    const Eigen::Vector3d& t_m = mPose_ptr->t;

    int backfill_count = 0;
    for (int mp_id : last_matched_kf_covis_mp_ids_) {
        if (!confirmed_mappoints_.count(mp_id)) continue;

        // ★★★ O(1) 检查：使用 hasObservation() 替代 O(n) 线性扫描 ★★★
        if (confirmed_mappoints_.at(mp_id)->hasObservation(current_kf_id)) continue;

        const auto& mp = confirmed_mappoints_.at(mp_id);
        const Eigen::Vector3d& P_w = mp->worldPoint();

        // 重投影到当前帧
        Eigen::Vector3d P_c = R_m * P_w + t_m;
        if (P_c.z() <= 1e-6) continue;

        double u = fx * P_c.x() / P_c.z() + cx;
        double v = fy * P_c.y() / P_c.z() + cy;
        if (u < 0 || u >= img_size.width || v < 0 || v >= img_size.height) continue;

        // 在当前帧特征点中查找最近匹配
        double best_dist2_m = best_dist2_max;
        int best_kpt_m = -1;
        for (int k = 0; k < static_cast<int>(kps.size()); k++) {
            double du = kps[k].pt.x - u, dv = kps[k].pt.y - v;
            double d2 = du*du + dv*dv;
            if (d2 < best_dist2_m) { best_dist2_m = d2; best_kpt_m = k; }
        }
        if (best_kpt_m < 0) continue;

        // 添加当前帧对该确认点的观测
        mp->addObservation({current_kf_id, best_kpt_m, kps[best_kpt_m].pt, P_w, -1.0,
                             cv::Mat()});  // 回填无描述子数据，置空
        obs_key_to_confirmed_mp_[{current_kf_id, best_kpt_m}] = mp_id;
        backfill_count++;
    }

    if (config_.verbose && backfill_count > 0) {
        std::cerr << "[Covis Prefill] 提前回填 " << backfill_count
                  << " 个确认点观测（matched_kf=" << last_loop_matched_kf_id_ << " → curr_kf=" << current_kf_id << "）" << std::endl;
    }
}

void CovisManager::backfillConfirmedObservations(const KeyFrame::Ptr& kf) {
    if (!config_.enable_loop_detection) return;
    if (old_kf_confirmed_obs_.empty()) return;
    if (config_.K.empty() || config_.K.rows < 3 || config_.K.cols < 3) return;  // ★★★ Bug Fix C2: 空 K 矩阵保护 ★★★

    const Pose& kf_pose = kf->pose;
    const auto& kps = (!kf->undistorted_keypoints.empty() && kf->undistorted_keypoints.size() == kf->keypoints.size())
        ? kf->undistorted_keypoints : kf->keypoints;
    const cv::Mat& kf_desc = kf->descriptors;  // ★★★ 获取当前关键帧描述子 ★★★
    int current_kf_id = kf->frame_id;

    const Eigen::Matrix3d R_wc = kf_pose.q.toRotationMatrix();
    const Eigen::Vector3d& t_wc = kf_pose.t;
    const double fx = config_.K.at<double>(0, 0);
    const double fy = config_.K.at<double>(1, 1);
    const double cx = config_.K.at<double>(0, 2);
    const double cy = config_.K.at<double>(1, 2);
    int img_w = config_.img_width;
    int img_h = config_.img_height;
    const cv::Size img_size(img_w, img_h);
    constexpr double reproj_search_radius = 25.0;
    const double best_dist2_max = reproj_search_radius * reproj_search_radius;

    int backfill_count = 0;

    // ★★★ 建立 matched_kf ↔ curr_kf 链接（使 Sim3Solver 能找到共视匹配点） ★★★
    if (last_loop_matched_kf_id_ >= 0 && !last_matched_kf_covis_mp_ids_.empty()) {
        const Pose* mPose_ptr = nullptr;
        Pose mPose_local;
        if (pose_graph_optimizer_ &&
            pose_graph_optimizer_->getOptimizedPose(last_loop_matched_kf_id_, &mPose_local)) {
            mPose_ptr = &mPose_local;
        } else if (pnp_original_poses_cache_.count(last_loop_matched_kf_id_)) {
            mPose_ptr = &pnp_original_poses_cache_.at(last_loop_matched_kf_id_);
        }
        if (mPose_ptr) {
            const Eigen::Matrix3d R_m = mPose_ptr->q.toRotationMatrix();
            const Eigen::Vector3d& t_m = mPose_ptr->t;
            for (int mp_id : last_matched_kf_covis_mp_ids_) {
                if (!confirmed_mappoints_.count(mp_id)) continue;
                // ★★★ O(1) 检查：使用 hasObservation() 替代 O(n) 线性扫描 ★★★
                if (confirmed_mappoints_.at(mp_id)->hasObservation(current_kf_id)) continue;
                const auto& mp = confirmed_mappoints_.at(mp_id);
                const Eigen::Vector3d& P_w = mp->worldPoint();
                Eigen::Vector3d P_c = R_m * P_w + t_m;
                if (P_c.z() <= 1e-6) continue;
                double u = fx * P_c.x() / P_c.z() + cx;
                double v = fy * P_c.y() / P_c.z() + cy;
                if (u < 0 || u >= img_size.width || v < 0 || v >= img_size.height) continue;
                double best_dist2_m = best_dist2_max;
                int best_kpt_m = -1;
                for (int k = 0; k < static_cast<int>(kps.size()); k++) {
                    double du = kps[k].pt.x - u, dv = kps[k].pt.y - v;
                    double d2 = du*du + dv*dv;
                    if (d2 < best_dist2_m) { best_dist2_m = d2; best_kpt_m = k; }
                }
                if (best_kpt_m < 0) continue;
                // ★★★ 添加描述子 ★★★
                cv::Mat curr_desc = (!kf_desc.empty() && best_kpt_m < kf_desc.rows)
                                 ? kf_desc.row(best_kpt_m) : cv::Mat();
                mp->addObservation({current_kf_id, best_kpt_m, kps[best_kpt_m].pt, P_w, -1.0,
                                   curr_desc});
                obs_key_to_confirmed_mp_[{current_kf_id, best_kpt_m}] = mp_id;
                backfill_count++;
            }
        }
    }

    // 遍历所有已滑出窗口的旧关键帧
    for (const auto& kf_kv : old_kf_confirmed_obs_) {
        int old_kf_id = kf_kv.first;
        const auto& obs_list = kf_kv.second;
        if (obs_list.empty()) continue;

        // 获取旧关键帧的世界位姿（优先 PGO 缓存，否则原始 PnP）
        Pose old_pose;
        bool have_pose = false;
        if (pose_graph_optimizer_ &&
            pose_graph_optimizer_->getOptimizedPose(old_kf_id, &old_pose)) {
            have_pose = true;
        } else if (pnp_original_poses_cache_.count(old_kf_id)) {
            old_pose = pnp_original_poses_cache_.at(old_kf_id);
            have_pose = true;
        }
        if (!have_pose) continue;

        for (const auto& obs : obs_list) {
            int mp_id = obs.first;
            // int old_kpt_idx = obs.second;  // 旧关键帧的特征索引（暂不需要）

            if (!confirmed_mappoints_.count(mp_id)) continue;
            const auto& mp = confirmed_mappoints_.at(mp_id);
            const Eigen::Vector3d& P_w = mp->worldPoint();

            // ★★★ O(1) 检查：使用 hasObservation() 替代 O(n) 线性扫描 ★★★
            if (mp->hasObservation(current_kf_id)) continue;

            // 重投影到当前关键帧
            Eigen::Vector3d P_c_cur = R_wc * P_w + t_wc;
            if (P_c_cur.z() <= 1e-6) continue;
            double u_cur = fx * P_c_cur.x() / P_c_cur.z() + cx;
            double v_cur = fy * P_c_cur.y() / P_c_cur.z() + cy;
            if (u_cur < 0 || u_cur >= img_size.width ||
                v_cur < 0 || v_cur >= img_size.height) continue;

            // 在当前帧特征点中查找最近的匹配
            double best_dist2 = best_dist2_max;
            int best_curr_kpt = -1;
            for (int kpt_idx = 0; kpt_idx < static_cast<int>(kps.size()); kpt_idx++) {
                double du = kps[kpt_idx].pt.x - u_cur;
                double dv = kps[kpt_idx].pt.y - v_cur;
                double dist2 = du * du + dv * dv;
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_curr_kpt = kpt_idx;
                }
            }

            if (best_curr_kpt < 0) continue;

            // 添加当前关键帧对该确认点的观测（★ 添加描述子 ★）
            cv::Mat curr_desc = (!kf_desc.empty() && best_curr_kpt < kf_desc.rows)
                             ? kf_desc.row(best_curr_kpt) : cv::Mat();
            mp->addObservation({current_kf_id, best_curr_kpt,
                                kps[best_curr_kpt].pt, P_w, -1.0,
                                curr_desc});
            obs_key_to_confirmed_mp_[{current_kf_id, best_curr_kpt}] = mp_id;
            backfill_count++;
        }
    }

    if (config_.verbose && backfill_count > 0) {
        std::cerr << "[Covis Backfill] 回填 " << backfill_count
                  << " 个确认点观测（来自 " << old_kf_confirmed_obs_.size()
                  << " 个旧关键帧）" << std::endl;
    }
}

} // namespace frontend
} // namespace sfc_core
