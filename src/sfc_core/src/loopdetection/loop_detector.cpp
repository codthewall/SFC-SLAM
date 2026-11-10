// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

// covis_manager.hpp 必须在 loop_detector.hpp 之前引入（打破 include 循环）。
// 当 loopdetection 头文件链解析时，CovisManager 的完整定义尚未出现。
// 通过在 cpp 层面先引入 covis_manager（它只 #include covis_core.hpp，不含 loopdetection），
// 让 CovisManager 在 loopdetection 头文件链解析完成前就已完整定义。
#include "sfc_core/frontend/covis_manager.hpp"
#include "sfc_core/loopdetection/loop_detector.hpp"
#include "sfc_core/loopdetection/sim3_optimizer.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#endif

namespace sfc_core {
namespace loopdetection {

// 跨平台 popcount：Windows 用 __popcnt，Linux/macOS 用 __builtin_popcount
#ifdef _WIN32
inline int popcnt(unsigned int x) { return __popcnt(x); }
#else
inline int popcnt(unsigned int x) { return __builtin_popcount(x); }
#endif

// ================================================================
// EM 降级路径的数值下限（R-018：全部具名，单位和判据写在注释里）
// ================================================================
// |t| 低于此值视为 EM 没给出平移（recoverPose 在纯旋转时返回零平移）。
constexpr double kEMTranslationEpsilonM = 1e-6;
// 样本投影到 EM 单位方向上必须为正且超过此值，否则该样本与 EM 解方向矛盾，丢弃。
constexpr double kEMBaselineMinProjectionM = 1e-3;
// 少于 3 对样本时中位数没有稳健性，宁可拒绝这次回环（D2-b）。
constexpr size_t kEMBaselineMinSamples = 3;
// 中位数基线低于此值视为退化（两帧几乎重合，方向不可信）。
constexpr double kEMBaselineMinMeters = 1e-3;

// ================================================================
// 共享辅助函数：世界坐标 → 相机坐标
// ================================================================
inline void worldToCameraPts(
        const std::vector<Eigen::Vector3d>& pts_world,
        const Eigen::Matrix3d& R_cw,
        const Eigen::Vector3d& t_cw,
        std::vector<Eigen::Vector3d>& pts_cam) {
    pts_cam.clear();
    pts_cam.reserve(pts_world.size());
    for (const auto& p_w : pts_world) {
        if (p_w.norm() < 1e-6) {
            pts_cam.emplace_back(0, 0, 0);
        } else {
            pts_cam.emplace_back(R_cw * p_w + t_cw);
        }
    }
}

// 从世界位姿提取 world→camera 变换矩阵的旋转和偏移
inline void extractW2CTransform(
        const ::sfc_core::frontend::Pose& pose_world,
        Eigen::Matrix3d* R_cw, Eigen::Vector3d* t_cw) {
    Eigen::Matrix3d R_wc = pose_world.q.toRotationMatrix();
    *R_cw = R_wc.transpose();
    *t_cw = -(*R_cw) * pose_world.t;
}

// SVD 正交化（ORB_SLAM3 风格），使旋转矩阵满足 SO(3) 约束
inline void svdOrthogonalize(Eigen::Matrix3d* R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(*R,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    *R = svd.matrixU() * svd.matrixV().transpose();
    if (R->determinant() < 0) R->col(2) *= -1;
}

// ================================================================
// kd-tree 辅助：O(log K) 最近邻搜索，加速 backfill
// 避免 O(K^2) 逐点遍历（K=关键点数量）
// ================================================================
class SimpleKDTree2D {
public:
    struct Node {
        float x = 0, y = 0;
        int idx = -1;
        int left = -1, right = -1;
    };
    std::vector<Node> nodes_;
    std::vector<int> root_indices_;

    void build(const std::vector<cv::KeyPoint>& kps) {
        nodes_.clear();
        root_indices_.clear();
        for (int i = 0; i < static_cast<int>(kps.size()); ++i) {
            Node n;
            n.x = static_cast<float>(kps[i].pt.x);
            n.y = static_cast<float>(kps[i].pt.y);
            n.idx = i;
            nodes_.push_back(n);
        }
        if (nodes_.empty()) return;
        for (size_t i = 0; i < nodes_.size(); ++i)
            root_indices_.push_back(buildRecursive(0, static_cast<int>(nodes_.size()), static_cast<int>(i)));
    }

    int nnSearch(float x, float y, float* out_dist2) const {
        int best_idx = -1;
        *out_dist2 = 1e9f;
        for (int rid : root_indices_)
            nnSearchRecursive(rid, x, y, 0, &best_idx, out_dist2);
        return best_idx;
    }

private:
    int buildRecursive(int l, int r, int depth) {
        if (l >= r) return -1;
        int mid = (l + r) >> 1;
        int axis = depth & 1;
        std::nth_element(nodes_.begin() + l, nodes_.begin() + mid, nodes_.begin() + r,
            [axis](const Node& a, const Node& b) {
                return axis == 0 ? a.x < b.x : a.y < b.y;
            });
        int node_idx = mid;
        nodes_[mid].left  = buildRecursive(l, mid, depth + 1);
        nodes_[mid].right = buildRecursive(mid + 1, r, depth + 1);
        return node_idx;
    }

    void nnSearchRecursive(int node_idx, float x, float y, int depth,
                           int* best_idx, float* best_dist2) const {
        if (node_idx < 0) return;
        const Node& node = nodes_[node_idx];
        float dx = node.x - x, dy = node.y - y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 < *best_dist2) { *best_dist2 = dist2; *best_idx = node.idx; }
        int axis = depth & 1;
        float diff = axis == 0 ? dx : dy;
        int first = diff < 0 ? node.left : node.right;
        int second = first == node.left ? node.right : node.left;
        nnSearchRecursive(first, x, y, depth + 1, best_idx, best_dist2);
        if (diff * diff < *best_dist2)
            nnSearchRecursive(second, x, y, depth + 1, best_idx, best_dist2);
    }
};

// ====================== LoopDetector ======================

LoopDetector::LoopDetector(const LoopDetectorConfig& config)
    : config_(config) {
    if (!config_.camera_K.empty() && config_.camera_K.rows == 3 &&
        config_.camera_K.cols == 3) {
        K_cv_ = config_.camera_K.clone();
    } else {
        K_cv_ = cv::Mat::eye(3, 3, CV_64F);
    }
    D_cv_ = config_.camera_D.empty() ? cv::Mat() : config_.camera_D.clone();

    place_recog_ = std::make_unique<PlaceRecognition>();

    // D3：几何验证参数整份来自 LoopDetectorConfig::geometric_verifier，
    // 这个成员由 loadAllConfigs() 从 config/geometric_verifier.yaml 灌入。
    // 原来这里有一串 "geo_config.xxx = config_.geo_xxx" 的搬运 —— 那是同一批参数
    // 的第二份声明（R-023），改 yaml 不生效、改代码两处要同步。
    GeometricVerifierConfig geo_config = config_.geometric_verifier;
    // verbose 只挂在 LoopDetectorConfig 上（回环层统一开关），几何验证跟随它。
    geo_config.verbose = config_.verbose;
    geometric_verifier_ = std::make_unique<GeometricVerifier>(geo_config);

    if (config_.verbose) {
        std::cerr << "[LoopDetector] 初始化完成: DBoW2 + GeometricVerifier (在线构建词袋)" << std::endl;
    }
}

void LoopDetector::addKeyFrame(int kf_id,
                               const std::vector<cv::KeyPoint>& keypoints,
                               const std::vector<Eigen::Vector3d>& pts3d_cam,
                               const cv::Mat& descriptors,
                               const sfc_core::frontend::Pose& pose) {
    KeyFrameEntry entry;
    entry.kf_id = kf_id;
    entry.descriptors = descriptors.clone();
    entry.pose = pose;
    entry.bow_vector = computeBoW(descriptors);
    entry.keypoints = keypoints;
    entry.pts3d = pts3d_cam;

    keyframe_db_[kf_id] = entry;
    keyframe_ids_.push_back(kf_id);

    if (place_recog_ && !place_recog_->isVocabularyReady()) {
        place_recog_->collectForVocabulary(descriptors);
        if (place_recog_->isReadyToBuild()) {
            place_recog_->tryBuildVocabulary(10, 6);
            if (place_recog_->isVocabularyReady()) {
                place_recog_->enableIncrementalVocabulary(25.0);
                { if (config_.verbose) std::cerr << "[LoopDetector] 词袋构建完成，启用增量更新" << std::endl; }
            }
        }
    }

    if (place_recog_ && place_recog_->isVocabularyReady()) {
        place_recog_->addKeyFrame(kf_id, descriptors, keypoints, pose.t);
    }
}

std::vector<int> LoopDetector::computeBoW(const cv::Mat& descriptors) const {
    std::vector<int> bow;
    bow.reserve(descriptors.rows * 4);
    for (int i = 0; i < descriptors.rows; ++i) {
        const uchar* row = descriptors.ptr<uchar>(i);
        for (int j = 0; j < 8; j += 2) {
            int word = row[j] | (row[j + 1] << 8);
            bow.push_back(word % 256);
        }
    }
    std::sort(bow.begin(), bow.end());
    return bow;
}

int LoopDetector::commonWordCount(const std::vector<int>& a,
                                  const std::vector<int>& b) const {
    std::vector<int> inter;
    inter.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::back_inserter(inter));
    return static_cast<int>(inter.size());
}

std::vector<cv::DMatch> LoopDetector::matchDescriptors(const cv::Mat& query_desc,
                                                        const cv::Mat& train_desc) const {
    if (query_desc.empty() || train_desc.empty()) return {};
    if (query_desc.rows < 2 || train_desc.rows < 2) return {};

    const int D = query_desc.cols;
    const int Q = query_desc.rows;
    const int T = train_desc.rows;

    std::vector<std::pair<int, int>> qnn1(Q, {-1, 9999}), qnn2(Q, {-1, 9999});
    std::vector<std::pair<int, int>> tnn1(T, {-1, 9999}), tnn2(T, {-1, 9999});

    for (int qi = 0; qi < Q; ++qi) {
        const uchar* qrow = query_desc.ptr<uchar>(qi);
        for (int ti = 0; ti < T; ++ti) {
            const uchar* trow = train_desc.ptr<uchar>(ti);
            int dist = 0;
            for (int k = 0; k < D; ++k) dist += popcnt(qrow[k] ^ trow[k]);

            if (dist < qnn2[qi].second) {
                if (dist < qnn1[qi].second) { qnn2[qi] = qnn1[qi]; qnn1[qi] = {ti, dist}; }
                else { qnn2[qi] = {ti, dist}; }
            }
            if (dist < tnn2[ti].second) {
                if (dist < tnn1[ti].second) { tnn2[ti] = tnn1[ti]; tnn1[ti] = {qi, dist}; }
                else { tnn2[ti] = {ti, dist}; }
            }
        }
    }

    std::vector<cv::DMatch> cross_verified;
    cross_verified.reserve(Q);
    for (int qi = 0; qi < Q; ++qi) {
        if (qnn1[qi].first < 0) continue;
        if (qnn1[qi].second > config_.max_match_distance) continue;
        int ti = qnn1[qi].first;
        if (tnn1[ti].first != qi) continue;
        cross_verified.push_back({qi, ti, static_cast<float>(qnn1[qi].second)});
    }

    if (config_.lowe_ratio_threshold >= 1.0) return cross_verified;

    std::vector<cv::DMatch> result;
    result.reserve(cross_verified.size());
    for (const cv::DMatch& m : cross_verified) {
        int d1 = qnn1[m.queryIdx].second;
        int d2 = qnn2[m.queryIdx].second;
        if (d2 > config_.max_match_distance) { result.push_back(m); continue; }
        if (d1 <= config_.lowe_ratio_threshold * d2) result.push_back(m);
    }
    return result;
}

bool LoopDetector::geometricVerification(
        const std::vector<cv::KeyPoint>& kp_query,
        const std::vector<cv::KeyPoint>& kp_train,
        const std::vector<Eigen::Vector3d>& pts3d_query_world,
        const std::vector<Eigen::Vector3d>& pts3d_train_world,
        const std::vector<cv::DMatch>& matches,
        int curr_kf_id,
        int matched_kf_id,
        Eigen::Matrix4d* out_relative_pose,
        int* out_inliers,
        bool* out_is_icp,
        const ::sfc_core::frontend::CovisManager* covis_mgr) const {

    if (matches.empty()) return false;

    auto it_curr = keyframe_db_.find(curr_kf_id);
    auto it_matched = keyframe_db_.find(matched_kf_id);
    if (it_curr == keyframe_db_.end() || it_matched == keyframe_db_.end()) {
        return false;
    }
    const auto& pose_curr = it_curr->second.pose;
    const auto& pose_matched = it_matched->second.pose;

    Eigen::Matrix3d R_cw_curr, R_cw_match;
    Eigen::Vector3d t_cw_curr, t_cw_match;
    extractW2CTransform(pose_curr, &R_cw_curr, &t_cw_curr);
    extractW2CTransform(pose_matched, &R_cw_match, &t_cw_match);

    std::vector<Eigen::Vector3d> pts3d_query_cam, pts3d_train_cam;
    worldToCameraPts(pts3d_query_world, R_cw_curr, t_cw_curr, pts3d_query_cam);
    worldToCameraPts(pts3d_train_world, R_cw_match, t_cw_match, pts3d_train_cam);

    std::vector<Eigen::Vector3d> pts3d_query_cam_valid;
    std::vector<Eigen::Vector3d> pts3d_train_cam_valid;
    std::vector<cv::DMatch> valid_matches;
    for (size_t i = 0; i < matches.size(); ++i) {
        // L7：这里原来用循环下标 i 去取 pts3d_query_cam[i] / pts3d_train_cam[i]，
        // 但这两个数组是按**关键点下标**排列的，matches[i] 只是第 i 个匹配，
        // 它的 queryIdx/trainIdx 才是关键点下标。两者一旦不同序（交叉匹配打乱顺序、
        // 或前面过滤掉过点），取到的就是别的点的深度 —— 后面所有 ICP 结果都建在错配上。
        const int qi = matches[i].queryIdx;
        const int ti = matches[i].trainIdx;
        if (qi < 0 || qi >= static_cast<int>(pts3d_query_cam.size())) continue;
        if (ti < 0 || ti >= static_cast<int>(pts3d_train_cam.size())) continue;
        // ★★★ 修复：使用相机坐标 z 轴（深度）而非原点距离 ★★★
        // norm() = sqrt(x²+y²+z²)，对 0.5m 深度但 0.5m 侧向偏移的点，norm≈0.7 > 0.1 被误拒
        // z() = 相机坐标系的深度值，这才是判断 3D 点是否在相机前方（有效深度）的正确方法
        if (pts3d_query_cam[qi].z() > 1e-6 && pts3d_train_cam[ti].z() > 1e-6) {
            // L6：传给 ICP 的两个 3D 数组是**压缩后**的（只含通过深度筛选的点），
            // 所以匹配索引必须同步重映射成压缩后的序号。原来这里保留原始关键点下标、
            // 却把压缩数组喂给 ICP，ICP 内部拿 queryIdx 直接索引 pts3d1，
            // 轻则越界被 continue 掉、重则取到另一个点的坐标 —— 等于用错配算内点率。
            const int compact_idx = static_cast<int>(pts3d_query_cam_valid.size());
            pts3d_query_cam_valid.push_back(pts3d_query_cam[qi]);
            pts3d_train_cam_valid.push_back(pts3d_train_cam[ti]);
            cv::DMatch compact_match = matches[i];
            compact_match.queryIdx = compact_idx;
            compact_match.trainIdx = compact_idx;
            valid_matches.push_back(compact_match);
        }
    }

    // ============================================================
    // ★★★ 核心修复：混合 3D-3D 匹配策略 ★★★
    //
    // 问题诊断（从日志观察）：
    //   1. 双向 KD-tree 重投影匹配 0% 成功（30px 半径太小，PnP 漂移导致重投影误差大）
    //   2. 直接 3D-3D 匹配依赖原始深度点（未 BA 精化），估计的 Sim3 旋转错误（38.6°）
    //   3. ICP 路径使用 dummy_matches（按索引对应），没有描述子验证，匹配对可能不对应
    //   4. 回环间隔 55 帧（KF=255 ↔ KF=200），漂移大，ICP 难以准确对齐
    //
    // 修复策略（两层）：
    //   第一层：用 confirmed map point 优先构建匹配（先从观测中找，再 KD-tree 搜索）
    //   第二层：用描述子匹配 + 原始深度点作为 fallback（不怕漂移）
    // ============================================================
    std::vector<::sfc_core::frontend::CovisMatch> covis_matches;
    // L2：阶段 2（描述子 fallback）在 covis_mgr 为空时也要执行，所以去重集合必须在这里，
    // 两个阶段共用同一份，否则同一个关键点会被重复计入对应点。
    std::unordered_set<int> used_curr_kpts;
    std::unordered_set<int> used_match_kpts;
    if (covis_mgr != nullptr) {
        const double fx = K_cv_.at<double>(0, 0);
        const double fy = K_cv_.at<double>(1, 1);
        const double cx = K_cv_.at<double>(0, 2);
        const double cy = K_cv_.at<double>(1, 2);
        const int img_w = static_cast<int>(cx * 2);
        const int img_h = static_cast<int>(cy * 2);

        SimpleKDTree2D kdtree_curr, kdtree_match;
        kdtree_curr.build(kp_query);
        kdtree_match.build(kp_train);

        // ---- 阶段 1：用 confirmed map point 构建匹配（最可靠）----
        // 策略：对每个 confirmed map point，先检查是否已有 curr/match 帧的观测，
        //       再用 KD-tree 扩大搜索范围（容忍 PnP 漂移导致的重投影误差）


        constexpr double reproj_search_radius_tight = 30.0;  // 有观测时用严格搜索
        constexpr double reproj_search_radius_wide = 80.0;   // 无观测时用宽松搜索
        const double tight_dist2_max = reproj_search_radius_tight * reproj_search_radius_tight;
        const double wide_dist2_max = reproj_search_radius_wide * reproj_search_radius_wide;

        int mp_from_observation = 0, mp_from_kdtree = 0;

        for (const auto& kv : covis_mgr->getAllConfirmedMapPoints()) {
            const auto& mp = kv.second;
            if (!mp) continue;

            Eigen::Vector3d pt_world = mp->worldPoint();
            if (pt_world.norm() < 1e-6) continue;

            int curr_kpt_idx = -1, match_kpt_idx = -1;
            bool seen_curr = false, seen_match = false;

            // 步骤 1a：检查 map point 是否已有当前帧和匹配帧的观测（最可靠）
            for (const auto& obs : mp->observations()) {
                if (obs.frame_id == curr_kf_id) {
                    seen_curr = true;
                    curr_kpt_idx = obs.kpt_idx;
                }
                if (obs.frame_id == matched_kf_id) {
                    seen_match = true;
                    match_kpt_idx = obs.kpt_idx;
                }
            }

            if (seen_curr && seen_match) {
                // 两个帧都有观测，直接使用（最高可靠性）
                if (used_curr_kpts.insert(curr_kpt_idx).second &&
                    used_match_kpts.insert(match_kpt_idx).second) {
                    covis_matches.push_back({pt_world, curr_kpt_idx, match_kpt_idx, -1.0});
                    mp_from_observation++;
                }
                continue;
            }

            // 步骤 1b：世界坐标重投影到相机坐标，检查是否可见
            Eigen::Vector3d P_c_curr = pose_curr.q.toRotationMatrix() * pt_world + pose_curr.t;
            if (P_c_curr.z() <= 1e-6) continue;
            Eigen::Vector3d P_c_match = pose_matched.q.toRotationMatrix() * pt_world + pose_matched.t;
            if (P_c_match.z() <= 1e-6) continue;

            double u_curr = fx * P_c_curr.x() / P_c_curr.z() + cx;
            double v_curr = fy * P_c_curr.y() / P_c_curr.z() + cy;
            double u_match = fx * P_c_match.x() / P_c_match.z() + cx;
            double v_match = fy * P_c_match.y() / P_c_match.z() + cy;

            if (u_curr < 0 || u_curr >= img_w || v_curr < 0 || v_curr >= img_h) continue;
            if (u_match < 0 || u_match >= img_w || v_match < 0 || v_match >= img_h) continue;

            // 步骤 1c：两步 KD-tree 搜索（容忍 PnP 漂移）
            //   第一步：用严格半径（30px）搜索（适用于当前帧已有观测的情况）
            //   第二步：用宽松半径（80px）搜索（适用于需要重新找对应点的情况）
            float dist2_curr = 1e10f, dist2_match = 1e10f;
            int best_curr_k = kdtree_curr.nnSearch(static_cast<float>(u_curr), static_cast<float>(v_curr), &dist2_curr);
            int best_match_k = kdtree_match.nnSearch(static_cast<float>(u_match), static_cast<float>(v_match), &dist2_match);

            bool curr_ok = (best_curr_k >= 0 && dist2_curr <= tight_dist2_max);
            bool match_ok = (best_match_k >= 0 && dist2_match <= tight_dist2_max);

            // 如果严格匹配失败，尝试宽松匹配
            if (!curr_ok || !match_ok) {
                if (!curr_ok && best_curr_k >= 0 && dist2_curr <= wide_dist2_max) curr_ok = true;
                if (!match_ok && best_match_k >= 0 && dist2_match <= wide_dist2_max) match_ok = true;
            }

            if (curr_ok && match_ok) {
                if (used_curr_kpts.insert(best_curr_k).second &&
                    used_match_kpts.insert(best_match_k).second) {
                    covis_matches.push_back({pt_world, best_curr_k, best_match_k, -1.0});
                    mp_from_kdtree++;
                }
            }
        }

        int total_mp_matches = mp_from_observation + mp_from_kdtree;
        if (total_mp_matches > 0) {
            { if (config_.verbose) std::cerr << "[GeoVerif Sim3] MapPoint 匹配: obs=" << mp_from_observation
                      << " kdtree=" << mp_from_kdtree
                      << " total=" << total_mp_matches << std::endl; }
        }

    }  // if (covis_mgr != nullptr)：只守阶段 1（L2 后阶段 2 与 Sim3 在守卫外）

    // L2：下面是阶段 2 —— 用**描述子 matches + keyframe_db_ 里已有的世界坐标点**
    // 构造对应点，不依赖 CovisManager。原实现把它写在 if (covis_mgr != nullptr) 里，
    // 而 Tracking::detectAndCorrectLoop() 传的是 nullptr，于是这段永远不执行、
    // covis_matches 恒空、Sim3 永不进入 —— 回环就只能退化成错的 ICP / 无尺度的 EM。
    // ---- 阶段 2：用描述子匹配 + 原始深度点作为 fallback ----
    // 这是最后的 fallback，适用于 confirmed map point 覆盖不足的情况
    int mp_fallback_count = 0;
    for (const auto& m : matches) {
        int qi = m.queryIdx, ti = m.trainIdx;
        if (used_curr_kpts.count(qi) || used_match_kpts.count(ti)) continue;
        if (qi < 0 || qi >= static_cast<int>(pts3d_query_world.size()) ||
            ti < 0 || ti >= static_cast<int>(pts3d_train_world.size())) continue;
        const Eigen::Vector3d& pt_curr = pts3d_query_world[qi];
        const Eigen::Vector3d& pt_matched = pts3d_train_world[ti];
        if (pt_curr.norm() < 1e-6 || pt_matched.norm() < 1e-6) continue;
        Eigen::Vector3d pc1 = R_cw_curr * pt_curr + t_cw_curr;
        Eigen::Vector3d pc2 = R_cw_match * pt_matched + t_cw_match;
        if (pc1.z() < 0.1 || pc1.z() > 15.0 || pc2.z() < 0.1 || pc2.z() > 15.0) continue;
        covis_matches.push_back({pt_curr, qi, ti, -1.0});
        mp_fallback_count++;
    }
    if (mp_fallback_count > 0) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 描述子 fallback: " << mp_fallback_count << " 个匹配点" << std::endl; }
    }

    if (static_cast<int>(covis_matches.size()) >= 1) {
        std::vector<Eigen::Vector3d> pts_curr, pts_matched;
        std::vector<int> curr_kpt_indices, matched_kpt_indices;
        // L1：这里必须传**世界坐标**。
        // geometricVerificationWithSim3() 的形参叫 curr_pts3d_world /
        // matched_pts3d_world，函数内部（:1165、:1192、:1202）已经把它们各做一次
        // world→camera。原来这里先转了 R_cw*p+t，函数里再转一次，等于把旋转和平移
        // 各施加两遍：Sim3 收到的点云整体被搬到相机背后，解出来的尺度/旋转都是噪声，
        // 这也是 covis 路径"永远失败、只能回落 ICP+EM"的根因。
        for (const auto& cm : covis_matches) {
            pts_curr.push_back(cm.pt_world);
            pts_matched.push_back(cm.pt_world);
            curr_kpt_indices.push_back(cm.curr_kpt_idx);
            matched_kpt_indices.push_back(cm.matched_kpt_idx);
        }

        Eigen::Matrix4d T_sim3;
        int sim3_inliers = 0;
        bool is_icp = false;
        bool sim3_ok = geometricVerificationWithSim3(
            pts_curr, curr_kpt_indices,
            pts_matched, matched_kpt_indices,
            pose_curr, pose_matched,
            kp_query, kp_train,  // ★★★ 传入原始关键点用于像素坐标 ★★★
            &T_sim3, &sim3_inliers, &is_icp);

        if (sim3_ok) {
            if (out_relative_pose) *out_relative_pose = T_sim3;
            if (out_inliers) *out_inliers = sim3_inliers;
            if (out_is_icp) *out_is_icp = is_icp;
            return true;
        }
    }

    // ============================================================
    // ★★★ 关键修复：当 hybrid Sim3 失败时，跳过 ICP+dummy_matches ★★★
    //
    // 问题诊断（从日志）：
    //   1. hybrid Sim3 用 covis_matches 失败（inlier<4），fallback 到 ICP
    //   2. ICP 用 dummy_matches（索引对应，不反映真实共视关系），导致错误旋转（38.6°）
    //   3. CovisManager 拒绝 38.6° 修正，回环完全失效
    //
    // 修复策略：
    //   - 当 covis_matches >= 2 但 hybrid Sim3 失败时，强制降级到 EM（放弃尺度，用方向）
    //   - 不再调用 ICP（ICP+dummy_matches 结果不可靠）
    //   - EM 已有固定 1.0m 尺度，比 ICP 的错误旋转安全得多
    // ============================================================
    bool skip_icp_dummy = (!covis_matches.empty() && covis_matches.size() < 3);
    if (skip_icp_dummy) {
        { if (config_.verbose) std::cerr << "[GeoVerif] hybrid Sim3 失败（covis_matches=" << covis_matches.size()
                  << "），跳过 ICP+dummy_matches，降级到 Essential Matrix（避免错误旋转）" << std::endl; }
    }

    // ============================================================
    // 方法 1: RGB-D 专用 3D-3D ICP（带真实尺度）
    // 仅当 covis_matches 为空（没有任何共视点）时才使用 ICP
    //
    // L6+L7：原来这里造了一份 dummy_matches(i,i) 喂给 ICP —— 索引一一对应，
    // 等于告诉 ICP"第 i 个点就是第 i 个点"，完全没有描述子约束；而算内点率时
    // 分母又用 matches.size()（真实匹配数），分子却是那份假匹配的内点数，
    // 两个数不同源，比例本身没有意义。
    // 现在直接用 valid_matches（真实描述子匹配 + 已按深度过滤），
    // 分母同步改成 valid_matches.size()。
    // ============================================================
    if (!skip_icp_dummy && geometric_verifier_ && !pts3d_query_cam_valid.empty() && !pts3d_train_cam_valid.empty()) {
        Eigen::Matrix4d T_icp;
        int icp_inliers = geometric_verifier_->verifyWithICP3D(
            pts3d_query_cam_valid, pts3d_train_cam_valid, valid_matches, K_cv_, &T_icp);

        if (icp_inliers > 0) {
            double inlier_ratio = valid_matches.empty()
                                ? 0.0
                                : static_cast<double>(icp_inliers) / static_cast<double>(valid_matches.size());
            double icp_t_norm = T_icp.block<3,1>(0,3).norm();
            { if (config_.verbose) std::cerr << "[GeoVerif] ★ 3D-3D ICP 验证通过！内点=" << icp_inliers
                      << " inlier_ratio=" << std::fixed << std::setprecision(2) << inlier_ratio
                      << " |t_icp|=" << icp_t_norm << " m" << std::endl; }

            Eigen::Matrix3d R_icp = T_icp.block<3,3>(0,0);
            svdOrthogonalize(&R_icp);
            T_icp.block<3,3>(0,0) = R_icp;

            double rot_trace = (R_icp.trace() - 1.0) / 2.0;
            rot_trace = std::max(-1.0, std::min(1.0, rot_trace));
            double rot_angle_deg = std::acos(rot_trace) * 180.0 / M_PI;
            { if (config_.verbose) std::cerr << "[GeoVerif] 旋转角度: " << rot_angle_deg << "°" << std::endl; }

            // ★★★ 关键修复：ICP 翻译阈值降到 3.0m（室内场景合理值）★★★
            // ICP+dummy_matches 产生的翻译经常达到 4-8m（错误的旋转导致）
            // 即使 covis_matches 为空，3.0m 也远大于 fr1/desk2 的真实回环距离
            // 拒绝不可靠的 ICP，直接降级到 EM（更安全）
            const double MAX_ICP_T_FOR_ROBUST_LOOP = 3.0;
            double max_icp_t = MAX_ICP_T_FOR_ROBUST_LOOP;

            // ★★★ 关键修复：删除 hybrid fallback 路径（ICP rotation + odometry translation）★★★
            // 这是导致 38.6° 错误旋转的根本原因
            // ICP 旋转在 dummy_matches 下不可靠，odometry 翻译已被 PnP 漂移污染
            // 任何 >3.0m 的 ICP 结果都应被拒绝（不是降级到 hybrid，而是完全拒绝）
            bool icp_passed = (icp_t_norm <= max_icp_t);

            // ★★★ 新增：ICP 旋转合理性检查 ★★★
            // 日志中 ICP 报告的旋转达到 38.6°（如 KF=255↔200），这远大于室内场景真实旋转
            // 室内回环通常是 PnP 漂移导致的平移误差累积，旋转应该很小（<25°）
            if (icp_passed && rot_angle_deg <= 25.0) {
                if (out_relative_pose) *out_relative_pose = T_icp;
                if (out_inliers) *out_inliers = icp_inliers;
                if (out_is_icp) *out_is_icp = true;
                { if (config_.verbose) std::cerr << std::string(80, '=') << "\n"; }
                return true;
            }

            if (!icp_passed) {
                { if (config_.verbose) std::cerr << "[GeoVerif] ICP |t|=" << icp_t_norm << "m > " << max_icp_t
                          << "m（室内场景不合理），跳过 ICP，降级到 Essential Matrix" << std::endl; }
            } else {
                { if (config_.verbose) std::cerr << "[GeoVerif] ICP 旋转=" << rot_angle_deg << "° > 25°（室内回环异常），"
                          << "跳过 ICP，降级到 Essential Matrix" << std::endl; }
            }
        } else {
            // L7：这里原来有一条「ICP 内点=0 → 采用 ICP 旋转 + 里程计尺度」的回退分支。
            // 内点为 0 意味着 ICP 的解没有任何数据支持，却把它的旋转矩阵拿来用、
            // 只把平移换成两帧里程计距离 —— 这正是上面注释里反复提到的 38.6° 错误旋转来源
            // （注释写着「已删除 hybrid fallback」，代码其实还在）。内点为 0 时唯一诚实的做法
            // 是不采用 ICP，直接落到后面的 Essential Matrix 降级路径。
            if (config_.verbose) {
                std::cerr << "[GeoVerif] ICP 内点为 0，不采用 ICP 结果，降级到 Essential Matrix" << std::endl;
            }
        }
    }

    // ICP 跳过/失败 → 降级到 Essential Matrix
    {
        { if (config_.verbose) std::cerr << "[GeoVerif] >>> 方法2: Essential Matrix（降级模式，容忍内点不足）\n"; }
        Eigen::Matrix4d T_em;
        int em_inliers = 0;
        if (geometric_verifier_) {
            em_inliers = geometric_verifier_->verifyWithEssential(
                kp_query, kp_train, matches, K_cv_, &T_em);
        }
        if (em_inliers > 0) {
            double t_norm_em = T_em.block<3,1>(0,3).norm();
            Eigen::Vector3d t_em_dir;
            if (t_norm_em > kEMTranslationEpsilonM) {
                t_em_dir = T_em.block<3,1>(0,3) / t_norm_em;
            } else {
                t_em_dir = Eigen::Vector3d::UnitX();
            }
            // D2：RGB-D 有深度，EM 的平移尺度不需要猜。
            //
            // EM 解出的 T_em 把 query 相机系映射到 train 相机系（p_train = R_em * p_query + t_em），
            // 所以每一对有效对应点都给出一个平移样本 t = p_train - R_em * p_query。
            // 把样本投影到 EM 给出的单位方向上取中位数，就是用深度直接测出的基线：
            // 与里程计漂移无关，也不是常数（原实现写死 kEMFallbackTranslationScale = 1.0，
            // 等于往图里塞一条尺度是假的回环边，CODE-RULES R-025 的反例）。
            //
            // D2-c 的帧间距离闸门复用同一批样本：separation_m 取样本模长的中位数，
            // 不再用 pose_curr.t - pose_matched.t —— 回环场景里这两帧的 PnP 结果恰恰漂得最厉害，
            // 拿一个已知在漂的量去判断"这两帧隔得够不够远"，闸门本身就没有判据。
            std::vector<double> baseline_samples;
            double separation_m = 0.0;
            if (t_norm_em > kEMTranslationEpsilonM) {
                const Eigen::Matrix3d R_em = T_em.block<3, 3>(0, 0);
                const size_t n_pairs = std::min(valid_matches.size(),
                                                std::min(pts3d_query_cam_valid.size(),
                                                         pts3d_train_cam_valid.size()));
                std::vector<double> norm_samples;
                baseline_samples.reserve(n_pairs);
                norm_samples.reserve(n_pairs);
                for (size_t k = 0; k < n_pairs; ++k) {
                    const Eigen::Vector3d t_sample =
                        pts3d_train_cam_valid[k] - R_em * pts3d_query_cam_valid[k];
                    if (!t_sample.allFinite()) continue;
                    norm_samples.push_back(t_sample.norm());
                    const double proj = t_sample.dot(t_em_dir);
                    if (proj > kEMBaselineMinProjectionM) baseline_samples.push_back(proj);
                }
                if (norm_samples.size() >= kEMBaselineMinSamples) {
                    const size_t mid = norm_samples.size() / 2;
                    std::nth_element(norm_samples.begin(),
                                     norm_samples.begin() + mid,
                                     norm_samples.end());
                    separation_m = norm_samples[mid];
                }
            }

            // D2-b：拿不到可靠基线就拒绝这次 EM 回环，而不是编一个数顶上。
            // 早期版本退回漂移的 frame_dist，中间版本退回常数 1.0 m —— 两者都会让一条
            // 尺度和真实几何无关的边进 PGO。与其给 PGO 一条假约束，不如这一次没有约束，
            // 下一个关键帧还会再试（回环检测每次关键帧都会跑）。
            if (baseline_samples.size() < kEMBaselineMinSamples ||
                separation_m <= kEMBaselineMinMeters) {
                { if (config_.verbose) std::cerr << "[GeoVerif] EM 拒绝：深度对应点不足以测基线（样本 "
                                                << baseline_samples.size() << " 对）\n"; }
                { if (config_.verbose) std::cerr << std::string(80, '=') << "\n"; }
                return false;
            }

            double baseline = 0.0;
            {
                const size_t mid = baseline_samples.size() / 2;
                std::nth_element(baseline_samples.begin(),
                                 baseline_samples.begin() + mid,
                                 baseline_samples.end());
                baseline = baseline_samples[mid];
            }
            if (!std::isfinite(baseline) || baseline <= kEMBaselineMinMeters) {
                { if (config_.verbose) std::cerr << "[GeoVerif] EM 拒绝：基线非有限或过小（"
                                                << baseline << " m）\n"; }
                { if (config_.verbose) std::cerr << std::string(80, '=') << "\n"; }
                return false;
            }
            T_em.block<3, 1>(0, 3) = baseline * t_em_dir;
            if (config_.verbose) {
                std::cerr << "[GeoVerif] EM 基线（深度实测，中位数）：" << baseline
                          << " m，帧间基线 " << separation_m
                          << " m（样本 " << baseline_samples.size() << " 对）" << std::endl;
            }
            // 到这里 T_em 的平移已有真实尺度；t_norm_em == 0 的分支（recoverPose 未给出平移）
            // 已被上面的样本检查挡掉，不再需要"补零"的兜底。

            // 帧间距离闸门：两端边界都来自 config（R-018/R-022）。
            // 原写法把上限 5.0 写死在局部常量里，yaml 的 frame_dist_max: 50.0 无人读取；
            // 下限则完全没有——极近的两帧（frame_dist≈0）走 EM 只会得到方向不可信的相对位姿。
            if (separation_m < config_.frame_dist_min) {
                { if (config_.verbose) std::cerr << "[GeoVerif] EM 拒绝：深度实测基线 " << separation_m
                                                << "m 低于下限 " << config_.frame_dist_min << "m\n"; }
                return false;
            }
            // 此检查基于匹配点深度实测基线，不适用于已修正的 ICP/Sim3 路径
            if (separation_m >= config_.frame_dist_max) {
                { if (config_.verbose) std::cerr << "[GeoVerif] EM 拒绝：深度实测基线 " << separation_m
                                                << "m 超过上限 " << config_.frame_dist_max << "m\n"; }
                return false;
            }

            if (out_relative_pose) *out_relative_pose = T_em;
            if (out_inliers) *out_inliers = em_inliers;
            if (out_is_icp) *out_is_icp = false;
            { if (config_.verbose) std::cerr << std::string(80, '=') << "\n"; }
            return true;
        }

        { if (config_.verbose) std::cerr << "[GeoVerif] FAIL: ICP 和 Essential Matrix 均失败，拒绝回环\n"; }
        { if (config_.verbose) std::cerr << std::string(80, '=') << "\n"; }
        return false;
    }

    { if (config_.verbose) std::cerr << std::string(80, '=') << "\n"; }
    return false;
}

LoopCandidate LoopDetector::detect(const std::vector<cv::KeyPoint>& curr_kps,
                                    const std::vector<Eigen::Vector3d>& curr_pts3d_cam,
                                    const cv::Mat& curr_desc,
                                    ::sfc_core::frontend::CovisManager* covis_mgr) {
    last_loop_candidate_ = LoopCandidate{};
    last_loop_candidate_.current_kf_id = keyframe_ids_.empty() ? 0 : keyframe_ids_.back();
    last_loop_candidate_.valid = false;

    // 这里只需要"至少还有别的帧可比较"。min_loop_separation 是"当前帧与候选帧在
    // 关键帧序列上至少隔多远"，它的正确用法在下面构造候选池时（i + separation < size），
    // 不是拿它当"数据库总数下限"。原写法要求库里至少 30 个关键帧才开始检测，
    // 而短序列（fr1_desk2 全程仅数十个关键帧）会因此整段不检测回环。
    if (keyframe_ids_.size() < 2) {
        return last_loop_candidate_;
    }

    auto it_curr = keyframe_db_.find(last_loop_candidate_.current_kf_id);
    if (it_curr == keyframe_db_.end()) {
        { if (config_.verbose) std::cerr << "[Loop] 未找到当前帧: " << last_loop_candidate_.current_kf_id << std::endl; }
        return last_loop_candidate_;
    }
    const auto& curr_pose_world = it_curr->second.pose;

    Eigen::Matrix3d R_wc_curr = curr_pose_world.q.toRotationMatrix();
    std::vector<Eigen::Vector3d> curr_pts3d_world;
    curr_pts3d_world.reserve(curr_pts3d_cam.size());
    for (const Eigen::Vector3d& p_cam : curr_pts3d_cam) {
        if (p_cam.norm() < 1e-6) {
            curr_pts3d_world.emplace_back(0, 0, 0);
        } else {
            Eigen::Vector3d p_world = R_wc_curr * p_cam + curr_pose_world.t;
            curr_pts3d_world.push_back(p_world);
        }
    }

    std::vector<int> candidates;
    for (size_t i = 0; i + config_.min_loop_separation < keyframe_ids_.size(); ++i) {
        candidates.push_back(keyframe_ids_[i]);
    }

    if (candidates.empty()) {
        { if (config_.verbose) std::cerr << "[Loop] Gate-1: 候选池为空" << std::endl; }
        return last_loop_candidate_;
    }

    // ★★★ 优化：一次性构建 kf_id → index 映射（O(K)），消除 O(N^2) 重复扫描 ★★★
    std::unordered_map<int, int> kf_id_to_idx;
    kf_id_to_idx.reserve(keyframe_ids_.size());
    int curr_idx_in_db = -1;
    for (size_t di = 0; di < keyframe_ids_.size(); ++di) {
        if (keyframe_ids_[di] == last_loop_candidate_.current_kf_id) curr_idx_in_db = static_cast<int>(di);
        kf_id_to_idx[keyframe_ids_[di]] = static_cast<int>(di);
    }

    std::vector<int> curr_bow = computeBoW(curr_desc);
    // 交给几何验证的候选上限：由 config_.candidate_pool_size 控制。
    // 原来是写死的 5，yaml 里的 candidate_pool_size: 30 完全没有消费者（R-018/R-022）。
    const int max_bow_candidates = std::max(1, config_.candidate_pool_size);
    std::vector<std::pair<int, int>> scored;

    bool use_dbow = (place_recog_ && place_recog_->isVocabularyReady());

    if (use_dbow) {
        { if (config_.verbose) std::cerr << "\n[Loop] ★★★ DBoW2 位置识别查询 ★★★" << std::endl; }

        std::vector<int> recent_matches;
        recent_matches.insert(recent_matches.end(),
                           recent_loop_matches_.begin(),
                           recent_loop_matches_.end());

        // D5-e：三个阈值来自配置，不再写死（R-018）。
        auto dbow_results = place_recog_->queryWithConsistency(
            curr_desc, curr_pose_world.t, recent_matches,
            config_.dbow_temporal_window,
            config_.dbow_max_results,
            config_.dbow_min_distance_m);

        if (dbow_results.empty()) {
            { if (config_.verbose) std::cerr << "[Loop]   DBoW2 无候选" << std::endl; }
        } else {
            { if (config_.verbose) std::cerr << "[Loop]   DBoW2 候选: " << dbow_results.size() << " 个" << std::endl; }
            // D5-e：候选上限用 candidate_pool_size，空间距离与帧间隔门槛用配置项。
            const double max_spatial_dist = config_.loop_max_spatial_dist_m;
            const int min_frames_gap = config_.min_frames_between_for_spatial_gate;
            const size_t max_bow_cands =
                std::min(static_cast<size_t>(max_bow_candidates), dbow_results.size());
            for (size_t i = 0; i < max_bow_cands; ++i) {
                const auto& r = dbow_results[i];

                auto it_cand = keyframe_db_.find(r.kf_id);
                if (it_cand != keyframe_db_.end()) {
                    const auto& cand_entry = it_cand->second;
                    Eigen::Vector3d delta = cand_entry.pose.t - curr_pose_world.t;
                    double spatial_dist = delta.norm();

                    auto it_cand_idx = kf_id_to_idx.find(r.kf_id);
                    int frames_between = -1;
                    if (it_cand_idx != kf_id_to_idx.end() && curr_idx_in_db >= 0) {
                        frames_between = std::abs(curr_idx_in_db - it_cand_idx->second);
                    }

                    if (spatial_dist > max_spatial_dist && frames_between >= 0 &&
                        frames_between < min_frames_gap) {
                        continue;
                    }
                }

                scored.emplace_back(r.kf_id, static_cast<int>(r.score * 1000));
            }
        }
    }

    if (scored.empty()) {
        { if (config_.verbose) std::cerr << "\n[Loop] 使用简单 BoW 匹配" << std::endl; }
        for (int cand_id : candidates) {
            // ★★★ Bug Fix: 使用 find() 替代 at()，避免异常 ★★★
            auto it = keyframe_db_.find(cand_id);
            if (it == keyframe_db_.end()) continue;
            const auto& entry = it->second;
            int common = commonWordCount(curr_bow, entry.bow_vector);
            if (common < config_.min_common_words) continue;
            scored.emplace_back(cand_id, common);
        }
    }

    if (scored.empty()) {
        { if (config_.verbose) std::cerr << "[Loop] Gate-2: BoW 过滤后无候选" << std::endl; }
        return last_loop_candidate_;
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (static_cast<int>(scored.size()) > max_bow_candidates) scored.resize(max_bow_candidates);

    { if (config_.verbose) std::cerr << "\n[Loop] ★★★ 候选: " << scored.size() << " 个，开始几何验证 ★★★\n"; }

    if (config_.enable_multi_candidate_verification && scored.size() > 1) {
        { if (config_.verbose) std::cerr << "\n[Loop] ★★★ 使用多候选验证 ★★★" << std::endl; }

        auto verified = verifyMultipleCandidates(
            curr_kps, curr_pts3d_world, curr_desc, scored, curr_pose_world, covis_mgr);

        if (verified.empty()) {
            { if (config_.verbose) std::cerr << "[Loop] 多候选验证: 所有候选都失败" << std::endl; }
            return last_loop_candidate_;
        }

        auto best = selectBestFromVerified(verified);

        if (best.kf_id < 0) {
            { if (config_.verbose) std::cerr << "[Loop] 多候选验证: 选择最佳候选失败" << std::endl; }
            return last_loop_candidate_;
        }

        recent_loop_matches_.push_back(best.kf_id);
        if (recent_loop_matches_.size() > 10) {
            recent_loop_matches_.erase(recent_loop_matches_.begin());
        }

        last_loop_candidate_ = LoopCandidate{
            .current_kf_id = last_loop_candidate_.current_kf_id,
            .matched_kf_id  = best.kf_id,
            .raw_matches   = best.raw_matches,
            .geometric_inliers = best.geometric_inliers,
            .inlier_ratio  = best.inlier_ratio,
            .relative_pose  = best.relative_pose,
            .valid          = true,
            .is_icp         = best.is_icp
        };
        return last_loop_candidate_;
    } else {
        { if (config_.verbose) std::cerr << "\n[Loop] ★★★ 使用单候选验证 ★★★" << std::endl; }

        for (const auto& cand : scored) {
            int cand_id = cand.first;
            int score = cand.second;

            auto it = keyframe_db_.find(cand_id);
            if (it == keyframe_db_.end()) continue;

            const auto& entry = it->second;
            std::vector<cv::DMatch> matches = matchDescriptors(curr_desc, entry.descriptors);

            if (static_cast<int>(matches.size()) < config_.min_geometric_inliers) {
                continue;
            }

            Eigen::Matrix4d relative_pose;
            int geometric_inliers = 0;

            const auto& matched_pose = entry.pose;
            Eigen::Matrix3d R_wc_matched = matched_pose.q.toRotationMatrix();
            Eigen::Vector3d t_wc_matched = matched_pose.t;
            std::vector<Eigen::Vector3d> entry_pts3d_world;
            entry_pts3d_world.reserve(entry.pts3d.size());
            for (const Eigen::Vector3d& p_cam : entry.pts3d) {
                if (p_cam.norm() < 1e-6) {
                    entry_pts3d_world.emplace_back(0, 0, 0);
                } else {
                    Eigen::Vector3d p_w = R_wc_matched * p_cam + t_wc_matched;
                    entry_pts3d_world.push_back(p_w);
                }
            }

            bool is_icp_flag = false;
            bool geo_ok = geometricVerification(
                curr_kps, entry.keypoints,
                curr_pts3d_world, entry_pts3d_world,
                matches,
                last_loop_candidate_.current_kf_id,
                cand_id,
                &relative_pose, &geometric_inliers,
                &is_icp_flag, covis_mgr);

            if (!geo_ok) {
                { if (config_.verbose) std::cerr << "[Loop] Gate-4: KF=" << cand_id << " 几何失败" << std::endl; }
                continue;
            }

            double inlier_ratio = static_cast<double>(geometric_inliers) / matches.size();

            recent_loop_matches_.push_back(cand_id);
            if (recent_loop_matches_.size() > 10) {
                recent_loop_matches_.erase(recent_loop_matches_.begin());
            }

            last_loop_candidate_ = LoopCandidate{
                .current_kf_id = last_loop_candidate_.current_kf_id,
                .matched_kf_id  = cand_id,
                .raw_matches   = static_cast<int>(matches.size()),
                .geometric_inliers = geometric_inliers,
                .inlier_ratio  = inlier_ratio,
                .relative_pose  = relative_pose,
                .valid          = true,
                .is_icp         = is_icp_flag
            };
            return last_loop_candidate_;
        }
    }

    { if (config_.verbose) std::cerr << "[Loop] 所有候选都失败" << std::endl; }
    return last_loop_candidate_;
}

std::vector<LoopDetector::VerifiedCandidate>
LoopDetector::verifyMultipleCandidates(
    const std::vector<cv::KeyPoint>& curr_kps,
    const std::vector<Eigen::Vector3d>& curr_pts3d_world,
    const cv::Mat& curr_desc,
    const std::vector<std::pair<int, int>>& bow_candidates,
    const frontend::Pose& curr_pose_world,
    ::sfc_core::frontend::CovisManager* covis_mgr) {

    std::vector<VerifiedCandidate> verified;
    verified.reserve(bow_candidates.size());

    { if (config_.verbose) std::cerr << "\n" << std::string(80, '=') << std::endl; }
    { if (config_.verbose) std::cerr << "[MultiCandidate] 开始多候选验证: " << bow_candidates.size() << " 个候选" << std::endl; }

    int candidates_to_verify = std::min(
        static_cast<int>(bow_candidates.size()),
        config_.max_candidates_to_verify);

    for (int i = 0; i < candidates_to_verify; ++i) {
        int cand_id = bow_candidates[i].first;
        int bow_score = bow_candidates[i].second;

        auto it = keyframe_db_.find(cand_id);
        if (it == keyframe_db_.end()) continue;

        const auto& entry = it->second;
        std::vector<cv::DMatch> matches = matchDescriptors(curr_desc, entry.descriptors);

        if (static_cast<int>(matches.size()) < config_.min_geometric_inliers) {
            continue;
        }

        const auto& matched_pose = entry.pose;
        Eigen::Matrix3d R_wc_matched = matched_pose.q.toRotationMatrix();
        Eigen::Vector3d t_wc_matched = matched_pose.t;
        std::vector<Eigen::Vector3d> entry_pts3d_world;
        entry_pts3d_world.reserve(entry.pts3d.size());
        for (const Eigen::Vector3d& p_cam : entry.pts3d) {
            if (p_cam.norm() < 1e-6) {
                entry_pts3d_world.emplace_back(0, 0, 0);
            } else {
                Eigen::Vector3d p_w = R_wc_matched * p_cam + t_wc_matched;
                entry_pts3d_world.push_back(p_w);
            }
        }

        Eigen::Matrix4d relative_pose;
        int geometric_inliers = 0;
        bool is_icp = false;
        bool geo_ok = geometricVerification(
            curr_kps, entry.keypoints,
            curr_pts3d_world, entry_pts3d_world,
            matches,
            last_loop_candidate_.current_kf_id,
            cand_id,
            &relative_pose, &geometric_inliers, &is_icp, covis_mgr);

        if (!geo_ok) {
            { if (config_.verbose) std::cerr << "[MultiCandidate]   几何验证失败" << std::endl; }
            continue;
        }

        double inlier_ratio = static_cast<double>(geometric_inliers) / matches.size();

        double quality_score =
            geometric_inliers * config_.geometric_inlier_weight +
            inlier_ratio * matches.size() * config_.inlier_ratio_weight +
            std::log1p(std::max(0.0, static_cast<double>(bow_score))) * config_.bow_score_weight;

        VerifiedCandidate vc;
        vc.kf_id = cand_id;
        vc.raw_matches = static_cast<int>(matches.size());
        vc.geometric_inliers = geometric_inliers;
        vc.inlier_ratio = inlier_ratio;
        vc.bow_score = bow_score;
        vc.relative_pose = relative_pose;
        vc.quality_score = quality_score;
        vc.is_icp = is_icp;

        verified.push_back(vc);
    }

    return verified;
}

LoopDetector::VerifiedCandidate
LoopDetector::selectBestFromVerified(
    const std::vector<VerifiedCandidate>& verified_candidates) {

    VerifiedCandidate best;
    best.kf_id = -1;
    best.quality_score = 0.0;

    if (verified_candidates.empty()) {
        { if (config_.verbose) std::cerr << "[MultiCandidate] 无验证通过的候选" << std::endl; }
        return best;
    }

    { if (config_.verbose) std::cerr << "\n" << std::string(80, '=') << std::endl; }
    { if (config_.verbose) std::cerr << "[MultiCandidate] 从 " << verified_candidates.size() << " 个候选中选择最佳" << std::endl; }

    std::vector<VerifiedCandidate> sorted = verified_candidates;
    // L11：原比较器不是严格弱序 —— "分数接近"时退化到 is_icp 比较，而 is_icp
    // 相等的两对又会被判为 a<b 且 b<a（std::sort 要求 comp(a,b) 与 comp(b,a) 不能
    // 同时为真，否则行为未定义，实践中表现为候选顺序随机、同一序列两次跑出不同回环）。
    // 改成：先按 is_icp 分层，同层内按质量分严格降序，完全等价时按 kf_id 兜底，
    // 任何一对候选都有确定的先后。
    std::sort(sorted.begin(), sorted.end(),
              [](const VerifiedCandidate& a, const VerifiedCandidate& b) {
                  if (a.is_icp != b.is_icp) return a.is_icp;
                  if (a.quality_score != b.quality_score) return a.quality_score > b.quality_score;
                  return a.kf_id < b.kf_id;
              });

    best = sorted[0];

    if (sorted.size() > 1) {
        double ratio = sorted[1].quality_score / best.quality_score;
        if (ratio > config_.best_candidate_ratio_threshold) {
            { if (config_.verbose) std::cerr << "[MultiCandidate] 警告: 多个候选分数接近，选择第一候选" << std::endl; }
        }
    }

    // L9：候选一致性检查原本是死代码（全仓零调用点），require_candidate_agreement /
    // max_spatial_dist_for_consistency / consistency_warning_threshold 三个配置键
    // 因此从来没有生效过。BoW 假阳性的典型形态是"几个候选各自指向场景里不同位置"——
    // 每个候选单独做几何验证都可能通过，但它们彼此在空间上矛盾。
    if (config_.require_candidate_agreement && sorted.size() >= 2) {
        const double consistency = computeCandidateConsistency(sorted);
        // 一致性低于阈值是"这次回环没通过"的决定，不是过程噪音，所以无条件输出。
        if (consistency < config_.consistency_warning_threshold) {
            std::cerr << "[MultiCandidate] 候选一致性 " << consistency
                      << " 低于阈值 " << config_.consistency_warning_threshold
                      << "（" << sorted.size() << " 个候选在空间上互相矛盾），拒绝本次回环"
                      << std::endl;
            best.kf_id = -1;
            return best;
        }
        { if (config_.verbose) std::cerr << "[MultiCandidate] 候选一致性 " << consistency
                                         << "（阈值 " << config_.consistency_warning_threshold
                                         << "），通过" << std::endl; }
    }

    if (static_cast<int>(verified_candidates.size()) < config_.min_candidates_to_accept) {
        best.kf_id = -1;
        return best;
    }

    return best;
}

double LoopDetector::computeCandidateConsistency(
    const std::vector<VerifiedCandidate>& candidates) {

    if (candidates.size() < 2) return 1.0;

    double total_consistency = 0.0;
    int comparisons = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            auto it_i = keyframe_db_.find(candidates[i].kf_id);
            auto it_j = keyframe_db_.find(candidates[j].kf_id);

            if (it_i == keyframe_db_.end() || it_j == keyframe_db_.end()) continue;

            const auto& pose_i = it_i->second.pose;
            const auto& pose_j = it_j->second.pose;

            Eigen::Vector3d delta = pose_i.t - pose_j.t;
            double spatial_dist = delta.norm();

            double consistency = std::max(0.0, 1.0 - spatial_dist / config_.max_spatial_dist_for_consistency);

            total_consistency += consistency;
            comparisons++;
        }
    }

    return comparisons > 0 ? total_consistency / comparisons : 1.0;
}

void LoopDetector::updateKeyFramePose(int kf_id, const frontend::Pose& pose) {
    auto it = keyframe_db_.find(kf_id);
    if (it != keyframe_db_.end()) {
        it->second.pose = pose;
    }
}

bool LoopDetector::geometricVerificationWithSim3(
        const std::vector<Eigen::Vector3d>& curr_pts3d_world,
        const std::vector<int>& curr_kpt_indices,
        const std::vector<Eigen::Vector3d>& matched_pts3d_world,
        const std::vector<int>& matched_kpt_indices,
        const frontend::Pose& pose_curr,
        const frontend::Pose& pose_matched,
        const std::vector<cv::KeyPoint>& kp_curr,
        const std::vector<cv::KeyPoint>& kp_matched,
        Eigen::Matrix4d* out_relative_pose,
        int* out_inliers,
        bool* out_is_icp) const {

    if (curr_pts3d_world.empty() || matched_pts3d_world.empty()) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 共视地图点为空，跳过 Sim3 验证" << std::endl; }
        return false;
    }

    int N = std::min(static_cast<int>(curr_pts3d_world.size()),
                      static_cast<int>(matched_pts3d_world.size()));
    if (N < 2) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 共视匹配数 N=" << N << " < 2，跳过" << std::endl; }
        return false;
    }

    if (pose_curr.t.norm() < 1e-6 || pose_matched.t.norm() < 1e-6) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 相机位姿无效（平移为零），跳过 Sim3 验证" << std::endl; }
        return false;
    }

    int valid_count = 0;
    std::vector<bool> pt_valid(N, false);
    Eigen::Matrix3d R_cw_curr, R_cw_matched;
    Eigen::Vector3d t_cw_curr, t_cw_matched;
    extractW2CTransform(pose_curr, &R_cw_curr, &t_cw_curr);
    extractW2CTransform(pose_matched, &R_cw_matched, &t_cw_matched);
    for (int i = 0; i < N; ++i) {
        if (curr_pts3d_world[i].norm() > 1e-6 && matched_pts3d_world[i].norm() > 1e-6) {
            Eigen::Vector3d pc1 = R_cw_curr * curr_pts3d_world[i] + t_cw_curr;
            Eigen::Vector3d pc2 = R_cw_matched * matched_pts3d_world[i] + t_cw_matched;
            if (pc1.z() > 1e-6 && pc2.z() > 1e-6) {
                valid_count++;
                pt_valid[i] = true;
            }
        }
    }

    { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 预统计：有效 3D 点=" << valid_count << "/" << N << std::endl; }

    if (valid_count < 2) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 有效共视点 " << valid_count << " < 2，跳过" << std::endl; }
        return false;
    }

    bool use_filtered = (valid_count < N);

    std::vector<Eigen::Vector3d> pts_cam_curr, pts_cam_matched;
    std::vector<cv::DMatch> solver_matches;
    if (use_filtered) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 过滤后使用 " << valid_count << " 个有效点构建 Sim3Solver" << std::endl; }
        pts_cam_curr.reserve(valid_count);
        pts_cam_matched.reserve(valid_count);
        solver_matches.reserve(valid_count);
        for (int i = 0; i < N; ++i) {
            if (!pt_valid[i]) continue;
            pts_cam_curr.push_back(R_cw_curr * curr_pts3d_world[i] + t_cw_curr);
            pts_cam_matched.push_back(R_cw_matched * matched_pts3d_world[i] + t_cw_matched);
            solver_matches.emplace_back(static_cast<int>(pts_cam_curr.size()) - 1,
                                       static_cast<int>(pts_cam_curr.size()) - 1, 0.0f);
        }
    } else {
        pts_cam_curr.reserve(N);
        pts_cam_matched.reserve(N);
        solver_matches.reserve(N);
        for (int i = 0; i < N; ++i) {
            Eigen::Vector3d pc1 = R_cw_curr * curr_pts3d_world[i] + t_cw_curr;
            Eigen::Vector3d pc2 = R_cw_matched * matched_pts3d_world[i] + t_cw_matched;
            pts_cam_curr.push_back(pc1);
            pts_cam_matched.push_back(pc2);
            solver_matches.emplace_back(i, i, 0.0f);
        }
    }

    if (static_cast<int>(pts_cam_curr.size()) < 3) {
        return false;
    }

    Sim3Solver solver(pts_cam_curr, pts_cam_matched, solver_matches, K_cv_, true);
    solver.setRansacParameters(0.99, 2, 300);

    std::vector<bool> vbInliers;
    int nInliers = 0;
    bool bNoMore = false;
    solver.iterate(300, bNoMore, vbInliers, nInliers);

    double inlier_ratio_check = (valid_count > 0) ? static_cast<double>(nInliers) / valid_count : 0.0;
    bool inliers_ok = (nInliers >= 4) || (nInliers >= 3 && inlier_ratio_check >= 0.50);
    if (!inliers_ok) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 内点不足（" << nInliers
                                        << "，内点率 " << inlier_ratio_check
                                        << "，要求 >=4 或 >=3 且内点率 >=0.50）" << std::endl; }
        return false;
    }

    Eigen::Matrix4d sim3_init = Eigen::Matrix4d::Identity();
    sim3_init.block<3, 3>(0, 0) = solver.getEstimatedRotation();
    sim3_init.block<3, 1>(0, 3) = solver.getEstimatedTranslation();

    { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 初始 Sim3: |t|="
                                    << sim3_init.block<3, 1>(0, 3).norm() << " m" << std::endl; }

    std::vector<MapPointMatch> opt_matches;
    opt_matches.reserve(nInliers);
    int solver_n = static_cast<int>(pts_cam_curr.size());

    // 构建世界到相机变换矩阵，用于 Sim3Optimizer
    Eigen::Matrix4d T_cw_curr = Eigen::Matrix4d::Identity();
    T_cw_curr.block<3, 3>(0, 0) = R_cw_curr;
    T_cw_curr.block<3, 1>(0, 3) = t_cw_curr;
    Eigen::Matrix4d T_cw_matched = Eigen::Matrix4d::Identity();
    T_cw_matched.block<3, 3>(0, 0) = R_cw_matched;
    T_cw_matched.block<3, 1>(0, 3) = t_cw_matched;

    for (int i = 0; i < solver_n; ++i) {
        if (i >= static_cast<int>(vbInliers.size()) || !vbInliers[i]) continue;
        const Eigen::Vector3d& pw = curr_pts3d_world[i];
        MapPointMatch m;
        m.kf1_kpt_idx = (i < static_cast<int>(curr_kpt_indices.size())) ? curr_kpt_indices[i] : i;
        m.kf2_kpt_idx = (i < static_cast<int>(matched_kpt_indices.size())) ? matched_kpt_indices[i] : i;
        m.pt_world = pw;
        m.pt_cam1 = pts_cam_curr[i];
        m.pt_cam2 = pts_cam_matched[i];

        // ★★★ 填充原始像素坐标，用于 Sim3Optimizer 的观测值 ★★★
        int kp1_idx = m.kf1_kpt_idx;
        int kp2_idx = m.kf2_kpt_idx;
        if (kp1_idx >= 0 && kp1_idx < static_cast<int>(kp_curr.size())) {
            m.px1 = Eigen::Vector2d(kp_curr[kp1_idx].pt.x, kp_curr[kp1_idx].pt.y);
        } else {
            m.px1 = Eigen::Vector2d(-1.0, -1.0);
        }
        if (kp2_idx >= 0 && kp2_idx < static_cast<int>(kp_matched.size())) {
            m.px2 = Eigen::Vector2d(kp_matched[kp2_idx].pt.x, kp_matched[kp2_idx].pt.y);
        } else {
            m.px2 = Eigen::Vector2d(-1.0, -1.0);
        }

        opt_matches.push_back(m);
    }

    Eigen::Matrix4d sim3_refined;
    int n_refined = Sim3Optimizer::refineWithTwoPass(
        sim3_init, opt_matches, K_cv_, true,
        8.0f, 5.0f, 10,
        T_cw_curr, T_cw_matched,  // ★★★ 传入真实相机位姿 ★★★
        sim3_refined);

    { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 精化后内点: " << n_refined << std::endl; }

    double refined_ratio = (static_cast<int>(pts_cam_curr.size()) > 0)
                             ? static_cast<double>(n_refined) / pts_cam_curr.size()
                             : 0.0;
    bool refined_ok = (n_refined >= 4) || (n_refined >= 3 && refined_ratio >= 0.50);
    if (!refined_ok) {
        { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 精化内点不足，验证失败" << std::endl; }
        return false;
    }

    // 旋转矩阵有效性验证
    {
        Eigen::Matrix3d R_check = sim3_refined.block<3, 3>(0, 0);
        svdOrthogonalize(&R_check);
        double det = R_check.determinant();
        if (std::abs(det - 1.0) > 0.05) {
            { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 拒绝回环：旋转行列式异常 det=" << det << std::endl; }
            return false;
        }

        double rot_trace = (R_check.trace() - 1.0) / 2.0;
        rot_trace = std::max(-1.0, std::min(1.0, rot_trace));
        double angle_deg = std::acos(rot_trace) * 180.0 / M_PI;
        if (angle_deg > 180.0) {
            { if (config_.verbose) std::cerr << "[GeoVerif Sim3] 拒绝回环：旋转角度过大 " << angle_deg << "°" << std::endl; }
            return false;
        }

        // 用正交化后的旋转矩阵替换 sim3_refined
        sim3_refined.block<3, 3>(0, 0) = R_check;
    }

    if (out_relative_pose) *out_relative_pose = sim3_refined;
    if (out_inliers) *out_inliers = n_refined;
    if (out_is_icp) *out_is_icp = true;
    return true;
}

}  // namespace loopdetection
}  // namespace sfc_core
