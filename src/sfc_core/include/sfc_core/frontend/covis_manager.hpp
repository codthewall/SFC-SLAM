// 必须在最开头清除 CMake project() 注入的 sfc_core 宏（g2o CMake 会注入 -Dsfc_core=1）
// 必须在所有 #include 之前执行，防止 loopdetection 头文件链中的 namespace sfc_core { 被错误展开为 namespace 1 {
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FRONTEND_COVIS_MANAGER_HPP_
#define SFC_CORE_FRONTEND_COVIS_MANAGER_HPP_

#include "sfc_core/loopdetection/loop_detector.hpp"
#include "sfc_core/loopdetection/pose_graph_optimizer.hpp"
#include "sfc_core/frontend/covis_core.hpp"
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>

namespace sfc_core {
namespace frontend {

struct CovisConfig {
    cv::Mat K;  // 相机内参（传递给 BA 边）
    cv::Mat D;  // 畸变系数（可选，回环几何校验等）
    // ★★★ 图像分辨率：用于重投影边界检查 ★★★
    // 必须正确设置！原来错误地从 cx*2 / cy*2 推断图像尺寸，
    // 对于 TUM RGB-D (cx=319.5, cy=239.5) 会得到 639x479 而非正确的 640x480
    int img_width = 640;
    int img_height = 480;
    // ★★★ 滑窗大小：增加默认值以保留更多历史 ★★★
    // 原值15帧约覆盖1秒（15fps），可能导致旋转累积后无法修正
    // 增加至30帧（约2秒），给BA更多修正机会，但会增加计算负担
    int window_size = 30;
    // ★★★ 滑窗保留历史关键帧数量 ★★★
    // 当关键帧滑出滑窗时，保留其轨迹信息用于回环修正
    // 设为window_size的2倍，确保足够的历史信息
    int max_history_kf_poses = 100;
    int min_covis_frames = 2;
    int candidate_confirm_threshold = 2;
    double max_reproj_error = 8.0;
    int max_candidates = 2000;  // ★★★ 从1000增加到2000，减少候选点过度剔除 ★★★
    int max_confirmed = 500;
    int candidate_check_interval = 5;
    int staleness_thresh = 30;
    // ★★★ 共视量化参数 ★★★
    // 3D 坐标量化格网大小：1m / quantize_scale = 格网尺寸（米）
    // 设为 20.0 = 5cm 格网（用于 O(1) 地图点哈希查找）
    double quantize_scale = 20.0;
    // 两点是否来自同一物理点的判断阈值（米）
    double same_point_distance_threshold = 0.10;
    // ★★★ 重投影搜索参数 ★★★
    // 3D 点重投影位置附近搜索匹配特征点的半径（像素）
    double reproj_search_radius = 12.0;
    // ★★★ 局部 BA 参数 ★★★
    // BA 最小边数阈值，只有边数 >= 此值时才执行 BA
    int ba_min_edges = 10;
    // BA 最大迭代次数（g2o Levenberg-Marquardt）
    int ba_max_iterations = 20;
    // ★★★ 局部 BA Huber δ（与 PnP BA 保持一致）★★★
    // 与 tracking.yaml 中的 pnp_huber_delta=2.45 一致。
    // chi2(2DoF, 0.95) = 5.991, delta ≈ 2.45 (95% 置信度)。
    // 原代码硬编码 huber->setDelta(5.0)，现改为 2.45 匹配 ORB_SLAM3。
    double ba_huber_delta = 2.45;
    // ★★★ PGO 修正阈值参数 ★★★
    // 最大允许平移修正（米），超过此值拒绝回环修正
    // ★★★ 从 5.0m 放宽到 10.0m：原值导致大量有效回环被拒绝
    // 但 YAML 文件中可配置更低值（如 5.0m），运行时以 YAML 为准
    double pgo_max_allowed_trans_corr = 5.0;
    // ★★★ 旋转修正拒绝阈值（度）★★★
    // 超过此值的旋转修正是异常的，正常回环修正的旋转通常 < 20°。
    // 设为 20°：允许合理的回环旋转，拒绝 57° 这类异常大修正。
    // 注意：pgo_max_allowed_rot_corr_deg=180° 是上限，此参数是真正的拒绝阈值。
    double pgo_max_allowed_rot_corr_deg_reject = 20.0;
    // 最大允许旋转修正（度），超过此值拒绝回环修正
    double pgo_max_allowed_rot_corr_deg = 180.0;
    // 平移修正警告阈值（米）
    double pgo_warn_trans_corr = 2.0;
    // 旋转修正警告阈值（度）
    double pgo_warn_rot_corr_deg = 45.0;
    // ★★★ EM模式尺度膨胀阈值 ★★★
    // Essential Matrix模式的平移向量|t|=1归一化，不携带真实尺度
    // 当回环距离与EM尺度的比值超过此阈值时，跳过回环修正
    // 室内场景：通常1-3倍正常，设为3.0允许合理的尺度变化，拒绝>3倍的异常膨胀
    double em_scale_explosion_threshold = 3.0;
    // 回环检测配置（0=禁用）
    int loop_detection_interval = 5;     // 每隔几帧检测一次
    int loop_min_inliers = 8;           // 从 12 降低到 8（允许 8+ 内点回环通过，阻止 5 内点垃圾回环）
    // ★★★ ICP 回环距离阈值 ★★★
    // ICP 验证通过后，相对位姿的平移量应在合理范围内。
    // 室内场景（fr1_desk2）：相机运动半径约 5-8m，回环时两帧距离通常 2-6m。
    // 设得太小（如 3.0m）会误杀真实回环；设得太大（如 10.0m）会让误检通过。
    // ★★★ 从 3.0m 放宽到 6.0m：原值导致大量 ICP 回环因室内房间跨度被误杀 ★★★
    double icp_loop_max_distance_m = 6.0;
    bool enable_loop_detection = true;
    bool enable_pgo = true;               // PGO 回环修正（关闭=保留纯 PnP 轨迹）
    bool enable_local_ba = true;          // 是否启用局部 BA（关闭=纯 PnP 里程计）
    // ★★★ BA 修正选项 ★★★
    // enable_full_ba_correction: 是否启用完整 BA 修正（包括旋转）
    // true: BA 同时修正位姿的旋转和平移（需要配合重力约束或其他绝对方向参考）
    // false: BA 只修正平移，保留原始旋转（默认，防止旋转误差累积）
    // 注意：开启此选项时，建议同时启用重力约束（use_gravity_constraint=true）
    bool enable_full_ba_correction = false;
    // BA 修正拒绝阈值（仅用于安全保护，正常 BA 修正通常远小于此值）
    // 当 enable_full_ba_correction=true 时，这些阈值用于拒绝异常大的修正
    double ba_max_trans_correction_m = 0.50;   // 最大允许平移修正（米），超过此值拒绝
    double ba_max_rot_correction_deg = 30.0;   // 最大允许旋转修正（度），超过此值拒绝
    // ★★★ 缓存大小限制 ★★★
    // 用于防止历史缓存无限增长（pnp_original_poses_cache_, pgo_optimized_poses_cache_等）
    int max_cache_size = 500;  // 最大缓存关键帧位姿数量
    bool enable_cache_cleanup = true;  // 启用自动缓存清理
    bool verbose = false;                 // 详细日志开关（默认关闭，避免每关键帧打印大量 PGO 修正信息）
};

class CovisManager {
public:
    virtual ~CovisManager() = default;

    explicit CovisManager(const CovisConfig& config = CovisConfig());

    // ==================== 核心接口 ====================

    // 新普通帧进来：不做共视管理，只返回当前已确认地图点的 3D→2D 对
    // 用于当前帧的绝对位姿解算（核心功能）
    // 返回：(world_pt, pixel_curr) 对
    std::vector<std::pair<Eigen::Vector3d, cv::Point2f>>
    getConfirmedMapPointsForPnP(const std::vector<cv::KeyPoint>& curr_kps,
                                const std::vector<cv::DMatch>& inlier_matches) const;

    // 新关键帧进来：执行完整的共视约束管理流程
    // kps: 关键帧特征点
    // desc: 关键帧描述子
    // pts3d_curr: 当前关键帧深度 3D 点（相机系）
    // depth_valid_curr: 当前关键帧深度有效性
    // matches_to_prev: 当前关键帧与「上一跟踪帧」的 PnP 内点匹配（Tracking 传入）
    // prev_kps / pts3d_prev: 上一跟踪帧的特征与深度反投 3D 点（相机系，非世界系！）
    // prev_frame_pose_world: 上一跟踪帧在世界系下的位姿，用于 P_c→P_w：P_w = R*p_c+t
    // depth_valid_prev: 上一跟踪帧深度有效性
    // kf_cache_ref: kf_data_cache_ 的引用，用于多关键帧匹配时的历史帧数据查询
    //   addKeyFrame 据此获取历史关键帧的 world_pose、pts_3d、depth_valid 等
    // matches_from_cache: 按历史关键帧索引分组的多关键帧匹配结果
    //   key = kf_cache_ref 中的索引，value = 该历史帧对当前关键帧的描述子匹配
    //   每个 match: queryIdx = 历史帧特征索引，trainIdx = 当前帧特征索引
    void addKeyFrame(const std::vector<cv::KeyPoint>& kps,
                    const std::vector<cv::KeyPoint>& undist_kps,
                    const cv::Mat& desc,
                    const std::vector<Eigen::Vector3d>& pts3d_curr,
                    const std::vector<bool>& depth_valid_curr,
                    const std::vector<cv::DMatch>& matches_to_prev,
                    const std::vector<cv::KeyPoint>& prev_kps,
                    const std::vector<cv::KeyPoint>& prev_undist_kps,
                    const std::vector<Eigen::Vector3d>& pts3d_prev,
                    const std::vector<bool>& depth_valid_prev,
                    const cv::Mat& prev_desc,                    // ★★★ Bug Fix 新增：前一帧描述子，用于存储观测 ★★★
                    int prev_frame_id,                           // ★★★ Bug Fix 新增：前一帧 ID ★★★
                    const Pose& prev_frame_pose_world,
                    const Pose& kf_pose,
                    int inlier_match_count,
                    const std::vector<KeyFrameCacheEntry>& kf_cache_ref,
                    const MultiKFMatchesByCache& matches_from_cache = {},
                    int tracking_frame_id = -1);

    // 每隔几帧调用一次：检查候选点，触发共视校验和淘汰
    void checkCandidates();

    // ==================== 查询接口 ====================
    const std::deque<KeyFrame::Ptr>& getKeyFrameWindow() const { return keyframe_window_; }
    int totalConfirmedMapPoints() const { return confirmed_mappoints_.size(); }
    int totalCandidates() const { return candidate_mappoints_.size(); }

    // 获取所有已确认地图点（用于 SparseMapper 全局地图构建）
    const std::unordered_map<int, MapPoint::Ptr>& getAllConfirmedMapPoints() const {
        return confirmed_mappoints_;
    }

    // 获取所有候选地图点（用于 SparseMapper 全局地图构建）
    const std::unordered_map<int, MapPoint::Ptr>& getAllCandidateMapPoints() const {
        return candidate_mappoints_;
    }

    // ==================== 局部 BA（关键帧滑动窗口版本）====================
    // 在关键帧窗口内运行 BA，同时优化位姿和地图点
    // is_keyframe_triggered: 是否由关键帧触发（关键帧时 BA 效果更好）
    bool runLocalBA(bool is_keyframe_triggered = false);

    // 获取 BA 优化后窗口内最新关键帧的位姿
    Pose getLatestOptimizedPose() const;

    // 获取 BA 已将哪些地图点的世界坐标写回 confirmed_mappoints_
    // 键：mappoint_id，值：BA 优化后的世界坐标（已更新到 confirmed_mappoints_）
    const std::unordered_map<int, Eigen::Vector3d>& getLastBAMapPointUpdates() const {
        return last_ba_mappoint_updates_;
    }

    // BA 自上次调用以来是否有新的位姿修正（超过跳变门限）
    bool hasRecentBAPoseUpdate() const { return recent_ba_pose_update_; }

    // 清除 BA 更新标记（Tracking 在消费后将标记复位）
    void clearRecentBAPoseUpdate() { recent_ba_pose_update_ = false; }

    // 获取滑动窗口中所有关键帧的当前位姿快照（用于同步 kf_data_cache_）
    std::map<int, Pose> getAllKeyFramePosesSnapshot() const;

    // ★ 同步 kf_data_cache_（kf_data_cache_ 中的 world_pose 和 pts_3d 需要始终保持一致）
    // 当回环检测完成后或 BA 运行后调用，确保 kf_data_cache_ 的 world_pose 是最新的，
    // 并且 pts_3d 经过了正确的坐标变换（world→camera）。
    void syncKeyFrameCache(std::vector<KeyFrameCacheEntry>& kf_cache);

    int runningFrames() const { return running_frames_; }

    /// 获取滑动窗口中最新关键帧的位姿（BA 修正后）
    Pose getLatestKeyframePose() const;

    /// 获取滑动窗口中最新关键帧的 frame_id
    int getLatestKeyframeId() const;

    /// 获取所有关键帧（滑动窗口 + PGO 图顶点，用于 SparseMapper 全局地图注册）
    std::vector<KeyFrame::Ptr> getAllKeyFrames() const;

    // ==================== 共视地图点查询（供 Sim3Solver 使用）===================
    // 获取滑动窗口中指定关键帧的全部共视确认地图点（世界坐标）
    // kf_id: 关键帧 ID
    // n_best: 返回最多前 n_best 个（按观测次数排序），默认 0=全部
    std::vector<std::pair<Eigen::Vector3d, int>> getCovisibleMapPoints(
        int kf_id, int n_best = 0) const;

    // 获取两帧之间的共视确认地图点匹配对
    // 两帧必须都在滑动窗口中
    // 返回：(pt_world, kpt_idx_in_curr_kf, kpt_idx_in_matched_kf) 元组
    // 注意：CovisMatch 定义在 covis_core.hpp（跳出 CovisManager 嵌套，打破 include 循环）
    std::vector<CovisMatch> getCovisibleMatches(int curr_kf_id, int matched_kf_id) const;

    // 获取两帧之间的共视确认地图点匹配对（扩展版，包含 kf_data_cache_ 中的历史帧）
    // getCovisibleMatches 只查滑动窗口（curr_kf 和 matched_kf 都必须在窗口内）。
    // 该方法扩展到缓存的历史关键帧：遍历 confirmed_mappoints_，对每个地图点检查它的
    // observations 中是否有 curr_kf_id 和 matched_kf_id。
    // 用于 Sim3Solver fallback：回环候选帧（matched_kf）通常不在滑动窗口内。
    std::vector<CovisMatch> getCovisibleMatchesWithCache(int curr_kf_id, int matched_kf_id) const;

    // 获取指定关键帧观测到的所有确认地图点 ID
    // 用于在 geometricVerification 内部实时 prefill matched_kf 的观测到 curr_kf
    // 解决上一次 prefill 用旧 matched_kf（与当前候选不匹配）的问题
    std::vector<int> getMapPointsObservedByKeyFrame(int kf_id) const;

    // ==================== 回环检测 + PGO ====================
    // 初始化回环检测模块（在 addKeyFrame 之前调用一次）
    void initLoopDetection(const loopdetection::LoopDetectorConfig& loop_cfg,
                           const loopdetection::PoseGraphOptimizerOptions& pgo_cfg);

    // 每当有新的关键帧加入时调用：执行回环检测，若检测到回环则修正位姿图
    // 返回：是否有新回环被检测并修正
    bool detectAndCorrectLoop();

    // 是否自上次调用以来检测到了新回环
    bool hasNewLoop() const;

    // 清除回环检测标志（在主循环消费 hasNewLoop() 后调用）
    void clearNewLoopFlag();

    // 获取 PGO 优化后的全部关键帧位姿（PGO 图中所有顶点）
    std::vector<std::pair<int, Pose>> getAllOptimizedKeyFramePoses() const;

    // ★★★ 新增：获取滑动窗口内关键帧的 PGO 修正后位姿 ★★★
    // 用于轨迹回写时，将最新 PGO 修正应用到所有关键帧（包括窗口内）
    // 返回 keyframe_window_ 中所有关键帧的位姿（已应用 PGO 修正）
    std::vector<std::pair<int, Pose>> getWindowKeyFramePoses() const;

    // 是否启用了回环检测
    bool isLoopDetectionEnabled() const { return config_.enable_loop_detection; }

    // 获取完整配置（用于调试日志输出）
    const CovisConfig& getConfig() const { return config_; }

    // 设置 PGO 顶点数上限（需要在 initLoopDetection 之后调用）
    void setMaxPGOVertices(int max_v) {
        if (pose_graph_optimizer_) {
            pose_graph_optimizer_->setMaxVertices(max_v);
        }
    }

    // 从 YAML 配置更新运行时参数（在 Tracking 构造后、开始跟踪前调用）
    void updateConfig(const CovisConfig& cfg) {
        // 允许运行时更新关键参数
        if (cfg.max_candidates > 0) config_.max_candidates = cfg.max_candidates;
        if (cfg.max_confirmed > 0) config_.max_confirmed = cfg.max_confirmed;
        if (cfg.candidate_check_interval > 0) config_.candidate_check_interval = cfg.candidate_check_interval;
        if (cfg.candidate_confirm_threshold > 0) config_.candidate_confirm_threshold = cfg.candidate_confirm_threshold;
        if (cfg.min_covis_frames > 0) config_.min_covis_frames = cfg.min_covis_frames;
        if (cfg.loop_detection_interval > 0) config_.loop_detection_interval = cfg.loop_detection_interval;
        if (cfg.loop_min_inliers > 0) config_.loop_min_inliers = cfg.loop_min_inliers;
        if (cfg.icp_loop_max_distance_m > 0) config_.icp_loop_max_distance_m = cfg.icp_loop_max_distance_m;
        if (cfg.pgo_max_allowed_trans_corr > 0) config_.pgo_max_allowed_trans_corr = cfg.pgo_max_allowed_trans_corr;
        if (cfg.pgo_max_allowed_rot_corr_deg > 0) config_.pgo_max_allowed_rot_corr_deg = cfg.pgo_max_allowed_rot_corr_deg;
        if (cfg.pgo_max_allowed_rot_corr_deg_reject > 0) config_.pgo_max_allowed_rot_corr_deg_reject = cfg.pgo_max_allowed_rot_corr_deg_reject;
        if (cfg.pgo_warn_trans_corr > 0) config_.pgo_warn_trans_corr = cfg.pgo_warn_trans_corr;
        if (cfg.pgo_warn_rot_corr_deg > 0) config_.pgo_warn_rot_corr_deg = cfg.pgo_warn_rot_corr_deg;
        if (cfg.staleness_thresh > 0) config_.staleness_thresh = cfg.staleness_thresh;
        if (cfg.quantize_scale > 0) config_.quantize_scale = cfg.quantize_scale;
        if (cfg.same_point_distance_threshold > 0) config_.same_point_distance_threshold = cfg.same_point_distance_threshold;
        if (cfg.reproj_search_radius > 0) config_.reproj_search_radius = cfg.reproj_search_radius;
        if (cfg.ba_min_edges > 0) config_.ba_min_edges = cfg.ba_min_edges;
        if (cfg.ba_max_iterations > 0) config_.ba_max_iterations = cfg.ba_max_iterations;
        if (cfg.ba_huber_delta > 0) config_.ba_huber_delta = cfg.ba_huber_delta;
        if (cfg.max_reproj_error > 0) config_.max_reproj_error = cfg.max_reproj_error;
        if (cfg.enable_local_ba != config_.enable_local_ba) config_.enable_local_ba = cfg.enable_local_ba;
        if (cfg.verbose) config_.verbose = cfg.verbose;
        // K 矩阵：只从 YAML 覆盖（避免覆盖 Tracking 构造函数已设置的正确 TUM 内参）
        if (!cfg.K.empty() && cfg.K.rows == 3 && cfg.K.cols == 3) {
            if (config_.K.empty() || config_.K.rows != 3 || config_.K.cols != 3) {
                config_.K = cfg.K.clone();
            }
        }
    }

    // 是否启用了 PGO 位姿回写（关闭=保留纯 PnP 轨迹）
    bool isPGOEnabled() const { return config_.enable_pgo; }

    // ★★★ 强制禁用局部 BA ★★★
    // 在 Tracking::updateConfigFromYAML 中调用，覆盖 YAML 中可能启用的 BA
    // 原因：CovisManager BA 每 5 帧运行一次（20 次迭代），非常耗时
    void forceDisableLocalBA() { config_.enable_local_ba = false; }

    // 回填已确认点的观测（geo_match 后调用，补充旧关键帧的观测到 confirmed 点）
    void backfillConfirmedObservations(const KeyFrame::Ptr& kf);

    // ★★★ 提前回填 matched_kf 的观测 ★★★
    // 在 detectAndCorrectLoop 开头调用，在 geometricVerification 之前运行
    // 将 last_loop_matched_kf_id_ 的共视确认点回填到当前帧，
    // 使 Sim3Solver 能够找到两帧之间的共视匹配对
    void prefillMatchedKFObservations(const KeyFrame::Ptr& curr_kf);

    // ★★★ 触发候选点晋升 ★★★
    // 在 geometricVerification 中对 backfill 后的候选点执行即时晋升
    // 声明为 const：内部使用 mutable + const_cast 突破 const 限制（仅此一处）
    void promoteCandidatesExternal() const;

private:
    // 核心：候选点共视聚合
    // 将候选点按"3D位置相近"聚合，同一 3D 点被多个关键帧观测到时聚合到一起
    void aggregateCandidatesBy3DProximity();

    // 核心：共视校验 + 升级
    // 对每个候选点，检查是否被 ≥ min_covis_frames 个关键帧共同观测
    void promoteCandidates();

    // 候选点淘汰
    void cullCandidates();

    // 已确认点的周期检查（淘汰长期未观测或误差超标的点）
    void cullConfirmedPoints();

    // 根据 3D 位置相似度，判断两个候选点是否可能来自同一物理点
    bool isSamePhysicalPoint(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const;

    // 获取当前已确认的地图点（用于 PnP）
    std::vector<std::pair<Eigen::Vector3d, cv::Point2f>>
    getAllConfirmedPairs(const std::vector<cv::KeyPoint>& curr_kps,
                        const std::vector<cv::DMatch>& inlier_matches) const;

    CovisConfig config_;

    // runLocalBA 将修正后的地图点世界坐标写入此 map（下次 runLocalBA 前由 CovisManager 自己清空）
    std::unordered_map<int, Eigen::Vector3d> last_ba_mappoint_updates_;

    // runLocalBA 是否修正了最新关键帧的位姿（超过跳变门限），供 Tracking 查询
    bool recent_ba_pose_update_ = false;
    // ★★★ Bug Fix B1: 记录 BA 实际是否写回了位姿（用于判断 pts_3d 是否需要变换）★★★
    // 在 runLocalBA 写回循环中设置，当至少有一个关键帧位姿被修正时为 true
    bool last_ba_actual_pose_update_ = false;
    int running_frames_ = 0;
    mutable bool promote_called_from_external_ = false;
    int next_mappoint_id_ = 0;
    int check_counter_ = 0;
    int skip_local_ba_frames_ = 0;  // 回环后跳过若干帧 runLocalBA，避免双重内存峰值

    // 关键帧滑动窗口（mutable：getAllOptimizedKeyFramePoses() 为 const 但需读取窗口状态）
    mutable std::deque<KeyFrame::Ptr> keyframe_window_;

    // ★★★ O(1) 关键帧窗口索引查找：替代 O(n) 线性遍历 keyframe_window_ ★★★
    // key: kf_id, value: index in keyframe_window_
    std::unordered_map<int, size_t> kf_id_to_window_idx_;

    // 候选地图点（尚未通过共视校验）
    std::unordered_map<int, MapPoint::Ptr> candidate_mappoints_;

    // 已确认的静态地图点（通过共视校验）
    std::unordered_map<int, MapPoint::Ptr> confirmed_mappoints_;

    // ★★★ PGO 位姿缓存：避免直接读 g2o 图（g2o 优化后内部状态可能损坏导致 SIGSEGV）
    // detectAndCorrectLoop 回写时同步写入，getAllOptimizedKeyFramePoses() 只读此缓存
    mutable std::unordered_map<int, Pose> pgo_optimized_poses_cache_;

    // ★★★ 原始 PnP 位姿缓存：永远存原始 PnP 结果，不被 PGO 覆盖
    // 轨迹回写和 getAllOptimizedKeyFramePoses() 永远用这个缓存，保证尺度不被 PGO 破坏
    mutable std::unordered_map<int, Pose> pnp_original_poses_cache_;

    // 候选点去重索引：量化世界坐标的uint64_t哈希 → mappoint_id
    // 优化：用整数哈希替代字符串序列化（省去to_string开销，O(1)查找）
    // 量化到5cm格网（QUANTIZE_SCALE=20.0，1m/5cm）
    // 格式：x(21bits)|y(21bits)|z(21bits)，每分量范围约±100m
    std::unordered_map<uint64_t, int> obs_key_to_mp_;

    // 反向索引：(frame_id, kpt_idx) → confirmed_mappoint_id
    // 用于 getConfirmedMapPointsForPnP 的 O(1) 查询
    std::map<std::pair<int, int>, int> obs_key_to_confirmed_mp_;

    // 旧关键帧的确认点观测记录（用于回环时回填共视确认点）
    // old_kf_confirmed_obs_[kf_id] = {{mp_id, kpt_idx}, ...}
    // 在关键帧淘汰时从 keyframe_window_ 提取，geo_match 后回填到 confirmed_mappoints_
    std::unordered_map<int, std::vector<std::pair<int, int>>> old_kf_confirmed_obs_;

    // 反向索引：{kf_id, kpt_idx} → mp_id（用于回填后快速查找）
    std::unordered_map<uint64_t, int> kpt_key_to_confirmed_mp_;

    // ★★★ Bug Fix C5 已确认非 bug：Sim3Solver 的 project() 对 camera-frame 输入
    // 正确执行 R_cw * P_cam + t（camera-to-world），后续再接 T12 相对变换。
    // 参数名 vP3Dw 为误命名，但逻辑正确。无需修改。★★★
    // 注意：quantizeWorldKey 在头文件中声明但从未被调用，已删除。
    // 相关功能使用 cpp 中的 quantizeWorldPointKey() 函数。

    // ==================== 回环检测 + PGO ====================
    // 注册外部的 LoopDetector（由外部拥有，由 CovisManager 使用）
    std::shared_ptr<loopdetection::LoopDetector> loop_detector_;
    std::shared_ptr<loopdetection::PoseGraphOptimizer> pose_graph_optimizer_;

    int loop_detect_counter_ = 0;
    bool loop_corrected_since_last_kf_ = false;

    // 记录上一次回环修正后的世界坐标偏移量（用于漂移校正）
    Eigen::Matrix4d last_correction_T_ = Eigen::Matrix4d::Identity();

    // ★★★ matched_kf 共视地图点缓存 ★★★
    // 存储最近一次回环检测到的 matched_kf 的共视确认地图点 ID 列表
    // 用于 backfillConfirmedObservations 建立 matched_kf ↔ curr_kf 链接，
    // 使 getCovisibleMatchesWithCache 能找到两帧共视的确认地图点
    std::vector<int> last_matched_kf_covis_mp_ids_;
    // matched_kf 的 frame_id（用于 backfill 时获取其位姿）
    int last_loop_matched_kf_id_ = -1;

    // ★★★ 方案A：回环共视点保护 ★★★
    // 被最近一次 matched_kf 观测过的候选点 ID（保护期内不允许被 cullCandidates 淘汰）
    std::unordered_set<int> protected_candidate_ids_;
    // 被最近一次 matched_kf 观测过的确认点 ID（保护期内不允许被 cullConfirmedPoints 淘汰）
    std::unordered_set<int> protected_confirmed_ids_;
    // 保护期倒计时：距上次回环检测后还需保护多少帧（回环后设为 candidate_check_interval × 3）
    int loop_protection_frames_ = 0;

    // PGO 修正所有关键帧位姿（滑动窗口内 + 历史关键帧）
    void applyPGOCorrection(const Eigen::Matrix4d& correction_base_T);

    // 计算两关键帧之间的相对位姿（T_from→to）
    static Eigen::Matrix4d computeRelativePose(const Pose& pose_from,
                                               const Pose& pose_to);

    // ★★★ Bug Fix: 添加线程安全保护 ★★★
    // 用于保护共享数据结构的互斥锁
    mutable std::mutex data_mutex_;

    // 受保护的成员变量访问接口
    template<typename Func>
    void forEachKeyFrame(Func&& func) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        for (auto& kf : keyframe_window_) {
            func(kf);
        }
    }
};

} // namespace frontend
} // namespace sfc_core

#endif // SFC_CORE_FRONTEND_COVIS_MANAGER_HPP_
