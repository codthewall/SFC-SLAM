#ifndef SFC_CORE_FRONTEND_TRACKING_HPP_
#define SFC_CORE_FRONTEND_TRACKING_HPP_

// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#include <memory>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/feature/octree_orb.hpp"
#include "sfc_core/feature/bow_adapter.hpp"
#include "sfc_core/feature/feature_matcher.hpp"
#include "sfc_core/frontend/robust_pnp_solver.hpp"
#include "sfc_core/mapping/map.hpp"
#include "sfc_core/mapping/key_frame.hpp"
#include "sfc_core/mapping/map_point.hpp"
#include "sfc_core/mapping/local_mapping.hpp"
#include "sfc_core/mapping/atlas.hpp"
#include "sfc_core/frontend/covis_core.hpp"
#include "sfc_core/loopdetection/loop_detector.hpp"
#include "sfc_core/loopdetection/pose_graph_optimizer.hpp"

namespace sfc_core { namespace frontend {

// ================================================================
// Tracking 配置
// ================================================================
struct TrackingConfig {
    feature::OctreeORBConfig extractor_config;
    feature::MatcherConfig matcher_config;
    cv::Mat K;
    cv::Mat D;
    // ★★★ TUM PNG 格式：depth_m = pixel * 0.0002（ORB_SLAM3 标准）
    // factor=5000，即 pixel / 5000 = 米
    double depth_scale = 0.0002;
    double min_depth = 0.1;
    double max_depth = 10.0;
    double depth_edge_thresh = 0.25;

    double min_keyframe_translation = 0.05;
    double min_keyframe_rotation = 0.05;
    int min_features = 50;

    int forced_keyframe_interval = 6;

    // ---------------- 关键帧插入策略 ----------------
    // 判定本体在 frontend/keyframe_policy.hpp 的 decideKeyframe()，这里是它的输入参数。
    // min_keyframe_translation / min_keyframe_rotation / forced_keyframe_interval
    // 继续作为同名的 YAML 键与 --kf_trans/--kf_rot 入口，不另设同义字段（R-019）。
    // 两关键帧之间的最小帧间隔（硬下限）：原实现的"间隔>=10帧"与"内点<参考点25%"
    // 都没有下限，于是 640 帧插出 562 个关键帧，局部地图全是零视差的相邻帧。
    int min_keyframe_gap_frames = 8;
    int keyframe_min_tracked_inliers = 120;   // 内点高于此值时不因相对内点率下降插帧
    double keyframe_weak_tracked_ratio = 0.30;  // 内点 < 参考活着点数×此比例 → 跟踪退化
    int keyframe_rescue_min_inliers = 40;     // 兜底插入：内点长期低于此值
    int keyframe_rescue_max_gap = 60;         //   且已间隔这么多帧时插一帧

    // ---------------- 局部地图与建点配额 ----------------
    // 局部关键帧窗口。20 太小：关键帧一密，窗口只覆盖不到 1 秒的视角，没有视差。
    int local_map_keyframe_window = 40;
    // 新建地图点的观察期（帧）。原值 3 太短：建点时 obs_count=1，3 帧内没被
    // 第二次观测到就整批判坏，地图长不起来。
    int map_point_grace_frames = 8;
    // 单个关键帧最多新建的地图点数。原值 100 让远处的强结构点抢不到配额。
    int max_new_map_points_per_kf = 300;
    // 注：纯 VO 路径不持有 PGO / LocalBA 开关。这两个能力分别由
    // loopdetection::PoseGraphOptimizerOptions 与 frontend::CovisConfig 提供，
    // TrackingConfig 里再放一份同名字段只会造成"配置了却不生效"的假象（R-019）。
    //
    // use_gravity_constraint 保留但尚未接线：applyGravityConstraint() 目前没有调用点。
    // 打算启用时，必须在同一改动里补上调用点，否则又是一个"配了不生效"的开关（R-022）。
    bool use_gravity_constraint = false;

    int pnp_max_ransac_iterations = 200;
    double pnp_initial_thresh_px = 8.0;
    double pnp_final_thresh_px = 4.0;
    int pnp_expansion_rounds = 2;
    double pnp_expansion_factor = 1.5;
    int pnp_ba_max_iterations = 15;
    bool pnp_use_huber_kernel = true;
    double pnp_huber_delta = 2.45;
    bool pnp_use_depth_weighting = false;

    // EMA 平滑参数：相对位姿的指数移动平均衰减系数。
    // 注：纯 VO 路径当前没有 EMA 消费者（tracking.cpp 里无读取点），
    // 保留字段是为了让 covis_manager 那条链路仍可按需接线（R-022）。
    double relative_t_ema_alpha = 0.9;

    // 静止检测死区（用于区分静止和运动状态）。
    // 注：同 relative_t_ema_alpha，纯 VO 路径当前无消费者。
    bool stationary_deadzone_enabled = false;
    double stationary_trans_deadzone_m = 0.01;   // 平移死区（米）
    double stationary_rot_deadzone_rad = 0.01;   // 旋转死区（弧度）

    // poseOptimization 后的最小内点数，低于此值视为本帧跟踪失败。
    // 原值 5：5 个点约束 6 个自由度，等于放任退化解通过（实测 267/640 帧就是
    // 在这种状态下被当成"跟踪成功"，位姿噪声直接进轨迹）。健康帧有数百内点。
    int pose_opt_min_inliers = 20;

    bool verbose_diag = false;

    // 关键点去畸变开关。R3 之后语义收敛为「相机模型要不要去畸变」，
    // 不再和"深度图是否已配准到 RGB"挂钩 —— 后者只决定深度采样用哪个像素，
    // 而 compute3DPoints() 已经固定用畸变像素采样深度、用理想像素反投影。
    // true : undistorted_keypoints = keypoints（不去畸变）。只有 D 为空时才物理正确。
    // false: undistorted_keypoints = undistortPoints 后重新像素化（理想像素）。推荐。
    bool disable_undistort = false;
};

// ================================================================
// 地图点数据结构（ORB_SLAM3 MapPoint 简化版）
// ================================================================
struct MapPointCandidate {
    int id = -1;
    Eigen::Vector3d world_point = Eigen::Vector3d::Zero();
    std::vector<std::pair<int, int>> observations;  // {frame_id, keypoint_index}
    cv::Mat descriptor;
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    bool is_bad = false;
    // 前端观测计数：本结构体是这条计数的事实来源，由 addObservationToMapPoint()
    // 在每次 searchLocalPoints() 成功匹配时 +1。后端 mapping::MapPoint::found_count_
    // 是另一套独立计数器，只做统计，不参与这里的建图/清理判定（R5）。
    int frontend_observation_count = 0;
    int found_count = 0;  // ★ ORB-SLAM3: 累计被当前帧 found（用于地图点质量评估）
    bool confirmed = false;  // 是否已确认（观测数>=2）
    int created_frame_id = -1;  // 创建该点的帧 ID（清理时的观察期基准）
    int last_observed_frame_id = -1;  // 最近一次被匹配到的帧 ID
};

// ================================================================
// Pure ORB_SLAM3 RGB-D Visual Odometry
// 完全遵循 ORB_SLAM3 的简化 VO 流水线
// ================================================================
// 注意：KeyFrameCacheEntry 在 covis_core.hpp 中定义，tracking.hpp 引用它
class Tracking {
public:
    ~Tracking();
    Tracking(const TrackingConfig& config);

    Pose track(const cv::Mat& rgb_img,
              const cv::Mat& depth_img,
              const cv::Mat& ir_img,
              double timestamp);

    std::vector<cv::KeyPoint> getCurrentKeypoints() const;
    std::vector<cv::KeyPoint> getCurrentUndistortedKeypoints() const;
    int getLastInlierCount() const { return last_inlier_count_; }
    bool isInitialized() const { return is_initialized_; }
    int getTrackingState() const { return static_cast<int>(mState_); }
    int getLastNGood() const { return mnMatchesInliers_; }
    /// 上一帧是否被判定为关键帧（由 track() 内部真实判定写入，供外部评测使用）
    bool isLastFrameKeyFrame() const { return mbLastFrameIsKeyFrame_; }

    // ================================================================
    // 完整 SLAM：回环检测 + 位姿图优化
    // ================================================================
    // 调用前 Tracking 是纯 VO（每帧只做 PnP + poseOptimization，误差单向累积）；
    // 调用后每建一个关键帧就做一次回环检测，命中则加回环边、跑 PGO，并把优化后的
    // 关键帧位姿与地图点一次写回前端缓存。
    void initLoopDetection(const loopdetection::LoopDetectorConfig& loop_cfg,
                          const loopdetection::PoseGraphOptimizerOptions& pgo_cfg);

    /// 回环检测是否已启用（initLoopDetection 之后为 true）
    bool isLoopDetectionEnabled() const { return loop_detector_ != nullptr; }
    /// 实际生效（通过几何校验并完成 PGO）的回环次数
    int getNumDetectedLoops() const { return mnDetectedLoops_; }
    /// 位姿图中当前的图内回环边数；未启用时为 0
    int getNumLoopEdges() const;

    const std::unordered_map<int, Eigen::Vector3d>& getLastBAMapPointUpdates() const;

    // ★★★ 获取所有确认的地图点（用于重力对齐等后处理）★★★
    std::vector<Eigen::Vector3d> getConfirmedMapPoints() const;

    // 关闭 LocalMapping 线程（显式调用，确保所有队列中的关键帧被处理）
    void shutDown() {
        if (mpLocalMapper_) mpLocalMapper_->shutDown();
    }

    // ★★★ 消费 LocalMapping 的 BA 结果，更新前端地图点 ★★★
    void consumeBAUpdates();

    // ★★★ 临时地图点管理（ORB-SLAM3 mlpTemporalPoints）★★★
    // 当前帧深度三角化产生的未确认地图点，用于跟踪。
    // E6：ID 列表与点池一起清，避免只清列表、点留在池里继续被搜到。
    std::vector<int>& getTemporalPoints() { return mlpTemporalPoints_; }
    void clearTemporalPoints() { clearTemporalMapPoints(); }

    CovisManager* getCovisManager() const { return nullptr; }  // 纯 VO，无 CovisManager

    // ★★★ OctreeORB 访问器（用于 ScaleFactors 等）★★★
    const feature::OctreeORB* getOctreeORB() const { return extractor_.get(); }

    // ★★★ BoW 访问器（用于重定位等）★★★
    const feature::BoWAdapter* getBoWAdapter() const { return bow_adapter_.get(); }
    void setBoWVocabulary(const std::string& voc_path);

private:
    // ===== 初始化 =====
    Pose initialize();

    // ===== 跟踪方法（返回 int = nGood，0 = 失败）=====
    int trackWithMotionModel();
    int trackReferenceKeyFrame();
    bool relocalization();
    void updateLastFrame();

    // ===== 局部地图 =====
    void updateLocalMap(bool add_current_frame = true);
    void searchLocalPoints();

    // ===== SearchByProjection（ORB_SLAM3 风格）=====
    int searchByProjection(Frame& CurrentFrame, const Frame& LastFrame,
                          float th, bool bMono);

    // ===== 三角化 =====
    void triangulateNewMapPoints();
    void cleanOldMapPoints();
    // 真正回收坏点（E6）：cleanOldMapPoints() 只标记 is_bad，本函数把标记过的点
    // 从 mAllMapPoints_ 里 erase，并清理各帧 map_point_ids / 索引数组里的悬空 id。
    // 只在单帧流程末尾、没有任何 MapPointCandidate* 被持有的时点调用。
    void retireBadMapPoints();

    // ===== 位姿优化 =====
    int poseOptimization();
    void updateMotionModel();

    // ===== 关键帧 =====
    bool needNewKeyFrame();
    void createNewKeyFrame();
    bool checkKeyFrame(Frame::Ptr frame, const PnPResult* pnp_result = nullptr, const Pose* smoothed_pose = nullptr);
    bool checkKeyFrameSimple(Frame::Ptr frame, const PnPResult* pnp_result = nullptr, const Pose* pose = nullptr);

    // ===== 回环检测 / PGO（仅在 loop_detector_ 非空时生效）=====
    // 关键帧入册：登记到 LoopDetector 的 BoW 数据库 + PGO 顶点 + 与上一关键帧的里程计边
    void registerKeyFrameForLoopDetection(const KeyFrameCacheEntry& kf);
    // 回环检测 + 位姿图优化 + 修正回写。返回是否真正完成了一次修正
    bool detectAndCorrectLoop();
    // 把 PGO 优化后的位姿写回前端缓存（mAllKeyFrames_ / mvpLocalKeyFrames_ / mpMap_）
    int applyOptimizedPosesToFrontend();

    // ===== 辅助 =====
    void compute3DPoints(Frame::Ptr frame, const cv::Mat& imDepth);
    // E6：清空临时点池（点 + ID 列表一起清）
    void clearTemporalMapPoints();
    void updateWorldPoints(Frame::Ptr frame);
    void checkReplacedInLastFrame();
    Pose applyGravityConstraint(const Pose& pose) const;
    std::vector<std::pair<Eigen::Vector3d, cv::Point2f>>
    searchLocalMap(const Frame::Ptr& curr_frame, const Pose& curr_world_pose);

    // ★★★ ORB-SLAM3 BoW 重定位候选帧搜索 ★★★
    // 扩展候选关键帧池：从整个 BoW 数据库中搜索与当前帧相似度最高的 N 个关键帧
    std::vector<KeyFrameCacheEntry> getBoWCandidateKeyFrames(int max_candidates = 20) const;

    // ===== 地图点管理（基于 ID，O(1) 查找）=====
    // permanent=false 的点是纯跟踪用的临时点（ORB-SLAM3 mlpTemporalPoints），
    // 不进 mNewMapPointIds_，因此不会被 cleanOldMapPoints() 判坏。
    int createMapPoint(const Eigen::Vector3d& world_pt, const cv::Mat& desc,
                       int frame_id, int kpt_idx, int obs_count, bool permanent = true);
    MapPointCandidate* getMapPoint(int id);
    void addObservationToMapPoint(int mp_id, int frame_id, int kpt_idx, const cv::Mat& desc = cv::Mat());

    TrackingConfig config_;

    // ★★★ Octree ORB 特征提取器（ORB-SLAM3 风格）★★★
    std::unique_ptr<feature::OctreeORB> extractor_;
    // ★★★ BoW 词袋适配器（ORB-SLAM3 风格重定位）★★★
    std::unique_ptr<feature::BoWAdapter> bow_adapter_;
    std::unique_ptr<feature::FeatureMatcher> matcher_;
    std::unique_ptr<RobustPnPSolver> robust_pnp_solver_;

    Frame::Ptr mLastFrame_;
    Frame::Ptr mCurrentFrame_;
    int next_frame_id_ = 0;
    bool is_initialized_ = false;
    int last_inlier_count_ = 0;

    // ===== 运动模型 =====
    Pose mVelocity_;
    bool mbVelocityValid_ = false;

    // ===== 跟踪状态 =====
    enum TrackState { NO_IMAGES_YET, NOT_INITIALIZED, OK, RECENTLY_LOST, LOST };
    TrackState mState_ = NOT_INITIALIZED;
    TrackState mLastProcessedState_ = NOT_INITIALIZED;
    int mnLastRelocFrameId_ = 0;
    int mnRelocConsecutiveFailures_ = 0;  // 连续 Reloc 失败帧数（渐进放宽阈值用）
    int mnBestRelocKFId_ = -1;  // RELOC 成功后使用的最佳参考关键帧 ID（防止用坏帧做 TrackRefKF）
    bool mbBestRelocKFValid_ = false;  // mBestRelocKF_ 是否有效
    KeyFrameCacheEntry mBestRelocKF_;    // RELOC 成功后缓存的最佳关键帧数据
    int mnLastKeyFrameId_ = 0;
    int mTrackedFr_ = 0;
    int mnMatchesInliers_ = 0;
    bool mbMapUpdated_ = false;
    bool mbVO_ = false;
    double mTimeStampLost_ = 0.0;  // 跟踪丢失时的时间戳

    // ===== 局部关键帧 =====
    // 只追加（push_back），不删除。最多 20 个，超过则从头删除
    std::deque<KeyFrameCacheEntry> mvpLocalKeyFrames_;
    // ★★★ 全局关键帧缓存（用于 BoW 重定位候选搜索）★★★
    // 保存所有关键帧的完整数据，不受 mvpLocalKeyFrames_ 20 个限制
    std::unordered_map<int, KeyFrameCacheEntry> mAllKeyFrames_;
    int mpReferenceKF_ = -1;

    // ===== 地图点（按 ID 索引的 unordered_map）=====
    // 使用 unordered_map<int, MapPointCandidate>：ID → 点。
    // 只放"永久点"（关键帧/三角化产出的地图点）。临时 VO 点在 mTemporalMapPoints_，
    // 生命周期只有一帧，池整体清空即回收（E6：以前临时点也塞在这里，每帧新建数百
    // 个且永不回收，是 mAllMapPoints_ 单调增长的唯一来源）。
    std::unordered_map<int, MapPointCandidate> mAllMapPoints_;
    int next_mp_id_ = 0;
    // 本轮新建（或本轮重建覆盖）的地图点 ID 集合。
    // cleanOldMapPoints() 只遍历这一批，不再每建一个关键帧就 O(N) 扫全图（R5）。
    std::vector<int> mNewMapPointIds_;

    // ===== E6：临时 VO 点池（ORB-SLAM3 mlpTemporalPoints 语义）=====
    // 与 mAllMapPoints_ 共用同一个 id 空间（next_mp_id_ 单调递增，绝不回收 id），
    // getMapPoint() 两处都查。清空时机见 clearTemporalMapPoints()。
    std::unordered_map<int, MapPointCandidate> mTemporalMapPoints_;

    // ===== E6：待退休的永久点 =====
    // cleanOldMapPoints() 判坏时登记，retireBadMapPoints() 在单帧流程末尾真正 erase。
    std::vector<int> mBadMapPointIds_;

    // ===== E6：世界坐标体素索引（近邻查重）=====
    // key = 体素坐标三轴打包，value = 该体素内的永久点 id。
    // 建新点前查同体素内是否有 < 5 cm 的已有点，命中则复用 id 而不是新建。
    std::unordered_map<int64_t, std::vector<int>> mVoxelMapPoints_;
    void addToVoxelIndex(int id, const Eigen::Vector3d& world_pt);
    void removeFromVoxelIndex(int id, const Eigen::Vector3d& world_pt);

    // ===== 当前帧的地图点（按特征索引）=====
    // 当前帧 mCurrentFrame_ 的特征点索引 → 地图点 ID（-1 表示无地图点）
    std::vector<int> mvpCurrentMapPointIds_;  // 长度 = 特征点数

    // ===== 上一帧的地图点（按特征索引）=====
    std::vector<int> mLastFrameMapPointIds_;

    // ===== LocalMapping 线程（ORB-SLAM3 后端）=====
    // 地图管理器
    mapping::Map::Ptr mpMap_;
    // LocalMapping 线程
    mapping::LocalMapping::Ptr mpLocalMapper_;
    // 地图点替换映射（用于 CheckReplacedInLastFrame）
    std::unordered_map<int, int> mReplaceMap_;  // old_mp_id → new_mp_id
    // BA 更新标记（ORB-SLAM3 mbMapUpdated_）
    bool mbMapUpdatedByBA_ = false;

    // ★★★ 临时地图点（ORB-SLAM3 mlpTemporalPoints）★★★
    // 当前帧深度三角化产生的未确认地图点，仅用于当前帧跟踪
    std::vector<int> mlpTemporalPoints_;

    // 上一帧是否为关键帧（track() 的真实判定结果，not "i % N == 0" 之类的推测）
    bool mbLastFrameIsKeyFrame_ = false;

    // ===== 回环检测 / 位姿图优化（initLoopDetection 之后非空）=====
    // 纯 VO 时两者都是 nullptr，track() 里所有回环相关分支都不执行——
    // 这样"不调用 initLoopDetection"和"调用"就是两条明确的行为路径，不靠配置开关猜。
    std::unique_ptr<loopdetection::LoopDetector> loop_detector_;
    std::unique_ptr<loopdetection::PoseGraphOptimizer> pose_graph_optimizer_;
    // 上一次做回环检测的关键帧 ID；用于按 FORCED 间隔节流（回环检测比建关键帧更贵）
    int mnLastLoopCheckKFId_ = -1;
    // D4-d：两次回环检测之间至少间隔多少个关键帧（来自 LoopDetectorConfig::loop_check_interval_kf）
    int mnLoopCheckIntervalKF_ = 1;
    // 上一个进入 PGO 的关键帧 ID（用于连里程计边）
    int mnLastPGOKeyFrameId_ = -1;
    // 成功完成的回环修正次数（仅统计 PGO 真正跑通并写回的）
    int mnDetectedLoops_ = 0;
    // PGO 优化后的地图点世界坐标，供 getLastBAMapPointUpdates() 对外暴露
    std::unordered_map<int, Eigen::Vector3d> last_ba_map_point_updates_;

    // ★★★ 多地图管理（ORB-SLAM3 Atlas）★★★
    mapping::Atlas::Ptr mpAtlas_;
};

} // namespace frontend
} // namespace sfc_core

#endif // SFC_CORE_FRONTEND_TRACKING_HPP_
