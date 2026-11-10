// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_LOOPDETECTION_LOOP_DETECTOR_HPP_
#define SFC_CORE_LOOPDETECTION_LOOP_DETECTOR_HPP_

#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/frontend/covis_core.hpp"
#include "sfc_core/loopdetection/place_recognition.hpp"
#include "sfc_core/loopdetection/geometric_verifier.hpp"
#include "sfc_core/loopdetection/sim3_solver.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>

namespace sfc_core {
namespace loopdetection {

// 前向声明：CovisManager 定义在 sfc_core::frontend 命名空间（covis_manager.hpp）
// 使用 ::sfc_core::frontend::CovisManager 在 loopdetection 命名空间内引用
struct LoopCandidate {
    int current_kf_id;      // 当前关键帧 ID
    int matched_kf_id;      // 匹配到的历史关键帧 ID
    int raw_matches;        // 描述子匹配数
    int geometric_inliers;  // 几何校验内点数
    double inlier_ratio;    // 内点率
    Eigen::Matrix4d relative_pose;  // 当前帧 → 匹配帧 的相对位姿（T_cw_cur.inverse() * T_cw_match）
    bool valid = false;     // 是否通过所有校验
    bool is_icp = false;     // ICP 3D-3D（有真实尺度）=true，Essential Matrix（尺度归一化）=false
};

struct LoopDetectorConfig {
    // 时间跨度约束：候选帧必须与当前帧至少隔这么多个关键帧（避免与最近邻误匹配）
    // 从 20 降低到 15，允许更近的回环，提高室内场景检出率
    int min_loop_separation = 15;

    // 描述子匹配参数
    int max_match_distance = 80;        // NN 匹配距离阈值（ORB: 0~256）
    double lowe_ratio_threshold = 1.0; // 1.0 = 禁用 Lowe 比率过滤（交叉一致性已足够）

    // BoW 预过滤
    int min_common_words = 6;           // 两帧 BoW vector 至少应有多少公共词

    // 几何校验
    // ransac_iterations / ransac_threshold_px / max_icp_inlier_ratio 原本也在这里，
    // 与 GeometricVerifierConfig 的同名字段是同一批参数的两份声明（R-023）：
    // 构造函数会把它们覆盖进几何验证器，于是 geometric_verifier.yaml 里配的值永远不生效。
    // 现在删掉，只留几何验证器那一份（见下面的 geometric_verifier 成员）。
    int min_geometric_inliers = 6;      // detect() 单候选路径的最少内点数（不是几何验证器内部阈值）

    // 匹配帧选择：从最近 N 个关键帧中选（排除太近的）
    int candidate_pool_size = 30;

    // ★★★ 多候选验证配置 ★★★
    // 当前只验证得分最高的候选，容易被假阳性欺骗
    // 启用后验证前 N 个候选，选择最优者
    bool enable_multi_candidate_verification = true;  // 启用多候选验证
    int max_candidates_to_verify = 5;                  // 最多验证的候选数
    // ★★★ P0-1: 修改回环确认条件 ★★★
    // 原值 2 过严，当只有1个候选通过时回环被拒绝，导致漂移无法修正
    // 改为 1：只要 1 个候选通过就认为有效，与 ORB_SLAM3 的 mnLoopNumCoincidences>=1 一致
    // 注意：这会降低鲁棒性，但在室内短回环场景下可以接受
    int min_candidates_to_accept = 1;
    double best_candidate_ratio_threshold = 0.85;      // 从 0.8 降低到 0.85，要求最佳候选更明显优于次优
    bool require_candidate_agreement = true;          // 启用候选一致性检查，过滤 BoW 假阳性

    // ★★★ 质量评分参数（YAML 可配置）★★★
    // 质量评分公式：quality = inliers * inlier_weight + ratio * matches * ratio_weight + log1p(bow) * bow_weight
    double geometric_inlier_weight = 1.0;   // 内点基础权重
    double inlier_ratio_weight = 2.0;         // 内点率加权系数
    double bow_score_weight = 0.5;            // BoW 分数对数权重

    // ★★★ Sim3 尺度参数 ★★★
    // 原 scale_min / scale_max 只服务于已删除的 correctLoopWithScale()（用漂移后的
    // 位姿距离当尺度真值）。现在尺度来自几何验证本身：ICP 有深度、EM 路径用
    // 3D 对应点投影中位数，两边都是实测值，不再需要配置里的夹紧区间。
    // D5-e：DBoW2 查询参数与候选筛选门槛（原来写死在 detect() 里，R-018）
    int dbow_temporal_window = 3;        // queryWithConsistency 的时序窗口
    int dbow_max_results = 10;           // queryWithConsistency 最多返回多少个候选
    double dbow_min_distance_m = 1.0;    // 与当前帧距离小于此值的候选直接丢弃
    double loop_max_spatial_dist_m = 8.0;  // 空间距离上限：超过它且帧间隔够近说明只是"路过"
    int min_frames_between_for_spatial_gate = 30;

    // D4-d：回环检测/PGO 的最小关键帧间隔。
    // 1 = 每个关键帧都检测（与改动前的实际行为一致）；调大可以显著降 CPU，
    // 代价是同一个回环要等更久才被发现。
    // 注意这与 CovisManager 的 loop_detection_interval（按帧计数）不是同一个量，
    // 两者分别作用于两条不同的回环路径。
    int loop_check_interval_kf = 1;  // 帧间隔小于此值才启用上面的空间距离闸门

    double frame_dist_min = 0.01;  // 最小帧间距离（米），低于此值不注入里程计尺度
    double frame_dist_max = 50.0;  // 最大帧间距离（米），高于此值不注入里程计尺度

    // ★★★ 候选一致性参数 ★★★
    double max_spatial_dist_for_consistency = 10.0;  // 计算一致性的最大空间距离（米）
    double consistency_warning_threshold = 0.5;     // 一致性警告阈值

    // 几何校验用相机内参（像素坐标下的 EPnP）；空则使用 3×3 单位阵（不推荐）
    cv::Mat camera_K;
    cv::Mat camera_D;

    // ★★★ 几何验证参数（D3）★★★
    // 直接持有 GeometricVerifierConfig，而不是把它的字段在本结构里再抄一遍。
    // 链路：config/geometric_verifier.yaml → loadGeometricVerifierConfig()
    //       → AllConfigs::geometric_verifier → AllConfigs::loop_detector.geometric_verifier
    //       → LoopDetector 构造函数 → GeometricVerifier。
    // 原实现这条链在第二跳就断了（yaml 读进 AllConfigs 后无人使用），构造函数里
    // 另写一份字面量 —— 改配置不生效、调参只能改源码重编译。
    GeometricVerifierConfig geometric_verifier;

    // 诊断输出开关。本文件原有 60 处无条件 std::cerr，回环检测在正常序列上
    // "没找到回环"是常态而非错误，每帧刷屏只会淹没真正的失败信息（R-017/R-021）。
    bool verbose = false;
};

class LoopDetector {
public:
    struct KeyFrameEntry {
        int kf_id;
        cv::Mat descriptors;
        sfc_core::frontend::Pose pose;
        std::vector<int> bow_vector;
        std::vector<cv::KeyPoint> keypoints;
        std::vector<Eigen::Vector3d> pts3d;
    };

    explicit LoopDetector(const LoopDetectorConfig& config = LoopDetectorConfig());

    // 当前关键帧进入：计算 BoW vector，存入历史
    void addKeyFrame(int kf_id,
                     const std::vector<cv::KeyPoint>& keypoints,
                     const std::vector<Eigen::Vector3d>& pts3d,
                     const cv::Mat& descriptors,
                     const sfc_core::frontend::Pose& pose);

    // 检测回环
    // covis_mgr: 可选，传入 ::sfc_core::frontend::CovisManager* 以启用 Sim3Solver 共视地图点几何验证
    // 注意：使用 ::sfc_core::frontend::CovisManager（无 class 关键字）引用 covis_core.hpp 中的前向声明
    LoopCandidate detect(const std::vector<cv::KeyPoint>& curr_kps,
                        const std::vector<Eigen::Vector3d>& curr_pts3d,
                        const cv::Mat& curr_desc,
                        ::sfc_core::frontend::CovisManager* covis_mgr = nullptr);

    int numKeyFrames() const { return keyframe_db_.size(); }
    bool hasNewLoop() const { return last_loop_candidate_.valid; }
    const LoopCandidate& lastCandidate() const { return last_loop_candidate_; }
    void clearNewLoopFlag() { last_loop_candidate_.valid = false; }

    // 获取关键帧数据库（用于在回环检测时用 PnP 世界位姿计算相对变换）
    const std::unordered_map<int, KeyFrameEntry>& getKeyFrameDb() const { return keyframe_db_; }

    // ★★★ 多候选验证 ★★★
    // 验证多个候选帧，选择几何一致性最好的
    // 返回验证成功的候选列表，按质量排序
    struct VerifiedCandidate {
        int kf_id;
        int raw_matches;
        int geometric_inliers;
        double inlier_ratio;
        double bow_score;
        Eigen::Matrix4d relative_pose;
        double quality_score;  // 综合质量分数
        bool is_icp = false;  // 是否使用 ICP 3D-3D（有真实尺度）；false=Essential Matrix（尺度归一化）
    };
    
    std::vector<VerifiedCandidate> verifyMultipleCandidates(
        const std::vector<cv::KeyPoint>& curr_kps,
        const std::vector<Eigen::Vector3d>& curr_pts3d_world,
        const cv::Mat& curr_desc,
        const std::vector<std::pair<int, int>>& bow_candidates,
        const frontend::Pose& curr_pose_world,
        ::sfc_core::frontend::CovisManager* covis_mgr = nullptr);
    
    // 从多个验证过的候选中选择最佳
    VerifiedCandidate selectBestFromVerified(
        const std::vector<VerifiedCandidate>& verified_candidates);
    
    // 检查候选之间的一致性（用于避免假阳性）
    double computeCandidateConsistency(const std::vector<VerifiedCandidate>& candidates);

    // ★ PGO 优化后同步位姿到 keyframe_db_
    // CovisManager 在 PGO 优化完成后调用此方法，用 PGO 优化后的位姿更新 keyframe_db_，
    // 确保下次 geometricVerification 时 world→camera 变换使用最新位姿。
    void updateKeyFramePose(int kf_id, const frontend::Pose& pose);

    // ★ Sim3Solver 几何验证（参考 ORB_SLAM3 Sim3Solver + OptimizeSim3 两轮精化）
    // 使用共视确认地图点（而非原始帧 3D 点），显著提升 ICP 成功率。
    //
    // covis_matches: 两帧之间的共视确认地图点匹配对
    //   （由 CovisManager::getCovisibleMatches() 准备好后在此处传入）
    // pose_curr / pose_matched: 两帧的当前世界位姿（用于 world→camera 变换）
    // out_relative_pose[out]: 输出的相对位姿
    // out_inliers[out]: 输出的几何内点数
    // out_is_icp[out]: 是否使用 ICP（有真实尺度）
    // 返回: true=验证通过
    //
    // 简化参数（避免循环 include CovisManager）：
    //   curr_pts3d_world[i]: 当前帧的共视确认地图点的世界坐标
    //   curr_kpt_indices[i]: 当前帧中该地图点对应的特征点索引
    //   matched_pts3d_world[i]: 匹配帧的共视确认地图点的世界坐标
    //   matched_kpt_indices[i]: 匹配帧中该地图点对应的特征点索引
    //   pose_curr / pose_matched: 两帧的世界位姿（用于 world→camera 变换）
    //   kp_curr / kp_matched: 两帧的原始关键点（用于提供像素坐标作为观测值）
    bool geometricVerificationWithSim3(
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
        bool* out_is_icp) const;

private:
    std::vector<int> computeBoW(const cv::Mat& descriptors) const;
    int commonWordCount(const std::vector<int>& a, const std::vector<int>& b) const;
    std::vector<cv::DMatch> matchDescriptors(const cv::Mat& query_desc,
                                            const cv::Mat& train_desc) const;
    bool geometricVerification(const std::vector<cv::KeyPoint>& kp_query,
                             const std::vector<cv::KeyPoint>& kp_train,
                             const std::vector<Eigen::Vector3d>& pts3d_query_world,
                             const std::vector<Eigen::Vector3d>& pts3d_train_world,
                             const std::vector<cv::DMatch>& matches,
                             int curr_kf_id,
                             int matched_kf_id,
                             Eigen::Matrix4d* out_relative_pose,
                             int* out_inliers,
                             bool* out_is_icp = nullptr,
                             const ::sfc_core::frontend::CovisManager* covis_mgr = nullptr) const;

    LoopDetectorConfig config_;

    std::unordered_map<int, KeyFrameEntry> keyframe_db_;
    std::vector<int> keyframe_ids_;
    LoopCandidate last_loop_candidate_;
    cv::Mat K_cv_;
    cv::Mat D_cv_;

    // DBoW2 风格的位置识别
    std::unique_ptr<PlaceRecognition> place_recog_;
    // RANSAC Essential Matrix 几何验证器
    std::unique_ptr<GeometricVerifier> geometric_verifier_;
    // 最近回环匹配历史（用于时序一致性检查）
    std::vector<int> recent_loop_matches_;
};

}  // namespace loopdetection
}  // namespace sfc_core

#endif  // SFC_CORE_LOOPDETECTION_LOOP_DETECTOR_HPP_
