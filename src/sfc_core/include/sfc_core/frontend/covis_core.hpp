// 必须在最开头执行：清除 CMake project() 注入的 sfc_core 宏
// 原因：g2o 的 CMakeLists.txt 设置了 g2o_DIR，且 project(sfc_core) 会生成 -Dsfc_core=1 宏。
// 若不提前 undef，标准库头文件（如 <unordered_set>）会被解析为:
//   std::unordered_set → sfc_core_1::frontend::std::unordered_set
// 从而破坏 std::hash、std::unordered_set 等模板的正常查找。
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FRONTEND_COVIS_CORE_HPP_
#define SFC_CORE_FRONTEND_COVIS_CORE_HPP_

#include "sfc_core/frontend/frame.hpp"
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace sfc_core {
namespace frontend {

// 前向声明 CovisManager（完整定义在 covis_manager.hpp）
// 这打破了 covis_core.hpp → loopdetector.hpp → covis_manager.hpp → covis_core.hpp 循环中的不完整类型问题。
// 当任何文件 #include covis_core.hpp 时，CovisManager 已是前向声明（而非未知类型），
// 使得 CovisManager 可被用作返回值、参数类型等完整类型要求的上下文中。
// covis_core.cpp 中通过 #include covis_manager.hpp 将此前向声明升级为完整定义。
class CovisManager;

// 候选点的聚合索引（用于多帧共视判断）
// key: 3D点的像素位置桶（u_bin, v_bin）→ 该桶内所有候选点的 id
// 用于判断：同一个 3D 点是否被多个关键帧共同观测到
struct CandidateBucket {
    int u_bin;   // 像素 u 离散桶（每 8 像素一个桶）
    int v_bin;   // 像素 v 离散桶
    int kpt_idx; // 该帧中该点的特征索引
    int frame_id;

    bool operator==(const CandidateBucket& other) const {
        return u_bin == other.u_bin && v_bin == other.v_bin &&
               kpt_idx == other.kpt_idx;
    }
};

// ============= MapPoint =============
// 单个地图点，携带多关键帧的共视观测记录
// 注意：此版本用于前端协方差追踪流水线。
// 后端地图管理使用 sfc_core::mapping::MapPoint（在 mapping/map_point.hpp 中定义），
// 它更完整，支持更多 ORB-SLAM3 功能（描述子计算、法向量更新等）。
// 两个版本服务于不同的目的，不应混淆使用。

struct MapPointObservation {
    int frame_id = -1;          // 观测到该点的关键帧 ID
    int kpt_idx = -1;          // 在该帧中的特征点索引
    cv::Point2f pixel;         // 该帧中该特征点的像素坐标
    Eigen::Vector3d pt_cam;    // 该帧相机系下的 3D 点坐标
    double reproj_error = -1.0; // 重投影误差（像素）
    // ★★★ 描述子（拷贝一份用于描述子二次校验）★★★
    // ORB_SLAM3 MapPoint 存储 KeyFrame 指针来获取描述子，
    // 此处直接存储描述子拷贝，避免需要回查 KeyFrame。
    cv::Mat descriptor;
};

enum class PointStatus {
    CANDIDATE,   // 候选点（观测次数不足）
    CONFIRMED,   // 确认点（通过共视校验，视为静态可靠点）
    DISCARDED    // 已淘汰（观测不足/误差超标/长期未观测）
};

class MapPoint {
public:
    using Ptr = std::shared_ptr<MapPoint>;

    virtual ~MapPoint() = default;

    MapPoint() = default;
    explicit MapPoint(int id, const Eigen::Vector3d& world_pt);

    int id() const { return id_; }
    const Eigen::Vector3d& worldPoint() const { return world_pt_; }
    void updateWorldPoint(const Eigen::Vector3d& new_pt);
    PointStatus status() const { return status_; }
    const std::vector<MapPointObservation>& observations() const { return observations_; }
    double avgReprojError() const { return avg_reproj_error_; }
    int obsCount() const { return obs_count_; }
    int lastObsFrameId() const { return last_obs_frame_id_; }

    // ★★★ 描述子管理（ORB_SLAM3 风格）★★★
    // 获取最佳描述子（观测中汉明距离之和最小的那个）
    const cv::Mat& getDescriptor() const { return descriptor_; }
    void computeDescriptor();
    // 更新最佳描述子（新增观测时重新计算）
    void updateDescriptor(const cv::Mat& desc) {
        if (!desc.empty()) descriptor_ = desc.clone();
    }
    // 获取 ORB 描述子距离
    static int descriptorDistance(const cv::Mat& a, const cv::Mat& b);

    // 添加一个观测记录
    void addObservation(const MapPointObservation& obs);

    // ★★★ O(1) 帧观测存在性检查：替代 O(n) 线性扫描 ★★★
    // 使用 obs_by_frame_ hash map，返回该帧是否已观测过此点
    bool hasObservation(int frame_id) const {
        return obs_by_frame_.find(frame_id) != obs_by_frame_.end();
    }

    // ★★★ O(1) 获取指定帧的观测索引：返回 observations_ 中的下标 ★★★
    // 若该帧未观测过此点，返回 -1
    int getObservationIndex(int frame_id) const {
        auto it = obs_by_frame_.find(frame_id);
        return (it != obs_by_frame_.end()) ? it->second : -1;
    }

    // 每帧更新重投影误差，返回更新后的均值
    double updateReprojError(double pixel_error);

    // 共视校验：是否被 ≥ min_common_frames 个关键帧共同观测
    bool passCovisibilityCheck(int min_common_frames = 3) const;

    // 提升状态（candidate → confirmed）
    void promote();

    // 淘汰（candidate/discarded → discarded）
    void discard();

    // 标记本帧已观测
    void markObserved(int frame_id);

private:
    int id_ = -1;
    Eigen::Vector3d world_pt_;                  // 世界坐标系 3D 点
    std::vector<MapPointObservation> observations_; // 被哪些关键帧观测到
    int obs_count_ = 0;                          // 总观测次数
    int last_obs_frame_id_ = -1;                // 最近一次观测的帧 ID
    int confirmed_frame_id_ = -1;                // 被确认为静态点的帧 ID
    PointStatus status_ = PointStatus::CANDIDATE;
    // ★★★ 修复：初始值从 1e9 改为 0 ★★★
    // 原因：updateReprojError() 从未被调用，avg_reproj_error_ 永远是 1e9，
    // 导致条件 avgReprojError() > max_reproj_error 永远为真，
    // cullConfirmedPoints() 每次都删掉所有 obs>5 的确认点。
    // 改为 0 后：avgReprojError() > 15.0 永远为假，reproj 删除条件失效，
    // 只靠 stale（陈旧）条件删除，保证确认点能稳定积累。
    // 注意：后续应实现真正的重投影误差跟踪（需要 PnP/BA 时传入 pixel_error）
    double avg_reproj_error_ = 0.0;             // 平均重投影误差
    std::set<int> observing_frames_;            // 观测到该点的帧 ID 集合（去重）
    std::unordered_map<int, int> obs_by_frame_;  // frame_id → observations_ index (O(1) lookup)
    // ★★★ 最佳描述子（从所有观测中选择 ORB 距离之和最小的那个）★★★
    // 每次新增观测时更新，用于 searchLocalMap 的描述子二次校验
    cv::Mat descriptor_;
};


// ============= KeyFrame =============
// 关键帧，携带本帧所有特征点和 3D 点信息
// 优化：包含空间格网索引，加速几何重投影匹配中的最近邻查找
// 注意：此版本用于前端协方差追踪流水线。
// 后端地图管理使用 sfc_core::mapping::KeyFrame（在 mapping/key_frame.hpp 中定义）。

// 空间格网条目：x坐标排序的特征点列表，用于O(log n)最近邻查找
// 替代原始的线性O(n)扫描（n=数百个特征点）
struct KeyPointGridEntry {
    float u;          // 像素x坐标
    int kpt_idx;     // 对应的特征点索引
};

struct KeyFrame {
    using Ptr = std::shared_ptr<KeyFrame>;

    virtual ~KeyFrame() = default;

    int frame_id = -1;                // 全局帧 ID（唯一递增）
    double timestamp = 0.0;

    // 位姿（世界坐标系）
    Pose pose;

    // 该帧所有特征点（来自 ORB 提取）
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_keypoints;
    cv::Mat descriptors;

    // 该帧所有 3D 点（来自深度图）
    std::vector<Eigen::Vector3d> pts_3d;
    std::vector<bool> depth_valid;

    // 该帧观测到的地图点列表
    // kpt_idx -> map_point_id（-1 表示该特征点尚未关联地图点）
    std::map<int, int> kpt_to_mappoint;

    // ★★★ Bug Fix: 关键帧关联的候选点列表 ★★★
    // 用于Sim3回环修正时，同时修正候选点的世界坐标
    // 原bug: kpt_to_mappoint只包含确认点，候选点被遗漏
    std::map<int, int> kpt_to_candidate;

    // ==================== 空间格网索引（优化） ====================
    // 按u坐标排序的特征点数组，用于O(log n)二分查找替代O(n)线性扫描
    std::vector<KeyPointGridEntry> kpt_grid_sorted;

    // ★★★ 缓存旋转矩阵：避免每次重投影时重复 quaternion→Matrix3d 计算 ★★★
    mutable Eigen::Matrix3d R_cached;
    mutable bool R_valid_ = false;

    // 当 pose 被外部修改（如 BA/PGO）后，同步更新缓存的旋转矩阵
    void syncRotationCache() const {
        R_cached = pose.q.toRotationMatrix();
        R_valid_ = true;
    }

    // 构建格网索引（addKeyFrame后调用一次）
    void buildSpatialIndex();

    // 成员函数
    static Ptr create(int frame_id, double timestamp,
                      const std::vector<cv::KeyPoint>& kps,
                      const std::vector<cv::KeyPoint>& undist_kps,
                      const cv::Mat& desc,
                      const std::vector<Eigen::Vector3d>& pts3d,
                      const std::vector<bool>& depth_valid,
                      const Pose& pose = Pose());
};

// 多关键帧匹配结果的聚合索引（配合 addKeyFrame 的 matches_from_cache 参数使用）
// key: kf_data_cache_ 中历史关键帧的索引，value: 该历史关键帧对当前关键帧的所有描述子匹配
// 用途：addKeyFrame 据此正确区分哪些匹配来自哪个历史关键帧，并为候选点添加对应 frame_id 的观测
using MultiKFMatchesByCache = std::map<int, std::vector<cv::DMatch>>;

// 关键帧缓存条目（用于多关键帧匹配）
// 存储一个历史关键帧的特征数据，供后续关键帧做描述子匹配和三角化
struct KeyFrameCacheEntry {
    int global_id;                          // 跟踪系统全局唯一关键帧 ID（从 next_kf_global_id_ 递增）
    int frame_id;                           // ★★★ Bug Fix #6: CovisManager 中的 frame_id，用于 BA 同步 ★★★
    std::vector<cv::KeyPoint> keypoints;    // 该关键帧的特征点
    std::vector<cv::KeyPoint> undistorted_keypoints;
    cv::Mat descriptors;                    // ORB 描述子
    std::vector<Eigen::Vector3d> pts_3d;     // 深度反投 3D 点（始终保持在相机坐标系，不随 world_pose 变化）
    std::vector<Eigen::Vector3d> pts_3d_world; // pts_3d 转换到世界坐标系（由 Tracking::updateWorldPoints 填充）
    std::vector<bool> depth_valid;          // 深度是否有效
    Pose world_pose;                         // 该关键帧的世界坐标系位姿
    // ★★★ 优化: 空间格网索引（O(log n) 二分查找替代 O(n) 线性扫描）★★★
    std::vector<KeyPointGridEntry> kpt_grid_sorted;

    // ★★★ 关键帧关联的地图点 ID 集合（ORB-SLAM3 风格）★★★
    // 用于 searchLocalPoints 高效获取局部关键帧的地图点
    std::unordered_set<int> map_point_ids;
};

// 共视匹配对：两帧共同观测到的确认地图点及其特征索引
// 定义在 covis_core.hpp（而非 covis_manager.hpp 嵌套类），
// 使得 loopdetection 命名空间中的 LoopDetector 可以在 CovisManager 仅前向声明时就使用此类型。
struct CovisMatch {
    Eigen::Vector3d pt_world;       // 地图点的世界坐标（BA 精化后）
    int curr_kpt_idx;                // 该点在当前关键帧中的特征索引
    int matched_kpt_idx;            // 该点在匹配关键帧中的特征索引
    double reproj_error;            // 该点在 matched 帧的重投影误差
};

} // namespace frontend
} // namespace sfc_core

// std::hash 特化必须放在全局命名空间（std）中
// 当编译器查找 std::hash<sfc_core::frontend::CandidateBucket> 时，
// ADL 会找到这里的特化（因为 CandidateBucket 的定义在 sfc_core::frontend 中）
namespace std {
template<>
struct hash<sfc_core::frontend::CandidateBucket> {
    size_t operator()(const sfc_core::frontend::CandidateBucket& b) const noexcept {
        return ((size_t)b.u_bin * 73856093) ^ ((size_t)b.v_bin * 19349663) ^
               ((size_t)b.kpt_idx * 83492791) ^ ((size_t)b.frame_id * 19998317);
    }
};
}  // namespace std

#endif // SFC_CORE_FRONTEND_COVIS_CORE_HPP_
