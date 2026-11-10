// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_LOOPDETECTION_POSE_GRAPH_OPTIMIZER_HPP_
#define SFC_CORE_LOOPDETECTION_POSE_GRAPH_OPTIMIZER_HPP_

#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/frontend/covis_core.hpp"
#include <Eigen/Dense>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/sba/edge_se3_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace sfc_core {
namespace loopdetection {

struct PoseGraphOptimizerOptions {
    int max_iterations = 50;
    double robust_kernel_delta = 5.0;   // Huber 核阈值（m），与 pose_graph_optimizer.yaml 一致
    bool verbose = false;

    // ★★★ 信息矩阵参数（与 pose_graph_optimizer.yaml 保持一致）★★★
    // 里程计边信息权重
    double odom_information_scale = 1.0;       // 里程计边基准缩放因子
    double odom_min_information_scale = 0.05;  // 里程计边最小缩放因子下限
    // ★★★ Bug Fix D6: 将 trans/rot 相对权重暴露为配置参数 ★★★
    double odom_trans_rot_ratio = 16.0;  // 平移/旋转相对权重比（原硬编码 16.0/1.0）

    // ICP 回环边信息权重（有真实尺度）
    double icp_rot_info = 1.0;     // ICP 旋转信息权重
    double icp_trans_info = 5.0;  // ICP 平移信息权重（约束尺度）

    // EM 回环边信息权重（无真实尺度，只有方向约束）
    double em_rot_info = 1.0;         // EM 旋转信息权重
    double em_trans_info = 0.2;      // EM 平移信息权重（弱方向约束）
    double em_min_rot_scale = 0.1;   // 最小旋转缩放因子
    double em_min_trans_info = 0.05; // 最小平移信息

    // 回环边信息缩放
    double loop_information_scale = 2.0;        // 回环边基准缩放因子
    double loop_min_information_scale = 0.05;   // 回环边最小缩放因子下限

    // D4-a：回环边的信息量必须随"支持它的内点对数"增长。
    //
    // 一个回环被 n 对观测支持时，位姿估计的方差按 1/n 收缩、信息矩阵按 n 增长。
    // 原实现只有 inlier_ratio 的一次项（且被 em_min_rot_scale 顶住下限），
    // 于是"2 对全内点"与"200 对全内点"拿到一模一样的权重 —— 最直接的置信度信号被丢了。
    // 现在额外乘一个计数因子 inliers / loop_inlier_count_ref，并夹在上述上下限之间，
    // 避免单一回环权重爆炸或归零。
    double loop_inlier_count_ref = 30.0;         // 内点数达到此值时计数因子 = 1.0
    double loop_inlier_count_min_factor = 0.25;  // 计数因子下限
    double loop_inlier_count_max_factor = 4.0;   // 计数因子上限

    // D4-c：里程计边的信息缩放区间。
    // 原来写死 std::min(2.0, std::max(0.5, ...))，30% 内点与 90% 内点的边权重几乎一样；
    // 而回环边再强，也只能通过里程计边把修正传播出去 —— 区间压平等于削弱整张图的响应。
    double odom_info_scale_min = 0.2;
    double odom_info_scale_max = 4.0;

    // ★★★ PGO 规模限制 ★★★
    int max_loop_edges = 50;   // 最多保留 N 条回环边（超出时移除最早的），防止 g2o 优化卡死
    int max_odom_edges = 100;  // 最多保留 N 条里程计边，防止重建时内存暴涨
    // ★★★ 从 200 提高到 500：TUM fr1/desk2 数据集约 10-15 个关键帧，
    //   500 足够容纳全部历史顶点，实现接近全局 PGO 的效果 ★★★
    int max_vertices = 500;    // 最多保留 N 个位姿顶点（超出时移除最早的），需 >= covis window_size
};

// 位姿图优化器（PGO）
// 仅优化位姿顶点（不含地图点），使用 g2o SE3 图优化
// 支持增量添加顶点和边，每次回环后执行一次全局优化
class PoseGraphOptimizer {
public:
    explicit PoseGraphOptimizer(const PoseGraphOptimizerOptions& opts);

    ~PoseGraphOptimizer();

    // 添加/更新位姿顶点（若已存在则更新其估计值）
    void addOrUpdateVertex(int kf_id, const sfc_core::frontend::Pose& pose);

    // 添加里程计边（两连续关键帧之间的相对约束）
    // inliers / total_matches：用于计算信息矩阵（内点率越高 → 信息矩阵权重越大）
    void addOdometryEdge(int kf_id_from, int kf_id_to,
                         const Eigen::Matrix4d& relative_T,
                         double information_scale = 1.0,
                         int inliers = 0, int total_matches = 0);

    // 添加回环边（回环检测得到的相对位姿约束）
    // geometric_inliers / raw_matches：用于计算信息矩阵（几何校验内点越多 → 越可靠）
    // is_icp: true=ICP 3D-3D（有真实尺度），false=Essential Matrix（尺度归一化，旋转可靠但平移尺度不可靠）
    //
    // 返回 true 表示这条约束**真的进入了 g2o 图**；false 表示被拒绝或去重（顶点缺失、
    // 尺度/正交性校验失败、同一对关键帧已有边、addEdge 失败）。
    // L4：原来是 void，去重时静默 return，调用方只能看到"检测到回环"而看不到"回环没生效"。
    bool addLoopEdge(int kf_id_from, int kf_id_to,
                     const Eigen::Matrix4d& relative_T,
                     double information_scale = 1.0,
                     int geometric_inliers = 0, int raw_matches = 0,
                     bool is_icp = false);

    // 执行一次位姿图优化（SE3 PGO）
    // 返回迭代次数（≤ 0 表示失败）
    //
    // D4-b：出参回传本次优化的 chi2 与实际迭代数。
    // 曾经的故障形态是 "odom 边整类被 setLevel(1) 移出优化 → 图上只剩 1 条回环边 →
    // chi2 恒为 0 → 顶点几乎不动却报优化成功"，而调用方只看得到迭代次数，看不出这件事。
    // chi2 == 0 现在会在 optimize() 内部显式告警。
    int optimize(double* out_chi2 = nullptr, int* out_iterations = nullptr);

    // 获取优化后的位姿
    bool getOptimizedPose(int kf_id, sfc_core::frontend::Pose* out_pose) const;

    // ★★★ 获取所有顶点的优化后位姿（用于 Essential Graph 风格的全量修正）★★★
    // 返回 optimizer_ 中所有顶点的 kf_id -> Pose 映射。
    // 用于在回环检测后直接读取 g2o 图中的所有优化结果，替代距离加权插值。
    void getAllOptimizedPoses(std::unordered_map<int, sfc_core::frontend::Pose>* out_poses) const;

    // 获取所有已添加顶点的 ID
    std::vector<int> getVertexIds() const;

    // 图中是否已有回环边
    bool hasLoopEdges() const { return !loop_edge_ids_.empty(); }
    int numLoopEdges() const { return static_cast<int>(loop_edge_ids_.size()); }
    int numOdometryEdges() const { return static_cast<int>(odom_edge_ids_.size()); }

    // 恢复被驱逐的顶点（addLoopEdge 需要引用历史关键帧时调用）
    // 当回环边引用的历史关键帧已被 PGO 滑动窗口驱逐时，将其重新加入 g2o 图。
    g2o::VertexSE3Expmap* restoreEvictedVertexIfNeeded(int kf_id);

    // 清除所有数据（用于重置）
    void clear();

    // 重建优化器：清除后重新初始化 g2o 算法（用于崩溃恢复）
    // 调用 clear() 后使用此方法替代重建整个 PoseGraphOptimizer 对象
    void reset();

    // 设置最大顶点数上限（运行时调整滑动窗口大小）
    void setMaxVertices(int max_v) { opts_.max_vertices = max_v; }

private:
    // 从 Pose 构建 g2o SE3 矩阵
    static Eigen::Isometry3d poseToIsometry(const sfc_core::frontend::Pose& pose);

    // 从 SE3 矩阵构建 Pose
    static sfc_core::frontend::Pose isometryToPose(const Eigen::Isometry3d& iso);

    // 检查是否存在 kf_id=0 的顶点
    bool hasKfIdZero() const;

    // 构建/获取顶点
    g2o::VertexSE3Expmap* getVertex(int kf_id);
    const g2o::VertexSE3Expmap* getVertex(int kf_id) const;

    // ★★★ 新增：重建时用 kf_id 作为 g2o v_id（不重置序号）★★★
    // rebuildOptimizer 调用此方法重建顶点，保证 v_id 与 edge 缓存中的 from_id/to_id 一致
    // 注意：signature 为非 const Pose&（rebuildOptimizer 需要 const_cast 去除 const）
    // force_v_id: >=0 时强制使用该 v_id（rebuild 时用顺序索引），<0 时用 kf_id（addLoopEdge 时用 kf_id）
    void addVertexWithId(int kf_id, sfc_core::frontend::Pose& pose, int force_v_id = -1);

    bool optimizer_initialized_ = false;

    // ★★★ 标记优化器是否需要重建 ★★★
    // 避免在 optimize() 中无条件调用 rebuildOptimizer()（会销毁 g2o optimizer）。
    // 仅在需要时（如 addLoopEdge 添加了新顶点）才触发增量重建。
    bool optimizerNeedsRebuild_ = false;

    // ★★★ 重建整个 SparseOptimizer（用于 reset() 中的完全重建）★★★
    void rebuildOptimizer();

    PoseGraphOptimizerOptions opts_;

    // g2o 优化器（unique_ptr 避免暴露模板）
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;

    // 顶点 ID 集合（有序）
    std::vector<int> vertex_ids_;
    std::unordered_map<int, int> kf_id_to_vertex_index_;

    // 自维护顶点指针：避免通过 optimizer_->vertex(idx) 访问 g2o 内部
    std::vector<g2o::VertexSE3Expmap*> vertex_ptrs_;

    // ★★★ 保存顶点记录（kf_id + 位姿）用于 rebuildOptimizer() 重建顶点 ★★★
    // vertex_poses_ 改为 vertex_records_，每个元素存储 {kf_id, pose}，
    // 保证即使 kf_id 与数组下标不对应时也能正确重建。
    // 注意：这是滑动窗口，会在驱逐顶点时被清理（erase），因此不能用于恢复已驱逐顶点。
    std::vector<std::pair<int, sfc_core::frontend::Pose>> vertex_records_;

    // ★★★ 永久保存所有顶点位姿（永不清理）★★★
    // 当回环边引用的历史关键帧被 PGO 滑动窗口驱逐后，需要用此表恢复其顶点。
    // 与 vertex_records_ 不同：此表只增不减，即使顶点被驱逐也不删除记录。
    std::unordered_map<int, sfc_core::frontend::Pose> all_vertex_poses_;

    // 回环边 ID（用于后续清理）
    std::vector<std::pair<int, int>> loop_edge_ids_;
    std::vector<std::pair<int, int>> odom_edge_ids_;

    // ★★★ 修复：保存 loop edge 的测量值和信息矩阵，用于 rebuild 重建 ★★★
    // 原始 addLoopEdge() 会计算 edge->information()，但该值在 optimizer_->clear() 后丢失。
    // SnapshotLoop / SavedLoop 中仅保存 ID，重建 loop edge 时只能使用单位阵，严重影响 PGO 精度。
    // 正确做法：在 addLoopEdge() 时缓存 rel_T、is_icp、information_scale、geometric_inliers、raw_matches，
    //          rebuild 时用缓存值重建完整的 loop edge（带真实测量值、RobustKernel 和信息矩阵）。
    struct LoopEdgeData {
        int from_id, to_id;
        Eigen::Matrix4d rel_T;
        bool is_icp;
        double information_scale;
        int geometric_inliers;
        int raw_matches;
        Eigen::Matrix<double, 6, 6> information;  // 预计算的信息矩阵
    };
    std::vector<LoopEdgeData> loop_edge_data_;

    // ★★★ 修复：缓存里程计边的信息矩阵，用于 rebuild 重建 ★★★
    // 原始 addOdometryEdge() 会根据 inlier_ratio 计算带缩放的信息矩阵，
    // 但 optimize()/reset() 中重建里程计边时丢失了该信息，统一使用单位阵。
    struct OdomEdgeData {
        int from_id, to_id;
        Eigen::Matrix4d rel_T;
        double information_scale;
        int inliers;
        int total_matches;
        Eigen::Matrix<double, 6, 6> information;  // 预计算的信息矩阵
    };
    std::vector<OdomEdgeData> odom_edge_data_;

    // ★★★ 图结构互斥锁（g2o SparseOptimizer 非线程安全）★★★
    mutable std::recursive_mutex mtx_;

    // 已驱逐的关键帧 ID（用于 skipVertex 标记）
    std::unordered_set<int> evicted_kf_ids_;

    // ★★★ O(1) 边存在性缓存：替代 hasEdgeBetweenVertices 的 O(n) 遍历 ★★★
    // key: (uint64_t)from_id | ((uint64_t)to_id << 20)
    // odom edges: key with extra bit 0, loop edges: key with extra bit 1
    std::unordered_set<uint64_t> odom_edge_key_set_;
    std::unordered_set<uint64_t> loop_edge_key_set_;

    // 构建边 key 的辅助函数（from_id 和 to_id 顺序无关紧要，只要与 check 函数一致）
    static uint64_t makeOdomEdgeKey(int from_id, int to_id) {
        int a = std::min(from_id, to_id);
        int b = std::max(from_id, to_id);
        return (static_cast<uint64_t>(a) & 0xFFFFF) |
               ((static_cast<uint64_t>(b) & 0xFFFFFULL) << 20) |
               (0ULL << 40);  // bit 40 = 0 表示 odom edge
    }
    static uint64_t makeLoopEdgeKey(int from_id, int to_id) {
        int a = std::min(from_id, to_id);
        int b = std::max(from_id, to_id);
        return (static_cast<uint64_t>(a) & 0xFFFFF) |
               ((static_cast<uint64_t>(b) & 0xFFFFFULL) << 20) |
               (1ULL << 40);  // bit 40 = 1 表示 loop edge
    }

    // 检查 kf_id 是否已被驱逐
    bool isKfIdEvicted(int kf_id) const;
};

}  // namespace loopdetection
}  // namespace sfc_core

#endif  // SFC_CORE_LOOPDETECTION_POSE_GRAPH_OPTIMIZER_HPP_
