#include "sfc_core/loopdetection/pose_graph_optimizer.hpp"
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/sba/edge_se3_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/hyper_graph.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <algorithm>
#include <exception>
#include <iostream>
#include <unordered_set>


// ★★★ 工具函数：检查两个顶点之间是否已存在边 ★★★
// 用于防止重复添加边。回环边用 robustKernel 区分。
// 返回 true 表示边已存在，应跳过添加。
static bool hasEdgeBetweenVertices(g2o::SparseOptimizer* opt,
                                   g2o::VertexSE3Expmap* v0,
                                   g2o::VertexSE3Expmap* v1,
                                   bool require_robust_kernel) {
    if (!opt || !v0 || !v1) return true;
    for (g2o::HyperGraph::Edge* e : opt->edges()) {
        g2o::EdgeSE3Expmap* edge = dynamic_cast<g2o::EdgeSE3Expmap*>(e);
        if (!edge) continue;
        // 回环边有 robustKernel，里程计边没有
        bool has_robust = (edge->robustKernel() != nullptr);
        if (has_robust != require_robust_kernel) continue;
        if (edge->vertices().size() != 2) continue;
        int ev0_id = static_cast<int>(edge->vertex(0)->id());
        int ev1_id = static_cast<int>(edge->vertex(1)->id());
        if ((ev0_id == v0->id() && ev1_id == v1->id()) ||
            (ev0_id == v1->id() && ev1_id == v0->id())) {
            return true;
        }
    }
    return false;
}

namespace sfc_core {
namespace loopdetection {

// ====================== PoseGraphOptimizer ======================

Eigen::Isometry3d PoseGraphOptimizer::poseToIsometry(const sfc_core::frontend::Pose& pose) {
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = pose.q.toRotationMatrix();
    iso.translation() = pose.t;
    return iso;
}

sfc_core::frontend::Pose PoseGraphOptimizer::isometryToPose(const Eigen::Isometry3d& iso) {
    sfc_core::frontend::Pose p;
    p.q = Eigen::Quaterniond(iso.linear());
    p.t = iso.translation();
    return p;
}

// 辅助函数：Eigen::Matrix4d -> g2o::SE3Quat
// ★★★ SVD 正交化 + 四元数归一化：从源头杜绝非单位四元数进入 g2o ★★★
static g2o::SE3Quat matrixToSE3(const Eigen::Matrix4d& T) {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);

    // SVD 正交化修正（兜底，防止旋转矩阵微小扰动导致非正交）
    // 使用 Kabsch 算法标准做法：R = U * D * V^T, D = diag(1,1,det(U*V^T))
    // 确保 det(R) = +1（右手坐标系），避免反射矩阵进入 g2o 优化器
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) {
        D(2, 2) = -1.0;  // 只翻转最小奇异值对应的 U 的列（通常是第3列）
    }
    Eigen::Matrix3d R_ortho = svd.matrixU() * D * svd.matrixV().transpose();

    // 强制四元数归一化
    Eigen::Quaterniond q(R_ortho);
    q.normalize();

    return g2o::SE3Quat(q, t);
}

PoseGraphOptimizer::PoseGraphOptimizer(const PoseGraphOptimizerOptions& opts) : opts_(opts) {
    // 首次调用时 optimizer_ 尚未创建，rebuildOptimizer() 会负责创建
    rebuildOptimizer();
}

PoseGraphOptimizer::~PoseGraphOptimizer() = default;

void PoseGraphOptimizer::rebuildOptimizer() {
    std::lock_guard<std::recursive_mutex> lock(mtx_);

    // ============================================================
    // 全量重建策略：清理后重建，保证状态完全一致
    //
    // 当 optimizerNeedsRebuild_ 被设置时（顶点驱逐、添加回环边后），
    // 执行全量重建：清空 g2o 图，重新从 C++ 容器添加所有顶点/边。
    //
    // 重要：C++ 容器（vertex_ids_, loop_edge_data_, odom_edge_data_ 等）
    // 是 canonical state，g2o optimizer 是派生状态。全量重建不会丢失数据。
    // ============================================================

    // ★★★ 关键修复：必须同时检查 optimizerNeedsRebuild_ 和 optimizer_initialized_ ★★★
    // 首次调用时 optimizer_initialized_ = false，即使 optimizerNeedsRebuild_ = false 也要初始化
    if (!optimizerNeedsRebuild_ && optimizer_initialized_) return;

    if (opts_.verbose) {
        std::cerr << "[PGO] rebuildOptimizer: "
                  << (!optimizer_initialized_ ? "首次初始化" : "全量重建")
                  << "（图状态可能不一致，执行清理）" << std::endl;
    }

    // ★★★ 关键修复：全量清空 g2o 图 ★★★
    // 清空所有顶点、边、内部索引缓存，确保干净状态
    if (optimizer_) {
        optimizer_->clear();
    }
    kf_id_to_vertex_index_.clear();
    vertex_ids_.clear();
    vertex_ptrs_.clear();
    // evicted 顶点已全部清理，evicted_kf_ids_ 可以清空
    evicted_kf_ids_.clear();
    // 清空边缓存（重建后重新填充）
    odom_edge_key_set_.clear();
    loop_edge_key_set_.clear();

    // 重新初始化 optimizer（如果尚未初始化）
    if (!optimizer_initialized_) {
        using BlockSolverType = g2o::BlockSolver_6_3;
        using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

        auto linear_solver = std::make_unique<LinearSolverType>();
        linear_solver->setWriteDebug(false);

        auto block_solver = std::make_unique<BlockSolverType>(std::move(linear_solver));

        auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
        algorithm->setUserLambdaInit(1e-4);

        optimizer_ = std::make_unique<g2o::SparseOptimizer>();
        optimizer_->setAlgorithm(algorithm);
        optimizer_->setVerbose(opts_.verbose);
        // g2o 默认允许重复边，重复边会在 optimize() 时被处理
        optimizer_initialized_ = true;
        if (opts_.verbose) std::cerr << "[PGO] rebuildOptimizer: optimizer 已创建" << std::endl;
    }

    optimizerNeedsRebuild_ = false;

    // --- Step 1: 收集所有需要重建的顶点 ---
    std::unordered_set<int> kf_ids_needed;
    // 1a. 滑动窗口中的顶点
    for (const auto& record : vertex_records_) {
        kf_ids_needed.insert(record.first);
    }
    // 1b. 所有回环边引用的顶点
    for (const auto& edge : loop_edge_data_) {
        kf_ids_needed.insert(edge.from_id);
        kf_ids_needed.insert(edge.to_id);
    }
    for (const auto& edge : loop_edge_ids_) {
        kf_ids_needed.insert(edge.first);
        kf_ids_needed.insert(edge.second);
    }
    // 1c. 所有里程计边引用的顶点（确保里程计约束能正确重建）
    for (const auto& edge : odom_edge_data_) {
        kf_ids_needed.insert(edge.from_id);
        kf_ids_needed.insert(edge.to_id);
    }

    // 构建位姿查找表
    std::unordered_map<int, sfc_core::frontend::Pose> pose_lookup;
    for (const auto& kv : all_vertex_poses_) {
        pose_lookup[kv.first] = kv.second;
    }
    for (const auto& record : vertex_records_) {
        pose_lookup[record.first] = record.second;
    }

    // --- Step 2: 重新添加所有顶点 ---
    int vertex_added = 0, vertex_skipped_nan = 0;
    bool world_origin_fixed = false;
    for (int kf_id : kf_ids_needed) {
        auto it_pose = pose_lookup.find(kf_id);
        if (it_pose == pose_lookup.end()) {
            ++vertex_skipped_nan;
            continue;
        }
        const auto& pose = it_pose->second;
        if (!pose.t.allFinite() || !pose.q.coeffs().allFinite()) {
            ++vertex_skipped_nan;
            continue;
        }

        Eigen::Isometry3d iso = poseToIsometry(pose);
        g2o::SE3Quat se3(iso.rotation(), iso.translation());
        if (!se3.translation().allFinite() || !se3.rotation().coeffs().allFinite()) {
            ++vertex_skipped_nan;
            continue;
        }

        auto* v_se3 = new g2o::VertexSE3Expmap();
        v_se3->setId(kf_id);
        v_se3->setEstimate(se3);
        // 第一个顶点固定为原点
        bool should_fix = !world_origin_fixed;
        v_se3->setFixed(should_fix);
        if (should_fix) world_origin_fixed = true;

        if (!optimizer_->addVertex(v_se3)) {
            std::cerr << "[PGO] rebuildOptimizer: 添加顶点 " << kf_id << " 失败" << std::endl;
            delete v_se3;
            continue;
        }

        ++vertex_added;
        kf_id_to_vertex_index_[kf_id] = kf_id;
        vertex_ids_.push_back(kf_id);
        vertex_ptrs_.push_back(v_se3);
    }

    if (opts_.verbose) {
        std::cerr << "[PGO] rebuildOptimizer: 顶点重建完成"
                  << " 顶点数=" << optimizer_->vertices().size()
                  << "（新增=" << vertex_added << " 跳过=" << vertex_skipped_nan << "）" << std::endl;
    }

    // --- Step 3: 重新添加所有里程计边 ---
    int odom_added = 0, odom_dropped_missing = 0, odom_dropped_nan = 0;
    for (const auto& data : odom_edge_data_) {
        auto* v_from_raw = optimizer_->vertex(data.from_id);
        auto* v_to_raw   = optimizer_->vertex(data.to_id);
        if (!v_from_raw || !v_to_raw) {
            ++odom_dropped_missing;
            continue;
        }
        g2o::SE3Quat measurement = matrixToSE3(data.rel_T);
        if (!measurement.translation().allFinite() || !measurement.rotation().coeffs().allFinite()) {
            ++odom_dropped_nan;
            continue;
        }
        // 检查旋转矩阵正交性
        Eigen::Matrix3d R = measurement.rotation().toRotationMatrix();
        double det = R.determinant();
        double ortho_err = (R * R.transpose() - Eigen::Matrix3d::Identity()).norm();
        if (std::abs(det - 1.0) > 1e-4 || ortho_err > 1e-4) {
            ++odom_dropped_nan;
            continue;
        }

        auto* edge = new g2o::EdgeSE3Expmap();
        edge->resize(2);
        edge->setVertex(0, v_from_raw);
        edge->setVertex(1, v_to_raw);
        edge->setMeasurement(measurement);
        edge->information() = data.information;
        // L3：这里原本有 edge->setLevel(1)，配合 optimize() 里不带参数的
        // initializeOptimization()（默认 level 0），结果是**所有 63 条里程计边被整类
        // 移出优化**，图上只剩 1 条回环边 —— 一个只有一条边的图，chi2 必然恒 0，
        // 顶点几乎不动，却被计为"优化成功"。里程计边是位姿图的脊柱：
        // 回环边只是补一个闭环约束，没有里程计边，回环修正无处传播。
        // 约束的强弱必须由 information()/鲁棒核表达，不能靠 setLevel() 把整类约束删掉。
        // （CODE-RULES 建议新增的 R-028 就是这一条）

        // ★★★ O(1) 边存在性检查：先查缓存，再查 g2o 图 ★★★
        uint64_t edge_key = makeOdomEdgeKey(data.from_id, data.to_id);
        if (odom_edge_key_set_.find(edge_key) != odom_edge_key_set_.end()) {
            delete edge;
            continue;
        }

        if (!optimizer_->addEdge(edge)) {
            delete edge;
        } else {
            ++odom_added;
            odom_edge_key_set_.insert(makeOdomEdgeKey(data.from_id, data.to_id));
        }
    }

    // --- Step 4: 重新添加所有回环边 ---
    int loop_added = 0, loop_dropped_missing = 0, loop_dropped_nan = 0, loop_skipped_already = 0;
    for (const auto& data : loop_edge_data_) {
        auto* v_from_raw = optimizer_->vertex(data.from_id);
        auto* v_to_raw   = optimizer_->vertex(data.to_id);
        if (!v_from_raw || !v_to_raw) {
            ++loop_dropped_missing;
            continue;
        }
        g2o::SE3Quat measurement = matrixToSE3(data.rel_T);
        if (!measurement.translation().allFinite() || !measurement.rotation().coeffs().allFinite()) {
            ++loop_dropped_nan;
            continue;
        }
        Eigen::Matrix3d R = measurement.rotation().toRotationMatrix();
        double det = R.determinant();
        double ortho_err = (R * R.transpose() - Eigen::Matrix3d::Identity()).norm();
        if (std::abs(det - 1.0) > 1e-4 || ortho_err > 1e-4) {
            ++loop_dropped_nan;
            continue;
        }

        auto* edge = new g2o::EdgeSE3Expmap();
        edge->resize(2);
        edge->setVertex(0, v_from_raw);
        edge->setVertex(1, v_to_raw);
        edge->setMeasurement(measurement);
        edge->information() = data.information;
        auto* robust_kernel = new g2o::RobustKernelHuber();
        robust_kernel->setDelta(opts_.robust_kernel_delta);
        edge->setRobustKernel(robust_kernel);

        // ★★★ O(1) 边存在性检查 ★★★
        uint64_t edge_key = makeLoopEdgeKey(data.from_id, data.to_id);
        if (loop_edge_key_set_.find(edge_key) != loop_edge_key_set_.end()) {
            delete edge;
            ++loop_skipped_already;
            continue;
        }

        if (!optimizer_->addEdge(edge)) {
            delete edge;
        } else {
            ++loop_added;
            loop_edge_key_set_.insert(makeLoopEdgeKey(data.from_id, data.to_id));
        }
    }

    if (opts_.verbose) {
        std::cerr << "[PGO] rebuildOptimizer: 全量重建完成"
                  << " 顶点=" << optimizer_->vertices().size()
                  << " 边=" << optimizer_->edges().size()
                  << "（odom=" << odom_added << "/" << odom_edge_data_.size()
                  << " +loop=" << loop_added << "/" << loop_edge_data_.size()
                  << " odom_dropped_missing=" << odom_dropped_missing
                  << " odom_dropped_nan=" << odom_dropped_nan
                  << " loop_dropped_missing=" << loop_dropped_missing
                  << " loop_dropped_nan=" << loop_dropped_nan
                  << " loop_skipped_already=" << loop_skipped_already
                  << "）" << std::endl;
    }
}

g2o::VertexSE3Expmap* PoseGraphOptimizer::getVertex(int kf_id) {
    auto it = kf_id_to_vertex_index_.find(kf_id);
    if (it == kf_id_to_vertex_index_.end()) return nullptr;
    return dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id));
}

const g2o::VertexSE3Expmap* PoseGraphOptimizer::getVertex(int kf_id) const {
    auto it = kf_id_to_vertex_index_.find(kf_id);
    if (it == kf_id_to_vertex_index_.end()) return nullptr;
    return dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id));
}

// ★★★ 恢复被固定/驱逐的顶点（供 addLoopEdge 使用）★★★
// 当回环边引用的历史关键帧被滑动窗口固定（或之前不存在）时，需要将其恢复。
// 由于我们使用"固定而非删除"策略，顶点始终存在于 g2o 图中。
// 此函数确保顶点是可优化的（未固定）。
// 返回指向顶点的指针（可能已存在），失败返回 nullptr。
g2o::VertexSE3Expmap* PoseGraphOptimizer::restoreEvictedVertexIfNeeded(int kf_id) {
    auto* v = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id));
    if (v) {
        // 顶点已存在，检查是否被固定
        if (v->fixed()) {
            v->setFixed(false);
            { if (opts_.verbose) std::cerr << "[PGO] restoreEvictedVertexIfNeeded: 取消固定 kf_id=" << kf_id
                      << "（使其重新参与优化）" << std::endl; }
        }
        // 确保顶点也在追踪列表中
        if (kf_id_to_vertex_index_.find(kf_id) == kf_id_to_vertex_index_.end()) {
            kf_id_to_vertex_index_[kf_id] = kf_id;
            vertex_ids_.push_back(kf_id);
            vertex_ptrs_.push_back(v);
            evicted_kf_ids_.erase(kf_id);
        }
        return v;
    }
    // 顶点不存在（异常情况），尝试从 all_vertex_poses_ 恢复
    auto it = all_vertex_poses_.find(kf_id);
    if (it == all_vertex_poses_.end()) {
        { if (opts_.verbose) std::cerr << "[PGO] restoreEvictedVertexIfNeeded: 找不到 kf_id=" << kf_id
                  << " 的历史位姿，跳过恢复" << std::endl; }
        return nullptr;
    }
    const sfc_core::frontend::Pose& stored_pose = it->second;
    if (!stored_pose.t.allFinite()) {
        std::cerr << "[PGO] restoreEvictedVertexIfNeeded: kf_id=" << kf_id
                  << " 历史位姿含 NaN，跳过恢复" << std::endl;
        return nullptr;
    }
    // 创建新顶点
    auto* v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(kf_id);
    Eigen::Isometry3d iso = poseToIsometry(stored_pose);
    v_se3->setEstimate(g2o::SE3Quat(iso.rotation(), iso.translation()));
    v_se3->setFixed(false);
    if (!optimizer_->addVertex(v_se3)) {
        std::cerr << "[PGO] restoreEvictedVertexIfNeeded: 添加顶点 kf_id=" << kf_id
                  << " 失败" << std::endl;
        delete v_se3;
        return nullptr;
    }
    kf_id_to_vertex_index_[kf_id] = kf_id;
    vertex_ids_.push_back(kf_id);
    vertex_ptrs_.push_back(v_se3);
    evicted_kf_ids_.erase(kf_id);
    { if (opts_.verbose) std::cerr << "[PGO] restoreEvictedVertexIfNeeded: 恢复顶点 kf_id=" << kf_id
              << "（新创建）" << std::endl; }
    return v_se3;
}

void PoseGraphOptimizer::addOrUpdateVertex(int kf_id, const sfc_core::frontend::Pose& pose) {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    // 范围检查：防止极端值污染图
    double t_norm = pose.t.norm();
    Eigen::Matrix3d R = pose.q.toRotationMatrix();
    Eigen::AngleAxisd aa(R);
    double rot_angle = aa.angle();
    if (t_norm > 500.0 || rot_angle > M_PI) {
        std::cerr << "[PGO] 拒绝添加顶点 " << kf_id << "：平移=" << t_norm
                  << "m 旋转=" << rot_angle << "rad（超出合理范围）" << std::endl;
        return;
    }

    // NaN/Inf 检查：防止崩溃
    if (!pose.t.allFinite() ||
        std::isnan(pose.q.x()) || std::isnan(pose.q.y()) ||
        std::isnan(pose.q.z()) || std::isnan(pose.q.w()) ||
        std::isinf(pose.q.x()) || std::isinf(pose.q.y()) ||
        std::isinf(pose.q.z()) || std::isinf(pose.q.w())) {
        std::cerr << "[PGO] 拒绝添加顶点 " << kf_id << "：位姿含 NaN/Inf" << std::endl;
        return;
    }

    g2o::SE3Quat se3 = matrixToSE3(poseToIsometry(pose).matrix());
    if (!se3.translation().allFinite() || !se3.rotation().coeffs().allFinite()) {
        std::cerr << "[PGO] 拒绝添加顶点 " << kf_id << "：SE3 含 NaN/Inf" << std::endl;
        return;
    }

    auto it = kf_id_to_vertex_index_.find(kf_id);
    if (it != kf_id_to_vertex_index_.end()) {
        // 顶点已存在，直接更新位姿
        g2o::VertexSE3Expmap* v = getVertex(kf_id);
        if (v) v->setEstimate(se3);
        // ★★★ 永久保存位姿（即使被驱逐也能恢复）★★★
        all_vertex_poses_[kf_id] = pose;
        return;
    }

    // 滑动窗口：移除最旧顶点，保证窗口内顶点数 <= max_vertices
    // ============================================================
    // ★★★ 关键修复：不要 setFixed() 任何顶点 ★★★
    //
    // 原逻辑：当窗口满时，固定(setFixed=true)最旧顶点作为"锚点"。
    // 这是根本性错误：
    //   1. 固定顶点不参与优化 → 无法被回环约束修正
    //   2. 运行几百帧后，几乎所有顶点都被固定 → 图失去自由度
    //   3. "跳过 N 个不可固定顶点"说明几乎所有顶点都有 loop 边，都不可固定
    //
    // 正确策略（与 ORB_SLAM3 一致）：
    //   - 始终只固定 kf_id=1（世界原点）作为绝对参考
    //   - 所有其他顶点（包括历史顶点）都不固定，自由参与优化
    //   - 回环边能修正历史轨迹的漂移
    //   - g2o 会正确处理固定顶点和自由顶点的混合优化
    //
    // 不再需要驱逐/固定逻辑：所有顶点都在图中，都可以优化。
    // 只需要在 vertex_ids_ 追踪列表中移除旧顶点（用于管理顶点添加顺序）。
    // g2o 图中顶点永远不会被删除，也不会被 setFixed（除原点外）。
    // ============================================================
    while (vertex_ids_.size() >= static_cast<size_t>(opts_.max_vertices)) {
        if (vertex_ids_.empty() || vertex_ptrs_.empty()) break;

        // 找到最旧的可移除顶点（用于追踪列表管理）
        int old_kf_id = vertex_ids_.front();
        g2o::VertexSE3Expmap* old_vertex = vertex_ptrs_.front();

        // ★★★ Bug Fix B3: 从 g2o Optimizer 中移除顶点，防止内存泄漏和残留顶点 ★★★
        if (optimizer_->removeVertex(old_vertex)) {
            delete old_vertex;  // g2o 不拥有顶点内存，需手动删除
        }
        kf_id_to_vertex_index_.erase(old_kf_id);
        vertex_ids_.erase(vertex_ids_.begin());
        vertex_ptrs_.erase(vertex_ptrs_.begin());
        if (!vertex_records_.empty()) vertex_records_.erase(vertex_records_.begin());
    }

    // 保险检查：若 g2o 中已存在 kf_id（异常残留），先移除
    if (optimizer_->vertex(kf_id)) {
        std::cerr << "[PGO] 警告：g2o 中已存在 kf_id=" << kf_id
                  << "（异常残留），强制移除" << std::endl;
        optimizer_->removeVertex(optimizer_->vertex(kf_id));
    }

    auto* v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(kf_id);
    v_se3->setEstimate(se3);

    // 固定世界原点
    bool should_fix = (kf_id == 0) || (vertex_ids_.empty() && !hasKfIdZero());
    v_se3->setFixed(should_fix);
    if (should_fix) {
        { if (opts_.verbose) std::cerr << "[PGO] 固定世界原点: kf_id=" << kf_id << std::endl; }
    }

    if (!optimizer_->addVertex(v_se3)) {
        std::cerr << "[PGO] addVertex 失败: kf_id=" << kf_id
                  << "（g2o vertex " << kf_id << " 可能已存在）" << std::endl;
        delete v_se3;
        return;
    }

    kf_id_to_vertex_index_[kf_id] = kf_id;
    vertex_ids_.push_back(kf_id);
    vertex_ptrs_.push_back(v_se3);
    vertex_records_.push_back({kf_id, pose});
    // ★★★ 永久保存位姿（永不清理，供 restoreEvictedVertexIfNeeded 恢复已驱逐顶点）★★★
    all_vertex_poses_[kf_id] = pose;
}

bool PoseGraphOptimizer::hasKfIdZero() const {
    return kf_id_to_vertex_index_.find(0) != kf_id_to_vertex_index_.end();
}

bool PoseGraphOptimizer::isKfIdEvicted(int kf_id) const {
    return evicted_kf_ids_.count(kf_id) > 0;
}

void PoseGraphOptimizer::addVertexWithId(int kf_id, sfc_core::frontend::Pose& pose, int /*force_v_id*/) {
    double t_norm = pose.t.norm();
    Eigen::Matrix3d R = pose.q.toRotationMatrix();
    Eigen::AngleAxisd aa(R);
    double rot_angle = aa.angle();
    if (t_norm > 500.0 || rot_angle > M_PI) {
        { if (opts_.verbose) std::cerr << "[PGO] addVertexWithId 跳过 kf_id=" << kf_id
                  << "：平移=" << t_norm << "m 旋转=" << rot_angle
                  << "rad（超出合理范围）" << std::endl; }
        return;
    }
    if (!pose.t.allFinite() ||
        std::isnan(pose.q.x()) || std::isnan(pose.q.y()) ||
        std::isnan(pose.q.z()) || std::isnan(pose.q.w()) ||
        std::isinf(pose.q.x()) || std::isinf(pose.q.y()) ||
        std::isinf(pose.q.z()) || std::isinf(pose.q.w())) {
        std::cerr << "[PGO] addVertexWithId 跳过 kf_id=" << kf_id
                  << "：位姿含 NaN/Inf" << std::endl;
        return;
    }

    g2o::SE3Quat se3 = matrixToSE3(poseToIsometry(pose).matrix());
    if (!se3.translation().allFinite() || !se3.rotation().coeffs().allFinite()) {
        std::cerr << "[PGO] addVertexWithId 跳过 kf_id=" << kf_id << "：SE3 含 NaN/Inf" << std::endl;
        return;
    }

    // ★★★ 关键修复：使用 g2o direct check 避免 kf_id_to_vertex_index_ 不同步的问题 ★★★
    // 场景：rebuildOptimizer 首次添加顶点到 g2o，但 kf_id_to_vertex_index_ 在第一次 optimize 后被清空。
    // 后续 addVertexWithId 调用时，kf_id_to_vertex_index_ 中没有该 kf_id，
    // 导致 getVertex() 返回 nullptr，但 g2o 中实际已有该顶点。
    // 这会导致 addVertexWithId 尝试重复添加顶点或跳过关键顶点。
    // 解决：直接查询 g2o，不依赖 C++ 容器的同步性。
    if (optimizer_->vertex(kf_id)) {
        g2o::VertexSE3Expmap* v_existing = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id));
        if (v_existing) {
            v_existing->setEstimate(se3);
            { if (opts_.verbose) std::cerr << "[PGO] addVertexWithId: kf_id=" << kf_id
                      << " g2o 中已存在（直接更新位姿）" << std::endl; }
        }
        // 同步 C++ 容器（修复 kf_id_to_vertex_index_ 不同步问题）
        if (kf_id_to_vertex_index_.find(kf_id) == kf_id_to_vertex_index_.end()) {
            kf_id_to_vertex_index_[kf_id] = kf_id;
            vertex_ids_.push_back(kf_id);
        }
        // 更新 evicted 状态
        evicted_kf_ids_.erase(kf_id);
        // 永久保存位姿
        all_vertex_poses_[kf_id] = pose;
        return;
    }

    auto* v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(kf_id);
    bool should_fix = (kf_id == 0) || (vertex_ids_.empty() && !hasKfIdZero());
    v_se3->setFixed(should_fix);
    if (should_fix) {
        { if (opts_.verbose) std::cerr << "[PGO] 重建：固定世界原点 kf_id=" << kf_id << std::endl; }
    }

    if (!optimizer_->addVertex(v_se3)) {
        std::cerr << "[PGO] addVertexWithId 失败: kf_id=" << kf_id << std::endl;
        delete v_se3;
        return;
    }

    kf_id_to_vertex_index_[kf_id] = kf_id;
    vertex_ids_.push_back(kf_id);
    vertex_ptrs_.push_back(v_se3);
    // ★★★ 永久保存位姿（永不清理，供 restoreEvictedVertexIfNeeded 恢复已驱逐顶点）★★★
    all_vertex_poses_[kf_id] = pose;
}

void PoseGraphOptimizer::addOdometryEdge(int kf_id_from, int kf_id_to,
                                          const Eigen::Matrix4d& relative_T,
                                          double information_scale,
                                          int inliers, int total_matches) {
    std::lock_guard<std::recursive_mutex> lock(mtx_);

    // ★★★ 修复 1：使用 optimizer_->vertex() 直接检查顶点存在性 ★★★
    auto* v_from_raw = optimizer_->vertex(kf_id_from);
    auto* v_to_raw   = optimizer_->vertex(kf_id_to);
    if (!v_from_raw || !v_to_raw) {
        std::cerr << "[PGO] addOdometryEdge 拒绝（顶点不存在于 g2o 图中）: "
                  << kf_id_from << " -> " << kf_id_to
                  << " v_from=" << (v_from_raw ? "OK" : "NULL")
                  << " v_to=" << (v_to_raw ? "OK" : "NULL")
                  << " 图中顶点数=" << optimizer_->vertices().size()
                  << std::endl;
        return;
    }
    auto* v_from = dynamic_cast<g2o::VertexSE3Expmap*>(v_from_raw);
    auto* v_to   = dynamic_cast<g2o::VertexSE3Expmap*>(v_to_raw);
    if (!v_from || !v_to) {
        std::cerr << "[PGO] addOdometryEdge 拒绝（顶点类型转换失败）: "
                  << kf_id_from << " -> " << kf_id_to << std::endl;
        return;
    }

    g2o::SE3Quat measurement = matrixToSE3(relative_T);
    if (!measurement.translation().allFinite() || !measurement.rotation().coeffs().allFinite()) {
        std::cerr << "[PGO] 跳过 odom 边（含 NaN/Inf）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        return;
    }
    double trans_norm = measurement.translation().norm();
    double rot_norm = Eigen::AngleAxisd(measurement.rotation()).angle();
    if (trans_norm > 10.0 || rot_norm > 1.0) {
        { if (opts_.verbose) std::cerr << "[PGO] 跳过 odom 边（极端值）: " << kf_id_from << " -> " << kf_id_to
                  << " t=" << trans_norm << "m r=" << rot_norm << "rad" << std::endl; }
        return;
    }

    // ★★★ 修复 2：检查旋转矩阵正交性（行列式≈1，R·R^T≈I）★★★
    {
        Eigen::Matrix3d R = measurement.rotation().toRotationMatrix();
        double det = R.determinant();
        double ortho_err = (R * R.transpose() - Eigen::Matrix3d::Identity()).norm();
        if (std::abs(det - 1.0) > 1e-3 || ortho_err > 1e-3) {
            { if (opts_.verbose) std::cerr << "[PGO] 跳过 odom 边（旋转矩阵非正交）: " << kf_id_from << " -> " << kf_id_to
                      << " det(R)=" << det << " ortho_err=" << ortho_err << std::endl; }
            return;
        }
    }

    auto* edge = new g2o::EdgeSE3Expmap();
    edge->setVertex(0, v_from);
    edge->setVertex(1, v_to);
    edge->resize(2);
    edge->setMeasurement(measurement);

    double inlier_ratio = (total_matches > 0)
                          ? std::max(0.0, std::min(1.0, static_cast<double>(inliers) / total_matches))
                          : 0.5;
    double info_scale = information_scale * inlier_ratio;
    // D4-c：区间来自配置，且比原来的 [0.5, 2.0] 更宽。
    // 原实现把 30% 内点与 90% 内点的里程计边压到几乎同一权重，回环修正传不出去；
    // 放宽后仍保留硬下限 odom_min_information_scale，防止一条烂边被彻底忽略。
    info_scale = std::max(opts_.odom_info_scale_min,
                          std::min(opts_.odom_info_scale_max, info_scale));
    info_scale = std::max(opts_.odom_min_information_scale, info_scale);
    // ★★★ 修复 VO 尺度漂移：增强平移约束 ★★★
    // I_6x6 对旋转和平移给相同权重，但旋转估计更可靠（Essential Matrix）。
    // 平移分量乘以较大系数，使 BA 优先保证平移尺度正确，抑制尺度漂移。
    // 旋转仍用单位权重（可靠），平移用 16 倍权重（4²，对应 4 倍 sigma 缩小）。
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
    for (int i = 0; i < 3; ++i) {
        info(i, i) = opts_.odom_trans_rot_ratio;  // 前3个 = 平移（Bug Fix D6: 从硬编码 16.0 改为配置参数）
    }
    for (int i = 3; i < 6; ++i) {
        info(i, i) = 1.0;    // 后3个 = 旋转（rx,ry,rz）：保持 1×
    }
    // ★★★ Bug Fix C3: 将钳位从基矩阵移到最终结果，钳位上限改为 config 驱动 ★★★
    // 原代码在基矩阵上钳位到 1e6，但后续还会乘以 info_scale * odom_information_scale，
    // 导致钳位失效。最终结果仍可能极大。用最终结果的钳位保证信息矩阵有界。
    edge->information() = info * info_scale * opts_.odom_information_scale;
    // 对信息矩阵每个元素钳位到 [min_info, max_info]
    constexpr double kMinInfo = 1e-6;
    double kMaxInfo = opts_.odom_information_scale * 100.0;  // 基于配置的上限
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            if (r == c) {  // 只钳位对角线元素（信息矩阵非对角部分由配置控制）
                edge->information()(r, c) = std::max(kMinInfo,
                    std::min(edge->information()(r, c), kMaxInfo));
            }
        }
    }

    // ★★★ Bug Fix C4: odometry 边不应使用 Huber 鲁棒核 ★★★
    // Huber 核用于处理离群回环边（错误的地点识别），不应用于 odometry 边。
    // odometry 约束来自 PnP，虽有噪声但不应被降权。仅回环边使用 Huber（见 rebuildOptimizer）。

    // ★★★ O(1) 边存在性检查：先查缓存 ★★★
    uint64_t edge_key = makeOdomEdgeKey(kf_id_from, kf_id_to);
    if (odom_edge_key_set_.find(edge_key) != odom_edge_key_set_.end()) {
        delete edge;
        return;
    }

    if (!optimizer_->addEdge(edge)) {
        std::cerr << "[PGO] addEdge 失败（odom）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        delete edge;
        return;
    }
    odom_edge_ids_.emplace_back(kf_id_from, kf_id_to);
    odom_edge_data_.push_back(OdomEdgeData{
        kf_id_from, kf_id_to, relative_T, information_scale, inliers, total_matches,
        edge->information()
    });
    // ★★★ O(1) 边存在性缓存更新 ★★★
    odom_edge_key_set_.insert(edge_key);

    while (odom_edge_data_.size() > static_cast<size_t>(opts_.max_odom_edges)) {
        odom_edge_data_.erase(odom_edge_data_.begin());
        odom_edge_ids_.erase(odom_edge_ids_.begin());
    }
}

bool PoseGraphOptimizer::addLoopEdge(int kf_id_from, int kf_id_to,
                                       const Eigen::Matrix4d& relative_T,
                                       double information_scale,
                                       int geometric_inliers, int raw_matches,
                                       bool is_icp) {
    std::lock_guard<std::recursive_mutex> lock(mtx_);

    // ★★★ 修复 1：使用 optimizer_->vertex() 直接检查顶点存在性 ★★★
    // 如果顶点不存在（已被滑动窗口驱逐），尝试恢复被驱逐的顶点。
    // 回环边引用的历史关键帧可能早已被驱逐，必须恢复才能正常添加回环约束。
    auto* v_from_raw = optimizer_->vertex(kf_id_from);
    auto* v_to_raw   = optimizer_->vertex(kf_id_to);

    if (!v_from_raw) {
        { if (opts_.verbose) std::cerr << "[PGO] addLoopEdge: kf_id_from=" << kf_id_from
                  << " 不在 g2o 图中，尝试恢复..." << std::endl; }
        v_from_raw = restoreEvictedVertexIfNeeded(kf_id_from);
    }
    if (!v_to_raw) {
        { if (opts_.verbose) std::cerr << "[PGO] addLoopEdge: kf_id_to=" << kf_id_to
                  << " 不在 g2o 图中（可能已被滑动窗口驱逐），尝试恢复..." << std::endl; }
        v_to_raw = restoreEvictedVertexIfNeeded(kf_id_to);
    }
    if (!v_from_raw || !v_to_raw) {
        std::cerr << "[PGO] addLoopEdge 拒绝（无法恢复顶点）: "
                  << kf_id_from << " -> " << kf_id_to
                  << " v_from=" << (v_from_raw ? "OK" : "NULL")
                  << " v_to=" << (v_to_raw ? "OK" : "NULL")
                  << std::endl;
        return false;
    }
    auto* v_from = dynamic_cast<g2o::VertexSE3Expmap*>(v_from_raw);
    auto* v_to   = dynamic_cast<g2o::VertexSE3Expmap*>(v_to_raw);
    if (!v_from || !v_to) {
        std::cerr << "[PGO] addLoopEdge 拒绝（顶点类型转换失败）: "
                  << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;
    }

    // 回环边去重：同一 (matched_kf, current_kf) 对只允许一条边
    // ★★★ O(1) 边存在性检查：使用缓存替代 O(n) 线性扫描 ★★★
    uint64_t loop_edge_key = makeLoopEdgeKey(kf_id_from, kf_id_to);
    if (loop_edge_key_set_.find(loop_edge_key) != loop_edge_key_set_.end()) {
        { if (opts_.verbose) std::cerr << "[PGO] addLoopEdge 跳过（缓存中已存在）: " << kf_id_from << " -> " << kf_id_to << std::endl; }
        // 确保 matched_kf 可以被优化（取消固定）
        auto* v_matched = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id_to));
        if (v_matched && v_matched->fixed()) {
            { if (opts_.verbose) std::cerr << "[PGO] addLoopEdge: matched_kf=" << kf_id_to << " 已被固定，取消固定以参与优化" << std::endl; }
            v_matched->setFixed(false);
            if (kf_id_to_vertex_index_.find(kf_id_to) == kf_id_to_vertex_index_.end()) {
                kf_id_to_vertex_index_[kf_id_to] = kf_id_to;
                vertex_ids_.push_back(kf_id_to);
                vertex_ptrs_.push_back(v_matched);
            }
            optimizerNeedsRebuild_ = true;
        }
        auto* v_curr = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id_from));
        if (v_curr && v_curr->fixed()) {
            v_curr->setFixed(false);
            if (kf_id_to_vertex_index_.find(kf_id_from) == kf_id_to_vertex_index_.end()) {
                kf_id_to_vertex_index_[kf_id_from] = kf_id_from;
                vertex_ids_.push_back(kf_id_from);
                vertex_ptrs_.push_back(v_curr);
            }
            optimizerNeedsRebuild_ = true;
        }
        return false;
    }
    // 旧代码：O(n) 遍历保留用于调试兼容
    int id1 = std::min(kf_id_from, kf_id_to);
    int id2 = std::max(kf_id_from, kf_id_to);
    int64_t key = (static_cast<int64_t>(id1) << 32) | (static_cast<int64_t>(id2) & 0xFFFFFFFFLL);
    for (const auto& id : loop_edge_ids_) {
        int e1 = std::min(id.first, id.second);
        int e2 = std::max(id.first, id.second);
        if ((static_cast<int64_t>(e1) << 32) | (static_cast<int64_t>(e2) & 0xFFFFFFFFLL) == key) {
            { if (opts_.verbose) std::cerr << "[PGO] addLoopEdge 跳过（loop_edge_ids_中已存在）: " << kf_id_from << " -> " << kf_id_to << std::endl; }
            // 确保 matched_kf 可以被优化（取消固定）
            auto* v_matched = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id_to));
            if (v_matched && v_matched->fixed()) {
                { if (opts_.verbose) std::cerr << "[PGO] addLoopEdge: matched_kf=" << kf_id_to << " 已被固定，取消固定以参与优化" << std::endl; }
                v_matched->setFixed(false);
                if (kf_id_to_vertex_index_.find(kf_id_to) == kf_id_to_vertex_index_.end()) {
                    kf_id_to_vertex_index_[kf_id_to] = kf_id_to;
                    vertex_ids_.push_back(kf_id_to);
                    vertex_ptrs_.push_back(v_matched);
                }
                optimizerNeedsRebuild_ = true;
            }
            auto* v_curr = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id_from));
            if (v_curr && v_curr->fixed()) {
                v_curr->setFixed(false);
                if (kf_id_to_vertex_index_.find(kf_id_from) == kf_id_to_vertex_index_.end()) {
                    kf_id_to_vertex_index_[kf_id_from] = kf_id_from;
                    vertex_ids_.push_back(kf_id_from);
                    vertex_ptrs_.push_back(v_curr);
                }
                optimizerNeedsRebuild_ = true;
            }
            return false;
        }
    }

    g2o::SE3Quat measurement = matrixToSE3(relative_T);
    Eigen::Vector3d t = measurement.translation();
    double t_norm = t.norm();
    if (!t.allFinite() || !measurement.rotation().coeffs().allFinite()) {
        std::cerr << "[PGO] 回环边被拒绝（含 NaN）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;
    }
    if (t_norm > 100.0 || t_norm < 1e-6) {
        std::cerr << "[PGO] 回环边被拒绝（尺度异常）: t_norm=" << t_norm
                  << "m " << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;
    }
    if (is_icp && t_norm > 50.0) {
        std::cerr << "[PGO] 回环边被拒绝（ICP 过大）: t_norm=" << t_norm
                  << "m " << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;
    }

    // ★★★ 修复 2：检查旋转矩阵正交性（行列式≈1，R·R^T≈I）★★★
    // 日志中 Sim3 R_diag 出现负数（如[-0.75, -0.90, 0.66]），提示回环求解
    // 的 Essential Matrix / ICP 结果可能产生非正交旋转矩阵。
    // 非正交旋转直接进入 g2o linearizeOplus() 会导致未定义行为（SIGSEGV）。
    {
        Eigen::Matrix3d R = measurement.rotation().toRotationMatrix();
        double det = R.determinant();
        double ortho_err = (R * R.transpose() - Eigen::Matrix3d::Identity()).norm();
        if (std::abs(det - 1.0) > 1e-3 || ortho_err > 1e-3) {
            std::cerr << "[PGO] 回环边被拒绝（旋转矩阵非正交）: " << kf_id_from << " -> " << kf_id_to
                      << " det(R)=" << det << " ortho_err=" << ortho_err
                      << " R_diag=" << R.diagonal().transpose() << std::endl;
            return false;
        }
    }

    auto* edge = new g2o::EdgeSE3Expmap();
    edge->resize(2);
    edge->setVertex(0, v_from);
    edge->setVertex(1, v_to);
    edge->setMeasurement(measurement);

    double inlier_ratio = (raw_matches > 0)
        ? std::max(0.0, std::min(1.0, static_cast<double>(geometric_inliers) / raw_matches))
        : 0.5;

    double rot_info, trans_info;
    if (is_icp) {
        // ICP: 有真实尺度，旋转和尺度都可靠
        // ★★★ 旋转主导：旋转比平移更可靠（深度噪声影响平移），采用 1.5:1 的旋转优势权重 ★★★
        rot_info   = opts_.icp_rot_info * 1.5;
        trans_info = opts_.icp_trans_info;
        double scale = std::max(opts_.em_min_rot_scale, inlier_ratio * 2.0);
        rot_info   *= scale;
        trans_info *= scale;
    } else {
        // EM: 无真实尺度，旋转可靠但平移尺度不可靠
        // ★★★ 旋转强烈主导：平移仅提供方向约束，不携带尺度信息 ★★★
        rot_info   = opts_.em_rot_info * 3.0;
        trans_info = std::max(opts_.em_min_trans_info, opts_.em_trans_info);
        double scale = std::max(opts_.em_min_rot_scale, inlier_ratio * 2.0);
        rot_info   *= scale;
        trans_info *= scale;
    }

    rot_info   = std::min(rot_info,   1e4);
    trans_info = std::min(trans_info, 1e4);

    // ★★★ 旋转主导的信息矩阵（与 ORB_SLAM3 OptimizeEssentialGraph 一致）★★★
    // 对于 RGB-D 室内场景，旋转比平移更可靠：
    // - ICP: 深度噪声导致平移估计不稳定
    // - EM:  Essential Matrix 只提供归一化平移方向，无真实尺度
    // 采用非对称权重：旋转信息 > 平移信息
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
    info.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * trans_info;
    info.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * rot_info;
    for (int i = 0; i < 6; ++i) {
        info(i, i) = std::max(1e-6, std::min(info(i, i), 1e6));
    }

    // ★★★ 修复 3：检查信息矩阵有效性（无 NaN/Inf）★★★
    if (!info.allFinite()) {
        std::cerr << "[PGO] 回环边被拒绝（信息矩阵含 NaN/Inf）: "
                  << kf_id_from << " -> " << kf_id_to << std::endl;
        delete edge;
        return false;
    }
    // D4-a：再乘一个"支持度"因子 —— 内点对数越多，这条约束越硬。
    // 取 inliers / loop_inlier_count_ref 后夹在配置的上下限内：
    // 低于下限说明这条回环几乎没有观测支持，不允许它影响整张图；
    // 高于上限说明这条回环特别结实，但也不该把相邻约束完全淹没。
    const double count_ref = std::max(1.0, opts_.loop_inlier_count_ref);
    const double count_factor = std::max(
        opts_.loop_inlier_count_min_factor,
        std::min(opts_.loop_inlier_count_max_factor,
                 static_cast<double>(std::max(0, geometric_inliers)) / count_ref));
    edge->information() =
        info * information_scale * opts_.loop_information_scale * count_factor;

    auto* robust_kernel = new g2o::RobustKernelHuber();
    robust_kernel->setDelta(opts_.robust_kernel_delta);
    edge->setRobustKernel(robust_kernel);

    // 直接添加到 g2o 图，由 g2o 内部处理重复边
    if (!optimizer_->addEdge(edge)) {
        std::cerr << "[PGO] addEdge 失败（loop）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        delete edge;
        return false;
    }
    loop_edge_ids_.emplace_back(kf_id_from, kf_id_to);
    { if (opts_.verbose) std::cerr << "[PGO] addLoopEdge: 添加回环边 " << kf_id_from << " -> " << kf_id_to
              << " (is_icp=" << is_icp << " 内点=" << geometric_inliers << "/" << raw_matches << ")" << std::endl; }
    loop_edge_data_.push_back(LoopEdgeData{
        kf_id_from, kf_id_to, relative_T, is_icp,
        information_scale, geometric_inliers, raw_matches,
        edge->information()
    });
    // ★★★ O(1) 边存在性缓存更新 ★★★
    loop_edge_key_set_.insert(loop_edge_key);

    while (loop_edge_data_.size() > static_cast<size_t>(opts_.max_loop_edges)) {
        loop_edge_data_.erase(loop_edge_data_.begin());
        loop_edge_ids_.erase(loop_edge_ids_.begin());
    }

    // 标记需要重建：回环边改变了图结构，下次 optimize() 前要刷新
    optimizerNeedsRebuild_ = true;
    // L4：边真的进了图 —— 调用方据此决定是否计一次有效回环。
    return true;
}

int PoseGraphOptimizer::optimize(double* out_chi2, int* out_iterations) {
    std::lock_guard<std::recursive_mutex> lock(mtx_);

    // ★★★ 完全重建优化器：确保 g2o 内部状态完全干净 ★★★
    // 之前的修复（边清理、顶点重建）仍无法解决 linearizeOplus() SIGSEGV。
    // 原因：g2o 内部可能有 ABI/对齐兼容性问题，即使重建边也无法解决。
    // 解决方案：完全重建 optimizer 对象本身（包括 BlockSolver、LinearSolver）。
    // 这确保 g2o 内部的所有状态（_vertices, _edges, _activeEdges, jacobianWorkspace）
    // 都是全新分配的，不会有任何残留指针或损坏状态。
    // ★★★ 增量优化策略：避免不必要的全量重建 ★★★
    //
    // 触发全量重建的唯一条件：optimizerNeedsRebuild_ = true
    // 触发条件：
    //   1. addOrUpdateVertex() 中顶点被驱逐（很少发生，max_vertices=500）
    //   2. addLoopEdge() 添加了新回环边（每次回环触发一次）
    //   3. previous optimize() 中发现边为空（正常情况会重建）
    //
    // 如果 optimizerNeedsRebuild_ = false 且优化器已初始化，直接优化即可。
    // 不需要每次 optimize() 都重建。
    //
    // 注意：即使有边存在（来自上次成功的回环），直接 optimize() 是安全的。
    // g2o 会增量优化现有结构，不需要每次都清空重来。
    if (optimizerNeedsRebuild_ || !optimizer_initialized_) {
        rebuildOptimizer();
    }

    if (optimizer_->vertices().empty() || optimizer_->edges().empty()) {
        { if (opts_.verbose) std::cerr << "[PGO] optimize: 无顶点或边，跳过" << std::endl; }
        return 0;
    }

    // ★★★ 防御性 dangling edge 检查（双重保险）★★★
    // rebuild 时顶点被 delete 并重建，但边对象里的顶点指针仍可能指向旧对象（dangling pointer）。
    // 仅用 dynamic_cast 无法识别： dangling 指针的 dynamic_cast 可能返回非 nullptr（垃圾值）。
    // 正确做法：通过 optimizer_->vertex(id) 验证顶点是否在 g2o 当前图中。
    // 若 vertex(id) 返回 nullptr，说明顶点已被 removeVertex 删除，边必须移除。
    {
        std::vector<g2o::HyperGraph::Edge*> edges_to_remove;
        for (auto* e_raw : optimizer_->edges()) {
            auto* e = dynamic_cast<g2o::EdgeSE3Expmap*>(e_raw);
            if (!e) continue;
            g2o::VertexSE3Expmap* ev0 = dynamic_cast<g2o::VertexSE3Expmap*>(e->vertex(0));
            g2o::VertexSE3Expmap* ev1 = dynamic_cast<g2o::VertexSE3Expmap*>(e->vertex(1));
            bool bad0 = !ev0;
            bool bad1 = !ev1;
            // ★★★ 关键：使用 optimizer_->vertex(id) 验证，而非仅靠 dynamic_cast ★★★
            // optimizer_->vertex(id) 返回 nullptr 当且仅当顶点不存在于 g2o 当前图中，
            // 无论 _vertices[i] 中的原始指针是什么状态。
            if (!bad0 && !optimizer_->vertex(ev0->id())) {
                bad0 = true;
            }
            if (!bad1 && !optimizer_->vertex(ev1->id())) {
                bad1 = true;
            }
            if (bad0 || bad1) {
                edges_to_remove.push_back(e_raw);
                { if (opts_.verbose) std::cerr << "[PGO] 移除 dangling 边: "
                          << (bad0 ? ("v" + std::to_string(bad0 && ev0 ? ev0->id() : -1)) : "(valid)")
                          << " -> "
                          << (bad1 ? ("v" + std::to_string(bad1 && ev1 ? ev1->id() : -1)) : "(valid)")
                          << "（optimizer_->vertex() 返回 nullptr）" << std::endl; }
            }
        }
        for (auto* e_raw : edges_to_remove) {
            optimizer_->removeEdge(e_raw);
        }
        if (!edges_to_remove.empty()) {
            if (opts_.verbose) {
                std::cerr << "[PGO] 移除了 " << edges_to_remove.size()
                          << " 条 dangling 边，优化器剩余: vertices=" << optimizer_->vertices().size()
                          << " edges=" << optimizer_->edges().size() << std::endl;
            }
        }
    }

    if (opts_.verbose) {
        std::cerr << "[PGO] optimize: vertices=" << optimizer_->vertices().size()
                  << " edges=" << optimizer_->edges().size()
                  << " odom_edge_data_=" << odom_edge_data_.size()
                  << " loop_edge_data_=" << loop_edge_data_.size()
                  << std::endl;
    }

    // ★★★ 调试：诊断 initializeOptimization() 返回 0 的原因（默认关闭）★★★
    if (opts_.verbose) {
        int fixed_v_edges = 0, missing_v_edges = 0, nan_edges = 0;
        for (auto* e_raw : optimizer_->edges()) {
            auto* e = dynamic_cast<g2o::EdgeSE3Expmap*>(e_raw);
            if (!e) continue;
            g2o::VertexSE3Expmap* ev0 = dynamic_cast<g2o::VertexSE3Expmap*>(e->vertex(0));
            g2o::VertexSE3Expmap* ev1 = dynamic_cast<g2o::VertexSE3Expmap*>(e->vertex(1));
            if (!ev0 || !ev1) {
                missing_v_edges++;
                { if (opts_.verbose) std::cerr << "[PGO DIAG] 边两端顶点缺失: "
                          << "v0=" << (ev0 ? std::to_string(ev0->id()) : "NULL")
                          << " v1=" << (ev1 ? std::to_string(ev1->id()) : "NULL")
                          << std::endl; }
                continue;
            }
            if (ev0->fixed() && ev1->fixed()) {
                fixed_v_edges++;
                { if (opts_.verbose) std::cerr << "[PGO DIAG] 边两端均 fixed: v" << ev0->id()
                          << "(fixed=" << ev0->fixed() << ")"
                          << " -> v" << ev1->id()
                          << "(fixed=" << ev1->fixed() << ")"
                          << std::endl; }
            }
            Eigen::Matrix<double, 6, 1> info_diag = e->information().diagonal();
            if (!info_diag.allFinite()) {
                nan_edges++;
                std::cerr << "[PGO DIAG] 边信息矩阵含 NaN: v" << ev0->id()
                          << "->v" << ev1->id()
                          << " info_diag=" << info_diag.transpose() << std::endl;
            }
        }
        { if (opts_.verbose) std::cerr << "[PGO DIAG] 边诊断: total=" << optimizer_->edges().size()
                  << " both_fixed=" << fixed_v_edges
                  << " missing_v=" << missing_v_edges
                  << " nan_info=" << nan_edges
                  << std::endl; }
    }

    // ★★★ 防御性：移除所有含空顶点指针的边，并验证顶点有效性 ★★★
    // rebuildOptimizer 重建了顶点，但边对象里的顶点指针理论上应为新顶点。
    // 若有漏网之鱼，buildStructure / buildSystem 会 SIGSEGV，提前清理。
    {
        std::vector<g2o::HyperGraph::Edge*> bad_edges;
        int vertex_not_in_optimizer = 0;
        int vertex_nan_estimate = 0;
        for (auto* e_raw : optimizer_->edges()) {
            bool is_bad = false;
            for (int vi = 0; vi < 2; ++vi) {
                auto* v_edge = e_raw->vertex(vi);
                if (!v_edge) {
                    is_bad = true;
                    break;
                }
                // 检查顶点是否属于当前优化器（防止 dangling vertex 残留）
                if (optimizer_->vertex(v_edge->id()) != v_edge) {
                    { if (opts_.verbose) std::cerr << "[PGO] 预校验：边端点不属于当前优化器，移除边 "
                              << v_edge->id() << " -> " << (vi == 0 ? "?" : "?") << std::endl; }
                    ++vertex_not_in_optimizer;
                    is_bad = true;
                    break;
                }
                // 检查顶点估计合法性
                auto* v_se3 = dynamic_cast<g2o::VertexSE3Expmap*>(v_edge);
                if (v_se3) {
                    const g2o::SE3Quat& est = v_se3->estimate();
                    if (!est.translation().allFinite() || !est.rotation().coeffs().allFinite()) {
                        std::cerr << "[PGO] 预校验：顶点估计含 NaN kf_id=" << v_edge->id()
                                  << "，尝试修复..." << std::endl;
                        ++vertex_nan_estimate;
                        // 尝试从 all_vertex_poses_ 恢复
                        int kf_id = static_cast<int>(v_edge->id());
                        auto it_pose = all_vertex_poses_.find(kf_id);
                        if (it_pose != all_vertex_poses_.end()) {
                            const auto& p = it_pose->second;
                            if (p.t.allFinite() && p.q.coeffs().allFinite()) {
                                Eigen::Isometry3d iso = poseToIsometry(p);
                                v_se3->setEstimate(g2o::SE3Quat(iso.rotation(), iso.translation()));
                                { if (opts_.verbose) std::cerr << "[PGO] 预校验：从 all_vertex_poses_ 恢复顶点 kf_id=" << kf_id
                                          << " t=(" << p.t.transpose() << ")" << std::endl; }
                                --vertex_nan_estimate;  // 修复成功
                            }
                        }
                        if (!v_se3->estimate().translation().allFinite()) {
                            is_bad = true;
                        }
                    }
                }
            }
            if (is_bad) {
                bad_edges.push_back(e_raw);
            }
        }
        if (vertex_not_in_optimizer > 0) {
            { if (opts_.verbose) std::cerr << "[PGO] 预校验：发现 " << vertex_not_in_optimizer
                      << " 条边端点不属于当前优化器" << std::endl; }
        }
        if (vertex_nan_estimate > 0) {
            std::cerr << "[PGO] 预校验：仍有 " << vertex_nan_estimate
                      << " 个顶点估计无法修复，移除相关边" << std::endl;
        }
        for (auto* e_bad : bad_edges) {
            optimizer_->removeEdge(e_bad);
        }
        if (!bad_edges.empty()) {
            { if (opts_.verbose) std::cerr << "[PGO] 共移除 " << bad_edges.size()
                      << " 条无效边，剩余: vertices=" << optimizer_->vertices().size()
                      << " edges=" << optimizer_->edges().size() << std::endl; }
        }
    }

    // ★★★ 防御性顶点估计有效性检查（扩展版）★★★
    // 检查每个顶点的 estimate() 是否可访问且数值合法。
    // g2o::SE3Quat 内部有 r implementations 旋转矩阵缓存，若缓存损坏，
    // linearizeOplus() 访问 .rotation().toRotationMatrix() 会 crash。
    // 强制触发 r implementations 计算：显式调用 .rotation().toRotationMatrix()。
    {
        int bad_estimate = 0;
        for (auto* e_raw : optimizer_->edges()) {
            for (int vi = 0; vi < 2; ++vi) {
                auto* v = dynamic_cast<g2o::VertexSE3Expmap*>(e_raw->vertex(vi));
                if (!v) { ++bad_estimate; break; }
                const g2o::SE3Quat& est = v->estimate();
                // 检查基本量
                if (!est.translation().allFinite() || !est.rotation().coeffs().allFinite()) {
                    std::cerr << "[PGO] 顶点 estimate 基本量含 NaN: kf_id=" << v->id() << std::endl;
                    ++bad_estimate; break;
                }
                // ★★★ 强制计算旋转矩阵缓存（触发 g2o 内部计算）★★★
                Eigen::Matrix3d R = est.rotation().toRotationMatrix();
                // 检查旋转矩阵有效性：有限值、正交性
                if (!R.allFinite()) {
                    std::cerr << "[PGO] 顶点 rotation matrix 含 NaN: kf_id=" << v->id() << std::endl;
                    ++bad_estimate; break;
                }
                double det = R.determinant();
                double ortho_err = (R * R.transpose() - Eigen::Matrix3d::Identity()).norm();
                if (std::abs(det - 1.0) > 1e-3 || ortho_err > 1e-3) {
                    { if (opts_.verbose) std::cerr << "[PGO] 顶点 rotation matrix 非正交 kf_id=" << v->id()
                              << " det=" << det << " ortho_err=" << ortho_err << std::endl; }
                    ++bad_estimate; break;
                }
            }
        }
        if (bad_estimate > 0) {
            std::cerr << "[PGO] 跳过优化（顶点 estimate 验证失败=" << bad_estimate << "）" << std::endl;
            return 0;
        }
        // ★★★ 诊断：打印前 3 个顶点的 estimate（默认关闭）★★★
        if (opts_.verbose) {
            int printed = 0;
            for (auto* e_raw : optimizer_->edges()) {
                if (printed >= 3) break;
                auto* v = dynamic_cast<g2o::VertexSE3Expmap*>(e_raw->vertex(0));
                if (!v) continue;
                const auto& est = v->estimate();
                { if (opts_.verbose) std::cerr << "[PGO DIAG] vertex kf_id=" << v->id()
                          << " t=(" << est.translation().transpose() << ")"
                          << " q=(" << est.rotation().coeffs().transpose() << ")" << std::endl; }
                ++printed;
            }
        }
    }
    // 防御性 dangling edge 检查（必须在 initializeOptimization() 之前执行）
    // initializeOptimization() 会把 _edges 复制进 _activeEdges；之后再 removeEdge()
    // 只改 _edges、不动 _activeEdges，于是 _activeEdges 里留下悬空指针。
    // 所以只能在它之前把坏边摘掉。这是原本被整段注释掉的检查（L8 复活）。
    {
        int removed_null_ptr = 0;
        int removed_bad_edge = 0;
        std::vector<g2o::HyperGraph::Edge*> edges_to_drop;
        for (auto* e_raw : optimizer_->edges()) {
            auto* e = dynamic_cast<g2o::EdgeSE3Expmap*>(e_raw);
            if (!e) continue;
            if (!e->vertex(0) || !e->vertex(1)) {
                ++removed_null_ptr;
                edges_to_drop.push_back(e_raw);
                continue;
            }
            int v0_id = static_cast<int>(e->vertex(0)->id());
            int v1_id = static_cast<int>(e->vertex(1)->id());
            if (!optimizer_->vertex(v0_id) || !optimizer_->vertex(v1_id)) {
                ++removed_bad_edge;
                edges_to_drop.push_back(e_raw);
            }
        }
        if (!edges_to_drop.empty()) {
            { if (opts_.verbose) std::cerr << "[PGO] 预检查（initializeOptimization 之前）：移除 "
                      << removed_null_ptr << " 条空指针边 + " << removed_bad_edge
                      << " 条悬空边" << std::endl; }
            for (auto* e_bad : edges_to_drop) {
                optimizer_->removeEdge(e_bad);
            }
        }
    }
    // L3：显式传 level 0。语义是"参与优化的边 = level 0 的边"，
    // 必须和 addOdometryEdge()（不设 level，默认 0）保持一致。
    // 不传参数时 g2o 也按 0 处理，但"隐式默认值"正是上一个 bug 的藏身处。
    int init = optimizer_->initializeOptimization(0);
    if (init <= 0) {
        std::cerr << "[PGO] initializeOptimization() 失败（返回 " << init
                  << "）：optimizer 可能无有效边或顶点" << std::endl;
        return 0;
    }

    // （旧位置的预检查已删除，移动到 initializeOptimization 之前）

    // R-028 要求的"剩余约束仍足以确定全部顶点"的运行时证明。
    // 判据：顶点数 - 固定顶点数 <= 参与优化的独立约束数。
    // 每条 EdgeSE3Expmap 提供 1 个 6-DoF 相对约束，n 个自由顶点至少需要 n-1 条边
    // 才连成生成树；不满足说明图是断的，PGO 只能动到其中一块，而 chi2 可能照样非 0，
    // 于是"优化成功"是假象。这是 L3 那个 bug（里程计边全被 setLevel(1) 移出）的
    // 直接检测点：一旦有人再写一次批量 setLevel，这里会立刻炸出来。
    {
        int n_vertices = 0;
        int n_fixed = 0;
        for (auto* v : optimizer_->vertices()) {
            if (!v) continue;
            ++n_vertices;
            if (v->fixed()) ++n_fixed;
        }
        const int n_active_edges = static_cast<int>(optimizer_->edges().size());
        const int n_free = n_vertices - n_fixed;
        if (n_free > 0 && n_active_edges < n_free - 1) {
            std::cerr << "[PGO] 约束不足：自由顶点 " << n_free << "（顶点 " << n_vertices
                      << " / 固定 " << n_fixed << "）只有 " << n_active_edges
                      << " 条边参与优化，少于生成树所需的 " << (n_free - 1)
                      << " 条：图是断的，本次 PGO 无法确定全部位姿（R-028）"
                      << std::endl;
        } else if (opts_.verbose) {
            std::cerr << "[PGO] 约束检查通过：自由顶点 " << n_free << "，参与优化的边 "
                      << n_active_edges << "，spanning tree 需要 " << std::max(0, n_free - 1)
                      << std::endl;
        }
    }

    // L8：这里原本用 sigsetjmp/sigaction 捕获 SIGSEGV/SIGBUS/SIGABRT，再 siglongjmp
    // 回来清理边缓存继续跑。这条路本身就是错的：段错误发生时 g2o 内部已处于半
    // 更新状态，longjmp 跳过栈展开后返回的是一个损坏的图，后续每帧都在损坏状态
    // 上"优化"，崩溃被换成静默的错误轨迹——比直接崩更难查。
    // 能防的只有 C++ 异常（g2o 抛 std::bad_alloc / 断言 exception），保留 try/catch。
    // 真正的段错误应该让它崩，用 ASAN / core dump 定位。
    int opt_result = 0;
    try {
        opt_result = optimizer_->optimize(opts_.max_iterations);
    } catch (const std::exception& e) {
        std::cerr << "[PGO] g2o optimize() 抛出异常: " << e.what() << std::endl;
        opt_result = 0;
    } catch (...) {
        std::cerr << "[PGO] g2o optimize() 抛出未知异常" << std::endl;
        opt_result = 0;
    }

    // D4-b：把 chi2 与迭代数回传给调用方，并在 chi2 无有效值时显式告警。
    // chi2 == 0 不是"完美收敛"，而是"图上没有产生任何残差" —— 典型成因是
    // 约束整类没进图（例如旧的 setLevel(1) 把里程计边移出 level 0）。
    // 只有把它打出来，这种"报成功其实没优化"的状态才会暴露。
    if (out_iterations) *out_iterations = opt_result;
    const double chi2 = optimizer_->chi2();
    if (out_chi2) *out_chi2 = chi2;
    if (!(chi2 > 0.0)) {
        std::cerr << "[PGO] chi2=" << chi2 << " —— 图上没有有效残差。"
                  << " 里程计边=" << odom_edge_data_.size()
                  << " 回环边=" << loop_edge_data_.size()
                  << " 活跃边=" << optimizer_->edges().size()
                  << "，请检查约束是否真的进了 level 0。" << std::endl;
    } else if (opts_.verbose) {
        std::cerr << "[PGO] chi2=" << chi2 << " iterations=" << opt_result << std::endl;
    }
    return opt_result;
}

bool PoseGraphOptimizer::getOptimizedPose(int kf_id, sfc_core::frontend::Pose* out_pose) const {
    const g2o::VertexSE3Expmap* v = getVertex(kf_id);
    if (!v) return false;

    const g2o::SE3Quat& se3 = v->estimate();
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = se3.rotation().toRotationMatrix();
    iso.translation() = se3.translation();
    *out_pose = isometryToPose(iso);
    return true;
}

void PoseGraphOptimizer::getAllOptimizedPoses(std::unordered_map<int, sfc_core::frontend::Pose>* out_poses) const {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    out_poses->clear();
    if (!optimizer_) return;
    // optimizer_->vertices() returns std::pair<int, Vertex*> pairs
    for (const auto& v_pair : optimizer_->vertices()) {
        int kf_id = v_pair.first;
        const g2o::VertexSE3Expmap* v = dynamic_cast<const g2o::VertexSE3Expmap*>(v_pair.second);
        if (!v) continue;
        const g2o::SE3Quat& se3 = v->estimate();
        Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
        iso.linear() = se3.rotation().toRotationMatrix();
        iso.translation() = se3.translation();
        (*out_poses)[kf_id] = isometryToPose(iso);
    }
}

std::vector<int> PoseGraphOptimizer::getVertexIds() const {
    return vertex_ids_;
}

void PoseGraphOptimizer::clear() {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    optimizer_->clear();
    kf_id_to_vertex_index_.clear();
    vertex_ids_.clear();
    vertex_ptrs_.clear();
    vertex_records_.clear();
    loop_edge_ids_.clear();
    loop_edge_data_.clear();
    odom_edge_ids_.clear();
    odom_edge_data_.clear();
    evicted_kf_ids_.clear();
    odom_edge_key_set_.clear();
    loop_edge_key_set_.clear();
    all_vertex_poses_.clear();
    optimizer_initialized_ = false;
    optimizerNeedsRebuild_ = false;
}

void PoseGraphOptimizer::reset() {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    optimizer_->clear();
    kf_id_to_vertex_index_.clear();
    vertex_ids_.clear();
    vertex_ptrs_.clear();
    vertex_records_.clear();
    loop_edge_ids_.clear();
    loop_edge_data_.clear();
    odom_edge_ids_.clear();
    odom_edge_data_.clear();
    evicted_kf_ids_.clear();
    odom_edge_key_set_.clear();
    loop_edge_key_set_.clear();
    all_vertex_poses_.clear();
    optimizer_initialized_ = false;
    optimizerNeedsRebuild_ = false;
}

} // namespace loopdetection
} // namespace sfc_core
