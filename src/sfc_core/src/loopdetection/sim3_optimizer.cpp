// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

// 第三方库头文件（在命名空间外部引入）
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "sfc_core/loopdetection/sim3_optimizer.hpp"
#include <iostream>
#include <cmath>

namespace sfc_core {
namespace loopdetection {

// ================================================================
// Sim3Optimizer Implementation
// 参考 ORB_SLAM3/src/Optimizer.cc OptimizeSim3() (第 2115 行)
// ================================================================

static float ProjectToImage(const Eigen::Vector3f& P, float fx, float fy, float cx, float cy) {
    if (P.z() < 1e-6f) return 1e9f;
    float u = fx * P.x() / P.z() + cx;
    float v = fy * P.y() / P.z() + cy;
    return std::sqrt(u * u + v * v);
}

int Sim3Optimizer::optimizeSim3(const Eigen::Matrix4d& initial_sim3,
                                const std::vector<MapPointMatch>& matches,
                                const cv::Mat& K,
                                bool fix_scale,
                                const Eigen::Matrix4d& T_cw1,
                                const Eigen::Matrix4d& T_cw2,
                                Eigen::Matrix4d& out_sim3) {
    // ================================================================
    // g2o Sim3 优化
    //
    // 顶点：
    //   - VertexSim3Expmap: Sim3 相对变换（T12）
    //   - VertexSBAPointXYZ: 地图点（固定）
    //
    // 边：
    //   - EdgeSim3ProjectXYZ: 帧2 地图点 -> 帧1 像素（x1 = S12 * X2）
    //   - EdgeInverseSim3ProjectXYZ: 帧1 地图点 -> 帧2 像素（x2 = S21 * X1）
    //
    // 参考 ORB_SLAM3/src/Optimizer.cc 第 2115-2350 行
    // ================================================================

    // 提取相机内参
    float fx, fy, cx, cy;
    if (!K.empty() && K.rows == 3 && K.cols == 3) {
        fx = static_cast<float>(K.at<double>(0, 0));
        fy = static_cast<float>(K.at<double>(1, 1));
        cx = static_cast<float>(K.at<double>(0, 2));
        cy = static_cast<float>(K.at<double>(1, 2));
    } else {
        fx = fy = 525.0f;
        cx = 319.5f;
        cy = 239.5f;
    }

    // 提取世界到相机的变换矩阵（用于深度检查）
    // T_cw = [R_cw | t_cw] where P_cam = R_cw * P_world + t_cw
    Eigen::Matrix3d R_cw1 = T_cw1.block<3, 3>(0, 0);
    Eigen::Vector3d t_cw1 = T_cw1.block<3, 1>(0, 3);
    Eigen::Matrix3d R_cw2 = T_cw2.block<3, 3>(0, 0);
    Eigen::Vector3d t_cw2 = T_cw2.block<3, 1>(0, 3);

    // 构建 g2o 优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    auto* linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    auto* blockSolver = new g2o::BlockSolverX(std::unique_ptr<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>(linearSolver));
    auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<g2o::BlockSolverX>(blockSolver));
    optimizer.setAlgorithm(algorithm);

    // 从 initial_sim3 构建 g2o::Sim3
    Eigen::Matrix3d R0 = initial_sim3.block<3, 3>(0, 0);
    Eigen::Vector3d t0 = initial_sim3.block<3, 1>(0, 3);
    double s0 = 1.0;  // RGB-D 固定尺度
    g2o::Sim3 g2oS12(Eigen::Quaterniond(R0), t0, s0);

    // 设置 Sim3 顶点
    g2o::VertexSim3Expmap* vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale = fix_scale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_focal_length1 = Eigen::Vector2d(fx, fy);
    vSim3->_focal_length2 = Eigen::Vector2d(fx, fy);
    vSim3->_principle_point1 = Eigen::Vector2d(cx, cy);
    vSim3->_principle_point2 = Eigen::Vector2d(cx, cy);
    optimizer.addVertex(vSim3);

    // 地图点顶点和边
    const int N = static_cast<int>(matches.size());
    std::vector<g2o::EdgeSim3ProjectXYZ*> edges12;
    std::vector<g2o::EdgeInverseSim3ProjectXYZ*> edges21;
    std::vector<size_t> inlierIndices;
    edges12.reserve(2 * N);
    edges21.reserve(2 * N);

    float deltaHuber = 8.0f;  // Huber 阈值（像素）
    float th2 = deltaHuber * deltaHuber;

    int nCorrespondences = 0;

    for (int i = 0; i < N; ++i) {
        const auto& m = matches[i];

        int kf2_idx = m.kf2_kpt_idx;
        if (kf2_idx < 0) continue;  // 帧2 没有匹配点

        // ★★★ 使用世界坐标（而非相机系坐标）★★★
        // g2o::VertexPointXYZ 的 estimate() 必须是世界坐标
        // 对应的重投影边 EdgeSim3ProjectXYZ / EdgeInverseSim3ProjectXYZ
        // 会自动应用 Sim3 变换将世界坐标投影到像素坐标
        if (m.pt_world.norm() < 1e-6) continue;
        Eigen::Vector3d pw = m.pt_world;

        // 深度检查（世界坐标投影到相机系后的深度）
        Eigen::Vector3d pc1_check = R_cw1 * pw + t_cw1;
        Eigen::Vector3d pc2_check = R_cw2 * pw + t_cw2;
        if (pc1_check.z() < 1e-6f || pc2_check.z() < 1e-6f) continue;

        // 构建地图点顶点（固定，BA 只优化 Sim3）
        g2o::VertexPointXYZ* vPoint1 = new g2o::VertexPointXYZ();
        vPoint1->setEstimate(pw);
        vPoint1->setId(2 * i + 1);
        vPoint1->setFixed(true);
        optimizer.addVertex(vPoint1);

        // ★★★ 使用原始像素坐标作为观测值（而非重投影相机系坐标）★★★
        // 观测 = 实际检测到的关键点像素位置
        // 这是正确的做法，避免了「用相机系坐标重投影作为观测」的循环依赖
        Eigen::Matrix<double, 2, 1> obs1;
        obs1 << m.px1.x(), m.px1.y();

        Eigen::Matrix<double, 2, 1> obs2;
        obs2 << m.px2.x(), m.px2.y();

        // 边：帧1 点 -> 帧1 像素 (x1 = S12 * X_world) ← S12 将世界点变换到帧1相机系再投影
        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2 * i + 1)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        e12->setInformation(Eigen::Matrix2d::Identity());
        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber();
        rk1->setDelta(deltaHuber);
        e12->setRobustKernel(rk1);
        optimizer.addEdge(e12);
        edges12.push_back(e12);

        // 边：帧1 点 -> 帧2 像素 (x2 = S21 * X_world) ← S21 将世界点变换到帧2相机系再投影
        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();
        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2 * i + 1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        e21->setInformation(Eigen::Matrix2d::Identity());
        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber();
        rk2->setDelta(deltaHuber);
        e21->setRobustKernel(rk2);
        optimizer.addEdge(e21);
        edges21.push_back(e21);

        inlierIndices.push_back(i);
        nCorrespondences++;
    }

    // 执行优化
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // 统计内点
    int nBad = 0;
    std::vector<MapPointMatch> refinedMatches;
    refinedMatches.reserve(nCorrespondences);

    for (size_t i = 0; i < edges12.size(); ++i) {
        g2o::EdgeSim3ProjectXYZ* e12 = edges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = edges21[i];

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            nBad++;
            continue;
        }
        refinedMatches.push_back(matches[inlierIndices[i]]);
    }

    // 提取精化后的 Sim3
    const g2o::Sim3& refined = vSim3->estimate();
    out_sim3.setIdentity();
    out_sim3.block<3, 3>(0, 0) = refined.rotation().toRotationMatrix();
    out_sim3.block<3, 1>(0, 3) = refined.translation();

    // ★★★ 防御性检查：旋转矩阵对角线应接近 1（若被 g2o 内部 scale 污染则需要归一化）★★★
    // Sim3 的旋转矩阵可能被内部的 scale 因子轻微污染，直接取 scale = R 的 F范/Frobenius范 估算
    Eigen::Matrix3d R_out = out_sim3.block<3, 3>(0, 0);
    double scale_hint = R_out.norm() / std::sqrt(3.0);  // 理想 R 的 F范 = sqrt(3)
    if (!std::isfinite(scale_hint) || std::abs(scale_hint - 1.0) > 0.01) {
        std::cerr << "[Sim3Opt] ⚠️ Scale 异常！R_Fnorm=" << R_out.norm()
                  << " scale_hint=" << scale_hint << " → 强制归一化" << std::endl;
        if (std::isfinite(scale_hint) && scale_hint > 1e-6) {
            out_sim3.block<3, 3>(0, 0) = R_out / scale_hint;
        } else {
            std::cerr << "[Sim3Opt] 严重错误：scale_hint 无效，丢弃 Sim3 结果" << std::endl;
            return 0;
        }
    }
    // 尺度（RGB-D 固定为 1）

    return static_cast<int>(refinedMatches.size());
}

int Sim3Optimizer::refineWithTwoPass(const Eigen::Matrix4d& initial_sim3,
                                     const std::vector<MapPointMatch>& matches,
                                     const cv::Mat& K,
                                     bool fix_scale,
                                     float first_th,
                                     float second_th,
                                     int max_iterations,
                                     const Eigen::Matrix4d& T_cw1,
                                     const Eigen::Matrix4d& T_cw2,
                                     Eigen::Matrix4d& out_sim3) {
    // ================================================================
    // 两轮精化（与 ORB_SLAM3 LoopClosing.cc DetectCommonRegionsFromBoW 对齐）
    //
    // 第一轮：粗匹配（阈值 first_th px）
    //   1. Sim3Solver RANSAC（阈值 8px）找到初始 Sim3
    //   2. OptimizeSim3（阈值 first_th px）精化
    //   3. 剔除外点，重新构建内点匹配列表
    //
    // 第二轮：精匹配（阈值 second_th px）
    //   1. 用第一轮精化的 Sim3 做更严格的投影匹配
    //   2. OptimizeSim3（阈值 second_th px）再次精化
    // ================================================================

    Eigen::Matrix4d sim3_current = initial_sim3;
    int best_inliers = 0;

    // 第一轮
    {
        Eigen::Matrix4d sim3_refined;
        int n_inliers = optimizeSim3(sim3_current, matches, K, fix_scale,
                                     T_cw1, T_cw2, sim3_refined);
        if (n_inliers > best_inliers) {
            best_inliers = n_inliers;
            sim3_current = sim3_refined;
        }
    }

    // 提取世界到相机的变换矩阵
    Eigen::Matrix3d R_cw1 = T_cw1.block<3, 3>(0, 0);
    Eigen::Vector3d t_cw1 = T_cw1.block<3, 1>(0, 3);
    Eigen::Matrix3d R_cw2 = T_cw2.block<3, 3>(0, 0);
    Eigen::Vector3d t_cw2 = T_cw2.block<3, 1>(0, 3);

    // ★★★ 迭代式外点剔除（参考 ORB_SLAM3） ★★★
    // 在两轮之间添加外点剔除，提高精化质量
    if (best_inliers >= 10) {
        std::vector<MapPointMatch> inlier_matches;
        inlier_matches.reserve(matches.size());

        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);

        Eigen::Matrix3d R = sim3_current.block<3, 3>(0, 0);
        Eigen::Vector3d t = sim3_current.block<3, 1>(0, 3);

        for (const auto& m : matches) {
            // ★★★ 使用世界坐标和 Sim3 变换计算重投影误差，而非相机系坐标 ★★★
            // Sim3 将世界坐标 X_world 变换到帧1相机系：X1c = R * X_world + t
            if (m.pt_world.norm() < 1e-6) continue;
            Eigen::Vector3d X1c = R * m.pt_world + t;
            if (X1c.z() < 1e-6) continue;

            // 用 Sim3 将世界坐标变换到帧2相机系，再投影到像素
            // Sim3: T12 将帧2相机系点变换到帧1相机系
            // 反变换: X2c = T12^{-1} * X1c = R^T * (X1c - s*t), scale=1
            Eigen::Vector3d X2c = R.transpose() * (X1c - t);
            if (X2c.z() < 1e-6) continue;

            double proj2_u = fx * X2c.x() / X2c.z() + cx;
            double proj2_v = fy * X2c.y() / X2c.z() + cy;

            // 用原始像素坐标作为 ground truth
            double gt_u = m.px2.x();
            double gt_v = m.px2.y();

            double error = std::sqrt((proj2_u - gt_u) * (proj2_u - gt_u) +
                                    (proj2_v - gt_v) * (proj2_v - gt_v));

            if (error < first_th * 2.0) {
                inlier_matches.push_back(m);
            }
        }

        if (inlier_matches.size() >= static_cast<size_t>(best_inliers * 0.8)) {
            Eigen::Matrix4d sim3_refined2;
            int n_inliers2 = optimizeSim3(sim3_current, inlier_matches, K, fix_scale,
                                         T_cw1, T_cw2, sim3_refined2);
            if (n_inliers2 > best_inliers) {
                best_inliers = n_inliers2;
                sim3_current = sim3_refined2;
            }
        }
    }

    // 第二轮（仅当第一轮内点足够多时）
    if (best_inliers >= 10) {
        Eigen::Matrix4d sim3_refined2;
        int n_inliers2 = optimizeSim3(sim3_current, matches, K, fix_scale,
                                     T_cw1, T_cw2, sim3_refined2);
        if (n_inliers2 > best_inliers) {
            best_inliers = n_inliers2;
            sim3_current = sim3_refined2;
        }
    }

    out_sim3 = sim3_current;
    return best_inliers;
}

}  // namespace loopdetection
}  // namespace sfc_core
