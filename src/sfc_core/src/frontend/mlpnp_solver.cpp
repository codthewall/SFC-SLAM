// ================================================================
//  MLPnPsolver 实现 - 基于 Bearing Vector 的 PnP 求解器
//  移植自 ORB_SLAM3，使用最小 6 点法 + 高斯-牛顿优化
// ================================================================

#include "sfc_core/frontend/mlpnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace sfc_core {
namespace frontend {

// ==================== RANSAC 参数设置 ====================

void MLPNPSolver::SetRansacParameters(double probability, int minInliers,
                                      int maxIterations, int minSet,
                                      float epsilon, float th2) {
    config_.probability = probability;
    config_.min_inliers = minInliers;
    config_.max_iterations = maxIterations;
    config_.min_set = minSet;
    config_.epsilon = epsilon;
    config_.threshold = th2;

    N = static_cast<int>(mvP2D.size());

    mvbInliersi.resize(N);

    int nMinInliers = static_cast<int>(N * config_.epsilon);
    if (nMinInliers < config_.min_inliers)
        nMinInliers = config_.min_inliers;
    if (nMinInliers < config_.min_set)
        nMinInliers = config_.min_set;
    config_.min_inliers = nMinInliers;

    if (config_.epsilon < static_cast<float>(config_.min_inliers) / N)
        config_.epsilon = static_cast<float>(config_.min_inliers) / N;

    int nIterations;
    if (config_.min_inliers == N)
        nIterations = 1;
    else
        nIterations = static_cast<int>(std::ceil(std::log(1 - config_.probability) /
                                                  std::log(1 - std::pow(config_.epsilon, 3))));

    config_.max_iterations = std::max(1, std::min(nIterations, config_.max_iterations));

    mvMaxError.resize(mvSigma2.size());
    for (size_t i = 0; i < mvSigma2.size(); i++)
        mvMaxError[i] = mvSigma2[i] * th2;
}

// ==================== 数据设置 ====================

void MLPNPSolver::SetData(const bearingVectors_t& bearings,
                         const points_t& points3d,
                         const std::vector<cv::Point2f>& pixels,
                         const std::vector<int>& keypoint_indices,
                         const std::vector<float>& sigma2,
                         const cv::Mat& K) {
    mvBearingVecs = bearings;
    mvP3Dw = points3d;
    mvP2D = pixels;
    mvKeyPointIndices.clear();
    mvAllIndices.clear();

    N = static_cast<int>(bearings.size());

    for (int i = 0; i < N; ++i) {
        mvKeyPointIndices.push_back(keypoint_indices.empty() ? i : keypoint_indices[i]);
        mvAllIndices.push_back(static_cast<size_t>(i));
    }

    mvSigma2 = sigma2;
    if (mvSigma2.empty()) {
        mvSigma2.resize(N, 1.0f);
    }

    if (!K.empty()) {
        mK = K.clone();
    }

    // 预分配内点容器
    mvbInliersi.resize(N);
    mvbBestInliers.resize(N);

    // 设置默认 RANSAC 参数
    SetRansacParameters();
}

// ==================== RANSAC 主循环 ====================

bool MLPNPSolver::iterate(int nIterations, bool& bNoMore,
                         std::vector<bool>& vbInliers,
                         int& nInliers, Eigen::Matrix4f& Tout) {
    Tout.setIdentity();
    bNoMore = false;
    vbInliers.clear();
    nInliers = 0;

    if (N < config_.min_inliers) {
        bNoMore = true;
        return false;
    }

    std::vector<size_t> vAvailableIndices = mvAllIndices;

    int nCurrentIterations = 0;
    while (mnIterations < config_.max_iterations || nCurrentIterations < nIterations) {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        bearingVectors_t bearingVecs(config_.min_set);
        points_t p3DS(config_.min_set);
        std::vector<int> indexes(config_.min_set);

        for (int i = 0; i < config_.min_set; ++i) {
            int randi = randomInt(static_cast<int>(vAvailableIndices.size()));
            int idx = static_cast<int>(vAvailableIndices[randi]);

            bearingVecs[i] = mvBearingVecs[idx];
            p3DS[i] = mvP3Dw[idx];
            // ★★★ Bug Fix: 使用正确的随机索引而非循环索引 ★★★
            // 原bug: indexes[i] = i 总是产生[0,1,2,...]，破坏了RANSAC的随机性
            // 正确做法: 保存实际随机选择的索引，用于CheckInliers中正确映射内点
            indexes[i] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // 协方差矩阵（当前版本不使用）
        cov3_mats_t covs(1);

        transformation_t result;
        computePose(bearingVecs, p3DS, covs, indexes, result);

        // 保存结果
        mRi[0][0] = result(0, 0);
        mRi[0][1] = result(0, 1);
        mRi[0][2] = result(0, 2);

        mRi[1][0] = result(1, 0);
        mRi[1][1] = result(1, 1);
        mRi[1][2] = result(1, 2);

        mRi[2][0] = result(2, 0);
        mRi[2][1] = result(2, 1);
        mRi[2][2] = result(2, 2);

        mti[0] = result(0, 3);
        mti[1] = result(1, 3);
        mti[2] = result(2, 3);

        CheckInliers();

        if (mnInliersi >= config_.min_inliers) {
            if (mnInliersi > mnBestInliers) {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3, 3, CV_64F, mRi);
                cv::Mat tcw(3, 1, CV_64F, mti);
                Rcw.convertTo(Rcw, CV_32F);
                tcw.convertTo(tcw, CV_32F);
                mBestTcw.setIdentity();
                mBestTcw.block<3, 3>(0, 0) <<
                    static_cast<float>(Rcw.at<double>(0, 0)), static_cast<float>(Rcw.at<double>(0, 1)), static_cast<float>(Rcw.at<double>(0, 2)),
                    static_cast<float>(Rcw.at<double>(1, 0)), static_cast<float>(Rcw.at<double>(1, 1)), static_cast<float>(Rcw.at<double>(1, 2)),
                    static_cast<float>(Rcw.at<double>(2, 0)), static_cast<float>(Rcw.at<double>(2, 1)), static_cast<float>(Rcw.at<double>(2, 2));
                mBestTcw.block<3, 1>(0, 3) <<
                    static_cast<float>(tcw.at<double>(0)), static_cast<float>(tcw.at<double>(1)), static_cast<float>(tcw.at<double>(2));

                Eigen::Matrix<double, 3, 3, Eigen::RowMajor> eigRcw(mRi[0]);
                Eigen::Vector3d eigtcw(mti);
            }

            if (Refine()) {
                nInliers = mnRefinedInliers;
                vbInliers.resize(mvKeyPointIndices.size(), false);
                for (int i = 0; i < N; i++) {
                    if (mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                Tout = mRefinedTcw;
                return true;
            }
        }
    }

    if (mnIterations >= config_.max_iterations) {
        bNoMore = true;
        if (mnBestInliers >= config_.min_inliers) {
            nInliers = mnBestInliers;
            vbInliers.resize(mvKeyPointIndices.size(), false);
            for (int i = 0; i < N; i++) {
                if (mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            Tout = mBestTcw;
            return true;
        }
    }

    return false;
}

// ==================== 内点检查 ====================

void MLPNPSolver::CheckInliers() {
    mnInliersi = 0;

    double fx = mK.empty() ? 525.0 : mK.at<double>(0, 0);
    double fy = mK.empty() ? 525.0 : mK.at<double>(1, 1);
    double cx = mK.empty() ? 319.5 : mK.at<double>(0, 2);
    double cy = mK.empty() ? 239.5 : mK.at<double>(1, 2);

    for (int i = 0; i < N; i++) {
        point_t p = mvP3Dw[i];
        cv::Point3f P3Dw(static_cast<float>(p(0)), static_cast<float>(p(1)), static_cast<float>(p(2)));
        cv::Point2f P2D = mvP2D[i];

        float xc = static_cast<float>(mRi[0][0]) * P3Dw.x +
                   static_cast<float>(mRi[0][1]) * P3Dw.y +
                   static_cast<float>(mRi[0][2]) * P3Dw.z + static_cast<float>(mti[0]);
        float yc = static_cast<float>(mRi[1][0]) * P3Dw.x +
                   static_cast<float>(mRi[1][1]) * P3Dw.y +
                   static_cast<float>(mRi[1][2]) * P3Dw.z + static_cast<float>(mti[1]);
        float zc = static_cast<float>(mRi[2][0]) * P3Dw.x +
                   static_cast<float>(mRi[2][1]) * P3Dw.y +
                   static_cast<float>(mRi[2][2]) * P3Dw.z + static_cast<float>(mti[2]);

        if (zc < 1e-6f) {
            mvbInliersi[i] = false;
            continue;
        }

        float u = fx * xc / zc + cx;
        float v = fy * yc / zc + cy;

        float distX = P2D.x - u;
        float distY = P2D.y - v;

        float error2 = distX * distX + distY * distY;

        if (error2 < mvMaxError[i]) {
            mvbInliersi[i] = true;
            mnInliersi++;
        } else {
            mvbInliersi[i] = false;
        }
    }
}

// ==================== 精化 ====================

bool MLPNPSolver::Refine() {
    std::vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for (size_t i = 0; i < mvbBestInliers.size(); i++) {
        if (mvbBestInliers[i]) {
            vIndices.push_back(static_cast<int>(i));
        }
    }

    bearingVectors_t bearingVecs;
    points_t p3DS;
    std::vector<int> indexes;

    for (size_t i = 0; i < vIndices.size(); i++) {
        int idx = vIndices[i];
        bearingVecs.push_back(mvBearingVecs[idx]);
        p3DS.push_back(mvP3Dw[idx]);
        indexes.push_back(static_cast<int>(i));
    }

    cov3_mats_t covs(1);
    transformation_t result;
    computePose(bearingVecs, p3DS, covs, indexes, result);

    CheckInliers();

    mnRefinedInliers = mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if (mnInliersi > config_.min_inliers) {
        cv::Mat Rcw(3, 3, CV_64F, mRi);
        cv::Mat tcw(3, 1, CV_64F, mti);
        Rcw.convertTo(Rcw, CV_32F);
        tcw.convertTo(tcw, CV_32F);
        mRefinedTcw.setIdentity();
        mRefinedTcw.block<3, 3>(0, 0) <<
            static_cast<float>(Rcw.at<double>(0, 0)), static_cast<float>(Rcw.at<double>(0, 1)), static_cast<float>(Rcw.at<double>(0, 2)),
            static_cast<float>(Rcw.at<double>(1, 0)), static_cast<float>(Rcw.at<double>(1, 1)), static_cast<float>(Rcw.at<double>(1, 2)),
            static_cast<float>(Rcw.at<double>(2, 0)), static_cast<float>(Rcw.at<double>(2, 1)), static_cast<float>(Rcw.at<double>(2, 2));
        mRefinedTcw.block<3, 1>(0, 3) <<
            static_cast<float>(tcw.at<double>(0)), static_cast<float>(tcw.at<double>(1)), static_cast<float>(tcw.at<double>(2));

        return true;
    }
    return false;
}

// ==================== MLPnP 核心算法 ====================

void MLPNPSolver::computePose(const bearingVectors_t& f,
                              const points_t& p,
                              const cov3_mats_t& covMats,
                              const std::vector<int>& indices,
                              transformation_t& result) {
    size_t numberCorrespondences = indices.size();
    assert(numberCorrespondences > 5);

    bool planar = false;
    std::vector<Eigen::MatrixXd> nullspaces(numberCorrespondences);
    Eigen::MatrixXd points3(3, static_cast<Eigen::Index>(numberCorrespondences));
    points_t points3v(numberCorrespondences);
    points4_t points4v(numberCorrespondences);

    for (size_t i = 0; i < numberCorrespondences; i++) {
        bearingVector_t f_current = f[indices[i]];
        points3.col(static_cast<Eigen::Index>(i)) = p[indices[i]];
        Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner>
            svd_f(f_current.transpose(), Eigen::ComputeFullV);
        nullspaces[i] = svd_f.matrixV().block(0, 1, 3, 2);
        points3v[i] = p[indices[i]];
    }

    // 检测是否为平面场景
    Eigen::Matrix3d planarTest = points3 * points3.transpose();
    Eigen::FullPivHouseholderQR<Eigen::Matrix3d> rankTest(planarTest);
    Eigen::Matrix3d eigenRot;
    eigenRot.setIdentity();

    if (rankTest.rank() == 2) {
        planar = true;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(planarTest);
        eigenRot = eigen_solver.eigenvectors().real();
        eigenRot.transposeInPlace();
        for (size_t i = 0; i < numberCorrespondences; i++)
            points3.col(static_cast<Eigen::Index>(i)) = eigenRot * points3.col(static_cast<Eigen::Index>(i));
    }

    // 随机模型
    Eigen::SparseMatrix<double> P(2 * static_cast<int>(numberCorrespondences),
                                  2 * static_cast<int>(numberCorrespondences));
    bool use_cov = false;
    P.setIdentity();

    if (covMats.size() == numberCorrespondences) {
        use_cov = true;
        int l = 0;
        for (size_t i = 0; i < numberCorrespondences; ++i) {
            cov2_mat_t temp = nullspaces[i].transpose() * covMats[i] * nullspaces[i];
            temp = temp.inverse().eval();
            P.coeffRef(l, l) = temp(0, 0);
            P.coeffRef(l, l + 1) = temp(0, 1);
            P.coeffRef(l + 1, l) = temp(1, 0);
            P.coeffRef(l + 1, l + 1) = temp(1, 1);
            l += 2;
        }
    }

    // 构建设计矩阵 A
    const int rowsA = static_cast<int>(2 * numberCorrespondences);
    int colsA = 12;
    Eigen::MatrixXd A;
    if (planar) {
        colsA = 9;
        A = Eigen::MatrixXd(rowsA, 9);
    } else {
        A = Eigen::MatrixXd(rowsA, 12);
    }
    A.setZero();

    if (planar) {
        for (size_t i = 0; i < numberCorrespondences; ++i) {
            point_t pt3_current = points3.col(static_cast<Eigen::Index>(i));

            A(2 * static_cast<Eigen::Index>(i), 0) = nullspaces[i](0, 0) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i) + 1, 0) = nullspaces[i](0, 1) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i), 1) = nullspaces[i](0, 0) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i) + 1, 1) = nullspaces[i](0, 1) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i), 2) = nullspaces[i](1, 0) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i) + 1, 2) = nullspaces[i](1, 1) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i), 3) = nullspaces[i](1, 0) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i) + 1, 3) = nullspaces[i](1, 1) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i), 4) = nullspaces[i](2, 0) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i) + 1, 4) = nullspaces[i](2, 1) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i), 5) = nullspaces[i](2, 0) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i) + 1, 5) = nullspaces[i](2, 1) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i), 6) = nullspaces[i](0, 0);
            A(2 * static_cast<Eigen::Index>(i) + 1, 6) = nullspaces[i](0, 1);
            A(2 * static_cast<Eigen::Index>(i), 7) = nullspaces[i](1, 0);
            A(2 * static_cast<Eigen::Index>(i) + 1, 7) = nullspaces[i](1, 1);
            A(2 * static_cast<Eigen::Index>(i), 8) = nullspaces[i](2, 0);
            A(2 * static_cast<Eigen::Index>(i) + 1, 8) = nullspaces[i](2, 1);
        }
    } else {
        for (size_t i = 0; i < numberCorrespondences; ++i) {
            point_t pt3_current = points3.col(static_cast<Eigen::Index>(i));

            A(2 * static_cast<Eigen::Index>(i), 0) = nullspaces[i](0, 0) * pt3_current[0];
            A(2 * static_cast<Eigen::Index>(i) + 1, 0) = nullspaces[i](0, 1) * pt3_current[0];
            A(2 * static_cast<Eigen::Index>(i), 1) = nullspaces[i](0, 0) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i) + 1, 1) = nullspaces[i](0, 1) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i), 2) = nullspaces[i](0, 0) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i) + 1, 2) = nullspaces[i](0, 1) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i), 3) = nullspaces[i](1, 0) * pt3_current[0];
            A(2 * static_cast<Eigen::Index>(i) + 1, 3) = nullspaces[i](1, 1) * pt3_current[0];
            A(2 * static_cast<Eigen::Index>(i), 4) = nullspaces[i](1, 0) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i) + 1, 4) = nullspaces[i](1, 1) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i), 5) = nullspaces[i](1, 0) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i) + 1, 5) = nullspaces[i](1, 1) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i), 6) = nullspaces[i](2, 0) * pt3_current[0];
            A(2 * static_cast<Eigen::Index>(i) + 1, 6) = nullspaces[i](2, 1) * pt3_current[0];
            A(2 * static_cast<Eigen::Index>(i), 7) = nullspaces[i](2, 0) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i) + 1, 7) = nullspaces[i](2, 1) * pt3_current[1];
            A(2 * static_cast<Eigen::Index>(i), 8) = nullspaces[i](2, 0) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i) + 1, 8) = nullspaces[i](2, 1) * pt3_current[2];
            A(2 * static_cast<Eigen::Index>(i), 9) = nullspaces[i](0, 0);
            A(2 * static_cast<Eigen::Index>(i) + 1, 9) = nullspaces[i](0, 1);
            A(2 * static_cast<Eigen::Index>(i), 10) = nullspaces[i](1, 0);
            A(2 * static_cast<Eigen::Index>(i) + 1, 10) = nullspaces[i](1, 1);
            A(2 * static_cast<Eigen::Index>(i), 11) = nullspaces[i](2, 0);
            A(2 * static_cast<Eigen::Index>(i) + 1, 11) = nullspaces[i](2, 1);
        }
    }

    // 最小二乘求解
    Eigen::MatrixXd AtPA;
    if (use_cov)
        AtPA = A.transpose() * P * A;
    else
        AtPA = A.transpose() * A;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(AtPA, Eigen::ComputeFullV);
    Eigen::MatrixXd result1 = svd_A.matrixV().col(colsA - 1);

    rotation_t Rout;
    translation_t tout;

    if (planar) {
        rotation_t tmp;
        tmp << 0.0, result1(0, 0), result1(1, 0),
               0.0, result1(2, 0), result1(3, 0),
               0.0, result1(4, 0), result1(5, 0);
        tmp.col(0) = tmp.col(1).cross(tmp.col(2));
        tmp.transposeInPlace();

        double scale = 1.0 / std::sqrt(std::abs(tmp.col(1).norm() * tmp.col(2).norm()));
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_R_frob(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
        rotation_t Rout1 = svd_R_frob.matrixU() * svd_R_frob.matrixV().transpose();
        if (Rout1.determinant() < 0)
            Rout1 *= -1.0;
        Rout1 = eigenRot.transpose() * Rout1;

        translation_t t = scale * translation_t(result1(6, 0), result1(7, 0), result1(8, 0));
        Rout1.transposeInPlace();
        Rout1 *= -1;
        if (Rout1.determinant() < 0.0)
            Rout1.col(2) *= -1;

        rotation_t R1, R2;
        R1.col(0) = Rout1.col(0);
        R1.col(1) = Rout1.col(1);
        R1.col(2) = Rout1.col(2);
        R2.col(0) = -Rout1.col(0);
        R2.col(1) = -Rout1.col(1);
        R2.col(2) = Rout1.col(2);

        std::vector<transformation_t, Eigen::aligned_allocator<transformation_t>> Ts(4);
        Ts[0].block<3, 3>(0, 0) = R1;
        Ts[0].block<3, 1>(0, 3) = t;
        Ts[1].block<3, 3>(0, 0) = R1;
        Ts[1].block<3, 1>(0, 3) = -t;
        Ts[2].block<3, 3>(0, 0) = R2;
        Ts[2].block<3, 1>(0, 3) = t;
        Ts[3].block<3, 3>(0, 0) = R2;
        Ts[3].block<3, 1>(0, 3) = -t;

        std::vector<double> normVal(4);
        for (int i = 0; i < 4; ++i) {
            point_t reproPt;
            double norms = 0.0;
            for (int p_idx = 0; p_idx < 6; ++p_idx) {
                reproPt = Ts[i].block<3, 3>(0, 0) * points3v[p_idx] + Ts[i].block<3, 1>(0, 3);
                reproPt = reproPt / reproPt.norm();
                norms += (1.0 - reproPt.transpose() * f[indices[p_idx]]);
            }
            normVal[i] = norms;
        }
        std::vector<double>::iterator findMinRepro = std::min_element(std::begin(normVal), std::end(normVal));
        int idx_min = static_cast<int>(std::distance(std::begin(normVal), findMinRepro));
        Rout = Ts[idx_min].block<3, 3>(0, 0);
        tout = Ts[idx_min].block<3, 1>(0, 3);
    } else {
        rotation_t tmp;
        tmp << result1(0, 0), result1(3, 0), result1(6, 0),
               result1(1, 0), result1(4, 0), result1(7, 0),
               result1(2, 0), result1(5, 0), result1(8, 0);
        double scale = 1.0 / std::pow(std::abs(tmp.col(0).norm() * tmp.col(1).norm() * tmp.col(2).norm()), 1.0 / 3.0);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_R_frob(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Rout = svd_R_frob.matrixU() * svd_R_frob.matrixV().transpose();
        if (Rout.determinant() < 0)
            Rout *= -1.0;
        tout = Rout * (scale * translation_t(result1(9, 0), result1(10, 0), result1(11, 0)));

        std::vector<double> error(2);
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Ts(2);
        for (int s = 0; s < 2; ++s) {
            error[s] = 0.0;
            Ts[s] = Eigen::Matrix4d::Identity();
            Ts[s].block<3, 3>(0, 0) = Rout;
            if (s == 0)
                Ts[s].block<3, 1>(0, 3) = tout;
            else
                Ts[s].block<3, 1>(0, 3) = -tout;
            Ts[s] = Ts[s].inverse().eval();
            for (int p_idx = 0; p_idx < 6; ++p_idx) {
                bearingVector_t v = Ts[s].block<3, 3>(0, 0) * points3v[p_idx] + Ts[s].block<3, 1>(0, 3);
                v = v / v.norm();
                error[s] += (1.0 - v.transpose() * f[indices[p_idx]]);
            }
        }
        if (error[0] < error[1])
            tout = Ts[0].block<3, 1>(0, 3);
        else
            tout = Ts[1].block<3, 1>(0, 3);
        Rout = Ts[0].block<3, 3>(0, 0);
    }

    // 高斯-牛顿优化
    rodrigues_t omega = rot2rodrigues(Rout);
    Eigen::VectorXd minx(6);
    minx[0] = omega[0];
    minx[1] = omega[1];
    minx[2] = omega[2];
    minx[3] = tout[0];
    minx[4] = tout[1];
    minx[5] = tout[2];

    mlpnp_gn(minx, points3v, nullspaces, P, use_cov);

    Rout = rodrigues2rot(rodrigues_t(minx[0], minx[1], minx[2]));
    tout = translation_t(minx[3], minx[4], minx[5]);

    result.block<3, 3>(0, 0) = Rout;
    result.block<3, 1>(0, 3) = tout;
}

// ==================== Rodrigues 转换 ====================

Eigen::Matrix3d MLPNPSolver::rodrigues2rot(const Eigen::Vector3d& omega) {
    rotation_t R = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d skewW;
    skewW << 0.0, -omega(2), omega(1),
             omega(2), 0.0, -omega(0),
            -omega(1), omega(0), 0.0;

    double omega_norm = omega.norm();

    if (omega_norm > std::numeric_limits<double>::epsilon())
        R = R + std::sin(omega_norm) / omega_norm * skewW
                + (1 - std::cos(omega_norm)) / (omega_norm * omega_norm) * (skewW * skewW);

    return R;
}

Eigen::Vector3d MLPNPSolver::rot2rodrigues(const Eigen::Matrix3d& R) {
    rodrigues_t omega;
    omega << 0.0, 0.0, 0.0;

    double trace = R.trace() - 1.0;
    // ★★★ Bug Fix: acos参数需要clamp到[-1,1] ★★★
    // 由于浮点误差，trace/2可能略微超过[-1,1]范围
    double trace_clamped = std::max(-1.0, std::min(1.0, trace / 2.0));
    double wnorm = std::acos(trace_clamped);
    
    if (wnorm > std::numeric_limits<double>::epsilon()) {
        omega[0] = (R(2, 1) - R(1, 2));
        omega[1] = (R(0, 2) - R(2, 0));
        omega[2] = (R(1, 0) - R(0, 1));
        double sc = wnorm / (2.0 * std::sin(wnorm));
        omega *= sc;
    }
    // else: wnorm接近0，返回零向量(无旋转)，符合预期
    return omega;
}

// ==================== 高斯-牛顿优化 ====================

void MLPNPSolver::mlpnp_gn(Eigen::VectorXd& x,
                            const points_t& pts,
                            const std::vector<Eigen::MatrixXd>& nullspaces,
                            const Eigen::SparseMatrix<double> Kll,
                            bool use_cov) {
    const int numObservations = static_cast<int>(pts.size());
    const int numUnknowns = 6;

    Eigen::VectorXd r(2 * numObservations);
    Eigen::VectorXd rd(2 * numObservations);
    Eigen::MatrixXd Jac(2 * numObservations, numUnknowns);
    Eigen::VectorXd g(numUnknowns, 1);
    Eigen::VectorXd dx(numUnknowns, 1);

    Jac.setZero();
    r.setZero();
    dx.setZero();
    g.setZero();

    int it_cnt = 0;
    bool stop = false;
    const int maxIt = 5;
    const double epsP = 1e-5;

    Eigen::MatrixXd JacTSKll;
    Eigen::MatrixXd A;

    while (it_cnt < maxIt && !stop) {
        mlpnp_residuals_and_jacs(x, pts, nullspaces, r, Jac, true);

        if (use_cov)
            JacTSKll = Jac.transpose() * Kll;
        else
            JacTSKll = Jac.transpose();

        A = JacTSKll * Jac;
        g = JacTSKll * r;

        Eigen::LDLT<Eigen::MatrixXd> chol(A);
        dx = chol.solve(g);

        if (dx.array().abs().maxCoeff() > 5.0 || dx.array().abs().minCoeff() > 1.0)
            break;

        Eigen::MatrixXd dl = Jac * dx;
        if (dl.array().abs().maxCoeff() < epsP) {
            stop = true;
            x = x - dx;
            break;
        } else {
            x = x - dx;
        }
        ++it_cnt;
    }
}

// ==================== 残差和雅可比计算 ====================

void MLPNPSolver::mlpnp_residuals_and_jacs(const Eigen::VectorXd& x,
                                           const points_t& pts,
                                           const std::vector<Eigen::MatrixXd>& nullspaces,
                                           Eigen::VectorXd& r,
                                           Eigen::MatrixXd& fjac,
                                           bool getJacs) {
    rodrigues_t w(x[0], x[1], x[2]);
    translation_t T(x[3], x[4], x[5]);

    rotation_t R = rodrigues2rot(w);
    int ii = 0;

    Eigen::MatrixXd jacs(2, 6);

    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        Eigen::Vector3d ptCam = R * pts[i] + T;
        ptCam /= ptCam.norm();

        r[ii] = nullspaces[i].col(0).transpose() * ptCam;
        r[ii + 1] = nullspaces[i].col(1).transpose() * ptCam;

        if (getJacs) {
            mlpnpJacs(pts[i], nullspaces[i].col(0), nullspaces[i].col(1), w, T, jacs);

            fjac(ii, 0) = jacs(0, 0);
            fjac(ii, 1) = jacs(0, 1);
            fjac(ii, 2) = jacs(0, 2);
            fjac(ii, 3) = jacs(0, 3);
            fjac(ii, 4) = jacs(0, 4);
            fjac(ii, 5) = jacs(0, 5);

            fjac(ii + 1, 0) = jacs(1, 0);
            fjac(ii + 1, 1) = jacs(1, 1);
            fjac(ii + 1, 2) = jacs(1, 2);
            fjac(ii + 1, 3) = jacs(1, 3);
            fjac(ii + 1, 4) = jacs(1, 4);
            fjac(ii + 1, 5) = jacs(1, 5);
        }
        ii += 2;
    }
}

// ==================== MLPnP 雅可比矩阵 ====================

void MLPNPSolver::mlpnpJacs(const point_t& pt,
                             const Eigen::Vector3d& nullspace_r,
                             const Eigen::Vector3d& nullspace_s,
                             const rodrigues_t& w,
                             const translation_t& t,
                             Eigen::MatrixXd& jacs) {
    double r1 = nullspace_r[0];
    double r2 = nullspace_r[1];
    double r3 = nullspace_r[2];

    double s1 = nullspace_s[0];
    double s2 = nullspace_s[1];
    double s3 = nullspace_s[2];

    double X1 = pt[0];
    double Y1 = pt[1];
    double Z1 = pt[2];

    double w1 = w[0];
    double w2 = w[1];
    double w3 = w[2];

    double t1 = t[0];
    double t2 = t[1];
    double t3 = t[2];

    double t5 = w1 * w1;
    double t6 = w2 * w2;
    double t7 = w3 * w3;
    double t8 = t5 + t6 + t7;
    double t9 = std::sqrt(t8);
    double t10 = std::sin(t9);
    double t11 = 1.0 / std::sqrt(t8);
    double t12 = std::cos(t9);
    double t13 = t12 - 1.0;
    double t14 = 1.0 / t8;
    double t16 = t10 * t11 * w3;
    double t17 = t13 * t14 * w1 * w2;
    double t19 = t10 * t11 * w2;
    double t20 = t13 * t14 * w1 * w3;
    double t24 = t6 + t7;
    double t27 = t16 + t17;
    double t28 = Y1 * t27;
    double t29 = t19 - t20;
    double t30 = Z1 * t29;
    double t31 = t13 * t14 * t24;
    double t32 = t31 + 1.0;
    double t33 = X1 * t32;
    double t15 = t1 - t28 + t30 + t33;
    double t21 = t10 * t11 * w1;
    double t22 = t13 * t14 * w2 * w3;
    double t45 = t5 + t7;
    double t53 = t16 - t17;
    double t54 = X1 * t53;
    double t55 = t21 + t22;
    double t56 = Z1 * t55;
    double t57 = t13 * t14 * t45;
    double t58 = t57 + 1.0;
    double t59 = Y1 * t58;
    double t18 = t2 + t54 - t56 + t59;
    double t34 = t5 + t6;
    double t38 = t19 + t20;
    double t39 = X1 * t38;
    double t40 = t21 - t22;
    double t41 = Y1 * t40;
    double t42 = t13 * t14 * t34;
    double t43 = t42 + 1.0;
    double t44 = Z1 * t43;
    double t23 = t3 - t39 + t41 + t44;
    double t25 = 1.0 / std::pow(t8, 3.0 / 2.0);
    double t26 = 1.0 / (t8 * t8);
    double t35 = t12 * t14 * w1 * w2;
    double t36 = t5 * t10 * t25 * w3;
    double t37 = t5 * t13 * t26 * w3 * 2.0;
    double t46 = t10 * t25 * w1 * w3;
    double t47 = t5 * t10 * t25 * w2;
    double t48 = t5 * t13 * t26 * w2 * 2.0;
    double t49 = t10 * t25;
    double t50 = t5 * t12 * t14;
    double t51 = t13 * t26 * w1 * w2 * w3 * 2.0;
    double t52 = t10 * t25 * w1 * w2 * w3;
    double t60 = t15 * t15;
    double t61 = t18 * t18;
    double t62 = t23 * t23;
    double t63 = t60 + t61 + t62;
    double t64 = t5 * t10 * t25;
    double t65 = 1.0 / std::sqrt(t63);
    double t66 = Y1 * r2 * t6;
    double t67 = Z1 * r3 * t7;
    double t68 = r1 * t1 * t5;
    double t69 = r1 * t1 * t6;
    double t70 = r1 * t1 * t7;
    double t71 = r2 * t2 * t5;
    double t72 = r2 * t2 * t6;
    double t73 = r2 * t2 * t7;
    double t74 = r3 * t3 * t5;
    double t75 = r3 * t3 * t6;
    double t76 = r3 * t3 * t7;
    double t77 = X1 * r1 * t5;
    double t78 = X1 * r2 * w1 * w2;
    double t79 = X1 * r3 * w1 * w3;
    double t80 = Y1 * r1 * w1 * w2;
    double t81 = Y1 * r3 * w2 * w3;
    double t82 = Z1 * r1 * w1 * w3;
    double t83 = Z1 * r2 * w2 * w3;
    double t84 = X1 * r1 * t6 * t12;
    double t85 = X1 * r1 * t7 * t12;
    double t86 = Y1 * r2 * t5 * t12;
    double t87 = Y1 * r2 * t7 * t12;
    double t88 = Z1 * r3 * t5 * t12;
    double t89 = Z1 * r3 * t6 * t12;
    double t90 = X1 * r2 * t9 * t10 * w3;
    double t91 = Y1 * r3 * t9 * t10 * w1;
    double t92 = Z1 * r1 * t9 * t10 * w2;
    double t102 = X1 * r3 * t9 * t10 * w2;
    double t103 = Y1 * r1 * t9 * t10 * w3;
    double t104 = Z1 * r2 * t9 * t10 * w1;
    double t105 = X1 * r2 * t12 * w1 * w2;
    double t106 = X1 * r3 * t12 * w1 * w3;
    double t107 = Y1 * r1 * t12 * w1 * w2;
    double t108 = Y1 * r3 * t12 * w2 * w3;
    double t109 = Z1 * r1 * t12 * w1 * w3;
    double t110 = Z1 * r2 * t12 * w2 * w3;
    double t93 = t66 + t67 + t68 + t69 + t70 + t71 + t72 + t73 + t74 + t75 + t76 + t77 + t78 + t79 + t80 + t81 + t82 + t83 + t84 + t85 + t86 + t87 + t88 + t89 + t90 + t91 + t92 - t102 - t103 - t104 - t105 - t106 - t107 - t108 - t109 - t110;
    double t94 = t10 * t25 * w1 * w2;
    double t95 = t6 * t10 * t25 * w3;
    double t96 = t6 * t13 * t26 * w3 * 2.0;
    double t97 = t12 * t14 * w2 * w3;
    double t98 = t6 * t10 * t25 * w1;
    double t99 = t6 * t13 * t26 * w1 * 2.0;
    double t100 = t6 * t10 * t25;
    double t101 = 1.0 / std::pow(t63, 3.0 / 2.0);
    double t111 = t6 * t12 * t14;
    double t112 = t10 * t25 * w2 * w3;
    double t113 = t12 * t14 * w1 * w3;
    double t114 = t7 * t10 * t25 * w2;
    double t115 = t7 * t13 * t26 * w2 * 2.0;
    double t116 = t7 * t10 * t25 * w1;
    double t117 = t7 * t13 * t26 * w1 * 2.0;
    double t118 = t7 * t12 * t14;
    double t119 = t13 * t24 * t26 * w1 * 2.0;
    double t120 = t10 * t24 * t25 * w1;
    double t121 = t119 + t120;
    double t122 = t13 * t26 * t34 * w1 * 2.0;
    double t123 = t10 * t25 * t34 * w1;
    double t131 = t13 * t14 * w1 * 2.0;
    double t124 = t122 + t123 - t131;
    double t139 = t13 * t14 * w3;
    double t125 = -t35 + t36 + t37 + t94 - t139;
    double t126 = X1 * t125;
    double t127 = t49 + t50 + t51 + t52 - t64;
    double t128 = Y1 * t127;
    double t129 = t126 + t128 - Z1 * t124;
    double t130 = t23 * t129 * 2.0;
    double t132 = t13 * t26 * t45 * w1 * 2.0;
    double t133 = t10 * t25 * t45 * w1;
    double t138 = t13 * t14 * w2;
    double t134 = -t46 + t47 + t48 + t113 - t138;
    double t135 = X1 * t134;
    double t136 = -t49 - t50 + t51 + t52 + t64;
    double t137 = Z1 * t136;
    double t140 = X1 * s1 * t5;
    double t141 = Y1 * s2 * t6;
    double t142 = Z1 * s3 * t7;
    double t143 = s1 * t1 * t5;
    double t144 = s1 * t1 * t6;
    double t145 = s1 * t1 * t7;
    double t146 = s2 * t2 * t5;
    double t147 = s2 * t2 * t6;
    double t148 = s2 * t2 * t7;
    double t149 = s3 * t3 * t5;
    double t150 = s3 * t3 * t6;
    double t151 = s3 * t3 * t7;
    double t152 = X1 * s2 * w1 * w2;
    double t153 = X1 * s3 * w1 * w3;
    double t154 = Y1 * s1 * w1 * w2;
    double t155 = Y1 * s3 * w2 * w3;
    double t156 = Z1 * s1 * w1 * w3;
    double t157 = Z1 * s2 * w2 * w3;
    double t158 = X1 * s1 * t6 * t12;
    double t159 = X1 * s1 * t7 * t12;
    double t160 = Y1 * s2 * t5 * t12;
    double t161 = Y1 * s2 * t7 * t12;
    double t162 = Z1 * s3 * t5 * t12;
    double t163 = Z1 * s3 * t6 * t12;
    double t164 = X1 * s2 * t9 * t10 * w3;
    double t165 = Y1 * s3 * t9 * t10 * w1;
    double t166 = Z1 * s1 * t9 * t10 * w2;
    double t183 = X1 * s3 * t9 * t10 * w2;
    double t184 = Y1 * s1 * t9 * t10 * w3;
    double t185 = Z1 * s2 * t9 * t10 * w1;
    double t186 = X1 * s2 * t12 * w1 * w2;
    double t187 = X1 * s3 * t12 * w1 * w3;
    double t188 = Y1 * s1 * t12 * w1 * w2;
    double t189 = Y1 * s3 * t12 * w2 * w3;
    double t190 = Z1 * s1 * t12 * w1 * w3;
    double t191 = Z1 * s2 * t12 * w2 * w3;
    double t167 = t140 + t141 + t142 + t143 + t144 + t145 + t146 + t147 + t148 + t149 + t150 + t151 + t152 + t153 + t154 + t155 + t156 + t157 + t158 + t159 + t160 + t161 + t162 + t163 + t164 + t165 + t166 - t183 - t184 - t185 - t186 - t187 - t188 - t189 - t190 - t191;
    double t168 = t13 * t26 * t45 * w2 * 2.0;
    double t169 = t10 * t25 * t45 * w2;
    double t170 = t168 + t169;
    double t171 = t13 * t26 * t34 * w2 * 2.0;
    double t172 = t10 * t25 * t34 * w2;
    double t176 = t13 * t14 * w2 * 2.0;
    double t173 = t171 + t172 - t176;
    double t174 = -t49 + t51 + t52 + t100 - t111;
    double t175 = X1 * t174;
    double t177 = t13 * t24 * t26 * w2 * 2.0;
    double t178 = t10 * t24 * t25 * w2;
    double t192 = t13 * t14 * w1;
    double t179 = -t97 + t98 + t99 + t112 - t192;
    double t180 = Y1 * t179;
    double t181 = t49 + t51 + t52 - t100 + t111;
    double t182 = Z1 * t181;
    double t193 = t13 * t26 * t34 * w3 * 2.0;
    double t194 = t10 * t25 * t34 * w3;
    double t195 = t193 + t194;
    double t196 = t13 * t26 * t45 * w3 * 2.0;
    double t197 = t10 * t25 * t45 * w3;
    double t200 = t13 * t14 * w3 * 2.0;
    double t198 = t196 + t197 - t200;
    double t199 = t7 * t10 * t25;
    double t201 = t13 * t24 * t26 * w3 * 2.0;
    double t202 = t10 * t24 * t25 * w3;
    double t203 = -t49 + t51 + t52 - t118 + t199;
    double t204 = Y1 * t203;
    double t205 = t1 * 2.0;
    double t206 = Z1 * t29 * 2.0;
    double t207 = X1 * t32 * 2.0;
    double t208 = t205 + t206 + t207 - Y1 * t27 * 2.0;
    double t209 = t2 * 2.0;
    double t210 = X1 * t53 * 2.0;
    double t211 = Y1 * t58 * 2.0;
    double t212 = t209 + t210 + t211 - Z1 * t55 * 2.0;
    double t213 = t3 * 2.0;
    double t214 = Y1 * t40 * 2.0;
    double t215 = Z1 * t43 * 2.0;
    double t216 = t213 + t214 + t215 - X1 * t38 * 2.0;

    jacs(0, 0) = t14 * t65 * (X1 * r1 * w1 * 2.0 + X1 * r2 * w2 + X1 * r3 * w3 + Y1 * r1 * w2 + Z1 * r1 * w3 + r1 * t1 * w1 * 2.0 + r2 * t2 * w1 * 2.0 + r3 * t3 * w1 * 2.0 + Y1 * r3 * t5 * t12 + Y1 * r3 * t9 * t10 - Z1 * r2 * t5 * t12 - Z1 * r2 * t9 * t10 - X1 * r2 * t12 * w2 - X1 * r3 * t12 * w3 - Y1 * r1 * t12 * w2 + Y1 * r2 * t12 * w1 * 2.0 - Z1 * r1 * t12 * w3 + Z1 * r3 * t12 * w1 * 2.0 + Y1 * r3 * t5 * t10 * t11 - Z1 * r2 * t5 * t10 * t11 + X1 * r2 * t12 * w1 * w3 - X1 * r3 * t12 * w1 * w2 - Y1 * r1 * t12 * w1 * w3 + Z1 * r1 * t12 * w1 * w2 - Y1 * r1 * t10 * t11 * w1 * w3 + Z1 * r1 * t10 * t11 * w1 * w2 - X1 * r1 * t6 * t10 * t11 * w1 - X1 * r1 * t7 * t10 * t11 * w1 + X1 * r2 * t5 * t10 * t11 * w2 + X1 * r3 * t5 * t10 * t11 * w3 + Y1 * r1 * t5 * t10 * t11 * w2 - Y1 * r2 * t5 * t10 * t11 * w1 - Y1 * r2 * t7 * t10 * t11 * w1 + Z1 * r1 * t5 * t10 * t11 * w3 - Z1 * r3 * t5 * t10 * t11 * w1 - Z1 * r3 * t6 * t10 * t11 * w1 + X1 * r2 * t10 * t11 * w1 * w3 - X1 * r3 * t10 * t11 * w1 * w2 + Y1 * r3 * t10 * t11 * w1 * w2 * w3 + Z1 * r2 * t10 * t11 * w1 * w2 * w3) - t26 * t65 * t93 * w1 * 2.0 - t14 * t93 * t101 * (t130 + t15 * (-X1 * t121 + Y1 * (t46 + t47 + t48 - t13 * t14 * w2 - t12 * t14 * w1 * w3) + Z1 * (t35 + t36 + t37 - t13 * t14 * w3 - t10 * t25 * w1 * w2)) * 2.0 + t18 * (t135 + t137 - Y1 * (t132 + t133 - t13 * t14 * w1 * 2.0)) * 2.0) * (1.0 / 2.0);

    jacs(0, 1) = t14 * t65 * (X1 * r2 * w1 + Y1 * r1 * w1 + Y1 * r2 * w2 * 2.0 + Y1 * r3 * w3 + Z1 * r2 * w3 + r1 * t1 * w2 * 2.0 + r2 * t2 * w2 * 2.0 + r3 * t3 * w2 * 2.0 - X1 * r3 * t6 * t12 - X1 * r3 * t9 * t10 + Z1 * r1 * t6 * t12 + Z1 * r1 * t9 * t10 + X1 * r1 * t12 * w2 * 2.0 - X1 * r2 * t12 * w1 - Y1 * r1 * t12 * w1 - Y1 * r3 * t12 * w3 - Z1 * r2 * t12 * w3 + Z1 * r3 * t12 * w2 * 2.0 - X1 * r3 * t6 * t10 * t11 + Z1 * r1 * t6 * t10 * t11 + X1 * r2 * t12 * w2 * w3 - Y1 * r1 * t12 * w2 * w3 + Y1 * r3 * t12 * w1 * w2 - Z1 * r2 * t12 * w1 * w2 - Y1 * r1 * t10 * t11 * w2 * w3 + Y1 * r3 * t10 * t11 * w1 * w2 - Z1 * r2 * t10 * t11 * w1 * w2 - X1 * r1 * t6 * t10 * t11 * w2 + X1 * r2 * t6 * t10 * t11 * w1 - X1 * r1 * t7 * t10 * t11 * w2 + Y1 * r1 * t6 * t10 * t11 * w1 - Y1 * r2 * t5 * t10 * t11 * w2 - Y1 * r2 * t7 * t10 * t11 * w2 + Y1 * r3 * t6 * t10 * t11 * w3 - Z1 * r3 * t5 * t10 * t11 * w2 + Z1 * r2 * t6 * t10 * t11 * w3 - Z1 * r3 * t6 * t10 * t11 * w2 + X1 * r2 * t10 * t11 * w2 * w3 + X1 * r3 * t10 * t11 * w1 * w2 * w3 + Z1 * r1 * t10 * t11 * w1 * w2 * w3) - t26 * t65 * t93 * w2 * 2.0 - t14 * t93 * t101 * (t18 * (Z1 * (-t35 + t94 + t95 + t96 - t13 * t14 * w3) - Y1 * t170 + X1 * (t97 + t98 + t99 - t13 * t14 * w1 - t10 * t25 * w2 * w3)) * 2.0 + t15 * (t180 + t182 - X1 * (t177 + t178 - t13 * t14 * w2 * 2.0)) * 2.0 + t23 * (t175 + Y1 * (t35 - t94 + t95 + t96 - t13 * t14 * w3) - Z1 * t173) * 2.0) * (1.0 / 2.0);

    jacs(0, 2) = t14 * t65 * (X1 * r3 * w1 + Y1 * r3 * w2 + Z1 * r1 * w1 + Z1 * r2 * w2 + Z1 * r3 * w3 * 2.0 + r1 * t1 * w3 * 2.0 + r2 * t2 * w3 * 2.0 + r3 * t3 * w3 * 2.0 + X1 * r2 * t7 * t12 + X1 * r2 * t9 * t10 - Y1 * r1 * t7 * t12 - Y1 * r1 * t9 * t10 + X1 * r1 * t12 * w3 * 2.0 - X1 * r3 * t12 * w1 + Y1 * r2 * t12 * w3 * 2.0 - Y1 * r3 * t12 * w2 - Z1 * r1 * t12 * w1 - Z1 * r2 * t12 * w2 + X1 * r2 * t7 * t10 * t11 - Y1 * r1 * t7 * t10 * t11 - X1 * r3 * t12 * w2 * w3 + Y1 * r3 * t12 * w1 * w3 + Z1 * r1 * t12 * w2 * w3 - Z1 * r2 * t12 * w1 * w3 + Y1 * r3 * t10 * t11 * w1 * w3 + Z1 * r1 * t10 * t11 * w2 * w3 - Z1 * r2 * t10 * t11 * w1 * w3 - X1 * r1 * t6 * t10 * t11 * w3 - X1 * r1 * t7 * t10 * t11 * w3 + X1 * r3 * t7 * t10 * t11 * w1 - Y1 * r2 * t5 * t10 * t11 * w3 - Y1 * r2 * t7 * t10 * t11 * w3 + Y1 * r3 * t7 * t10 * t11 * w2 + Z1 * r1 * t7 * t10 * t11 * w1 + Z1 * r2 * t7 * t10 * t11 * w2 - Z1 * r3 * t5 * t10 * t11 * w3 - Z1 * r3 * t6 * t10 * t11 * w3 - X1 * r3 * t10 * t11 * w2 * w3 + X1 * r2 * t10 * t11 * w1 * w2 * w3 + Y1 * r1 * t10 * t11 * w1 * w2 * w3) - t26 * t65 * t93 * w3 * 2.0 - t14 * t93 * t101 * (t18 * (Z1 * (t46 - t113 + t114 + t115 - t13 * t14 * w2) - Y1 * t198 + X1 * (t49 + t51 + t52 + t118 - t7 * t10 * t25)) * 2.0 + t23 * (X1 * (-t97 + t112 + t116 + t117 - t13 * t14 * w1) + Y1 * (-t46 + t113 + t114 + t115 - t13 * t14 * w2) - Z1 * t195) * 2.0 + t15 * (t204 + Z1 * (t97 - t112 + t116 + t117 - t13 * t14 * w1) - X1 * (t201 + t202 - t13 * t14 * w3 * 2.0)) * 2.0) * (1.0 / 2.0);

    jacs(0, 3) = r1 * t65 - t14 * t93 * t101 * t208 * (1.0 / 2.0);
    jacs(0, 4) = r2 * t65 - t14 * t93 * t101 * t212 * (1.0 / 2.0);
    jacs(0, 5) = r3 * t65 - t14 * t93 * t101 * t216 * (1.0 / 2.0);

    jacs(1, 0) = t14 * t65 * (X1 * s1 * w1 * 2.0 + X1 * s2 * w2 + X1 * s3 * w3 + Y1 * s1 * w2 + Z1 * s1 * w3 + s1 * t1 * w1 * 2.0 + s2 * t2 * w1 * 2.0 + s3 * t3 * w1 * 2.0 + Y1 * s3 * t5 * t12 + Y1 * s3 * t9 * t10 - Z1 * s2 * t5 * t12 - Z1 * s2 * t9 * t10 - X1 * s2 * t12 * w2 - X1 * s3 * t12 * w3 - Y1 * s1 * t12 * w2 + Y1 * s2 * t12 * w1 * 2.0 - Z1 * s1 * t12 * w3 + Z1 * s3 * t12 * w1 * 2.0 + Y1 * s3 * t5 * t10 * t11 - Z1 * s2 * t5 * t10 * t11 + X1 * s2 * t12 * w1 * w3 - X1 * s3 * t12 * w1 * w2 - Y1 * s1 * t12 * w1 * w3 + Z1 * s1 * t12 * w1 * w2 + X1 * s2 * t10 * t11 * w1 * w3 - X1 * s3 * t10 * t11 * w1 * w2 - Y1 * s1 * t10 * t11 * w1 * w3 + Z1 * s1 * t10 * t11 * w1 * w2 - X1 * s1 * t6 * t10 * t11 * w1 - X1 * s1 * t7 * t10 * t11 * w1 + X1 * s2 * t5 * t10 * t11 * w2 + X1 * s3 * t5 * t10 * t11 * w3 + Y1 * s1 * t5 * t10 * t11 * w2 - Y1 * s2 * t5 * t10 * t11 * w1 - Y1 * s2 * t7 * t10 * t11 * w1 + Z1 * s1 * t5 * t10 * t11 * w3 - Z1 * s3 * t5 * t10 * t11 * w1 - Z1 * s3 * t6 * t10 * t11 * w1 + Y1 * s3 * t10 * t11 * w1 * w2 * w3 + Z1 * s2 * t10 * t11 * w1 * w2 * w3) - t14 * t101 * t167 * (t130 + t15 * (Y1 * (t46 + t47 + t48 - t113 - t138) + Z1 * (t35 + t36 + t37 - t94 - t139) - X1 * t121) * 2.0 + t18 * (t135 + t137 - Y1 * (-t131 + t132 + t133)) * 2.0) * (1.0 / 2.0) - t26 * t65 * t167 * w1 * 2.0;

    jacs(1, 1) = t14 * t65 * (X1 * s2 * w1 + Y1 * s1 * w1 + Y1 * s2 * w2 * 2.0 + Y1 * s3 * w3 + Z1 * s2 * w3 + s1 * t1 * w2 * 2.0 + s2 * t2 * w2 * 2.0 + s3 * t3 * w2 * 2.0 - X1 * s3 * t6 * t12 - X1 * s3 * t9 * t10 + Z1 * s1 * t6 * t12 + Z1 * s1 * t9 * t10 + X1 * s1 * t12 * w2 * 2.0 - X1 * s2 * t12 * w1 - Y1 * s1 * t12 * w1 - Y1 * s3 * t12 * w3 - Z1 * s2 * t12 * w3 + Z1 * s3 * t12 * w2 * 2.0 - X1 * s3 * t6 * t10 * t11 + Z1 * s1 * t6 * t10 * t11 + X1 * s2 * t12 * w2 * w3 - Y1 * s1 * t12 * w2 * w3 + Y1 * s3 * t12 * w1 * w2 - Z1 * s2 * t12 * w1 * w2 + X1 * s2 * t10 * t11 * w2 * w3 - Y1 * s1 * t10 * t11 * w2 * w3 + Y1 * s3 * t10 * t11 * w1 * w2 - Z1 * s2 * t10 * t11 * w1 * w2 - X1 * s1 * t6 * t10 * t11 * w2 + X1 * s2 * t6 * t10 * t11 * w1 - X1 * s1 * t7 * t10 * t11 * w2 + Y1 * s1 * t6 * t10 * t11 * w1 - Y1 * s2 * t5 * t10 * t11 * w2 - Y1 * s2 * t7 * t10 * t11 * w2 + Y1 * s3 * t6 * t10 * t11 * w3 - Z1 * s3 * t5 * t10 * t11 * w2 + Z1 * s2 * t6 * t10 * t11 * w3 - Z1 * s3 * t6 * t10 * t11 * w2 + X1 * s3 * t10 * t11 * w1 * w2 * w3 + Z1 * s1 * t10 * t11 * w1 * w2 * w3) - t26 * t65 * t167 * w2 * 2.0 - t14 * t101 * t167 * (t18 * (X1 * (t97 + t98 + t99 - t112 - t192) + Z1 * (-t35 + t94 + t95 + t96 - t139) - Y1 * t170) * 2.0 + t15 * (t180 + t182 - X1 * (-t176 + t177 + t178)) * 2.0 + t23 * (t175 + Y1 * (t35 - t94 + t95 + t96 - t139) - Z1 * t173) * 2.0) * (1.0 / 2.0);

    jacs(1, 2) = t14 * t65 * (X1 * s3 * w1 + Y1 * s3 * w2 + Z1 * s1 * w1 + Z1 * s2 * w2 + Z1 * s3 * w3 * 2.0 + s1 * t1 * w3 * 2.0 + s2 * t2 * w3 * 2.0 + s3 * t3 * w3 * 2.0 + X1 * s2 * t7 * t12 + X1 * s2 * t9 * t10 - Y1 * s1 * t7 * t12 - Y1 * s1 * t9 * t10 + X1 * s1 * t12 * w3 * 2.0 - X1 * s3 * t12 * w1 + Y1 * s2 * t12 * w3 * 2.0 - Y1 * s3 * t12 * w2 - Z1 * s1 * t12 * w1 - Z1 * s2 * t12 * w2 + X1 * s2 * t7 * t10 * t11 - Y1 * s1 * t7 * t10 * t11 - X1 * s3 * t12 * w2 * w3 + Y1 * s3 * t12 * w1 * w3 + Z1 * s1 * t12 * w2 * w3 - Z1 * s2 * t12 * w1 * w3 - X1 * s3 * t10 * t11 * w2 * w3 + Y1 * s3 * t10 * t11 * w1 * w3 + Z1 * s1 * t10 * t11 * w2 * w3 - Z1 * s2 * t10 * t11 * w1 * w3 - X1 * s1 * t6 * t10 * t11 * w3 - X1 * s1 * t7 * t10 * t11 * w3 + X1 * s3 * t7 * t10 * t11 * w1 - Y1 * s2 * t5 * t10 * t11 * w3 - Y1 * s2 * t7 * t10 * t11 * w3 + Y1 * s3 * t7 * t10 * t11 * w2 + Z1 * s1 * t7 * t10 * t11 * w1 + Z1 * s2 * t7 * t10 * t11 * w2 - Z1 * s3 * t5 * t10 * t11 * w3 - Z1 * s3 * t6 * t10 * t11 * w3 + X1 * s2 * t10 * t11 * w1 * w2 * w3 + Y1 * s1 * t10 * t11 * w1 * w2 * w3) - t26 * t65 * t167 * w3 * 2.0 - t14 * t101 * t167 * (t18 * (Z1 * (t46 - t113 + t114 + t115 - t138) - Y1 * t198 + X1 * (t49 + t51 + t52 + t118 - t199)) * 2.0 + t23 * (X1 * (-t97 + t112 + t116 + t117 - t192) + Y1 * (-t46 + t113 + t114 + t115 - t138) - Z1 * t195) * 2.0 + t15 * (t204 + Z1 * (t97 - t112 + t116 + t117 - t192) - X1 * (-t200 + t201 + t202)) * 2.0) * (1.0 / 2.0);

    jacs(1, 3) = s1 * t65 - t14 * t101 * t167 * t208 * (1.0 / 2.0);
    jacs(1, 4) = s2 * t65 - t14 * t101 * t167 * t212 * (1.0 / 2.0);
    jacs(1, 5) = s3 * t65 - t14 * t101 * t167 * t216 * (1.0 / 2.0);
}

// ==================== 便捷接口 ====================

bool MLPNPSolver::solve(const points_t& points3d,
                        const bearingVectors_t& bearings,
                        const std::vector<cv::Point2f>& pixels,
                        Eigen::Matrix4f& output_pose,
                        std::vector<bool>& inliers,
                        int& n_inliers) {
    if (points3d.size() < 6) {
        return false;
    }

    std::vector<int> keypoint_indices(points3d.size());
    std::vector<float> sigma2(points3d.size(), 1.0f);
    for (size_t i = 0; i < points3d.size(); ++i) {
        keypoint_indices[i] = static_cast<int>(i);
    }

    SetData(bearings, points3d, pixels, keypoint_indices, sigma2, mK);

    SetRansacParameters(config_.probability, config_.min_inliers,
                       config_.max_iterations, config_.min_set,
                       config_.epsilon, config_.threshold);

    bool bNoMore = false;
    Eigen::Matrix4f T_out;
    bool ok = iterate(5, bNoMore, inliers, n_inliers, T_out);

    if (ok) {
        output_pose = T_out;
        return true;
    }

    return false;
}

} // namespace frontend
} // namespace sfc_core
