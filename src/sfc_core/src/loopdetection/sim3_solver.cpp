// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/loopdetection/sim3_solver.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace sfc_core {
namespace loopdetection {

// ================================================================
// Sim3Solver Implementation
// 参考 ORB_SLAM3/src/Sim3Solver.cc
// ================================================================

Sim3Solver::Sim3Solver(const std::vector<Eigen::Vector3d>& kf1_pts3d_cam,
                       const std::vector<Eigen::Vector3d>& kf2_pts3d_cam,
                       const std::vector<cv::DMatch>& matches,
                       const cv::Mat& K,
                       bool fixScale)
    : mbFixScale_(fixScale) {

    // 提取相机内参
    if (!K.empty() && K.rows == 3 && K.cols == 3) {
        fx_ = static_cast<float>(K.at<double>(0, 0));
        fy_ = static_cast<float>(K.at<double>(1, 1));
        cx_ = static_cast<float>(K.at<double>(0, 2));
        cy_ = static_cast<float>(K.at<double>(1, 2));
    } else {
        fx_ = fy_ = 525.0f;
        cx_ = 319.5f;
        cy_ = 239.5f;
    }

    // 初始化默认值（防止野值）
    mnBestInliers_ = 0;
    mnInliersi_ = 0;
    ms12i = 1.0f;
    mBestScale_ = 1.0f;
    mvbBestInliers_.clear();
    mvbInliersi_.clear();

    mvX3Dc1_ = kf1_pts3d_cam;
    mvX3Dc2_ = kf2_pts3d_cam;
    mMatches_ = matches;

    N_ = static_cast<int>(matches.size());
    mvAllIndices_.reserve(N_);
    mvnIndices1_.reserve(N_);

    // 预计算每个匹配点的图像坐标和误差阈值
    for (int i = 0; i < N_; ++i) {
        mvAllIndices_.push_back(i);
        mvnIndices1_.push_back(i);

        // 扩大阈值以适应 prefill 共视匹配的叠加噪声（位姿漂移 + 重投影用旧位姿）
        // backfill 的重投影误差比正常共视匹配更大（没有经过 BA 精化）
        // 20.0 px² ≈ 4.5 px RMS，在容忍噪声和保证验证严格性之间取得平衡
        mvnMaxError1_.push_back(20.0f);
        mvnMaxError2_.push_back(20.0f);
    }

    // 预计算匹配点的图像坐标（用于 CheckInliers）
    mvP1im1_.reserve(N_);
    mvP2im2_.reserve(N_);
    for (const auto& m : matches) {
        int idx1 = m.queryIdx;
        int idx2 = m.trainIdx;

        Eigen::Vector2f p1_im, p2_im;
        if (idx1 >= 0 && idx1 < static_cast<int>(kf1_pts3d_cam.size())) {
            const auto& pc1 = kf1_pts3d_cam[idx1];
            if (pc1.z() > 0.05f) {
                float invz1 = 1.0f / pc1.z();
                if (std::isfinite(invz1)) {
                    p1_im.x() = fx_ * pc1.x() * invz1 + cx_;
                    p1_im.y() = fy_ * pc1.y() * invz1 + cy_;
                }
            }
        }
        if (idx2 >= 0 && idx2 < static_cast<int>(kf2_pts3d_cam.size())) {
            const auto& pc2 = kf2_pts3d_cam[idx2];
            if (pc2.z() > 0.05f) {
                float invz2 = 1.0f / pc2.z();
                if (std::isfinite(invz2)) {
                    p2_im.x() = fx_ * pc2.x() * invz2 + cx_;
                    p2_im.y() = fy_ * pc2.y() * invz2 + cy_;
                }
            }
        }
        mvP1im1_.push_back(p1_im);
        mvP2im2_.push_back(p2_im);
    }
}

void Sim3Solver::setRansacParameters(double probability, int minInliers, int maxIterations) {
    mRansacProb_ = static_cast<float>(probability);
    mRansacMinInliers_ = minInliers;
    mRansacMaxIts_ = maxIterations;

    N_ = static_cast<int>(mMatches_.size());
    mvbInliersi_.assign(N_, false);

    // 根据内点率和概率自适应计算所需迭代次数
    // ★★★ P2-7: 修复 epsilon 计算错误 ★★★
    // 原代码 epsilon = minInliers / N_ 计算的是「内点率」
    // RANSAC 迭代公式中的 epsilon 应该是「外点率」，即 1 - 内点率
    float epsilon = (N_ > 0) ? 1.0f - static_cast<float>(minInliers) / N_ : 1.0f;
    // 防止外点率过小（此时公式中 log(1 - (1-ε)^3) 接近 0，导致迭代次数爆炸）
    if (epsilon < 0.01f) epsilon = 0.01f;
    int nIterations = 1;
    if (epsilon < 1.0f) {
        double log_prob = std::log(1.0 - std::pow(1.0 - epsilon, 3.0));
        if (log_prob < -50.0) log_prob = -50.0;  // 防止 exp 溢出
        nIterations = static_cast<int>(std::ceil(std::log(1.0 - mRansacProb_) / log_prob));
    }
    mRansacMaxIts_ = std::max(1, std::min(nIterations, maxIterations));
    mnIterations_ = 0;
}

void Sim3Solver::computeCentroid(Eigen::Matrix3f& P, Eigen::Matrix3f& Pr, Eigen::Vector3f& C) {
    C = P.rowwise().sum() / P.cols();
    for (int i = 0; i < P.cols(); ++i)
        Pr.col(i) = P.col(i) - C;
}

void Sim3Solver::computeSim3(Eigen::Matrix3f& P1, Eigen::Matrix3f& P2) {
    // ================================================================
    // Horn 1987, Closed-form solution of absolute orientation using unit quaternions
    // 输入：P1, P2 — 每列一个 3D 点（共 3 列，已去质心）
    // 输出：mR12i（旋转）, mt12i（平移）, ms12i（尺度）, mT12i / mT21i（4x4 矩阵）
    // ================================================================

    // Step 1: Compute centroid
    Eigen::Matrix3f Pr1, Pr2;
    Eigen::Vector3f O1, O2;
    computeCentroid(P1, Pr1, O1);
    computeCentroid(P2, Pr2, O2);

    // Step 2: Compute M matrix
    Eigen::Matrix3f M = Pr2 * Pr1.transpose();

    // Step 3: Compute N matrix (quaternion-based)
    // N = [ N11 N12 N13 N14
    //       N12 N22 N23 N24
    //       N13 N23 N33 N34
    //       N14 N24 N34 N44 ]
    Eigen::Matrix4f N = Eigen::Matrix4f::Zero();
    N(0, 0) = M(0, 0) + M(1, 1) + M(2, 2);
    N(1, 1) = M(0, 0) - M(1, 1) - M(2, 2);
    N(2, 2) = -M(0, 0) + M(1, 1) - M(2, 2);
    N(3, 3) = -M(0, 0) - M(1, 1) + M(2, 2);
    N(0, 1) = N(1, 0) = M(1, 2) - M(2, 1);
    N(0, 2) = N(2, 0) = M(2, 0) - M(0, 2);
    N(0, 3) = N(3, 0) = M(0, 1) - M(1, 0);
    N(1, 2) = N(2, 1) = M(0, 1) + M(1, 0);
    N(1, 3) = N(3, 1) = M(2, 0) + M(0, 2);
    N(2, 3) = N(3, 2) = M(1, 2) + M(2, 1);

    // Step 4: Eigenvector of the highest eigenvalue → 四元数提取
    // ★★★ 修复：参考 Horn 1987 + ORB_SLAM3 Sim3Solver.cc 第 362-368 行 ★★★
    // 关键修正：
    //   1. 找到最大特征值后，验证其对应的特征向量是否给出正确的旋转
    //   2. 如果 det(R) < 0，说明四元数符号需要翻转（q 和 -q 代表同一旋转）
    //   3. 用 SVD 正交化作为最终保障
    Eigen::EigenSolver<Eigen::Matrix4f> eigSolver(N);
    Eigen::Vector4f eval = eigSolver.eigenvalues().real();
    Eigen::Matrix4f evec = eigSolver.eigenvectors().real();

    int maxIndex = 0;
    eval.maxCoeff(&maxIndex);

    // 提取四元数 [q0, q1, q2, q3] — 注意：Eigen 列优先，evec.col(maxIndex) 得到特征向量
    Eigen::Vector4f q = evec.col(maxIndex).normalized();

    // 将四元数转换为旋转矩阵
    float q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);
    mR12i <<
        1-2*(q2*q2+q3*q3),  2*(q1*q2-q0*q3),  2*(q1*q3+q0*q2),
        2*(q1*q2+q0*q3),  1-2*(q1*q1+q3*q3),  2*(q2*q3-q0*q1),
        2*(q1*q3-q0*q2),  2*(q2*q3+q0*q1),  1-2*(q1*q1+q2*q2);

    // ★★★ 关键修复：检查旋转矩阵行列式 ★★★
    // 如果 det(R) < 0，说明四元数符号取反了（q 和 -q 是等价的旋转）
    // 翻转四元数符号即可修正（等价于对旋转轴取反）
    double det = mR12i.cast<double>().determinant();
    if (det < 0) {
        q0 = -q0; q1 = -q1; q2 = -q2; q3 = -q3;
        mR12i <<
            1-2*(q2*q2+q3*q3),  2*(q1*q2-q0*q3),  2*(q1*q3+q0*q2),
            2*(q1*q2+q0*q3),  1-2*(q1*q1+q3*q3),  2*(q2*q3-q0*q1),
            2*(q1*q3-q0*q2),  2*(q2*q3+q0*q1),  1-2*(q1*q1+q2*q2);
    }

    // 验证旋转矩阵有效性（SVD 正交化作为最后防线）
    {
        double det2 = mR12i.cast<double>().determinant();
        double ortho_err = (mR12i.cast<double>() * mR12i.cast<double>().transpose()
                          - Eigen::Matrix3d::Identity()).norm();
        if (std::abs(det2 - 1.0) > 1e-6 || ortho_err > 1e-6) {
            std::cerr << "[Sim3Solver] 警告：旋转矩阵不正交 det=" << det2
                      << " ortho_err=" << ortho_err << "，执行 SVD 修正" << std::endl;
            Eigen::Matrix3d R = mR12i.cast<double>();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R,
                Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
            // Horn 1987: 如果 det < 0，最小奇异值对应列取负
            // 修正后: R = U * D * V^T, 其中 D = diag(1,1,-1)
            if (R.determinant() < 0) {
                D(2, 2) = -1.0;
            }
            R = svd.matrixU() * D * svd.matrixV().transpose();
            mR12i = R.cast<float>();
        }
    }

    // Step 5: Rotate set 2
    Eigen::Matrix3f P3 = mR12i * Pr2;

    // Step 6: Scale
    if (!mbFixScale_) {
        double nom = (Pr1.array() * P3.array()).sum();
        double den = (P3.array() * P3.array()).sum();
        ms12i = static_cast<float>(nom / std::max(den, 1e-10));
    } else {
        ms12i = 1.0f;
    }

    // Step 7: Translation
    mt12i = O1 - ms12i * mR12i * O2;

    // Step 8: Build 4x4 matrices
    mT12i.setIdentity();
    mT12i.block<3, 3>(0, 0) = ms12i * mR12i;
    mT12i.block<3, 1>(0, 3) = mt12i;

    mT21i.setIdentity();
    Eigen::Matrix3f sRinv = (1.0f / ms12i) * mR12i.transpose();
    mT21i.block<3, 3>(0, 0) = sRinv;
    mT21i.block<3, 1>(0, 3) = -sRinv * mt12i;
}

void Sim3Solver::checkInliers() {
    // 将帧2 的 3D 点用 mT12i 变换后投影到帧1 的图像平面
    // 将帧1 的 3D 点用 mT21i 变换后投影到帧2 的图像平面
    // 比较投影位置与原始图像位置，误差 < 阈值则计为内点

    mnInliersi_ = 0;
    mvbInliersi_.assign(N_, false);

    std::vector<Eigen::Vector2f> vP1im2, vP2im1;
    vP1im2.reserve(N_);
    vP2im1.reserve(N_);

    std::vector<Eigen::Vector3f> pts1_transformed(N_);
    std::vector<Eigen::Vector3f> pts2_transformed(N_);

    // frame2 的相机点用 Sim3 变换到 frame1 相机系，再用 frame1 内参投影
    Eigen::Matrix4f T = mT12i;
    for (int i = 0; i < N_; ++i) {
        const Eigen::Vector3f& P = mvX3Dc2_[i].cast<float>();
        if (P.z() < 1e-6f) {
            pts2_transformed[i] = Eigen::Vector3f::Zero();
        } else {
            pts2_transformed[i] = T.block<3, 3>(0, 0) * P + T.block<3, 1>(0, 3);
        }
    }
    fromCameraToImage(pts2_transformed, vP2im1);

    // frame1 的相机点用 Sim3 逆变换到 frame2 相机系，再用 frame2 内参投影
    T = mT21i;
    for (int i = 0; i < N_; ++i) {
        const Eigen::Vector3f& P = mvX3Dc1_[i].cast<float>();
        if (P.z() < 1e-6f) {
            pts1_transformed[i] = Eigen::Vector3f::Zero();
        } else {
            pts1_transformed[i] = T.block<3, 3>(0, 0) * P + T.block<3, 1>(0, 3);
        }
    }
    fromCameraToImage(pts1_transformed, vP1im2);

    // 比较投影位置与 ground-truth 图像坐标
    for (int i = 0; i < N_; ++i) {
        if (i >= static_cast<int>(mvP1im1_.size()) ||
            i >= static_cast<int>(mvP2im2_.size()) ||
            i >= static_cast<int>(vP1im2.size()) ||
            i >= static_cast<int>(vP2im1.size()))
            continue;

        Eigen::Vector2f dist1 = mvP1im1_[i] - vP2im1[i];
        Eigen::Vector2f dist2 = vP1im2[i] - mvP2im2_[i];
        float err1 = dist1.dot(dist1);
        float err2 = dist2.dot(dist2);

        if (err1 < mvnMaxError1_[i] && err2 < mvnMaxError2_[i]) {
            mvbInliersi_[i] = true;
            mnInliersi_++;
        }
    }
}

void Sim3Solver::project(const std::vector<Eigen::Vector3f>& vP3Dw,
                         std::vector<Eigen::Vector2f>& vP2D,
                         const Eigen::Matrix4f& Tcw) const {
    vP2D.resize(vP3Dw.size());
    for (size_t i = 0; i < vP3Dw.size(); ++i) {
        const Eigen::Vector3f& P = vP3Dw[i];
        if (P.z() < 0.05f || !std::isfinite(P.z())) {
            vP2D[i] = Eigen::Vector2f::Zero();
            continue;
        }
        float invz = 1.0f / P.z();
        vP2D[i].x() = fx_ * P.x() * invz + cx_;
        vP2D[i].y() = fy_ * P.y() * invz + cy_;
    }
}

void Sim3Solver::fromCameraToImage(const std::vector<Eigen::Vector3f>& vP3Dc,
                                   std::vector<Eigen::Vector2f>& vP2D) const {
    vP2D.resize(vP3Dc.size());
    for (size_t i = 0; i < vP3Dc.size(); ++i) {
        const Eigen::Vector3f& P = vP3Dc[i];
        if (P.z() < 0.05f || !std::isfinite(P.z())) {
            vP2D[i] = Eigen::Vector2f::Zero();
            continue;
        }
        float invz = 1.0f / P.z();
        vP2D[i].x() = fx_ * P.x() * invz + cx_;
        vP2D[i].y() = fy_ * P.y() * invz + cy_;
    }
}

Eigen::Matrix4f Sim3Solver::iterate(int nIterations, bool& bNoMore,
                                     std::vector<bool>& vbInliers, int& nInliers) {
    bNoMore = false;
    vbInliers = std::vector<bool>(N_, false);
    nInliers = 0;

    if (N_ < mRansacMinInliers_) {
        bNoMore = true;
        return Eigen::Matrix4f::Identity();
    }

    std::vector<size_t> availableIndices = mvAllIndices_;
    Eigen::Matrix3f P3Dc1i, P3Dc2i;

    // 预分配随机数生成器
    std::mt19937 rng(static_cast<unsigned>(N_ * 7919));

    int nCurrentIterations = 0;
    Eigen::Matrix4f bestSim3 = Eigen::Matrix4f::Identity();

    while (mnIterations_ < mRansacMaxIts_ && nCurrentIterations < nIterations) {
        nCurrentIterations++;
        mnIterations_++;

        availableIndices = mvAllIndices_;
        if (availableIndices.empty()) {
            std::cerr << "[Sim3Solver] availableIndices 为空，跳过迭代" << std::endl;
            return Eigen::Matrix4f::Identity();
        }

        // 随机采样 3 个点
        for (int i = 0; i < 3; ++i) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(availableIndices.size()) - 1);
            int randi = dist(rng);
            int idx = static_cast<int>(availableIndices[randi]);

            P3Dc1i.col(i) = mvX3Dc1_[idx].cast<float>();
            P3Dc2i.col(i) = mvX3Dc2_[idx].cast<float>();

            availableIndices[randi] = availableIndices.back();
            availableIndices.pop_back();
        }

        // Horn 1987 闭式求解
        computeSim3(P3Dc1i, P3Dc2i);

        // 内点检查
        checkInliers();

        if (mnInliersi_ >= mnBestInliers_) {
            mvbBestInliers_ = mvbInliersi_;
            mnBestInliers_ = mnInliersi_;
            mBestT12_ = mT12i;
            mBestRotation_ = mR12i;
            mBestTranslation_ = mt12i;
            mBestScale_ = ms12i;

            if (mnInliersi_ > mRansacMinInliers_) {
                nInliers = mnInliersi_;
                vbInliers = mvbBestInliers_;
                return mBestT12_;
            } else {
                bestSim3 = mBestT12_;
            }
        }
    }

    if (mnIterations_ >= mRansacMaxIts_)
        bNoMore = true;

    // 如果找到了最佳内点但没超过 minInliers，仍然返回最佳结果
    if (mnBestInliers_ > 0) {
        nInliers = mnBestInliers_;
        vbInliers = mvbBestInliers_;
        return mBestT12_;
    }

    return bestSim3;
}

Eigen::Matrix4f Sim3Solver::find(std::vector<bool>& vbInliers, int& nInliers) {
    bool bNoMore = false;
    return iterate(mRansacMaxIts_, bNoMore, vbInliers, nInliers);
}

}  // namespace loopdetection
}  // namespace sfc_core
