// L9 收尾：把两份死代码处理掉。
//   (a) computeCandidateConsistency() 接线到 selectBestFromVerified()
//       —— 顺带让 require_candidate_agreement / max_spatial_dist_for_consistency /
//          consistency_warning_threshold 三个配置键第一次真正生效。
//   (b) correctLoopWithScale() 删除
//       —— 它算的是 scale = 位姿距离 / 相对平移长度，也就是拿"漂移后的里程计距离"
//          当尺度真值；那正是 EM 路径刚刚被替换掉的那个错误来源（现在改用 3D 对应点
//          投影到 t_em_dir 的中位数）。留着它等于给尺度留第二份互相矛盾的真值。
//          删掉后 scale_min / scale_max 也一并从 LoopDetectorConfig 移除
//          （yaml 里没有这两个键，config_loader 也不读，属于纯内部字段）。
const fs = require('fs');

// ---------- (a) 接线一致性检查 ----------
const cppPath = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
let src = fs.readFileSync(cppPath, 'utf8');

const anchor = `    if (static_cast<int>(verified_candidates.size()) < config_.min_candidates_to_accept) {
        best.kf_id = -1;
        return best;
    }

    return best;
}`;

const replacement = `    // L9：候选一致性检查原本是死代码（全仓零调用点），require_candidate_agreement /
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
}`;

if (src.indexOf(anchor) < 0) throw new Error('未找到 selectBestFromVerified 尾部锚点');
if (src.indexOf(anchor, src.indexOf(anchor) + 1) >= 0) throw new Error('锚点重复');
src = src.replace(anchor, replacement);

// ---------- (b) 删除 correctLoopWithScale 实现 ----------
const funcStart = src.indexOf('double LoopDetector::correctLoopWithScale(');
if (funcStart < 0) throw new Error('未找到 correctLoopWithScale 实现');
const nextFunc = src.indexOf('std::vector<LoopDetector::VerifiedCandidate>\nLoopDetector::verifyMultipleCandidates(', funcStart);
if (nextFunc < 0) throw new Error('未找到 correctLoopWithScale 之后的函数起点');
src = src.slice(0, funcStart) + src.slice(nextFunc);

fs.writeFileSync(cppPath, src);
console.log('[L9] computeCandidateConsistency 已接线；correctLoopWithScale 实现已删除');

// ---------- (b2) 删除头文件声明 + 两个 scale 配置键 ----------
const hppPath = 'src/sfc_core/include/sfc_core/loopdetection/loop_detector.hpp';
let hpp = fs.readFileSync(hppPath, 'utf8');

const declBlock = `    // ★ Sim(3) 回环校正 - 估计并应用尺度校正
    // 返回估计的尺度因子
    double correctLoopWithScale(int curr_kf_id, int matched_kf_id,
                                const Eigen::Matrix4d& relative_pose,
                                int inliers);

`;
if (hpp.indexOf(declBlock) < 0) throw new Error('未找到 correctLoopWithScale 声明块');
hpp = hpp.replace(declBlock, '');

const scaleBlock = `    // ★★★ Sim3 尺度估计参数 ★★★
    double scale_min = 0.5;    // 最小尺度因子
    double scale_max = 2.0;    // 最大尺度因子
`;
const scaleNew = `    // ★★★ Sim3 尺度参数 ★★★
    // 原 scale_min / scale_max 只服务于已删除的 correctLoopWithScale()（用漂移后的
    // 位姿距离当尺度真值）。现在尺度来自几何验证本身：ICP 有深度、EM 路径用
    // 3D 对应点投影中位数，两边都是实测值，不再需要配置里的夹紧区间。
`;
if (hpp.indexOf(scaleBlock) < 0) throw new Error('未找到 scale_min/scale_max 配置块');
hpp = hpp.replace(scaleBlock, scaleNew);

fs.writeFileSync(hppPath, hpp);
console.log('[L9] 头文件：声明与 scale_min/scale_max 已清理');
