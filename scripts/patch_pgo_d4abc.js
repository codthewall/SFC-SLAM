// D4-a / D4-b / D4-c：位姿图优化的约束强度与可观测性
//
// D4-a  回环边信息矩阵按"内点对数"加权，不再只由内点率决定；
// D4-b  optimize() 出参回传 chi2 与迭代数，chi2 == 0 显式告警；
// D4-c  里程计边的信息缩放区间从写死的 [0.5, 2.0] 改为可配置且更宽
//       （原区间把 30% 内点与 90% 内点的边压成同一个权重，回环边再强也拉不动轨迹）。
const fs = require('fs');

function patch(path, pairs) {
  let s = fs.readFileSync(path, 'utf8');
  for (const [from, to, label] of pairs) {
    if (s.indexOf(from) < 0) throw new Error(path + ' :: 未找到片段 ' + label);
    s = s.replace(from, to);
  }
  fs.writeFileSync(path, s);
  console.log('[D4] patched', path);
}

// ---------- 1. 头文件：新增配置字段 + optimize() 出参 ----------
patch('src/sfc_core/include/sfc_core/loopdetection/pose_graph_optimizer.hpp', [
  [
`    // 回环边信息缩放
    double loop_information_scale = 2.0;        // 回环边基准缩放因子
    double loop_min_information_scale = 0.05;   // 回环边最小缩放因子下限`,
`    // 回环边信息缩放
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
    double odom_info_scale_max = 4.0;`,
    'options fields',
  ],
  [
`    // 执行一次位姿图优化（SE3 PGO）
    // 返回迭代次数（≤ 0 表示失败）
    int optimize();`,
`    // 执行一次位姿图优化（SE3 PGO）
    // 返回迭代次数（≤ 0 表示失败）
    //
    // D4-b：出参回传本次优化的 chi2 与实际迭代数。
    // 曾经的故障形态是 "odom 边整类被 setLevel(1) 移出优化 → 图上只剩 1 条回环边 →
    // chi2 恒为 0 → 顶点几乎不动却报优化成功"，而调用方只看得到迭代次数，看不出这件事。
    // chi2 == 0 现在会在 optimize() 内部显式告警。
    int optimize(double* out_chi2 = nullptr, int* out_iterations = nullptr);`,
    'optimize signature',
  ],
]);

// ---------- 2. 实现：addLoopEdge 计数因子 ----------
patch('src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp', [
  [
`    edge->information() = info * information_scale * opts_.loop_information_scale;`,
`    // D4-a：再乘一个"支持度"因子 —— 内点对数越多，这条约束越硬。
    // 取 inliers / loop_inlier_count_ref 后夹在配置的上下限内：
    // 低于下限说明这条回环几乎没有观测支持，不允许它影响整张图；
    // 高于上限说明这条回环特别结实，但也不该把相邻约束完全淹没。
    const double count_ref = std::max(1.0, opts_.loop_inlier_count_ref);
    const double count_factor = std::max(
        opts_.loop_inlier_count_min_factor,
        std::min(opts_.loop_inlier_count_max_factor,
                 static_cast<double>(std::max(0, geometric_inliers)) / count_ref));
    edge->information() =
        info * information_scale * opts_.loop_information_scale * count_factor;`,
    'loop count factor',
  ],
  // ---------- 3. 实现：addOdometryEdge 缩放区间 ----------
  [
`    double info_scale = information_scale * inlier_ratio;
    info_scale = std::min(2.0, std::max(0.5, std::max(opts_.odom_min_information_scale, info_scale)));`,
`    double info_scale = information_scale * inlier_ratio;
    // D4-c：区间来自配置，且比原来的 [0.5, 2.0] 更宽。
    // 原实现把 30% 内点与 90% 内点的里程计边压到几乎同一权重，回环修正传不出去；
    // 放宽后仍保留硬下限 odom_min_information_scale，防止一条烂边被彻底忽略。
    info_scale = std::max(opts_.odom_info_scale_min,
                          std::min(opts_.odom_info_scale_max, info_scale));
    info_scale = std::max(opts_.odom_min_information_scale, info_scale);`,
    'odom info clamp',
  ],
  // ---------- 4. 实现：optimize() 签名与 chi2 出参 ----------
  [
`int PoseGraphOptimizer::optimize() {`,
`int PoseGraphOptimizer::optimize(double* out_chi2, int* out_iterations) {`,
    'optimize def',
  ],
  [
`        std::cerr << "[PGO] g2o optimize() 抛出未知异常" << std::endl;
        opt_result = 0;
    }
    return opt_result;`,
`        std::cerr << "[PGO] g2o optimize() 抛出未知异常" << std::endl;
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
    return opt_result;`,
    'optimize chi2',
  ],
]);

console.log('[D4-a/b/c] done');
