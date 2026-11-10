// D1-c：回环只修正关键帧位姿，不搬地图点（路线 A）。
//
// 原实现按"每个点只搬一次"的做法，用点的第一个观测关键帧的 delta 去变换世界坐标。
// 这仍然是错的：一个点被 K 个关键帧观测，那些关键帧各自拿到不同的 delta，
// 用其中一个 delta 搬点，等于宣布另外 K-1 个观测从此指向错误的世界位置 ——
// 地图里于是长出重复结构，下一次位姿优化又被这些错点拖走。
//
// 路线 A 的取舍：关键帧位姿是图优化的输出（可信），地图点的世界坐标不是
// （它只是一堆三角化结果）。回环只写回位姿；地图点保持原坐标，让后续的
// poseOptimization / BA 自己在新位姿下重新收敛。短期重投影误差会升高，
// 判断标准是 ATE 不是肉眼（见 PLAN 风险节）。
const fs = require('fs');

const p = 'src/sfc_core/src/frontend/tracking.cpp';
let s = fs.readFileSync(p, 'utf8');

const startMark = '    // 再把地图点搬到新坐标系。必须按"每个点只搬一次"来做：';
const endMark = '    // 当前帧本身就是刚建的关键帧，位姿要跟优化结果对齐；';

const i0 = s.indexOf(startMark);
const i1 = s.indexOf(endMark);
if (i0 < 0 || i1 < 0 || i1 < i0) throw new Error('未找到搬地图点代码块');

const replacement = `    // D1-c：不搬地图点（路线 A）。
    // 原实现在这里按"点的第一个观测关键帧"取一份 delta，把地图点世界坐标批量变换一次。
    // 一个点通常被 K 个关键帧共同观测，而这些关键帧在本次 PGO 里拿到的 delta 并不相同；
    // 用其中一个的 delta 搬点，等于宣布另外 K-1 个观测从此指向错误的世界位置，
    // 地图里长出重复结构，下一轮位姿优化再被这些错点拖回去。
    // 关键帧位姿是图优化的输出，可信；地图点世界坐标只是一堆三角化结果，不可信。
    // 所以回环只写回位姿，地图点留给后续 poseOptimization / BA 在新位姿下重新收敛。
    // 短期重投影误差会升高，判断标准是 ATE（PLAN 风险节已声明）。
    const size_t n_points_moved = 0;   // 保留日志字段，语义改成"本次未搬点"

`;

s = s.slice(0, i0) + replacement + s.slice(i1);
fs.writeFileSync(p, s);
console.log('[D1-c] 已移除地图点搬移逻辑');
