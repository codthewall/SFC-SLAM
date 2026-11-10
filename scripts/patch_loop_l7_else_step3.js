#!/usr/bin/env node
// 一次性补丁第三步：清掉 L7 回退分支的残留代码。
// 前两次按括号配平删除时，首行 `} else if (...) {` 与残留体首行各自内部括号平衡，
// 导致只删掉了单行。这里改成"按内容锚点定位 + 显式行区间删除"，并先做内容断言。
const fs = require('fs');
const path = process.argv[2] || 'src/sfc_core/src/loopdetection/loop_detector.cpp';

const lines = fs.readFileSync(path, 'utf8').split('\n');

// 锚点 1：残留体第一行（老分支里的 R_icp 取值）
let start = -1;
for (let i = 0; i < lines.length; ++i) {
    if (lines[i].includes('Eigen::Matrix3d R_icp = T_icp.block<3,3>(0,0);')) { start = i; break; }
}
if (start < 0) { console.error('未找到残留体起点'); process.exit(1); }

// 锚点 2：残留体最后一行是 return true;
let ret = -1;
for (let i = start; i < Math.min(start + 40, lines.length); ++i) {
    if (lines[i].trim() === 'return true;') { ret = i; break; }
}
if (ret < 0) { console.error('未找到残留体 return true'); process.exit(1); }

// 从 return true 往下：应依次是 3 个关闭括号（if frame_dist / if allFinite / 原 else-if 的 }）
// 再加 1 个属于外层 if 的 }，一共要删掉 return 之后的 3 行，保留第 4 个 } 作为外层 if 收尾。
const after = lines.slice(ret + 1, ret + 6);
const closing = [];
for (let i = 0; i < after.length; ++i) {
    const t = after[i].trim();
    if (t === '}') closing.push(ret + 1 + i);
    else break;
}
if (closing.length < 3) {
    console.error(`return true 之后只找到 ${closing.length} 个连续关闭括号，与预期 3 个不符，中止`);
    process.exit(1);
}

// 删除 start..closing[2]（残留体 + 它的 3 个收尾括号），保留后面那个外层 if 的 }
const end = closing[2];
const removed = lines.slice(start, end + 1);
if (!removed.some(l => l.includes('T_hybrid'))) {
    console.error('待删区间里没有 T_hybrid，锚点可能错位，中止');
    process.exit(1);
}

const out = lines.slice(0, start).concat(lines.slice(end + 1));
fs.writeFileSync(path, out.join('\n'), 'utf8');
console.log(`已删除 ${path} 第 ${start + 1}-${end + 1} 行（共 ${end - start + 1} 行）`);
