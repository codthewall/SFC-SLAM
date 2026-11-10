#!/usr/bin/env node
// 一次性补丁：删除 L7 遗留的 "ICP 内点=0 → ICP 旋转 + 里程计尺度" 回退分支。
// 该分支在 dummy_matches 被移除后已无法编译（引用了不存在的变量），
// 且其语义本身就是"内点为零仍然采用 ICP 的旋转"，属于注释里声称已删除、
// 实际仍存在的 38.6° 错误旋转来源。
const fs = require('fs');
const path = process.argv[2] || 'src/sfc_core/src/loopdetection/loop_detector.cpp';

const raw = fs.readFileSync(path, 'utf8');
const lines = raw.split('\n');

// 定位起点：`} else if (icp_inliers == 0 && dummy_matches.size() >= 2) {`
let start = -1;
for (let i = 0; i < lines.length; ++i) {
    if (lines[i].includes('icp_inliers == 0 && dummy_matches.size() >= 2')) { start = i; break; }
}
if (start < 0) {
    console.error('未找到目标分支起点，文件可能已被修改');
    process.exit(1);
}

// 从起点向下做括号配平，找到该 else-if 分支的结束行
let depth = 0;
let end = -1;
let started = false;
for (let i = start; i < lines.length; ++i) {
    for (const ch of lines[i]) {
        if (ch === '{') { depth++; started = true; }
        else if (ch === '}') { depth--; }
    }
    if (started && depth === 0) { end = i; break; }
}
if (end < 0) {
    console.error('括号配平失败，未找到分支结尾');
    process.exit(1);
}

const replacement = [
    '        } else {',
    '            // L7：这里原来有一条「ICP 内点=0 → 采用 ICP 旋转 + 里程计尺度」的回退分支。',
    '            // 内点为 0 意味着 ICP 的解没有任何数据支持，却把它的旋转矩阵拿来用、',
    '            // 只把平移换成两帧里程计距离 —— 这正是上面注释里反复提到的 38.6° 错误旋转来源',
    '            // （注释写着「已删除 hybrid fallback」，代码其实还在）。内点为 0 时唯一诚实的做法',
    '            // 是不采用 ICP，直接落到后面的 Essential Matrix 降级路径。',
    '            if (config_.verbose) {',
    '                std::cerr << "[GeoVerif] ICP 内点为 0，不采用 ICP 结果，降级到 Essential Matrix" << std::endl;',
    '            }',
    '        }',
];

const out = lines.slice(0, start).concat(replacement, lines.slice(end + 1));
fs.writeFileSync(path, out.join('\n'), 'utf8');
console.log(`已替换 ${path} 第 ${start + 1}-${end + 1} 行（共 ${end - start + 1} 行）`);
