#!/usr/bin/env node
// 一次性补丁（按行号直接删除）：清掉 L7 回退分支的残留代码。
//
// 现场结构（1-based）：
//   609  } else {
//   615      if (config_.verbose) {
//   617      }
//   618  }                              <- 新 else 块收尾
//   619  Eigen::Matrix3d R_icp = ...;   <- 以下全是残留体
//   635      return true;
//   636      }                            <- if (frame_dist ...)
//   637  }                                <- if (R_icp.allFinite())
//   638  }                                <- 原 else-if 的收尾，现在孤立
//   639  }                                <- 外层 if 收尾（保留）
// 删除 619..638（1-based），保留 639。
const fs = require('fs');
const path = process.argv[2] || 'src/sfc_core/src/loopdetection/loop_detector.cpp';

const lines = fs.readFileSync(path, 'utf8').split('\n');

const START_1BASED = 619;
const END_1BASED = 638;

const startIdx = START_1BASED - 1;
const endIdx = END_1BASED - 1;

const assertions = [
    [startIdx, 'Eigen::Matrix3d R_icp = T_icp.block<3,3>(0,0);'],
    [startIdx + 10, 'T_hybrid.block<3,3>(0,0) = R_icp;'],
    [startIdx + 16, 'return true;'],
    [endIdx, '}'],
];
for (const [i, expected] of assertions) {
    const actual = (lines[i] || '').trim();
    if (actual !== expected) {
        console.error(`第 ${i + 1} 行断言失败：期望 "${expected}"，实际 "${actual}"`);
        process.exit(1);
    }
}
if ((lines[endIdx + 1] || '').trim() !== '}') {
    console.error(`第 ${endIdx + 2} 行应为外层 if 的收尾 }，实际 "${(lines[endIdx + 1] || '').trim()}"`);
    process.exit(1);
}

const out = lines.slice(0, startIdx).concat(lines.slice(endIdx + 1));
fs.writeFileSync(path, out.join('\n'), 'utf8');
console.log(`已删除第 ${START_1BASED}-${END_1BASED} 行（共 ${END_1BASED - START_1BASED + 1} 行）`);
