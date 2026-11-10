#!/usr/bin/env node
// 一次性补丁第三步（按行号精确删除）：清掉 L7 回退分支的残留代码。
//
// 现场结构（1-based）：
//   609  } else {
//   615      if (config_.verbose) {
//   617      }
//   618  }                      <- 新的 else 块收尾
//   619  Eigen::Matrix3d R_icp = ...;   <- 以下全是残留体
//   ...
//   636      }
//   637      }
//   638  }                      <- 原 else-if 的收尾，现在成了孤立的 }
//   639  }                      <- 外层 if 收尾（保留）
// 因此删除 619..638。
const fs = require('fs');
const path = process.argv[2] || 'src/sfc_core/src/loopdetection/loop_detector.cpp';

const lines = fs.readFileSync(path, 'utf8').split('\n');

const startIdx = lines.findIndex(l => l.includes('Eigen::Matrix3d R_icp = T_icp.block<3,3>(0,0);'));
if (startIdx < 0) { console.error('未找到残留体起点'); process.exit(1); }

// 从起点往下找到"残留体结束"：连续两个 } 之后紧跟一个单独 } 的行，
// 取到那个单独 } 为止（即 638 行那个孤立括号）
let idx = startIdx;
let seenT_hybrid = false;
while (idx < lines.length) {
    if (lines[idx].includes('T_hybrid')) seenT_hybrid = true;
    if (seenT_hybrid && lines[idx].trim() === '}' && lines[idx + 1] !== undefined &&
        lines[idx + 1].trim() === '}') {
        // closeIdx 是残留体最后一个 }, 孤立的那个 } 在 idx+1
        break;
    }
    idx++;
}
if (!seenT_hybrid) { console.error('区间内未见 T_hybrid，锚点可能错位'); process.exit(1); }

// 再往后多删一行（原 else-if 的孤立收尾 }）
const endIdx = idx + 1;
if (lines[endIdx].trim() !== '}') { console.error('预期孤立收尾括号但没找到'); process.exit(1); }

const out = lines.slice(0, startIdx).concat(lines.slice(endIdx + 1));
fs.writeFileSync(path, out.join('\n'), 'utf8');
console.log(`已删除第 ${startIdx + 1}-${endIdx + 1} 行（共 ${endIdx - startIdx + 1} 行）`);
