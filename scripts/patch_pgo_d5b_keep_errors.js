// D5-b 收尾：按 R-011/R-021 把"错误/拒绝类"日志的守卫撤掉。
//
// guard_diag_logs.js 的 KEEP_ERROR_RE 是按 tracking.cpp 的措辞写的
// （FATAL|exception|错误：|点数不足|未设置相机内参），覆盖不到 PGO 里
// "拒绝…/…失败/…含 NaN/…含 NaN/Inf/异常" 这类句式，于是它们被一起加了守卫。
// 这些是"某个顶点/边没能进图"的真实故障，静默掉会让位姿图少约束而无人知晓，
// 因此必须无条件输出。这里按语句内容把它们改回裸 std::cerr。
const fs = require('fs');

const p = 'src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp';
const lines = fs.readFileSync(p, 'utf8').split('\n');

// 需要保持无条件的语义关键词
const KEEP = /拒绝|失败|NaN|Inf|异常|错误|FATAL|无法|不存在/;

let changed = 0;
for (let i = 0; i < lines.length; i++) {
  const line = lines[i];
  if (!line.includes('{ if (opts_.verbose) std::cerr')) continue;

  // 取整条语句（可能跨行到分号）
  let stmt = line;
  let j = i;
  while (!stmt.includes(';') && j + 1 < lines.length) {
    j++;
    stmt += '\n' + lines[j];
  }

  if (!KEEP.test(stmt)) continue;

  // 撤掉守卫：'{ if (opts_.verbose) ' + body + ' }' -> body
  const prefix = '{ if (opts_.verbose) ';
  const at = lines[i].indexOf(prefix);
  if (at < 0) continue;

  // 单行闭合（同一行以 ' }' 结尾）
  if (j === i && lines[i].trimEnd().endsWith('}')) {
    const head = lines[i].slice(0, at);
    let body = lines[i].slice(at + prefix.length);
    body = body.replace(/\s*\}\s*$/, '');
    lines[i] = head + body;
    changed++;
    continue;
  }

  // 多行：去掉首行前缀，去掉末行结尾的 ' }'
  const head = lines[i].slice(0, at);
  let body = lines[i].slice(at + prefix.length);
  lines[i] = head + body;
  const lastLine = lines[j];
  lines[j] = lastLine.replace(/\s*\}\s*$/, '');
  changed++;
}

fs.writeFileSync(p, lines.join('\n'));
console.log(`[D5-b] 撤销守卫的错误类日志: ${changed} 处`);
