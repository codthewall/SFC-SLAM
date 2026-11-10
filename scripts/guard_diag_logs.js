// 给无条件的 std::cerr 日志加上 config_.verbose_diag 守卫（对应 CODE-RULES R-017）。
//
// 设计要点（都是为了让改动可静态验证、不依赖编译器）：
// 1) 用 `config_.verbose_diag` 而非新增全局变量：本文件的 std::cerr 都在 Tracking 成员函数内，
//    config_ 必然可见；引入 gVerboseDiag 反而要求额外声明与初始化顺序。
// 2) 一律用复合语句 `{ if (...) std::cerr << ...; }` 包裹。
//    这样即使原语句是 `if (cond) std::cerr << ...; else ...;` 的裸体分支，
//    插入后 else 仍绑定外层 if，不会因为悬挂 else 改变语义。
// 3) 已经处于 verbose_diag 守卫内的语句跳过，避免重复判断。
//
// 用法: node scripts/guard_diag_logs.js <file> [--apply] [--guard=<表达式>]
//
// --guard 指定守卫表达式，默认 config_.verbose_diag（tracking.cpp 用）。
// robust_pnp_solver.cpp 的开关是 config_.verbose，用 --guard=config_.verbose。
const fs = require('fs');

const target = process.argv[2];
const apply = process.argv.includes('--apply');
const guardArg = process.argv.find((a) => a.startsWith('--guard='));
if (!target) {
  console.error('usage: node scripts/guard_diag_logs.js <file> [--apply] [--guard=<expr>]');
  process.exit(2);
}

const src = fs.readFileSync(target, 'utf8');
const CERR = 'std::cerr';
const GUARD = guardArg ? guardArg.slice('--guard='.length) : 'config_.verbose_diag';
// 判定"已守卫"的正则：只看开关的标识部分，避免把完整表达式写进正则被转义问题绊住。
const GUARD_RE = new RegExp(GUARD.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

// 错误/异常类日志必须始终可见（CODE-RULES R-011/R-012：不吞异常、对外返回可读错误），
// 因此这些语句不加深色开关守卫，保持无条件输出。
const KEEP_ERROR_RE = /FATAL|exception|错误：|点数不足|未设置相机内参/;

// 从 start 起找语句结束分号，跳过注释/字符串，并要求圆括号与花括号都回到起始深度。
// 必须跟踪花括号：否则 `[this]() { int n = 0; ... }()` 这类 lambda 里的分号
// 会被误判成语句结束，导致守卫插进 lambda 体内（首次运行就踩到了）。
function findStatementEnd(text, start) {
  let i = start;
  let paren = 0;
  let brace = 0;
  while (i < text.length) {
    const c = text[i];
    const n = text[i + 1];
    if (c === '/' && n === '/') {
      const nl = text.indexOf('\n', i);
      i = nl === -1 ? text.length : nl;
      continue;
    }
    if (c === '/' && n === '*') {
      const end = text.indexOf('*/', i + 2);
      i = end === -1 ? text.length : end + 2;
      continue;
    }
    if (c === '"' || c === "'") {
      const quote = c;
      i++;
      while (i < text.length) {
        if (text[i] === '\\') i += 2;
        else if (text[i] === quote) { i++; break; }
        else i++;
      }
      continue;
    }
    if (c === '(') paren++;
    else if (c === ')') paren--;
    else if (c === '{') brace++;
    else if (c === '}') brace--;
    else if (c === ';' && paren === 0 && brace === 0) return i;
    i++;
  }
  return -1;
}

// 向上找最近的、未闭合的 `if (...) {` 是否已经带 verbose_diag 守卫。
// 只回看当前语句到上一处语句结束之间的区域，避免跨作用域误判。
function alreadyGuarded(text, stmtStart) {
  const windowStart = Math.max(0, stmtStart - 300);
  const before = text.slice(windowStart, stmtStart);
  const lastStmtEnd = Math.max(before.lastIndexOf(';'), before.lastIndexOf('}'));
  const tail = lastStmtEnd >= 0 ? before.slice(lastStmtEnd) : before;
  return GUARD_RE.test(tail);
}

// 该 std::cerr 是否位于某个类成员函数内（决定 config_ 是否可用）。
// 匹配任意 `Type ClassName::method(`，不限于 Tracking。
function enclosingIsMemberFn(text, pos) {
  const before = text.slice(0, pos);
  const m = before.match(/\n([A-Za-z_][\w:<>\*&\s]*?)\s+\w+::\w+\s*\(/g);
  if (!m || m.length === 0) return null;
  return m[m.length - 1];
}

const edits = [];
let searchFrom = 0;
const skipped = [];
while (true) {
  const idx = src.indexOf(CERR, searchFrom);
  if (idx === -1) break;
  searchFrom = idx + 1;

  const end = findStatementEnd(src, idx);
  if (end === -1) { skipped.push([idx, 'no-stmt-end']); continue; }

  if (alreadyGuarded(src, idx)) { skipped.push([idx, 'already-guarded']); continue; }

  const stmt = src.slice(idx, end + 1);
  if (KEEP_ERROR_RE.test(stmt)) { skipped.push([idx, 'always-visible-error']); continue; }

  const fn = enclosingIsMemberFn(src, idx);
  if (!fn) { skipped.push([idx, 'no-member-scope']); continue; }

  edits.push({ start: idx, end, fn: fn.trim() });
}

const lineOf = (text, pos) => text.slice(0, pos).split('\n').length;

console.log(`file=${target}`);
console.log(`candidates=${edits.length} skipped=${skipped.length}`);
for (const [pos, why] of skipped) {
  console.log(`  SKIP line ${lineOf(src, pos)}: ${why}`);
}

let out = src;
for (let i = edits.length - 1; i >= 0; i--) {
  const e = edits[i];
  out = out.slice(0, e.start) + `{ if (${GUARD}) ` + out.slice(e.start, e.end + 1) + ' }' + out.slice(e.end + 1);
}

if (!apply) {
  for (const e of edits) {
    console.log(`  line ${lineOf(src, e.start)} [${e.fn}]: ` +
      src.slice(e.start, e.end + 1).replace(/\s+/g, ' ').slice(0, 100));
  }
  console.log('dry-run only; pass --apply to write');
  process.exit(0);
}

fs.writeFileSync(target, out);
console.log(`applied=${edits.length}`);
