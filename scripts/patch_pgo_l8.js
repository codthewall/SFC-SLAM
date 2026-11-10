// L8：删除 pose_graph_optimizer.cpp 里三大块"用调试代替修复"的残留
//   1) :1199-1318 的 120 行块注释（崩溃诊断 + 被注释掉的 dangling 预检查）
//      —— 其中的 dangling edge 预检查是有价值的（removeEdge 必须在
//      initializeOptimization() 之前），所以复活成真代码，其余诊断永不留存。
//   2) :1326-1424 的 sigsetjmp / sigaction + SIGSEGV 捕获
//      —— g2o 段错误不是可恢复状态，longjmp 回来后的图已损坏，
//      捕获只会把崩溃换成"静默给出错误轨迹"。保留 try/catch 处理 C++ 异常。
//   3) 文件顶部 :14-26 的 sigjmp_buf / sigsegv_handler 定义
// 本脚本按内容断言定位，不依赖行号漂移。
const fs = require('fs');

const path = 'src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp';
let src = fs.readFileSync(path, 'utf8');
const before = src.split('\n').length;

function replaceOnce(text, needle, replacement, label) {
    const at = text.indexOf(needle);
    if (at < 0) throw new Error(`[L8] 未找到片段：${label}`);
    if (text.indexOf(needle, at + 1) >= 0) throw new Error(`[L8] 片段重复出现：${label}`);
    return text.slice(0, at) + replacement + text.slice(at + needle.length);
}

// ---- 1) 顶部信号防护定义 ----
const headStart = src.indexOf('// ====== SIGSEGV 防护（仅 POSIX/Linux，非 Windows）======');
const headEnd = src.indexOf('#endif // _WIN32', headStart);
if (headStart < 0 || headEnd < 0) throw new Error('[L8] 未找到顶部信号防护块');
src = src.slice(0, headStart) + src.slice(headEnd + '#endif // _WIN32\n'.length);
src = src.replace('#include <csetjmp>\n', '');
src = src.replace('#include <csignal>\n', '');

// ---- 2) 120 行块注释 → 复活的 dangling 预检查 ----
const blockStart = src.indexOf('/*\n    // ★★★ 崩溃前诊断');
const blockEnd = src.indexOf('*/\n    int init = optimizer_->initializeOptimization();');
if (blockStart < 0 || blockEnd < 0) throw new Error('[L8] 未找到块注释 1199-1318');
const revived = `    // 防御性 dangling edge 检查（必须在 initializeOptimization() 之前执行）
    // initializeOptimization() 会把 _edges 复制进 _activeEdges；之后再 removeEdge()
    // 只改 _edges、不动 _activeEdges，于是 _activeEdges 里留下悬空指针。
    // 所以只能在它之前把坏边摘掉。这是原本被整段注释掉的检查（L8 复活）。
    {
        int removed_null_ptr = 0;
        int removed_bad_edge = 0;
        std::vector<g2o::HyperGraph::Edge*> edges_to_drop;
        for (auto* e_raw : optimizer_->edges()) {
            auto* e = dynamic_cast<g2o::EdgeSE3Expmap*>(e_raw);
            if (!e) continue;
            if (!e->vertex(0) || !e->vertex(1)) {
                ++removed_null_ptr;
                edges_to_drop.push_back(e_raw);
                continue;
            }
            int v0_id = static_cast<int>(e->vertex(0)->id());
            int v1_id = static_cast<int>(e->vertex(1)->id());
            if (!optimizer_->vertex(v0_id) || !optimizer_->vertex(v1_id)) {
                ++removed_bad_edge;
                edges_to_drop.push_back(e_raw);
            }
        }
        if (!edges_to_drop.empty()) {
            std::cerr << "[PGO] 预检查（initializeOptimization 之前）：移除 "
                      << removed_null_ptr << " 条空指针边 + " << removed_bad_edge
                      << " 条悬空边" << std::endl;
            for (auto* e_bad : edges_to_drop) {
                optimizer_->removeEdge(e_bad);
            }
        }
    }
`;
src = src.slice(0, blockStart) + revived + src.slice(blockEnd + '*/\n'.length);

// ---- 3) sigsetjmp / SIGSEGV 捕获 → 纯 try/catch ----
const sigStart = src.indexOf('#ifndef _WIN32\n    // ★★★ 信号防护（仅 POSIX/Linux）');
const sigEnd = src.indexOf('#endif // _WIN32\n    return opt_result;');
if (sigStart < 0 || sigEnd < 0) throw new Error('[L8] 未找到信号防护块');
const plainTry = `    // L8：这里原本用 sigsetjmp/sigaction 捕获 SIGSEGV/SIGBUS/SIGABRT，再 siglongjmp
    // 回来清理边缓存继续跑。这条路本身就是错的：段错误发生时 g2o 内部已处于半
    // 更新状态，longjmp 跳过栈展开后返回的是一个损坏的图，后续每帧都在损坏状态
    // 上"优化"，崩溃被换成静默的错误轨迹——比直接崩更难查。
    // 能防的只有 C++ 异常（g2o 抛 std::bad_alloc / 断言 exception），保留 try/catch。
    // 真正的段错误应该让它崩，用 ASAN / core dump 定位。
    int opt_result = 0;
    try {
        opt_result = optimizer_->optimize(opts_.max_iterations);
    } catch (const std::exception& e) {
        std::cerr << "[PGO] g2o optimize() 抛出异常: " << e.what() << std::endl;
        opt_result = 0;
    } catch (...) {
        std::cerr << "[PGO] g2o optimize() 抛出未知异常" << std::endl;
        opt_result = 0;
    }
`;
src = src.slice(0, sigStart) + plainTry + src.slice(sigEnd + '#endif // _WIN32\n'.length);

// ---- L3 收尾：initializeOptimization 显式 level 0 ----
src = replaceOnce(
    src,
    '    int init = optimizer_->initializeOptimization();',
    `    // L3：显式传 level 0。语义是"参与优化的边 = level 0 的边"，
    // 必须和 addOdometryEdge()（不设 level，默认 0）保持一致。
    // 不传参数时 g2o 也按 0 处理，但"隐式默认值"正是上一个 bug 的藏身处。
    int init = optimizer_->initializeOptimization(0);`,
    'initializeOptimization');

fs.writeFileSync(path, src);
const after = src.split('\n').length;
console.log(`[L8] ${path}: ${before} -> ${after} 行`);
