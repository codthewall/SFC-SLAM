// L4：addLoopEdge 返回 bool —— 让调用方能区分"边真的进了图"和"被去重/校验静默丢弃"
// 原实现 void + 去重后静默 return：调用方无法知道回环是否生效，于是
// mnDetectedLoops_ 涨了、loop_edge_data_ 没涨，日志看起来正常而轨迹不动。
const fs = require('fs');

function patchFile(path, pairs) {
    let src = fs.readFileSync(path, 'utf8');
    for (const [needle, replacement, label] of pairs) {
        const at = src.indexOf(needle);
        if (at < 0) throw new Error(`${path}: 未找到片段 ${label}`);
        if (src.indexOf(needle, at + 1) >= 0) throw new Error(`${path}: 片段重复 ${label}`);
        src = src.slice(0, at) + replacement + src.slice(at + needle.length);
    }
    fs.writeFileSync(path, src);
    console.log(`[L4] patched ${path}`);
}

// ---------- 头文件 ----------
patchFile('src/sfc_core/include/sfc_core/loopdetection/pose_graph_optimizer.hpp', [[
`    // 添加回环边（回环检测得到的相对位姿约束）
    // geometric_inliers / raw_matches：用于计算信息矩阵（几何校验内点越多 → 越可靠）
    // is_icp: true=ICP 3D-3D（有真实尺度），false=Essential Matrix（尺度归一化，旋转可靠但平移尺度不可靠）
    void addLoopEdge(int kf_id_from, int kf_id_to,`,
`    // 添加回环边（回环检测得到的相对位姿约束）
    // geometric_inliers / raw_matches：用于计算信息矩阵（几何校验内点越多 → 越可靠）
    // is_icp: true=ICP 3D-3D（有真实尺度），false=Essential Matrix（尺度归一化，旋转可靠但平移尺度不可靠）
    //
    // 返回 true 表示这条约束**真的进入了 g2o 图**；false 表示被拒绝或去重（顶点缺失、
    // 尺度/正交性校验失败、同一对关键帧已有边、addEdge 失败）。
    // L4：原来是 void，去重时静默 return，调用方只能看到"检测到回环"而看不到"回环没生效"。
    bool addLoopEdge(int kf_id_from, int kf_id_to,`, 'header signature']]);

// ---------- 实现 ----------
patchFile('src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp', [
[`void PoseGraphOptimizer::addLoopEdge(int kf_id_from, int kf_id_to,`,
 `bool PoseGraphOptimizer::addLoopEdge(int kf_id_from, int kf_id_to,`, 'impl signature'],

// 顶点恢复失败
[`                  << " v_to=" << (v_to_raw ? "OK" : "NULL")
                  << std::endl;
        return;`,
 `                  << " v_to=" << (v_to_raw ? "OK" : "NULL")
                  << std::endl;
        return false;`, 'restore-fail return'],

// 顶点类型转换失败
[`        std::cerr << "[PGO] addLoopEdge 拒绝（顶点类型转换失败）: "
                  << kf_id_from << " -> " << kf_id_to << std::endl;
        return;`,
 `        std::cerr << "[PGO] addLoopEdge 拒绝（顶点类型转换失败）: "
                  << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;`, 'cast-fail return'],

// 缓存去重（第一处）
[`                optimizerNeedsRebuild_ = true;
            }
        }
        auto* v_curr = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id_from));
        if (v_curr && v_curr->fixed()) {
            v_curr->setFixed(false);
            if (kf_id_to_vertex_index_.find(kf_id_from) == kf_id_to_vertex_index_.end()) {
                kf_id_to_vertex_index_[kf_id_from] = kf_id_from;
                vertex_ids_.push_back(kf_id_from);
                vertex_ptrs_.push_back(v_curr);
            }
            optimizerNeedsRebuild_ = true;
        }
        return;
    }`,
 `                optimizerNeedsRebuild_ = true;
            }
        }
        auto* v_curr = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer_->vertex(kf_id_from));
        if (v_curr && v_curr->fixed()) {
            v_curr->setFixed(false);
            if (kf_id_to_vertex_index_.find(kf_id_from) == kf_id_to_vertex_index_.end()) {
                kf_id_to_vertex_index_[kf_id_from] = kf_id_from;
                vertex_ids_.push_back(kf_id_from);
                vertex_ptrs_.push_back(v_curr);
            }
            optimizerNeedsRebuild_ = true;
        }
        // 这一对关键帧已经有回环约束了：不是新约束，返回 false。
        return false;
    }`, 'cache-dup return'],

// loop_edge_ids_ 去重
[`                optimizerNeedsRebuild_ = true;
            }
            return;
        }
    }`,
 `                optimizerNeedsRebuild_ = true;
            }
            return false;
        }
    }`, 'ids-dup return'],

// NaN
[`        std::cerr << "[PGO] 回环边被拒绝（含 NaN）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        return;`,
 `        std::cerr << "[PGO] 回环边被拒绝（含 NaN）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;`, 'nan return'],

// 尺度异常
[`                  << "m " << kf_id_from << " -> " << kf_id_to << std::endl;
        return;
    }
    if (is_icp && t_norm > 50.0) {`,
 `                  << "m " << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;
    }
    if (is_icp && t_norm > 50.0) {`, 'scale return'],

// ICP 过大
[`        std::cerr << "[PGO] 回环边被拒绝（ICP 过大）: t_norm=" << t_norm
                  << "m " << kf_id_from << " -> " << kf_id_to << std::endl;
        return;`,
 `        std::cerr << "[PGO] 回环边被拒绝（ICP 过大）: t_norm=" << t_norm
                  << "m " << kf_id_from << " -> " << kf_id_to << std::endl;
        return false;`, 'icp-large return'],

// 非正交
[`                      << " R_diag=" << R.diagonal().transpose() << std::endl;
            return;`,
 `                      << " R_diag=" << R.diagonal().transpose() << std::endl;
            return false;`, 'ortho return'],

// 信息矩阵 NaN
[`        delete edge;
        return;
    }
    edge->information() =`,
 `        delete edge;
        return false;
    }
    edge->information() =`, 'info-nan return'],

// addEdge 失败
[`        std::cerr << "[PGO] addEdge 失败（loop）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        delete edge;
        return;
    }`,
 `        std::cerr << "[PGO] addEdge 失败（loop）: " << kf_id_from << " -> " << kf_id_to << std::endl;
        delete edge;
        return false;
    }`, 'addEdge-fail return'],

// 成功出口
[`    // 标记需要重建：回环边改变了图结构，下次 optimize() 前要刷新
    optimizerNeedsRebuild_ = true;
}`,
 `    // 标记需要重建：回环边改变了图结构，下次 optimize() 前要刷新
    optimizerNeedsRebuild_ = true;
    return true;
}`, 'success return'],
]);
