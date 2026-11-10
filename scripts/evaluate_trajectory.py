#!/usr/bin/env python3
"""TUM 轨迹评测：ATE / RPE / 路径总长。

为什么需要它：本仓库此前的验收标准是"看日志里的路径总长"，而路径总长对
单帧纬度跳变极其敏感（fr1_desk2 曾跑到 478m，真实约 5–12m），既不能说明
位姿对不对，也不能跨次运行比较。ATE / RPE 才是 SLAM 的通用口径。

用法：
    python scripts/evaluate_trajectory.py \
        --est sfc_traj.tum \
        --ref /path/to/fr1_desk2/groundtruth.txt \
        --align --max-diff 0.02

    # 只算路径总长，不做对齐（快速 sanity check）
    python scripts/evaluate_trajectory.py --est sfc_traj.tum --path-length-only

依赖 numpy（脚本只用它做 SVD 与矩阵运算，不依赖 evo）。
"""

import argparse
import sys

import numpy as np


def load_tum(path):
    """读取 TUM 轨迹：timestamp tx ty tz qx qy qz qw。

    返回 (tstamps[N], positions[N,3], quats[N,4] order=xyzw)。
    空行、以 # 开头的注释行会被跳过；列数不足的行直接报错而不是静默丢弃，
    因为静默丢弃会让人误以为"评测通过了"。
    """
    tstamps, positions, quats = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ValueError("%s:%d expected 8 columns, got %d: %s"
                                 % (path, lineno, len(parts), line))
            tstamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            quats.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not tstamps:
        raise ValueError("%s contains no trajectory samples" % path)
    return np.array(tstamps), np.array(positions), np.array(quats)


def associate(est_ts, ref_ts, max_diff):
    """按时间戳配对，返回下标对 (est_idx, ref_idx)。

    采用双指针：两边时间戳都已排序（TUM 文件天然有序）。不做插值 ——
    SLAM 输出是离散位姿，插值会把误差抹平。
    """
    order_e = np.argsort(est_ts)
    order_r = np.argsort(ref_ts)
    est_ts, ref_ts = est_ts[order_e], ref_ts[order_r]

    pairs = []
    i = j = 0
    while i < len(est_ts) and j < len(ref_ts):
        diff = est_ts[i] - ref_ts[j]
        if abs(diff) <= max_diff:
            pairs.append((order_e[i], order_r[j]))
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return pairs


def umeyama(src, dst, with_scale):
    """求把 src 对齐到 dst 的相似变换 (R, t, s)。

    src/dst 均为 [N,3]。算法见 Umeyama 1991；with_scale=False 时强制 s=1，
    这也是 ORB-SLAM3 的默认口径（单目才需要估尺度，RGB-D 不需要）。
    """
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama needs >=3 point pairs with matching shapes")
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    # 修正反射：det 为负时会得到镜像解，必须翻转最后一列
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1.0
    rot = u @ s_fix @ vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        scale = float(np.trace(np.diag(d) @ s_fix) / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    trans = mu_dst - scale * (rot @ mu_src)
    return rot, trans, scale


def apply_sim3(points, rot, trans, scale):
    return (scale * (rot @ points.T)).T + trans


def rotation_angle_deg(q_xyzw):
    """四元数 -> 旋转角（度）。用于 RPE 的旋转误差。"""
    w = np.clip(np.abs(q_xyzw[:, 3]), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def path_length(positions):
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def stats_block(errors, label):
    if len(errors) == 0:
        return "%s: no valid samples" % label
    return ("%s: rmse=%.4f mean=%.4f median=%.4f std=%.4f min=%.4f max=%.4f n=%d"
            % (label, float(np.sqrt((errors ** 2).mean())), float(errors.mean()),
               float(np.median(errors)), float(errors.std()),
               float(errors.min()), float(errors.max()), len(errors)))


def compute_ate(est_pos, ref_pos, do_align, with_scale):
    if do_align:
        rot, trans, scale = umeyama(est_pos, ref_pos, with_scale)
        aligned = apply_sim3(est_pos, rot, trans, scale)
    else:
        rot, trans, scale = np.eye(3), np.zeros(3), 1.0
        aligned = est_pos
    errors = np.linalg.norm(aligned - ref_pos, axis=1)
    return errors, aligned, (rot, trans, scale)


def compute_rpe(est_pos, ref_pos, est_q, ref_q, delta):
    """RPE：相邻 delta 帧之间的相对位姿误差（平移 m / 旋转 deg）。

    只比较"相对运动"，不比较绝对位置 —— 因此对起始点选择不敏感，
    是判断局部一致性（尺度、漂移速度）的主要指标。
    """
    if delta <= 0:
        raise ValueError("--delta must be positive")
    n = len(est_pos)
    if n <= delta:
        return np.zeros(0), np.zeros(0)

    idx = np.arange(0, n - delta)
    d_est = est_pos[idx + delta] - est_pos[idx]
    d_ref = ref_pos[idx + delta] - ref_pos[idx]
    trans_err = np.linalg.norm(d_est - d_ref, axis=1)

    # 旋转误差：R_est_rel * R_ref_rel^-1 的角度。四元数共轭即逆变换。
    qe0, qe1 = est_q[idx], est_q[idx + delta]
    qr0, qr1 = ref_q[idx], ref_q[idx + delta]
    rel_e = quat_mul(qe1, quat_conj(qe0))
    rel_r = quat_mul(qr1, quat_conj(qr0))
    diff = quat_mul(rel_e, quat_conj(rel_r))
    rot_err = rotation_angle_deg(diff)
    return trans_err, rot_err


def quat_conj(q):
    out = q.copy()
    out[:, :3] = -out[:, :3]
    return out


def quat_mul(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="TUM trajectory ATE/RPE evaluation")
    ap.add_argument("--est", required=True, help="estimated trajectory (TUM format)")
    ap.add_argument("--ref", help="ground-truth trajectory (TUM format)")
    ap.add_argument("--align", action="store_true",
                    help="run Umeyama alignment (default: raw error, no alignment)")
    ap.add_argument("--align-scale", action="store_true",
                    help="estimate scale during alignment (monocular only; keep off for RGB-D)")
    ap.add_argument("--max-diff", type=float, default=0.02,
                    help="timestamp association tolerance in seconds (TUM official default 0.02)")
    ap.add_argument("--delta", type=int, default=1,
                    help="RPE frame gap (default 1 = consecutive frames)")
    ap.add_argument("--path-length-only", action="store_true",
                    help="only print estimated path length; --ref not required")
    ap.add_argument("--save-aligned", help="write the aligned trajectory to this path")
    args = ap.parse_args(argv)

    est_ts, est_pos, est_q = load_tum(args.est)
    print("[est] %s: %d poses, path length %.4f m" % (args.est, len(est_ts), path_length(est_pos)))

    if args.path_length_only:
        return 0
    if not args.ref:
        ap.error("--ref is required unless --path-length-only is set")

    ref_ts, ref_pos, ref_q = load_tum(args.ref)
    print("[ref] %s: %d poses, path length %.4f m" % (args.ref, len(ref_ts), path_length(ref_pos)))

    pairs = associate(est_ts, ref_ts, args.max_diff)
    if len(pairs) < 3:
        print("ERROR: only %d associated frames (tolerance %.3fs); cannot evaluate.\n"
              "Check that both trajectories share the same time base." % (len(pairs), args.max_diff))
        return 2
    print("[assoc] %d pairs / est %d / ref %d"
          % (len(pairs), len(est_ts), len(ref_ts)))

    ei = np.array([p[0] for p in pairs])
    ri = np.array([p[1] for p in pairs])
    ate_err, aligned, (rot, trans, scale) = compute_ate(
        est_pos[ei], ref_pos[ri], args.align, args.align_scale)

    print(stats_block(ate_err, "ATE(m)"))
    if args.align:
        print("[align] scale=%.6f, t=%s" % (scale, np.array2string(trans, precision=4)))

    rpe_t, rpe_r = compute_rpe(aligned, ref_pos[ri], est_q[ei], ref_q[ri], args.delta)
    print(stats_block(rpe_t, "RPE_trans(m, delta=%d)" % args.delta))
    print(stats_block(rpe_r, "RPE_rot(deg, delta=%d)" % args.delta))

    # 单帧最大位移：诊断"轨迹总长暴涨"时第一眼看这个
    est_steps = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
    ref_steps = np.linalg.norm(np.diff(ref_pos[ri], axis=0), axis=1)
    if len(est_steps):
        print("[step] est per-frame step median %.4f / max %.4f m; ref median %.4f / max %.4f m"
              % (np.median(est_steps), est_steps.max(),
                 np.median(ref_steps), ref_steps.max()))

    if args.save_aligned:
        with open(args.save_aligned, "w", encoding="utf-8") as f:
            f.write("# timestamp tx ty tz qx qy qz qw (aligned to ref)\n")
            for k, idx in enumerate(ei):
                q = est_q[idx]
                f.write("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n"
                        % (est_ts[idx], aligned[k, 0], aligned[k, 1], aligned[k, 2],
                           q[0], q[1], q[2], q[3]))
        print("[save] wrote %s" % args.save_aligned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
