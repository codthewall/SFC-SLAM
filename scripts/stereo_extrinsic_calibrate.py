#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双目 / IR+RGB 双相机外参标定脚本
--------------------------------
内参直接写在代码里（来自 left.yaml / right.yaml）。

【重要】OpenCV 的 pattern 是「内角点」个数，不是方格个数！
  常见 9×6 格棋盘 → 内角点为 (9-1)×(6-1) = 8×5
  数法：黑白交界处的内角点，横向数、纵向数各减 1。

推荐命令（9×6 方格棋盘）：
  python3 stereo_extrinsic_calibrate.py \\
    --data_dir ... --squares 9x6 --square_size 0.025

或显式写内角点：
  --pattern_cols 8 --pattern_rows 5

IR 图散斑重时会自动做多档预处理 + 多种角点检测，无需改阈值。

依赖：numpy, opencv-python (>=4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# 硬编码内参（来源：left.yaml / right.yaml）
# =============================================================================
IMAGE_SIZE = (640, 480)          # (width, height)

# ---- IR 相机（left）----
K_IR = np.array(
    [[547.89324, 0.0, 322.76494],
     [0.0, 549.98672, 272.8274],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
D_IR = np.array([-0.086391, 0.119919, 0.012090, 0.000095, 0.0], dtype=np.float64)

# ---- RGB 相机（right）----
K_RGB = np.array(
    [[575.65516, 0.0, 322.40668],
     [0.0, 578.42405, 265.89652],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
D_RGB = np.array([0.077171, -0.057755, 0.014456, 0.004023, 0.0], dtype=np.float64)
# =============================================================================


def build_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """cols x rows 个内角点的 3D 坐标（Z=0 平面）。"""
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp = np.hstack([grid, np.zeros((rows * cols, 1), np.float32)])
    return objp * float(square_m)


def collect_image_pairs(
    data_dir: Path,
    left_prefix: str,
    right_prefix: str,
    ext: str,
) -> list[tuple[Path, Path]]:
    """按编号匹配左右图像路径。"""
    pairs: list[tuple[Path, Path]] = []
    for lp in sorted(data_dir.glob(f"{left_prefix}*{ext}")):
        suffix = lp.name[len(left_prefix) :]
        rp = data_dir / f"{right_prefix}{suffix}"
        if rp.is_file():
            pairs.append((lp, rp))
    return pairs


def _preprocess_variants(gray: np.ndarray) -> list[np.ndarray]:
    """IR 散斑/噪声：多档预处理，依次尝试。"""
    g = gray.astype(np.uint8)
    out: list[np.ndarray] = [g]
    # CLAHE 增强局部对比
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out.append(clahe.apply(g))
    # 中值滤波去椒盐散斑
    out.append(cv2.medianBlur(g, 5))
    out.append(cv2.medianBlur(clahe.apply(g), 3))
    # 双边保边平滑
    out.append(cv2.bilateralFilter(g, 5, 50, 50))
    # 轻度高斯
    out.append(cv2.GaussianBlur(g, (5, 5), 0))
    return out


def find_chessboard_corners_robust(
    gray: np.ndarray,
    pattern: tuple[int, int],
    criteria: tuple,
) -> np.ndarray | None:
    """
    多策略检测棋盘内角点：
    1) findChessboardCorners + 标准 flags
    2) 多种预处理后重复（角点亚像素在原始灰度上细化，更稳）
    3) findChessboardCornersSB（OpenCV 4.3+，对噪声更稳）
    """
    g0 = gray.astype(np.uint8)
    n_expected = pattern[0] * pattern[1]

    cb_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    def try_one(img: np.ndarray, refine_on: np.ndarray) -> np.ndarray | None:
        ok, corners = cv2.findChessboardCorners(img, pattern, cb_flags)
        if ok and corners is not None and len(corners) == n_expected:
            return cv2.cornerSubPix(
                refine_on, corners, (11, 11), (-1, -1), criteria
            )
        return None

    for img in _preprocess_variants(gray):
        c = try_one(img, g0)
        if c is not None:
            return c

    # ChessboardSB：对散斑 IR 往往比经典方法更稳
    # Python 签名: findChessboardCornersSB(image, patternSize[, corners[, flags]])
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        for img in _preprocess_variants(gray):
            ret, corners = cv2.findChessboardCornersSB(img, pattern, None, sb_flags)
            if ret and corners is not None and len(corners) == n_expected:
                corners = corners.astype(np.float32)
                return cv2.cornerSubPix(
                    g0, corners, (11, 11), (-1, -1), criteria
                )

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="双相机外参标定（内参硬编码）")
    ap.add_argument(
        "--data_dir",
        type=str,
        default="/home/syf/VSLAM/camera_yaml/calibrationdata",
        help="标定图像目录",
    )
    ap.add_argument(
        "--left_prefix", type=str, default="left-",
        help="左图文件名前缀（默认: left-）",
    )
    ap.add_argument(
        "--right_prefix", type=str, default="right-",
        help="右图文件名前缀（默认: right-）",
    )
    ap.add_argument(
        "--ext", type=str, default=".png",
        help="图像扩展名（默认: .png）",
    )
    ap.add_argument(
        "--squares",
        type=str,
        default=None,
        metavar="WxH",
        help="棋盘方格数 宽x高，如 9x6（自动换算内角点为 8x5）。若指定则忽略 pattern_cols/rows",
    )
    ap.add_argument(
        "--pattern_cols",
        type=int,
        default=8,
        help="棋盘内角点列数（横向）；默认 8（对应 9 列方格）",
    )
    ap.add_argument(
        "--pattern_rows",
        type=int,
        default=5,
        help="棋盘内角点行数（纵向）；默认 5（对应 6 行方格）",
    )
    ap.add_argument(
        "--square_size", type=float, default=0.025,
        help="方格边长（米）；T 的物理单位依赖此值",
    )
    ap.add_argument(
        "--out_dir", type=str, default=None,
        help="结果输出目录（默认同 data_dir）",
    )
    ap.add_argument(
        "--no_fix_intrinsic", action="store_true",
        help="放开内参一起优化（默认固定已标定内参）",
    )
    args = ap.parse_args()

    if args.squares:
        s = args.squares.lower().replace(" ", "").replace("*", "x")
        parts = s.split("x")
        if len(parts) != 2:
            print("[ERROR] --squares 格式应为 宽x高，例如 9x6")
            return 1
        try:
            sh, sv = int(parts[0]), int(parts[1])
        except ValueError:
            print("[ERROR] --squares 应为整数，例如 9x6")
            return 1
        pattern_cols, pattern_rows = sh - 1, sv - 1
        if pattern_cols < 2 or pattern_rows < 2:
            print("[ERROR] 方格数过小，内角点至少为 2x2")
            return 1
    else:
        pattern_cols, pattern_rows = args.pattern_cols, args.pattern_rows

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir else data_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] 棋盘内角点 pattern = {pattern_cols} x {pattern_rows} "
        f"（OpenCV 要求的是内角点个数；9x6 方格 → 8x5 内角点）"
    )

    pairs = collect_image_pairs(data_dir, args.left_prefix, args.right_prefix, args.ext)
    if not pairs:
        print(
            f"[ERROR] 在 {data_dir} 未找到配对图像 "
            f"({args.left_prefix}* 与 {args.right_prefix}*){args.ext}"
        )
        return 1

    pattern_size = (pattern_cols, pattern_rows)
    objp = build_object_points(pattern_cols, pattern_rows, args.square_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-6)

    obj_points: list[np.ndarray] = []
    img_points_l: list[np.ndarray] = []
    img_points_r: list[np.ndarray] = []

    for p_l, p_r in pairs:
        im_l = cv2.imread(str(p_l), cv2.IMREAD_GRAYSCALE)
        im_r = cv2.imread(str(p_r), cv2.IMREAD_GRAYSCALE)
        if im_l is None or im_r is None:
            print(f"[SKIP] 读取失败: {p_l.name}")
            continue

        c_l = find_chessboard_corners_robust(im_l, pattern_size, criteria)
        c_r = find_chessboard_corners_robust(im_r, pattern_size, criteria)
        if c_l is None or c_r is None:
            print(f"[SKIP] 棋盘检测不完整: {p_l.name}")
            continue

        obj_points.append(objp.copy())
        img_points_l.append(c_l)
        img_points_r.append(c_r)

    n_ok = len(obj_points)
    if n_ok < 3:
        print(f"[ERROR] 有效图像对仅 {n_ok} 对，至少需要 3 对以上。")
        return 1

    print(f"[INFO] 使用 {n_ok} 对图像，image_size={IMAGE_SIZE}")
    print(f"[INFO] IR 相机: fx={K_IR[0,0]:.3f} fy={K_IR[1,1]:.3f}  cx={K_IR[0,2]:.3f}  cy={K_IR[1,2]:.3f}")
    print(f"[INFO] RGB相机: fx={K_RGB[0,0]:.3f} fy={K_RGB[1,1]:.3f}  cx={K_RGB[0,2]:.3f}  cy={K_RGB[1,2]:.3f}")

    flags = 0 if args.no_fix_intrinsic else cv2.CALIB_FIX_INTRINSIC
    criteria_sc = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)

    rms, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_l,
        img_points_r,
        K_IR.copy(),
        D_IR.copy(),
        K_RGB.copy(),
        D_RGB.copy(),
        IMAGE_SIZE,
        criteria=criteria_sc,
        flags=flags,
    )

    # OpenCV: P2 = R * P1 + T  （点从 IR 相机坐标变换到 RGB 相机坐标）
    print("\n" + "=" * 50)
    print("标定结果（IR → RGB）")
    print("=" * 50)
    print(f"重投影 RMS: {rms:.5f} px")
    print(f"\n旋转矩阵 R (IR→RGB):\n{R}")
    print(f"\n平移向量 T (米): {T.ravel()}")
    print(f"  ||T|| = {np.linalg.norm(T):.5f} m （若 square_size 正确即为基线长度）")
    print(f"\n本质矩阵 E:\n{E}")
    print(f"\n基础矩阵 F:\n{F}")

    # 立体校正
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1o, D1o, K2o, D2o, IMAGE_SIZE, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
    )

    # ---- 保存 npz（完整数据，可 later 加载重算）----
    np.savez(
        out_dir / "stereo_extrinsic.npz",
        RMS=rms,
        R=R, T=T, E=E, F=F,
        K1=K1o, D1=D1o, K2=K2o, D2=D2o,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        image_size=IMAGE_SIZE,
    )

    # ---- 保存 JSON（便于阅读）----
    def to_list(x):
        return x.tolist() if hasattr(x, "tolist") else x

    with open(out_dir / "stereo_extrinsic.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "RMS": float(rms),
                "R": to_list(R),
                "T": to_list(T.ravel()),
                "E": to_list(E),
                "F": to_list(F),
                "image_size": list(IMAGE_SIZE),
                "baseline_m": float(np.linalg.norm(T)),
            },
            f, indent=2, ensure_ascii=False,
        )

    # ---- 保存 YAML（R/T 便于在 ROS / C++ 中直接使用）----
    yaml_out = out_dir / "stereo_extrinsics.yaml"
    with open(yaml_out, "w", encoding="utf-8") as yf:
        yf.write("%YAML:1.0\n")
        yf.write("# IR (left) → RGB (right) 外参\n")
        yf.write(f"# RMS: {rms}\n")
        yf.write(f"# ||T|| (baseline): {float(np.linalg.norm(T))} m\n")
        yf.write("rotation_matrix:\n  rows: 3\n  cols: 3\n  data: [")
        yf.write(", ".join(f"{float(v):.12f}" for v in R.ravel(order="C")))
        yf.write("]\n")
        yf.write("translation_vector:\n  rows: 3\n  cols: 1\n  data: [")
        yf.write(", ".join(f"{float(v):.12f}" for v in T.ravel()))
        yf.write("]\n")
        yf.write("Q_matrix:\n  rows: 4\n  cols: 4\n  data: [")
        yf.write(", ".join(f"{float(v):.12f}" for v in Q.ravel(order="C")))
        yf.write("]\n")

    print(f"\n[OK] 输出目录: {out_dir}")
    print(f"     npz : {out_dir / 'stereo_extrinsic.npz'}")
    print(f"     json: {out_dir / 'stereo_extrinsic.json'}")
    print(f"     yaml: {yaml_out}")
    print("\n提示: T 的物理单位（米）只有在 --square_size 与真实方格尺寸一致时才准确。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
