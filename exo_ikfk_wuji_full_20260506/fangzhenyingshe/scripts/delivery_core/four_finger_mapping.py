from __future__ import annotations

import math

import numpy as np
from scipy import optimize


IK_RETRY_ERROR_M = 1e-4
IK_EQUAL_ERROR_TOL_M = 1e-7


def configure_chinese_font() -> None:
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError:
        return

    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def mm_to_m(value_mm: float) -> float:
    return value_mm / 1000.0


def project_point(point: np.ndarray, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    delta = point - origin
    return np.array([np.dot(delta, x_axis), np.dot(delta, y_axis)], dtype=np.float64)


def build_human_polyline(base_xy: np.ndarray, lengths: tuple[float, float, float], angles: tuple[float, float, float]) -> np.ndarray:
    l1, l2, l3 = lengths
    q1, q2, q3 = angles
    p0 = np.asarray(base_xy, dtype=np.float64)
    p1 = p0 + np.array([l1 * math.cos(q1), l1 * math.sin(q1)], dtype=np.float64)
    p2 = p1 + np.array([l2 * math.cos(q1 + q2), l2 * math.sin(q1 + q2)], dtype=np.float64)
    p3 = p2 + np.array([l3 * math.cos(q1 + q2 + q3), l3 * math.sin(q1 + q2 + q3)], dtype=np.float64)
    return np.vstack([p0, p1, p2, p3])


def exo_tip_frame_2d(exo_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tip = np.asarray(exo_points[-1], dtype=np.float64)
    prev = np.asarray(exo_points[-2], dtype=np.float64)
    tangent = tip - prev
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12:
        raise ValueError("外骨骼末端前一段长度过小，无法建立末端局部坐标系。")
    local_x = tangent / norm
    local_y = np.array([-local_x[1], local_x[0]], dtype=np.float64)
    return tip, local_x, local_y


def target_tip_from_local_offset(exo_points: np.ndarray, tip_offset_local: np.ndarray) -> np.ndarray:
    tip_origin, local_x, local_y = exo_tip_frame_2d(exo_points)
    offset_local = np.asarray(tip_offset_local, dtype=np.float64)
    return tip_origin + offset_local[0] * local_x + offset_local[1] * local_y


def transform_four_finger_base_offset(base_xy: np.ndarray) -> np.ndarray:
    base_xy = np.asarray(base_xy, dtype=np.float64)
    if base_xy.shape != (2,):
        raise ValueError(f"四指 base 输入必须是二维向量，当前形状为 {base_xy.shape}")
    return np.array([base_xy[1], -base_xy[0]], dtype=np.float64)


def four_finger_base_offset_from_mm(base_dx_mm: float, base_dy_mm: float) -> np.ndarray:
    return transform_four_finger_base_offset(
        np.array([mm_to_m(base_dx_mm), mm_to_m(base_dy_mm)], dtype=np.float64)
    )


def recover_four_finger_base_input(base_offset: np.ndarray) -> np.ndarray:
    base_offset = np.asarray(base_offset, dtype=np.float64)
    if base_offset.shape != (2,):
        raise ValueError(f"四指 base_offset 必须是二维向量，当前形状为 {base_offset.shape}")
    return np.array([-base_offset[1], base_offset[0]], dtype=np.float64)


def human_joint_bounds(flexion_sign: int) -> tuple[np.ndarray, np.ndarray]:
    if flexion_sign not in (-1, 1):
        raise ValueError("flexion_sign 只能是 -1 或 1")

    down = float(flexion_sign)
    joint_ranges = [
        (-down * math.radians(60.0), down * math.radians(90.0)),
        (0.0, down * math.radians(100.0)),
        (0.0, down * math.radians(90.0)),
    ]
    bounds_lower = np.array([min(lo, hi) for lo, hi in joint_ranges], dtype=np.float64)
    bounds_upper = np.array([max(lo, hi) for lo, hi in joint_ranges], dtype=np.float64)
    return bounds_lower, bounds_upper


def human_ik_seed_candidates(
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    initial_angles: np.ndarray,
    flexion_sign: int,
) -> list[np.ndarray]:
    down = float(flexion_sign)
    midpoint = 0.5 * (bounds_lower + bounds_upper)
    raw_candidates = [
        initial_angles,
        np.zeros(3, dtype=np.float64),
        midpoint,
        np.array([midpoint[0], bounds_lower[1], bounds_lower[2]], dtype=np.float64),
        np.array([midpoint[0], bounds_upper[1], bounds_upper[2]], dtype=np.float64),
        np.array([bounds_lower[0], midpoint[1], midpoint[2]], dtype=np.float64),
        np.array([bounds_upper[0], midpoint[1], midpoint[2]], dtype=np.float64),
        np.array([0.0, down * math.radians(25.0), down * math.radians(18.0)], dtype=np.float64),
        np.array([down * math.radians(30.0), down * math.radians(45.0), down * math.radians(30.0)], dtype=np.float64),
        np.array([-down * math.radians(25.0), down * math.radians(40.0), down * math.radians(28.0)], dtype=np.float64),
    ]

    candidates: list[np.ndarray] = []
    for candidate in raw_candidates:
        clipped = np.clip(np.asarray(candidate, dtype=np.float64), bounds_lower, bounds_upper)
        if not any(np.allclose(clipped, existing, atol=1e-9, rtol=0.0) for existing in candidates):
            candidates.append(clipped)
    return candidates


def solve_human_ik(
    target_tip_xy: np.ndarray,
    lengths: tuple[float, float, float],
    base_xy: np.ndarray,
    initial_angles: tuple[float, float, float],
    flexion_sign: int,
) -> tuple[np.ndarray, float]:
    bounds_lower, bounds_upper = human_joint_bounds(flexion_sign)
    target_tip_xy = np.asarray(target_tip_xy, dtype=np.float64)
    base_xy = np.asarray(base_xy, dtype=np.float64)
    initial = np.clip(np.asarray(initial_angles, dtype=np.float64), bounds_lower, bounds_upper)

    def residual(q: np.ndarray) -> np.ndarray:
        human_points = build_human_polyline(base_xy, lengths, (float(q[0]), float(q[1]), float(q[2])))
        tip_error = human_points[-1] - target_tip_xy
        return np.array([tip_error[0], tip_error[1]], dtype=np.float64)

    def solve_from(seed: np.ndarray) -> tuple[np.ndarray, float, float]:
        result = optimize.least_squares(
            residual,
            x0=seed,
            bounds=(bounds_lower, bounds_upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )
        solved_angles = np.asarray(result.x, dtype=np.float64)
        solved_error = float(
            np.linalg.norm(build_human_polyline(base_xy, lengths, tuple(solved_angles))[-1] - target_tip_xy)
        )
        continuity_cost = float(np.linalg.norm(solved_angles - initial))
        return solved_angles, solved_error, continuity_cost

    best_angles, best_error, best_continuity = solve_from(initial)
    if best_error > IK_RETRY_ERROR_M:
        for seed in human_ik_seed_candidates(bounds_lower, bounds_upper, initial, flexion_sign)[1:]:
            solved_angles, solved_error, continuity_cost = solve_from(seed)
            if solved_error < best_error - IK_EQUAL_ERROR_TOL_M or (
                abs(solved_error - best_error) <= IK_EQUAL_ERROR_TOL_M and continuity_cost < best_continuity
            ):
                best_angles = solved_angles
                best_error = solved_error
                best_continuity = continuity_cost

    return best_angles, best_error


def reachability_summary(target_tip_xy: np.ndarray, base_xy: np.ndarray, lengths: tuple[float, float, float]) -> tuple[float, float, bool]:
    target_distance = float(np.linalg.norm(np.asarray(target_tip_xy, dtype=np.float64) - np.asarray(base_xy, dtype=np.float64)))
    max_reach = float(sum(lengths))
    return target_distance, max_reach, target_distance <= max_reach + 1e-9
