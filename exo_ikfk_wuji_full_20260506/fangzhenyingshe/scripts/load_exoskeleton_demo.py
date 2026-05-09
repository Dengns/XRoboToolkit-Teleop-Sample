#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""外骨骼 URDF 的 MuJoCo 最小演示脚本。

功能：
1. 自动把中文路径下的外骨骼资源镜像到 ASCII 临时目录。
2. 处理当前 URDF 在 MuJoCo 中编译时需要的 STL 平铺问题。·
3. 支持列出关节信息。
4. 支持用 MuJoCo viewer 观察简单的关节驱动动画。

当前默认演示对象为：
    third_party/io_mocap_description/blender_human_skeleton_v4.urdf
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "third_party" / "io_mocap_description" / "blender_human_skeleton_v4.urdf"
RIGHT_INDEX_JOINTS = [
    "joint_RightSkeletonIndex1",
    "joint_RightSkeletonIndex2",
    "joint_RightSkeletonIndex3",
    "joint_RightSkeletonIndex4",
]


def infer_mesh_variant(urdf_path: Path) -> str:
    """根据 URDF 文件名推断 mesh 子目录。"""
    name = urdf_path.name.lower()
    if "v4" in name:
        return "v4"
    if "v2" in name:
        return "v2"
    return "old"


def apply_joint_limit_overrides(
    urdf_path: Path,
    limit_overrides: dict[str, tuple[float, float, float, float]] | None,
) -> None:
    """在复制出的 URDF 上覆写关节角度/速度/力矩限制。"""
    if not limit_overrides:
        return

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name or name not in limit_overrides:
            continue
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        lower, upper, effort, velocity = limit_overrides[name]
        limit.set("lower", f"{lower:.6f}")
        limit.set("upper", f"{upper:.6f}")
        limit.set("effort", f"{effort:.6f}")
        limit.set("velocity", f"{velocity:.6f}")

    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def build_ascii_bundle(
    urdf_path: Path,
    limit_overrides: dict[str, tuple[float, float, float, float]] | None = None,
) -> Path:
    """把 URDF 与所需 STL 复制到 ASCII 临时目录。

    MuJoCo 3.6.0 在当前机器上对中文路径敏感，且该外骨骼 URDF
    编译阶段会按 STL 文件名在模型目录中查找网格，因此这里将需要
    的 STL 平铺复制到同一目录，确保能稳定编译。
    """

    if not urdf_path.exists():
        raise FileNotFoundError(f"找不到 URDF: {urdf_path}")

    source_root = urdf_path.parent
    bundle_root = Path(tempfile.gettempdir()) / "fangzhenyingshe_mujoco_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    target_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{urdf_path.stem}_",
            dir=str(bundle_root),
        )
    )

    target_urdf = target_dir / urdf_path.name
    shutil.copy2(urdf_path, target_urdf)

    mesh_variant = infer_mesh_variant(urdf_path)
    mesh_dir = source_root / "meshes" / mesh_variant
    copied_meshes = 0

    if mesh_dir.exists():
        for stl_path in mesh_dir.glob("*.STL"):
            shutil.copy2(stl_path, target_dir / stl_path.name)
            copied_meshes += 1
    else:
        # 兼容已经自包含的精简 URDF：mesh 文件与 URDF 放在同一目录。
        for pattern in ("*.STL", "*.stl"):
            for stl_path in source_root.glob(pattern):
                shutil.copy2(stl_path, target_dir / stl_path.name)
                copied_meshes += 1

    if copied_meshes == 0:
        raise FileNotFoundError(f"找不到可复制的 mesh 资源，URDF: {urdf_path}")

    apply_joint_limit_overrides(target_urdf, limit_overrides)
    return target_urdf


def load_model_from_urdf(
    urdf_path: Path,
    limit_overrides: dict[str, tuple[float, float, float, float]] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    """准备资源并编译 MuJoCo 模型。"""
    bundle_urdf = build_ascii_bundle(urdf_path, limit_overrides=limit_overrides)
    spec = mujoco.MjSpec.from_file(str(bundle_urdf))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, bundle_urdf


def joint_rows(model: mujoco.MjModel) -> list[dict[str, object]]:
    """提取关节信息表。"""
    rows: list[dict[str, object]] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        qpos_adr = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        rows.append(
            {
                "id": joint_id,
                "name": name,
                "qpos_adr": qpos_adr,
                "lower": float(lower),
                "upper": float(upper),
                "movable": abs(float(upper) - float(lower)) > 1e-9,
            }
        )
    return rows


def print_joint_table(model: mujoco.MjModel) -> None:
    """输出关节摘要。"""
    rows = joint_rows(model)
    print("关节列表：")
    for row in rows:
        status = "可动" if row["movable"] else "锁定"
        print(
            f"[{row['id']:02d}] {row['name']:<28} "
            f"qpos={row['qpos_adr']:<2} "
            f"range=({row['lower']:.4f}, {row['upper']:.4f}) {status}"
        )

    movable = [row["name"] for row in rows if row["movable"]]
    print()
    print(f"可动关节数量：{len(movable)} / {len(rows)}")
    if movable:
        print("当前可动关节：")
        for name in movable:
            print(f"- {name}")


def choose_demo_joints(model: mujoco.MjModel, joint_names: list[str] | None) -> list[dict[str, object]]:
    """选择用于演示的关节。"""
    rows = joint_rows(model)
    movable_rows = [row for row in rows if row["movable"]]
    if joint_names:
        wanted = set(joint_names)
        selected = [row for row in movable_rows if row["name"] in wanted]
        missing = [name for name in joint_names if name not in {row["name"] for row in selected}]
        if missing:
            raise ValueError(f"未找到这些可动关节：{', '.join(missing)}")
        return selected

    # 当前 v4 模型实测只有右手拇指链存在非零关节范围。
    return movable_rows


def launch_slider_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    selected_rows: list[dict[str, object]],
    dt: float,
) -> None:
    """打开 MuJoCo 3D viewer，并用滑条实时调关节角。"""
    try:
        import matplotlib
        # Windows + PyQt 在当前环境下会命中 Qt platform plugin 初始化问题，
        # 这里强制切到 TkAgg，避免滑条窗口依赖 Qt。
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        import mujoco.viewer as viewer
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 slider viewer 所需图形依赖，请检查 matplotlib / mujoco.viewer") from exc

    fig_height = max(2.8, 1.2 + 0.8 * len(selected_rows))
    fig = plt.figure("外骨骼关节滑条", figsize=(8, fig_height))
    fig.subplots_adjust(left=0.28, right=0.95, top=0.96, bottom=0.10)

    sliders: list[Slider] = []
    for idx, row in enumerate(selected_rows):
        top = 0.86 - idx * 0.16
        axis = fig.add_axes([0.28, top, 0.62, 0.05])
        slider = Slider(
            ax=axis,
            label=row["name"],
            valmin=float(row["lower"]),
            valmax=float(row["upper"]),
            valinit=0.0,
        )
        sliders.append(slider)

    reset_ax = fig.add_axes([0.05, 0.10, 0.14, 0.07])
    reset_button = Button(reset_ax, "重置")
    reset_button.on_clicked(lambda _event: [slider.reset() for slider in sliders])

    print("已打开滑条窗口。拖动滑条即可实时修改食指关节角。")
    print("关闭任一窗口即可结束。")

    with viewer.launch_passive(model, data) as handle:
        while handle.is_running() and plt.fignum_exists(fig.number):
            for slider, row in zip(sliders, selected_rows):
                data.qpos[int(row["qpos_adr"])] = slider.val
            mujoco.mj_forward(model, data)
            handle.sync()
            plt.pause(dt)

    plt.close(fig)


def build_limit_overrides(unlock_right_index: bool) -> dict[str, tuple[float, float, float, float]] | None:
    """根据命令行参数构建临时关节限制覆写。"""
    if not unlock_right_index:
        return None

    # 先完全放开右手食指四个关节，便于观察动作变化。
    return {joint_name: (-3.14, 3.14, 100.0, 1.0) for joint_name in RIGHT_INDEX_JOINTS}


def animate_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    selected_rows: list[dict[str, object]],
    steps: int,
    dt: float,
) -> None:
    """无 viewer 的简单动画，便于在终端验证。"""
    print("开始无界面动画演示。")
    for step in range(steps):
        phase = step * dt
        for idx, row in enumerate(selected_rows):
            lower = float(row["lower"])
            upper = float(row["upper"])
            center = 0.5 * (lower + upper)
            amplitude = 0.35 * (upper - lower)
            data.qpos[int(row["qpos_adr"])] = center + amplitude * math.sin(phase + idx * 0.35)
        mujoco.mj_forward(model, data)

    print("动画结束，当前关节状态：")
    for row in selected_rows:
        value = data.qpos[int(row["qpos_adr"])]
        print(f"- {row['name']}: {value:.4f}")


def animate_with_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    selected_rows: list[dict[str, object]],
    dt: float,
) -> None:
    """带 viewer 的关节驱动演示。"""
    try:
        import mujoco.viewer as viewer
    except ImportError as exc:
        raise RuntimeError("当前环境没有可用的 mujoco.viewer，请改用 --headless") from exc

    print("打开 MuJoCo viewer。按 Ctrl+C 可退出。")
    with viewer.launch_passive(model, data) as handle:
        start = time.perf_counter()
        while handle.is_running():
            phase = time.perf_counter() - start
            for idx, row in enumerate(selected_rows):
                lower = float(row["lower"])
                upper = float(row["upper"])
                center = 0.5 * (lower + upper)
                amplitude = 0.35 * (upper - lower)
                data.qpos[int(row["qpos_adr"])] = center + amplitude * math.sin(phase + idx * 0.35)
            mujoco.mj_forward(model, data)
            handle.sync()
            time.sleep(dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="载入外骨骼 URDF 并演示 MuJoCo 中的关节操纵。")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="待载入的外骨骼 URDF 路径。默认使用 blender_human_skeleton_v4.urdf",
    )
    parser.add_argument(
        "--mode",
        choices=("list", "animate", "sliders"),
        default="list",
        help="list 输出关节信息；animate 执行动画演示；sliders 打开 3D viewer 与滑条调节窗口。",
    )
    parser.add_argument(
        "--joint",
        action="append",
        help="指定要驱动的关节名，可重复传入。默认驱动所有可动关节。",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="动画模式下不打开 viewer，只做终端验证。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="headless 模式动画步数，默认 240。",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="动画时间步长，默认 0.02 秒。",
    )
    parser.add_argument(
        "--unlock-right-index",
        action="store_true",
        help="临时放开右手食指 4 个关节的角度限制，不修改原始 URDF 文件。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    limit_overrides = build_limit_overrides(args.unlock_right_index)
    model, data, bundle_urdf = load_model_from_urdf(args.urdf, limit_overrides=limit_overrides)
    print(f"已加载模型：{model.njnt} 个关节，{model.nq} 个 qpos，资源镜像：{bundle_urdf.parent}")

    if args.mode == "list":
        print_joint_table(model)
        return 0

    selected_joint_names = args.joint
    if args.mode == "sliders" and not selected_joint_names and args.unlock_right_index:
        selected_joint_names = RIGHT_INDEX_JOINTS

    selected_rows = choose_demo_joints(model, selected_joint_names)
    if not selected_rows:
        raise RuntimeError("没有找到可演示的可动关节。")

    print("本次演示关节：")
    for row in selected_rows:
        print(f"- {row['name']}")

    if args.mode == "sliders":
        launch_slider_viewer(model, data, selected_rows, args.dt)
    elif args.headless:
        animate_headless(model, data, selected_rows, args.steps, args.dt)
    else:
        animate_with_viewer(model, data, selected_rows, args.dt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断，退出演示。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
