from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "third_party" / "io_mocap_description" / "blender_human_skeleton_v4.urdf"
RIGHT_INDEX_JOINTS = [
    "joint_RightSkeletonIndex1",
    "joint_RightSkeletonIndex2",
    "joint_RightSkeletonIndex3",
    "joint_RightSkeletonIndex4",
]


def referenced_mesh_basenames(urdf_path: Path) -> set[str]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    names: set[str] = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            names.add(Path(filename).name)
    return names


def infer_mesh_variant(urdf_path: Path) -> str:
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
    if not urdf_path.exists():
        raise FileNotFoundError(f"找不到 URDF: {urdf_path}")

    source_root = urdf_path.parent
    bundle_root = Path(tempfile.gettempdir()) / "fangzhenyingshe_mujoco_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    target_dir = Path(tempfile.mkdtemp(prefix=f"{urdf_path.stem}_", dir=str(bundle_root)))
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
        for pattern in ("*.STL", "*.stl"):
            for stl_path in source_root.glob(pattern):
                shutil.copy2(stl_path, target_dir / stl_path.name)
                copied_meshes += 1

    if copied_meshes == 0:
        raise FileNotFoundError(f"找不到可复制的 mesh 资源，URDF: {urdf_path}")

    apply_joint_limit_overrides(target_urdf, limit_overrides)

    # Windows assets can hide case mismatches that MuJoCo exposes on Linux.
    copied_by_lower = {path.name.lower(): path for path in target_dir.glob("*") if path.is_file()}
    for mesh_name in referenced_mesh_basenames(target_urdf):
        target_mesh = target_dir / mesh_name
        if target_mesh.exists():
            continue
        source_mesh = copied_by_lower.get(mesh_name.lower())
        if source_mesh is not None:
            shutil.copy2(source_mesh, target_mesh)

    return target_urdf


def load_model_from_urdf(
    urdf_path: Path,
    limit_overrides: dict[str, tuple[float, float, float, float]] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    bundle_urdf = build_ascii_bundle(urdf_path, limit_overrides=limit_overrides)
    spec = mujoco.MjSpec.from_file(str(bundle_urdf))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, bundle_urdf
