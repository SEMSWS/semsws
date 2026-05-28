"""Auto-preflight: runs partition_mesh and semsws_export_model once before
per-shot YAMLs are materialised, returning dict overrides that redirect
mesh:/material: at the freshly written shared directories.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from .layout import Layout
from .run_config import RunConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mesh preflight
# ---------------------------------------------------------------------------

def mesh_needs_partition(mesh: dict, ranks_per_shot: int) -> bool:
    if ranks_per_shot <= 1:
        return False
    return str(mesh.get("type", "")) != "prepart_mfem"


def _resolve_launcher_executable(launcher: str) -> str:
    """Map a `RunConfig.effective_launcher()` tag to an actual executable."""
    return {"local": "mpirun"}.get(launcher, launcher)


def _resolve_companion_binary(semsws_binary: Path, name: str) -> Path:
    sibling = semsws_binary.resolve().parent / name
    if not sibling.exists():
        which = shutil.which(name)
        if which is not None:
            return Path(which)
        raise FileNotFoundError(
            f"{name} not found next to {semsws_binary}; rebuild SEMSWS or "
            f"add {name} to PATH"
        )
    return sibling


def _build_partition_mesh_cmd(
    *, mesh: dict, ranks: int, output_dir: Path, binary: Path,
) -> list[str]:
    cmd: list[str] = [str(binary), "-n", str(ranks)]
    if mesh.get("partition"):
        cmd += ["-p", str(mesh["partition"])]
        if mesh["partition"] == "cartesian":
            grid = mesh.get("partition_grid")
            if not grid:
                raise ValueError(
                    "mesh.partition='cartesian' requires partition_grid"
                )
            cmd += ["-g", ",".join(str(int(x)) for x in grid)]
    mtype = str(mesh.get("type", ""))
    if mtype == "internal":
        cmd += ["--internal", "--dim", str(int(mesh.get("dim", 2)))]
        origin = mesh.get("origin")
        if origin and any(float(x) != 0.0 for x in origin):
            cmd += ["--origin", ",".join(repr(float(x)) for x in origin)]
        size = mesh.get("size")
        if not size:
            raise ValueError("internal mesh requires `size`")
        cmd += ["--size", ",".join(repr(float(x)) for x in size)]
        elements = mesh.get("elements")
        if not elements:
            raise ValueError("internal mesh requires `elements`")
        cmd += ["--elements", ",".join(str(int(x)) for x in elements)]
        if mesh.get("attr_y_threshold") is not None:
            cmd += ["--attr-y-threshold",
                    repr(float(mesh["attr_y_threshold"]))]
        cmd += [str(output_dir)]
    elif mtype == "external":
        if not mesh.get("file"):
            raise ValueError("external mesh requires `file`")
        cmd += [str(mesh["file"]), str(output_dir)]
    else:
        raise ValueError(
            f"unsupported mesh.type for partition preflight: {mtype!r}"
        )
    return cmd


def run_mesh_preflight(
    *,
    mesh: dict,
    rc: RunConfig,
    layout: Layout,
    launcher: str = "mpirun",
) -> dict:
    """Run partition_mesh and return a `mesh:` dict pointing at the shared
    partition directory."""
    output_dir = layout.shared_mesh_partitions_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pm_bin = _resolve_companion_binary(rc.binary, "partition_mesh")
    launcher_exe = _resolve_launcher_executable(launcher)
    cmd = [launcher_exe, "-n", str(rc.ranks_per_shot)] + _build_partition_mesh_cmd(
        mesh=mesh, ranks=rc.ranks_per_shot,
        output_dir=output_dir, binary=pm_bin,
    )
    log.info("[mesh-preflight] %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(layout.workdir),
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        log.error("[mesh-preflight] stdout:\n%s", proc.stdout)
        log.error("[mesh-preflight] stderr:\n%s", proc.stderr)
        raise RuntimeError(
            f"partition_mesh preflight failed (rc={proc.returncode})")
    if proc.stdout.strip():
        log.info("[mesh-preflight] %s", proc.stdout.strip())

    return {
        "type": "prepart_mfem",
        "prepart_mfem": {
            "directory": str(output_dir),
            "nparts": int(rc.ranks_per_shot),
        },
        # Carry forward optional metadata that downstream cares about.
        **{k: mesh[k] for k in ("max_freq", "ppw") if k in mesh},
    }


# ---------------------------------------------------------------------------
# Material preflight
# ---------------------------------------------------------------------------

def material_needs_export(material: dict) -> bool:
    mtype = str(material.get("type", ""))
    if mtype in ("isotropic_acoustic", "isotropic_elastic"):
        # The data lives under `vp` / `vs` / `rho` (and friends); look at
        # those for `format: grid` or any explicit grid sources.
        for key in ("vp", "vs", "rho", "qkappa", "qmu"):
            sub = material.get(key)
            if isinstance(sub, dict):
                if str(sub.get("format", "")) == "grid":
                    return True
                if str(sub.get("type", "")) == "grid":
                    return True
        # by_attribute_mixed style: look one level deeper.
        if isinstance(material.get("by_attribute_mixed"), dict):
            for v in material["by_attribute_mixed"].values():
                if isinstance(v, dict) and (
                    str(v.get("format", "")) == "grid"
                    or str(v.get("type", "")) == "grid"
                ):
                    return True
    return False


def _write_preflight_yaml(
    *,
    template: dict,
    layout: Layout,
    inputs_h5: Path,
    shot_id_for_preflight: int = 0,
) -> Path:
    """Write a self-contained YAML for the export tool. The tool ignores
    sources/receivers/output, but the validator demands them, so we emit
    a reasonable shot YAML."""
    from .template import render_shot_yaml
    out = layout.shared_model_dir / "preflight.yaml"
    return render_shot_yaml(
        template=template, shot_id=shot_id_for_preflight,
        inputs_h5=inputs_h5, out_path=out,
        output_directory=layout.shared_model_dir,
    )


def run_material_preflight(
    *,
    template: dict,
    rc: RunConfig,
    layout: Layout,
    inputs_h5: Path,
    launcher: str = "mpirun",
) -> dict:
    """Run semsws_export_model and return a material override dict pointing
    at the shared BP directory."""
    bp_dir = layout.shared_model_bp_dir
    bp_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = _write_preflight_yaml(
        template=template, layout=layout, inputs_h5=inputs_h5,
    )
    export_bin = _resolve_companion_binary(rc.binary, "semsws_export_model")
    launcher_exe = _resolve_launcher_executable(launcher)
    cmd = [launcher_exe, "-n", str(rc.ranks_per_shot),
           str(export_bin),
           "-config", str(yaml_path),
           "-out", str(bp_dir)]
    log.info("[material-preflight] %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(layout.shared_model_dir),
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        log.error("[material-preflight] stdout:\n%s", proc.stdout)
        log.error("[material-preflight] stderr:\n%s", proc.stderr)
        raise RuntimeError(
            f"semsws_export_model preflight failed (rc={proc.returncode})")
    if proc.stdout.strip():
        log.info("[material-preflight] %s", proc.stdout.strip())

    files: dict[str, str] = {}
    material = template.get("material", {})
    if isinstance(material.get("vp"), dict):
        files["vp"] = str((bp_dir / "vp.bp").resolve())
    if isinstance(material.get("vs"), dict):
        files["vs"] = str((bp_dir / "vs.bp").resolve())
    if isinstance(material.get("rho"), dict):
        files["rho"] = str((bp_dir / "rho.bp").resolve())
    if isinstance(material.get("qkappa"), dict):
        files["qkappa"] = str((bp_dir / "qkappa.bp").resolve())
    if isinstance(material.get("qmu"), dict):
        files["qmu"] = str((bp_dir / "qmu.bp").resolve())

    out = dict(material)
    # Replace each parameter sub-block with adios2 reference.
    for k in list(files.keys()):
        out[k] = {"format": "adios2", "file": files[k]}
    return out
