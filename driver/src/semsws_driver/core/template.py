"""Per-shot YAML rendering from the user template.

Deep-copies the user template, redirects sources/receivers at the bundled
HDF5 with the per-shot id, forces `receivers.output.formats` to
`[{type: hdf5}]`, strips `run:`, and writes `<workdir>/shots/shot_NNNN/config.yaml`.
Optional mesh/material overrides from the preflight stage are merged in.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

from .layout import Layout
from .run_config import RunConfig


def load_template(path: Path) -> dict:
    """Load and return the user template as a YAML dict.

    `run:` is optional. Callers that need a RunConfig (single-shot driver)
    should invoke `extract_run_config(template)` which validates presence.
    Multi-config callers (fwi-driver) supply `run:` from a separate file.
    """
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    return data


def extract_run_config(template: dict) -> RunConfig:
    """Pull `run:` out of the loaded template and parse into a RunConfig."""
    if "run" not in template:
        raise ValueError("template missing required `run:` section")
    return RunConfig.from_dict(dict(template["run"]))


def render_shot_yaml(
    *,
    template: dict,
    shot_id: int,
    inputs_h5: Path,
    out_path: Path,
    mesh_override: Optional[dict] = None,
    material_override: Optional[dict] = None,
    output_directory: Optional[Path] = None,
) -> Path:
    """Materialise one shot's config.yaml from the template.

    `inputs_h5` is the path the SEMSWS process will see (typically a
    relative path from the shot's workdir to `<workdir>/inputs/observations.h5`,
    or an absolute path).

    `mesh_override` / `material_override`, when provided, replace the
    template's `mesh:` / `material:` blocks (used by preflight to redirect
    to a shared partition / BP directory).
    """
    cfg = copy.deepcopy(template)

    # Strip driver-only keys before SEMSWS sees the YAML.
    cfg.pop("run", None)

    # Source / receiver inputs come from the bundled HDF5.
    inputs_str = str(inputs_h5)
    cfg["sources"] = {
        "file": inputs_str,
        "shot_id": int(shot_id),
    }
    rcv = {}
    # Preserve user's `receivers.type` if present; receivers geometry comes
    # from sources.file (same HDF5 bundle) automatically.
    user_rcv = template.get("receivers") or {}
    if isinstance(user_rcv, dict) and "type" in user_rcv:
        rcv["type"] = user_rcv["type"]
    rcv["output"] = {
        "formats": ["hdf5"],
        "filename": "seismograms",
    }
    cfg["receivers"] = rcv

    if mesh_override is not None:
        cfg["mesh"] = copy.deepcopy(mesh_override)
    if material_override is not None:
        cfg["material"] = copy.deepcopy(material_override)

    # SEMSWS validates simulation.directory; driver knows the per-shot
    # workdir so we auto-fill it. User overrides anything they explicitly set.
    if output_directory is not None:
        sim = cfg.setdefault("simulation", {})
        sim.setdefault("directory", str(output_directory))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return out_path


def render_all_shots(
    *,
    template: dict,
    shot_ids: list[int],
    layout: Layout,
    inputs_h5: Path,
    mesh_override: Optional[dict] = None,
    material_override: Optional[dict] = None,
) -> list[Path]:
    """Write one config.yaml per shot under <workdir>/shots/shot_NNNN/."""
    out: list[Path] = []
    for sid in shot_ids:
        out.append(render_shot_yaml(
            template=template,
            shot_id=sid,
            inputs_h5=inputs_h5,
            out_path=layout.shot_config(sid),
            mesh_override=mesh_override,
            material_override=material_override,
            output_directory=layout.shot_dir(sid),
        ))
    return out
