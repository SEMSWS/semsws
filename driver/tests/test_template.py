"""Template loading + shot YAML synthesis tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from semsws_driver.core.layout import Layout
from semsws_driver.core.template import (
    extract_run_config,
    load_template,
    render_all_shots,
    render_shot_yaml,
)


@pytest.fixture
def user_template() -> dict:
    return {
        "name": "demo",
        "simulation": {"dim": 2, "order": 4, "mode": "forward",
                       "steps": 1000, "dt": 1e-3},
        "mesh": {"type": "internal", "dim": 2,
                 "origin": [0.0, 0.0], "size": [1000.0, 1000.0],
                 "elements": [10, 10]},
        "material": {"type": "isotropic_elastic",
                     "vp": {"type": "constant", "value": 3000.0},
                     "vs": {"type": "constant", "value": 1700.0},
                     "rho": {"type": "constant", "value": 2200.0}},
        "device": {"type": "cpu"},
        "receivers": {"type": ["VEL"]},
        "sources": {},  # placeholder, driver overrides
        "run": {"binary": "/path/to/semsws",
                 "scheduler": "local",
                 "ranks_per_shot": 1},
    }


def test_load_template(tmp_path: Path, user_template: dict):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(user_template))
    loaded = load_template(p)
    assert loaded["name"] == "demo"
    assert loaded["run"]["binary"] == "/path/to/semsws"


def test_load_template_rejects_missing_run(tmp_path: Path, user_template: dict):
    user_template.pop("run")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(user_template))
    with pytest.raises(ValueError, match="run"):
        load_template(p)


def test_extract_run_config(user_template: dict):
    rc = extract_run_config(user_template)
    assert str(rc.binary) == "/path/to/semsws"
    assert rc.scheduler == "local"


def test_render_shot_yaml_strips_run(tmp_path: Path, user_template: dict):
    out = tmp_path / "shot_0001" / "config.yaml"
    render_shot_yaml(
        template=user_template, shot_id=1,
        inputs_h5=Path("../../inputs/observations.h5"),
        out_path=out,
    )
    body = yaml.safe_load(out.read_text())
    assert "run" not in body, "driver-only run section must be stripped"


def test_render_shot_yaml_overrides_sources_receivers(
    tmp_path: Path, user_template: dict,
):
    out = tmp_path / "shot_0007" / "config.yaml"
    render_shot_yaml(
        template=user_template, shot_id=7,
        inputs_h5=Path("/abs/path/observations.h5"),
        out_path=out,
    )
    body = yaml.safe_load(out.read_text())
    assert body["sources"] == {"file": "/abs/path/observations.h5",
                                "shot_id": 7}
    # Preserve user's `type` list; receivers geometry inherits from sources.file
    assert body["receivers"]["type"] == ["VEL"]
    # Force HDF5 output
    assert body["receivers"]["output"]["formats"] == ["hdf5"]
    # No file/format/shot_id under receivers — auto-inherits from sources.file
    assert "file" not in body["receivers"]
    assert "format" not in body["receivers"]
    assert "shot_id" not in body["receivers"]


def test_render_shot_yaml_keeps_mesh_material_unmodified(
    tmp_path: Path, user_template: dict,
):
    out = tmp_path / "shot_0001" / "config.yaml"
    render_shot_yaml(
        template=user_template, shot_id=1,
        inputs_h5=Path("/x.h5"), out_path=out,
    )
    body = yaml.safe_load(out.read_text())
    assert body["mesh"] == user_template["mesh"]
    assert body["material"] == user_template["material"]


def test_render_with_mesh_material_override(
    tmp_path: Path, user_template: dict,
):
    out = tmp_path / "shot_0001" / "config.yaml"
    new_mesh = {"type": "prepart_mfem", "directory": "/shared/parts",
                 "max_freq": 30.0, "ppw": 5.0}
    new_mat = {"type": "isotropic_elastic",
                "vp": {"format": "adios2", "files": {"vp": "/x/vp.bp"}}}
    render_shot_yaml(
        template=user_template, shot_id=1,
        inputs_h5=Path("/x.h5"), out_path=out,
        mesh_override=new_mesh, material_override=new_mat,
    )
    body = yaml.safe_load(out.read_text())
    assert body["mesh"] == new_mesh
    assert body["material"] == new_mat


def test_render_all_shots(tmp_path: Path, user_template: dict):
    L = Layout.at(tmp_path)
    L.make_skeleton()
    paths = render_all_shots(
        template=user_template,
        shot_ids=[0, 1, 5],
        layout=L,
        inputs_h5=Path("/abs/observations.h5"),
    )
    assert paths == [L.shot_config(0), L.shot_config(1), L.shot_config(5)]
    for p in paths:
        assert p.exists()
    body = yaml.safe_load(L.shot_config(5).read_text())
    assert body["sources"]["shot_id"] == 5
