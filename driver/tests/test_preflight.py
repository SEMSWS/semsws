"""Preflight detection + dispatch tests (subprocess monkeypatched)."""

from __future__ import annotations

from pathlib import Path

import pytest

from semsws_driver.core import preflight
from semsws_driver.core.layout import Layout
from semsws_driver.core.run_config import RunConfig


def test_mesh_needs_partition_skips_single_rank():
    assert preflight.mesh_needs_partition({"type": "internal"}, ranks_per_shot=1) is False


def test_mesh_needs_partition_skips_already_partitioned():
    assert preflight.mesh_needs_partition({"type": "prepart_mfem"}, ranks_per_shot=4) is False


def test_mesh_needs_partition_true_for_internal_multi_rank():
    assert preflight.mesh_needs_partition({"type": "internal"}, ranks_per_shot=4) is True
    assert preflight.mesh_needs_partition({"type": "external", "file": "x.mesh"},
                                          ranks_per_shot=4) is True


def test_material_needs_export_true_for_grid():
    mat = {"type": "isotropic_elastic",
           "vp":  {"format": "grid", "file": "vp.txt"},
           "vs":  {"type": "constant", "value": 1700.0},
           "rho": {"type": "constant", "value": 2200.0}}
    assert preflight.material_needs_export(mat) is True


def test_material_needs_export_false_for_constant():
    mat = {"type": "isotropic_elastic",
           "vp":  {"type": "constant", "value": 3000.0},
           "vs":  {"type": "constant", "value": 1700.0},
           "rho": {"type": "constant", "value": 2200.0}}
    assert preflight.material_needs_export(mat) is False


def test_material_needs_export_handles_by_attribute_mixed():
    mat = {"type": "isotropic_elastic",
           "by_attribute_mixed": {
               "1": {"format": "grid", "file": "x.txt"},
               "2": {"type": "constant", "value": 2.0},
           }}
    assert preflight.material_needs_export(mat) is True


def test_run_mesh_preflight_invokes_subprocess(tmp_path: Path, monkeypatch):
    L = Layout.at(tmp_path)
    L.make_skeleton()

    captured = []

    def fake_run(cmd, **kw):
        captured.append(cmd)
        # Pretend partition_mesh succeeded.
        class P:
            returncode = 0
            stdout = ""
            stderr = ""
        return P()

    fake_bin = tmp_path / "fakebin" / "partition_mesh"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    fake_semsws = fake_bin.parent / "semsws"
    fake_semsws.write_text("dummy")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    rc = RunConfig.from_dict({"binary": str(fake_semsws), "ranks_per_shot": 4,
                              "scheduler": "local"})
    mesh = {"type": "internal", "dim": 2, "size": [100.0, 100.0],
            "elements": [4, 4], "partition": "cartesian",
            "partition_grid": [2, 2], "max_freq": 30.0, "ppw": 5.0}
    out = preflight.run_mesh_preflight(
        mesh=mesh, rc=rc, layout=L, launcher="mpirun",
    )
    assert out["type"] == "prepart_mfem"
    # SEMSWS expects mesh.prepart_mfem.{directory, nparts} sub-block.
    assert out["prepart_mfem"]["nparts"] == 4
    assert Path(out["prepart_mfem"]["directory"]) == L.shared_mesh_partitions_dir
    assert out["max_freq"] == 30.0   # carry-through metadata
    assert captured, "subprocess should have been invoked"
    cmd = captured[0]
    assert cmd[0] == "mpirun"
    assert "-n" in cmd
    assert str(fake_bin) in cmd


def test_launcher_local_tag_resolves_to_mpirun(tmp_path: Path, monkeypatch):
    """Regression: launcher='local' must be mapped to mpirun (or similar)
    before subprocess.run; otherwise execve fails with FileNotFoundError."""
    L = Layout.at(tmp_path)
    L.make_skeleton()
    captured = []

    def fake_run(cmd, **kw):
        captured.append(list(cmd))
        class P:
            returncode = 0; stdout = ""; stderr = ""
        return P()

    fake_bin = tmp_path / "fakebin" / "partition_mesh"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    fake_semsws = fake_bin.parent / "semsws"
    fake_semsws.write_text("dummy")
    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    rc = RunConfig.from_dict({"binary": str(fake_semsws), "ranks_per_shot": 4,
                              "scheduler": "local"})
    mesh = {"type": "internal", "dim": 2, "size": [100.0, 100.0],
            "elements": [4, 4], "partition": "cartesian",
            "partition_grid": [2, 2]}
    preflight.run_mesh_preflight(
        mesh=mesh, rc=rc, layout=L,
        launcher="local",   # logical tag, not an executable
    )
    assert captured[0][0] == "mpirun", \
        f"first arg must be a real executable, got {captured[0][0]!r}"


def test_partition_mesh_command_demands_partition_grid_for_cartesian():
    with pytest.raises(ValueError, match="partition_grid"):
        preflight._build_partition_mesh_cmd(
            mesh={"type": "internal", "dim": 2,
                  "size": [100.0, 100.0], "elements": [4, 4],
                  "partition": "cartesian"},
            ranks=4,
            output_dir=Path("/tmp/out"),
            binary=Path("/tmp/partition_mesh"),
        )
