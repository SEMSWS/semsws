"""CLI smoke tests via the argparse main()."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from semsws_driver.cli.main import build_parser, main
from semsws_driver.io import hdf5_schema as S
from semsws_driver.io.hdf5_v2 import (
    ReceiverEntry, ShotEntry, SourceEntry, write_v2,
)


def _make_min_hdf5(path: Path, n_shots: int = 1):
    nt = 4
    shots = []
    for sid in range(n_shots):
        src = SourceEntry(
            source_id=1, type=S.SOURCE_TYPE_PRESSURE,
            position=[0.0, 0.0], stf=np.zeros(nt),
        )
        rec = ReceiverEntry(receiver_id=1, position=[10.0, 0.0])
        shots.append(ShotEntry(shot_id=sid, sources=[src], receivers=[rec]))
    write_v2(path, shots, dt=1e-3, n_samples=nt, space_dim=2)


def _make_template(binary: Path) -> dict:
    return {
        "name": "smoke",
        "simulation": {"dim": 2, "order": 4, "mode": "forward",
                       "steps": 4, "dt": 1e-3},
        "mesh": {"type": "internal", "dim": 2,
                 "origin": [0.0, 0.0], "size": [100.0, 100.0],
                 "elements": [4, 4]},
        "material": {"type": "isotropic_acoustic",
                     "vp":  {"type": "constant", "value": 1500.0},
                     "rho": {"type": "constant", "value": 1000.0}},
        "device": {"type": "cpu"},
        "receivers": {"type": ["PS"]},
        "sources": {},
        "run": {"binary": str(binary),
                 "scheduler": "local",
                 "ranks_per_shot": 1},
    }


def test_help_runs():
    p = build_parser()
    args = p.parse_args(["run", "--config", "x.yaml",
                         "--inputs", "y.h5", "--workdir", "z"])
    assert args.cmd == "run"


def test_dry_run_produces_shot_yamls(tmp_path: Path):
    inputs = tmp_path / "obs.h5"
    _make_min_hdf5(inputs, n_shots=2)
    binary = tmp_path / "fakebin" / "semsws"
    binary.parent.mkdir(parents=True)
    binary.write_text("dummy")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(_make_template(binary)))

    workdir = tmp_path / "wd"
    rc = main(["dry-run",
               "--config", str(cfg),
               "--inputs", str(inputs),
               "--workdir", str(workdir)])
    assert rc == 0
    assert (workdir / "shots" / "shot_0000" / "config.yaml").exists()
    assert (workdir / "shots" / "shot_0001" / "config.yaml").exists()
    body = yaml.safe_load(
        (workdir / "shots" / "shot_0001" / "config.yaml").read_text())
    assert "run" not in body
    assert body["sources"]["shot_id"] == 1


def test_merge_emits_merged_file(tmp_path: Path):
    # Build two minimal per-shot files under shots/shot_NNNN/seismograms.h5.
    workdir = tmp_path / "wd"
    workdir.mkdir()
    for sid in (0, 1):
        d = workdir / "shots" / f"shot_{sid:04d}"
        d.mkdir(parents=True)
        f = d / "seismograms.h5"
        # Re-use the writer to make valid v2 single-shot files.
        nt = 4
        rec = ReceiverEntry(
            receiver_id=1, position=[0.0, 0.0],
            channels={S.CHANNEL_PRESSURE: np.zeros(nt, dtype=np.float32)},
        )
        write_v2(f, [ShotEntry(shot_id=sid,
                                sources=[SourceEntry(
                                    source_id=1, type=S.SOURCE_TYPE_PRESSURE,
                                    position=[0.0, 0.0], stf=np.zeros(nt))],
                                receivers=[rec])],
                 dt=1e-3, n_samples=nt, space_dim=2)
    out = workdir / "results" / "merged.h5"
    rc = main(["merge", "--workdir", str(workdir), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    with h5py.File(str(out), "r") as f:
        assert "0000" in f[S.GROUP_SHOTS]
        assert "0001" in f[S.GROUP_SHOTS]
