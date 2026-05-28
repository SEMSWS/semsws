"""End-to-end smoke: 1 shot, 1 receiver, 2D acoustic with the real SEMSWS.

Skipped automatically when the binary is not built. Runs in /tmp; should
finish in well under 10 s on any laptop.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from semsws_driver import run
from semsws_driver.io import hdf5_schema as S
from semsws_driver.io.hdf5_v2 import (
    ReceiverEntry, ShotEntry, SourceEntry, write_v2,
)

def _find_binary() -> Path | None:
    """Locate `semsws` for smoke tests.

    Honours $SEMSWS_BINARY if set, else `which semsws` (e.g. after
    `spack load semsws`). Tests skip when nothing is found rather than
    using a hard-coded developer path.
    """
    env = os.environ.get("SEMSWS_BINARY")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    found = shutil.which("semsws")
    return Path(found) if found else None


@pytest.fixture
def binary() -> Path:
    b = _find_binary()
    if b is None:
        pytest.skip("SEMSWS binary not built")
    return b


def _ricker_samples(n: int, dt: float, freq: float, delay: float) -> np.ndarray:
    t = np.arange(n) * dt - delay
    x = (np.pi * freq * t) ** 2
    return ((1.0 - 2.0 * x) * np.exp(-x)).astype(np.float64)


def _make_inputs(path: Path, *, nt: int, dt: float):
    src = SourceEntry(
        source_id=1, type=S.SOURCE_TYPE_PRESSURE,
        position=[500.0, 500.0],
        stf=_ricker_samples(nt, dt, freq=10.0, delay=0.1),
    )
    rec = ReceiverEntry(
        receiver_id=1, position=[700.0, 500.0], label="REC1",
    )
    write_v2(path, [ShotEntry(shot_id=0, sources=[src], receivers=[rec])],
             dt=dt, n_samples=nt, space_dim=2)


def _make_template(binary: Path) -> dict:
    return {
        "name": "smoke_2d_acoustic",
        "simulation": {"dimension": 2, "order": 4, "mode": "forward",
                       "steps": 200, "dt": 5e-4, "cfl_factor": 0.5,
                       "log_interval": 200},
        "vis": {"wavefield": {"enabled": False}},
        "mesh": {"type": "internal",
                 "origin": [0.0, 0.0], "size": [1000.0, 1000.0],
                 "elements": [10, 10],
                 "max_freq": 30.0, "ppw": 5.0},
        "material": {"type": "isotropic_acoustic", "format": "constant",
                     "vp": 1500.0, "rho": 1000.0},
        "device": {"type": "cpu"},
        "boundary": {
            "absorbing": {"sides": [], "thickness": 0.0, "alpha": 0.0},
        },
        "receivers": {"type": ["PS"]},
        "sources": {},
        "run": {"binary": str(binary),
                 "scheduler": "local",
                 "launcher": "local",
                 "ranks_per_shot": 1,
                 "shots_per_job": 1,
                 "device_kind": "cpu"},
    }


def test_run_smoke_2d_acoustic(tmp_path: Path, binary: Path):
    inputs = tmp_path / "obs.h5"
    _make_inputs(inputs, nt=200, dt=5e-4)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(_make_template(binary)))
    workdir = tmp_path / "work"

    res = run(config=cfg, inputs=inputs, workdir=workdir)

    # End-to-end success (binary actually ran).
    assert len(res.shot_outcomes) == 1
    rc = res.shot_outcomes[0].return_code
    if rc != 0:
        # Surface logs to help debug.
        log = res.layout.shot_stderr(0)
        if log.exists():
            tail = log.read_text()[-2000:]
            pytest.fail(f"shot returned rc={rc}; stderr tail:\n{tail}")
        pytest.fail(f"shot returned rc={rc}")

    seis = res.layout.shot_seismograms(0)
    assert seis.exists(), f"no {seis}"
    with h5py.File(str(seis), "r") as f:
        # Schema basics
        fv = f.attrs[S.ATTR_FORMAT_VERSION]
        if isinstance(fv, bytes):
            fv = fv.decode()
        assert str(fv) == "2.0"
        assert int(np.asarray(f.attrs[S.ATTR_SPACE_DIM])) == 2
        # Receiver group exists under /shots/0000/receivers/<exactly one>.
        # SEMSWS C++ currently uses the user-friendly label as group name;
        # the F8 ID-based naming is not yet enforced on the writer side.
        rec_grp = f[f"{S.GROUP_SHOTS}/0000/{S.GROUP_RECEIVERS}"]
        rec_names = list(rec_grp.keys())
        assert len(rec_names) == 1
        rec = rec_grp[rec_names[0]]
        assert S.CHANNEL_PRESSURE in rec, \
            f"missing PS dataset; rec keys: {list(rec.keys())}"
        ps = np.asarray(rec[S.CHANNEL_PRESSURE])
        assert ps.shape == (200,)
        # Some non-zero waveform should have arrived at the receiver
        assert np.any(np.abs(ps) > 1e-12)

    # Manifest written, with per-shot start time populated.
    assert res.manifest_path is not None and res.manifest_path.exists()
    import json
    m = json.loads(res.manifest_path.read_text())
    forbidden = {"config_yaml_sha256", "inputs_h5_sha256",
                 "semsws_binary_sha256", "seismograms_sha256"}
    assert not (set(m.keys()) & forbidden)
    assert not (set(m["shots"][0].keys()) & forbidden)
    assert m["shots"][0]["started_at_utc"] is not None
    assert "T" in m["shots"][0]["started_at_utc"]
    # Launch argv: actual mpirun/srun command for the shot.
    argv = m["shots"][0]["launch_argv"]
    assert argv, "launch_argv must be populated"
    assert any("mpirun" in a or "srun" in a for a in argv)
    assert "-config" in argv
