"""Verify HDF5 output naming reflects input shot_id.

For each parametrised case, generates a synthetic v2.0 input HDF5 with a
known shot_id and one or more sources, runs semsws, and asserts:

  input shot_id=N, sources [i_0, ..., i_{M-1}]
    → output file: seis<N:04d>.h5
    → internal:    /shots/<N:04d>/   with @shot_id = N

Usage:
    pytest test/consistency/test_hdf5_shot_id_naming.py --build-dir ./build -v
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

TEST_ROOT = Path(__file__).parent.parent
BASE_CONFIG = TEST_ROOT / "waveform" / "2D" / "elastic" / "config.yaml"
SHORT_STEPS = 200


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _ricker_samples(nt: int, dt: float, f0: float, t0: float,
                    amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(nt, dtype=np.float64) * dt - t0
    arg = (np.pi * f0) ** 2 * t * t
    s = (1.0 - 2.0 * arg) * np.exp(-arg)
    return amplitude * s


def _write_input_hdf5(path: Path, *, shot_id: int, source_ids: list[int],
                      space_dim: int, dt: float, n_samples: int,
                      receiver_pos: tuple[float, float]) -> None:
    """Synthesize a v2.0 input HDF5 with given shot_id and source ids.

    All sources are pressure type at distinct positions; STF is the
    same Ricker for each. Receiver list is a single PS receiver.
    """
    samples = _ricker_samples(n_samples, dt, f0=10.0, t0=0.15)
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "2.0"
        f.attrs["dt"] = float(dt)
        f.attrs["t0"] = 0.0
        f.attrs["n_samples"] = np.int64(n_samples)
        f.attrs["space_dim"] = np.int32(space_dim)
        f.attrs["coord_system"] = "cartesian"
        f.attrs["units"] = "SI"

        gshot = f.create_group(f"shots/{shot_id:04d}")
        gshot.attrs["shot_id"] = np.int32(shot_id)

        gsrc = gshot.create_group("sources")
        for idx, sid in enumerate(source_ids):
            s = gsrc.create_group(f"S{sid:04d}")
            s.attrs["id"] = np.int32(sid)
            s.attrs["label"] = ""
            s.attrs["type"] = "pressure"
            # distinct positions to keep the assemble deterministic
            s.attrs["position"] = np.asarray(
                [500.0 + 50.0 * idx, 200.0], dtype=np.float64)[:space_dim]
            s.create_dataset("stf", data=samples.astype(np.float64))

        grecv = gshot.create_group("receivers")
        r = grecv.create_group("R001")
        r.attrs["position"] = np.asarray(receiver_pos, dtype=np.float64)[:space_dim]
        # @types omitted → reader falls back to YAML receivers.type


def _materialise_config(base: Path, dst: Path, *, outdir: Path,
                        device: str, h5_input: Path,
                        shot_id: int) -> None:
    """Strip the inline source/receiver lists from base config and point
    sources / receivers at the synthesized HDF5."""
    with open(base) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("simulation", {})["steps"] = SHORT_STEPS
    cfg["simulation"]["directory"] = str(outdir)
    vis = cfg.get("vis") or {}
    if "wavefield" in vis:
        vis["wavefield"]["enabled"] = False
    cfg.setdefault("device", {})["type"] = device

    cfg["sources"] = {
        "file": str(h5_input),
        "shot_id": shot_id,
    }

    parent_type = cfg["receivers"]["type"]
    cfg["receivers"] = {
        "type": parent_type,
        "output": {
            "formats": ["hdf5"],
            "filename": "seis",
        },
    }

    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _run(executable: Path, np_procs: int, config: Path,
         mpi_cmd: str, cwd: Path, timeout: int = 600) -> tuple[bool, str]:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(executable),
               "--config", str(config)]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(executable),
               "--config", str(config)]
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0:
            return False, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


def _assert_h5_shot(path: Path, expected_shot_id: int,
                    expected_source_ids: list[int]) -> None:
    """Open an output HDF5 and check /shots/<NNNN>/ key, @shot_id, and
    that the only source(s) inside match the expected list."""
    assert path.exists(), f"missing output {path}"
    with h5py.File(path, "r") as f:
        shots = f["/shots"]
        keys = sorted(shots.keys())
        assert keys == [f"{expected_shot_id:04d}"], (
            f"{path}: /shots/ keys = {keys}, expected "
            f"['{expected_shot_id:04d}']"
        )
        gshot = shots[f"{expected_shot_id:04d}"]
        attr_shot_id = int(gshot.attrs["shot_id"])
        assert attr_shot_id == expected_shot_id, (
            f"{path}: /shots/<NNNN>/ @shot_id = {attr_shot_id}, "
            f"expected {expected_shot_id}"
        )

        if "sources" in gshot:
            src_keys = sorted(gshot["sources"].keys())
            expected_keys = sorted(f"S{sid:04d}" for sid in expected_source_ids)
            assert src_keys == expected_keys, (
                f"{path}: /shots/.../sources/ keys = {src_keys}, "
                f"expected {expected_keys}"
            )


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shot_id", [0, 7, 42])
def test_simultaneous_uses_shot_id_for_filename_and_internal(
    shot_id: int,
    executable: Path,
    np_procs: int,
    device_type: str,
    mpi_cmd: str,
    keep_results: bool,
    tmp_path_factory,
):
    """Filename suffix = input shot_id; internal /shots/<NNNN>/ also uses
    input shot_id; all sources fire together into one output file."""
    if not BASE_CONFIG.exists():
        pytest.skip(f"base config not found: {BASE_CONFIG}")

    work = tmp_path_factory.mktemp(f"sim_shot{shot_id}")
    h5_input = work / "input.h5"
    source_ids = [1, 2, 3]
    _write_input_hdf5(h5_input, shot_id=shot_id, source_ids=source_ids,
                      space_dim=2, dt=1.0e-3, n_samples=SHORT_STEPS,
                      receiver_pos=(800.0, 200.0))

    outdir = work / "out"
    outdir.mkdir()
    cfg = work / "config.yaml"
    _materialise_config(BASE_CONFIG, cfg, outdir=outdir, device=device_type,
                        h5_input=h5_input, shot_id=shot_id)

    ok, msg = _run(executable.resolve(), np_procs, cfg, mpi_cmd, cwd=work)
    assert ok, f"semsws run failed:\n{msg}"

    expected_file = outdir / f"seis{shot_id:04d}.h5"
    assert expected_file.exists(), (
        f"simultaneous output should be seis{shot_id:04d}.h5; "
        f"actual files: {sorted(p.name for p in outdir.iterdir())}"
    )
    # No other seis*.h5 should exist (only one file in simultaneous)
    extra = [p.name for p in outdir.glob("seis*.h5")
             if p.name != expected_file.name]
    assert not extra, f"unexpected extra HDF5 files: {extra}"

    _assert_h5_shot(expected_file, shot_id, source_ids)

    if not keep_results:
        shutil.rmtree(work, ignore_errors=True)


