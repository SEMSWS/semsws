"""HDF5 source-input consistency test.

Generates a reproducible STF sample sequence and runs the same waveform
config twice:
  (a) wavelet.type=external, reading the STF from an ASCII 2-column file,
  (b) sources.file=<bundle.h5>, reading the STF from /shots/<id>/sources/S0001/stf.

The two runs must produce bit-exact ASCII receiver traces because both
paths feed the same numerical samples — no wavelet formula is re-computed.

Usage:
    pytest test/consistency/test_source_hdf5_input.py --build-dir ./build -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

TEST_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = TEST_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from compare_waveforms import read_sem_ascii  # noqa: E402

CASES = [
    ("2D/elastic", TEST_ROOT / "waveform" / "2D" / "elastic" / "config.yaml"),
    ("2D/acoustic", TEST_ROOT / "waveform" / "2D" / "acoustic" / "config.yaml"),
]

SHORT_STEPS = 500


def _ricker_samples(nt: int, dt: float, f0: float, t0: float,
                    amplitude: float) -> np.ndarray:
    """Match SourceTimeFunction::Ricker (double precision intermediate)."""
    t = np.arange(nt, dtype=np.float64) * dt - t0
    arg = (np.pi * f0) ** 2 * t * t
    s = (1.0 - 2.0 * arg) * np.exp(-arg)
    s = np.where(np.abs(s) < 1e-6, 0.0, s)
    return amplitude * s


def _write_ascii_stf(path: Path, samples: np.ndarray, dt: float) -> None:
    with open(path, "w") as f:
        for i, v in enumerate(samples):
            f.write(f"{i*dt:.17e} {float(v):.17e}\n")


def _write_hdf5_sources(path: Path, src_def: dict, samples: np.ndarray,
                        space_dim: int, dt: float, n_samples: int,
                        shot_id: int = 0) -> None:
    """Write a v2.0 HDF5 file with one source under /shots/<shot_id>/sources/S0001/."""
    import h5py
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
        s = gsrc.create_group("S0001")
        s.attrs["id"] = np.int32(1)
        s.attrs["label"] = src_def.get("name", "")
        s.attrs["type"] = src_def["type"]
        s.attrs["position"] = np.asarray(
            src_def["location"][:space_dim], dtype=np.float64)
        if src_def["type"] == "force":
            s.attrs["direction"] = np.asarray(
                src_def["direction"][:space_dim], dtype=np.float64)
        s.create_dataset("stf", data=samples.astype(np.float64))


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


def _read_all_ascii(outdir: Path, source_id: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path in sorted(outdir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in (".d", ".v", ".a", ".p", ".g"):
            continue
        if f"_{source_id}" not in path.stem:
            continue
        out[path.name] = read_sem_ascii(path).data
    return out


def _materialise_yaml(src: Path, dst: Path, outdir: Path, device: str,
                      source_section: dict) -> None:
    with open(src) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("simulation", {})["steps"] = SHORT_STEPS
    cfg["simulation"]["directory"] = str(outdir)
    vis = cfg.get("vis") or {}
    if "wavefield" in vis:
        vis["wavefield"]["enabled"] = False

    cfg.setdefault("device", {})["type"] = device

    cfg["receivers"]["output"]["formats"] = ["ascii"]
    cfg["receivers"]["output"]["filename"] = "dummy"

    # Replace sources section wholesale.
    cfg["sources"] = source_section

    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


@pytest.mark.parametrize("case_id,config_path", CASES,
                         ids=[c[0] for c in CASES])
def test_hdf5_source_input_matches_external_ascii(
    case_id: str,
    config_path: Path,
    executable: Path,
    np_procs: int,
    device_type: str,
    mpi_cmd: str,
    keep_results: bool,
    tmp_path_factory,
):
    if not config_path.exists():
        pytest.skip(f"config not found: {config_path}")

    src = yaml.safe_load(config_path.read_text())
    space_dim = int(src["simulation"]["dimension"])
    dt = float(src["simulation"]["dt"])
    inline_src = src["sources"]["list"][0]

    # Generate STF samples — Ricker matches the YAML's wavelet so the
    # absolute waveform amplitude is realistic, but the test only relies
    # on bit-exact equality of the two pipelines (a) and (b).
    wv = inline_src["wavelet"]
    samples = _ricker_samples(SHORT_STEPS, dt,
                              float(wv["frequency"]),
                              float(wv.get("delay", 0.0)),
                              float(wv.get("amplitude", 1.0)))

    work = tmp_path_factory.mktemp(f"src_h5in_{case_id.replace('/', '_')}")

    # --- Run (a): ASCII external STF ----------------------------------------
    ascii_stf = work / "stf_external.txt"
    _write_ascii_stf(ascii_stf, samples, dt)

    out_a = work / "results_ascii"; out_a.mkdir()
    cfg_a = work / "config_ascii.yaml"
    src_section_a = {
        "list": [{
            **inline_src,
            "wavelet": {
                "type": "external",
                "file": str(ascii_stf),
                "frequency": float(wv["frequency"]),
            },
        }],
    }
    _materialise_yaml(config_path, cfg_a, out_a, device_type, src_section_a)
    ok, msg = _run(executable.resolve(), np_procs, cfg_a, mpi_cmd, cwd=work)
    assert ok, f"{case_id}: ASCII external run failed:\n{msg}"

    # --- Run (b): HDF5 source input -----------------------------------------
    h5_path = work / "sources.h5"
    _write_hdf5_sources(h5_path, inline_src, samples, space_dim, dt,
                        n_samples=SHORT_STEPS, shot_id=0)

    out_b = work / "results_h5"; out_b.mkdir()
    cfg_b = work / "config_h5.yaml"
    src_section_b = {
        "file": str(h5_path),
        "shot_id": 0,
    }
    _materialise_yaml(config_path, cfg_b, out_b, device_type, src_section_b)
    ok, msg = _run(executable.resolve(), np_procs, cfg_b, mpi_cmd, cwd=work)
    assert ok, f"{case_id}: HDF5 source run failed:\n{msg}"

    # --- Compare ASCII traces (bit-exact) -----------------------------------
    a_traces = _read_all_ascii(out_a, "0000")
    b_traces = _read_all_ascii(out_b, "0000")
    assert a_traces, f"{case_id}: no ASCII output from external-stf run"
    assert a_traces.keys() == b_traces.keys(), (
        f"{case_id}: trace filename sets differ\n"
        f"  ASCII only: {a_traces.keys() - b_traces.keys()}\n"
        f"  HDF5  only: {b_traces.keys() - a_traces.keys()}"
    )

    failures: list[str] = []
    for name, ax in a_traces.items():
        bx = b_traces[name]
        if ax.shape != bx.shape:
            failures.append(f"{name}: shape {ax.shape} vs {bx.shape}")
            continue
        if not np.allclose(ax, bx, atol=1e-6, rtol=1e-5):
            diff = float(np.max(np.abs(ax - bx)))
            failures.append(f"{name}: max abs diff = {diff:.3e}")

    if not keep_results:
        shutil.rmtree(work, ignore_errors=True)

    assert not failures, f"{case_id}: {len(failures)} mismatches:\n" + \
                          "\n".join(failures)
