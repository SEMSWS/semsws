"""
Q → ∞ limit test for 3D viscoacoustic kernel.

Runs the 3D acoustic_analytic config twice:
  (A) Pure acoustic (attenuation disabled)                — reference
  (B) Viscoacoustic with Qkappa → ∞ (e.g. 9999)            — test

When Q is very large the Generalized Zener correction and memory-variable
contribution vanish, so (B) should reproduce (A) to high accuracy. Any
significant deviation indicates a bug in the viscoacoustic 3D integrator.

Usage:
    pytest test/consistency/test_visco_qlimit_3d.py --build-dir ./build -v
    pytest test/consistency/test_visco_qlimit_3d.py --build-dir ./build --device hip -v
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


BASE_CONFIG = TEST_ROOT / "waveform" / "3D" / "acoustic_analytic" / "config.yaml"

# Q-limit parameters: Qkappa large enough to make the Zener correction
# negligible but still numerically well-defined.
QKAPPA_LIMIT = 9999.0
F0_HZ = 1.0      # reference frequency for Q-band (must match wavelet center)
N_UNITS = 3

# L2 tolerance: the unrelaxed correction ~ O(1/Q) leaves a tiny phase/amplitude
# offset that accumulates over the run. 2 % is generous but catches real bugs.
L2_THRESHOLD = 0.02


def mutate_config(
    src: Path,
    dst: Path,
    outdir: Path,
    device: str,
    *,
    enable_attenuation: bool,
) -> None:
    """Write a mutated copy of src with acoustic or visco-acoustic settings."""
    with open(src) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("simulation", {})["directory"] = str(outdir)
    vis = cfg.get("vis") or {}
    if "wavefield" in vis:
        vis["wavefield"]["enabled"] = False

    cfg.setdefault("device", {})["type"] = device

    att = cfg["material"].setdefault("attenuation", {})
    if enable_attenuation:
        att.clear()
        att.update({
            "enabled": True,
            "f0": F0_HZ,
            "n_units": N_UNITS,
            "qkappa": QKAPPA_LIMIT,
        })
    else:
        att["enabled"] = False

    # Force ASCII so the comparison path is simple. Canonical form:
    # `receivers.output.formats: [{type: ascii}]` — scalar `format:`
    # shorthand is rejected by YamlConfig::Validate since the config
    # schema cleanup.
    recv_out = cfg["receivers"].setdefault("output", {})
    recv_out.pop("format", None)
    recv_out["formats"] = ["ascii"]
    recv_out["filename"] = "seismograms"

    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def run_simulation(
    executable: Path,
    np_procs: int,
    config: Path,
    mpi_cmd: str,
    cwd: Path,
    timeout: int = 600,
) -> tuple[bool, str]:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(executable),
               "--config", str(config)]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(executable),
               "--config", str(config)]
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True,
                                text=True, timeout=timeout)
        if result.returncode != 0:
            return False, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


def _collect_pressure_traces(outdir: Path) -> dict[str, np.ndarray]:
    """Read all R*_0001.p receiver files into {name: trace}."""
    traces: dict[str, np.ndarray] = {}
    for path in sorted(outdir.iterdir()):
        if path.suffix != ".p" or not path.is_file():
            continue
        stem = path.stem
        # Strip source-id suffix ("_0001") if present.
        if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
            name = stem.rsplit("_", 1)[0]
        else:
            name = stem
        traces[name] = read_sem_ascii(path).data
    return traces


def test_visco_acoustic_3d_qlimit(
    executable: Path,
    np_procs: int,
    device_type: str,
    mpi_cmd: str,
    keep_results: bool,
):
    """Visco-acoustic 3D with Q→∞ must match pure acoustic 3D."""
    if not BASE_CONFIG.exists():
        pytest.skip(f"base config not found: {BASE_CONFIG}")

    # Results go next to the test file so --keep-results leaves them findable.
    work = Path(__file__).parent / "results_visco_qlimit_3d"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    ac_out = work / "acoustic"
    vi_out = work / "visco"
    ac_out.mkdir()
    vi_out.mkdir()

    ac_cfg = work / "config_acoustic.yaml"
    vi_cfg = work / "config_visco.yaml"
    mutate_config(BASE_CONFIG, ac_cfg, ac_out, device_type,
                  enable_attenuation=False)
    mutate_config(BASE_CONFIG, vi_cfg, vi_out, device_type,
                  enable_attenuation=True)

    exe = executable.resolve()
    ok, msg = run_simulation(exe, np_procs, ac_cfg, mpi_cmd, cwd=work)
    assert ok, f"acoustic simulation failed:\n{msg}"

    ok, msg = run_simulation(exe, np_procs, vi_cfg, mpi_cmd, cwd=work)
    assert ok, f"viscoacoustic simulation failed:\n{msg}"

    ac_traces = _collect_pressure_traces(ac_out)
    vi_traces = _collect_pressure_traces(vi_out)
    assert ac_traces, f"no acoustic receiver files in {ac_out}"
    assert vi_traces, f"no viscoacoustic receiver files in {vi_out}"
    assert ac_traces.keys() == vi_traces.keys(), (
        "receiver sets differ between runs:\n"
        f"  acoustic only: {ac_traces.keys() - vi_traces.keys()}\n"
        f"  visco only:    {vi_traces.keys() - ac_traces.keys()}"
    )

    failures: list[str] = []
    for name, ac in ac_traces.items():
        vi = vi_traces[name]
        assert ac.shape == vi.shape, (
            f"{name}: sample count mismatch {ac.shape} vs {vi.shape}"
        )
        ref_norm = float(np.linalg.norm(ac))
        if ref_norm < 1e-30:
            err = float(np.linalg.norm(vi))
        else:
            err = float(np.linalg.norm(ac - vi) / ref_norm)
        if err > L2_THRESHOLD:
            failures.append(f"{name}: L2 error = {err:.4e}")
        else:
            print(f"  {name}: L2 error = {err:.4e}  [PASS]")

    if not keep_results:
        shutil.rmtree(work, ignore_errors=True)

    assert not failures, (
        f"Q→∞ limit mismatch (threshold {L2_THRESHOLD}): {len(failures)} receivers:\n"
        + "\n".join(failures)
    )
