"""
HDF5 moment-tensor source-input consistency test (Stage 4).

Runs the same waveform config twice with a moment_tensor source:
  (a) inline YAML  : `sources.list[0].type=moment_tensor` + `wavelet.type=ricker`
  (b) HDF5 source  : `sources.file=<bundle.h5>` pointing at a v2.0 file with
                     `/shots/0/sources/S0001/moment_tensor` + `/stf` scalar.

The Python writer generates the Ricker STF samples that the C++ Ricker
factory would compute, so the two runs feed identical samples through the
simulator and produce bit-exact ASCII traces.

Covers 2D (force inline → MT inline + HDF5) and 3D (existing MT inline →
HDF5).

Usage:
    pytest test/consistency/test_source_mt_hdf5_input.py --build-dir ./build -v
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
    # (case_id, base_config, mt_components_override)
    # 2D: force config → swap to MT with these (Mxx, Myy, Mxy) components.
    ("2D/elastic",
     TEST_ROOT / "waveform" / "2D" / "elastic" / "config.yaml",
     {"Mxx": 1.0e9, "Myy": -1.0e9, "Mxy": 0.5e9}),
    # 3D: already MT — pass an explicit override matching the inline values
    # so the HDF5 file uses the same components.
    ("3D/elastic",
     TEST_ROOT / "waveform" / "3D" / "elastic" / "config.yaml",
     {"Mxx": 1.0e10, "Myy": 1.0e10, "Mzz": 1.0e10,
      "Mxy": -1.0e10, "Mxz": 1.0e10, "Myz": -1.0e10}),
]

SHORT_STEPS = 500


def _ricker_samples(nt: int, dt: float, f0: float, t0: float,
                    amplitude: float) -> np.ndarray:
    """Match SourceTimeFunction::Ricker bit-for-bit when SEMSWS is built
    with real_t=float. C++ casts each parameter from real_t (float) to
    double via static_cast, which preserves the *float* rounding (e.g.
    dt=0.001 → float ~0.001000000047 → promoted to double with that
    rounding). We replicate that here."""
    dt_d = float(np.float32(dt))
    f0_d = float(np.float32(f0))
    t0_d = float(np.float32(t0))
    amp_d = float(np.float32(amplitude))

    pi2 = np.pi * np.pi
    f02 = f0_d * f0_d
    t = np.arange(nt, dtype=np.float64) * dt_d - t0_d
    arg = pi2 * f02 * t * t
    s = (1.0 - 2.0 * arg) * np.exp(-arg)
    s = np.where(np.abs(s) < 1e-6, 0.0, s)
    return amp_d * s


def _write_hdf5_mt_source(path: Path, src_def: dict, samples: np.ndarray,
                          mt_components: dict, space_dim: int, dt: float,
                          n_samples: int, shot_id: int = 0) -> None:
    """Write a v2.0 HDF5 file with one moment_tensor source under
    /shots/<shot_id>/sources/S0001/, plus scalar /stf."""
    import h5py
    canonical_3d = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]
    canonical_2d = ["Mxx", "Myy", "Mxy"]
    canonical = canonical_3d if space_dim == 3 else canonical_2d
    mt_values = np.asarray([float(mt_components.get(k, 0.0)) for k in canonical],
                           dtype=np.float64)

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
        s.attrs["type"] = "moment_tensor"
        s.attrs["position"] = np.asarray(
            src_def["location"][:space_dim], dtype=np.float64)

        ds = s.create_dataset("moment_tensor", data=mt_values)
        # Variable-length string array for component_order.
        str_dt = h5py.string_dtype(encoding="utf-8")
        ds.attrs["component_order"] = np.asarray(canonical, dtype=str_dt)
        ds.attrs["coord_system"] = "xyz"

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

    cfg["sources"] = source_section

    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _build_inline_mt_section(base_src_section: dict,
                             mt_components: dict,
                             wavelet_overrides: dict | None = None) -> dict:
    """Take a 'sources' section and rewrite list[0] as moment_tensor with
    the given components and (optionally) a tweaked wavelet."""
    base_list = base_src_section.get("list", [])
    base = dict(base_list[0]) if base_list else {}
    base["type"] = "moment_tensor"
    base["moment_tensor"] = {k: float(v) for k, v in mt_components.items()}
    if wavelet_overrides:
        base["wavelet"] = {**base.get("wavelet", {}), **wavelet_overrides}
    # Strict validator (B7) rejects `direction` on moment_tensor sources.
    base.pop("direction", None)
    return {
        "list": [base],
    }


@pytest.mark.parametrize("case_id,config_path,mt_components", CASES,
                         ids=[c[0] for c in CASES])
def test_hdf5_mt_input_matches_yaml(
    case_id: str,
    config_path: Path,
    mt_components: dict,
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
    wv = inline_src["wavelet"]

    # Generate the same samples both runs will see. C++ inline YAML uses
    # SourceTimeFunction::Ricker; this Python equivalent is computed in
    # f64 and cast to real_t at the same point, so the two paths match.
    samples = _ricker_samples(SHORT_STEPS, dt,
                              float(wv["frequency"]),
                              float(wv.get("delay", 0.0)),
                              float(wv.get("amplitude", 1.0)))

    work = tmp_path_factory.mktemp(f"mt_h5in_{case_id.replace('/', '_')}")

    # --- (a) inline YAML moment_tensor ---------------------------------------
    inline_section = _build_inline_mt_section(src["sources"], mt_components)
    out_a = work / "results_yaml"; out_a.mkdir()
    cfg_a = work / "config_yaml.yaml"
    _materialise_yaml(config_path, cfg_a, out_a, device_type, inline_section)
    ok, msg = _run(executable.resolve(), np_procs, cfg_a, mpi_cmd, cwd=work)
    assert ok, f"{case_id}: YAML MT run failed:\n{msg}"

    # --- (b) HDF5 moment_tensor -----------------------------------------------
    h5_path = work / "sources.h5"
    _write_hdf5_mt_source(h5_path, inline_src, samples, mt_components,
                          space_dim, dt, n_samples=SHORT_STEPS, shot_id=0)

    out_b = work / "results_h5"; out_b.mkdir()
    cfg_b = work / "config_h5.yaml"
    src_section_b = {
        "file": str(h5_path),
        "shot_id": 0,
    }
    _materialise_yaml(config_path, cfg_b, out_b, device_type, src_section_b)
    ok, msg = _run(executable.resolve(), np_procs, cfg_b, mpi_cmd, cwd=work)
    assert ok, f"{case_id}: HDF5 MT run failed:\n{msg}"

    # --- Compare ASCII traces ------------------------------------------------
    # Both (a) and (b) use shot_id=0 (defaults / explicit) → suffix "0000".
    suffix_str = "0000"
    a_traces = _read_all_ascii(out_a, suffix_str)
    b_traces = _read_all_ascii(out_b, suffix_str)
    assert a_traces, f"{case_id}: no ASCII output from YAML MT run"
    assert a_traces.keys() == b_traces.keys(), (
        f"{case_id}: trace filename sets differ\n"
        f"  YAML only: {a_traces.keys() - b_traces.keys()}\n"
        f"  HDF5 only: {b_traces.keys() - a_traces.keys()}"
    )

    failures: list[str] = []
    for name, ax in a_traces.items():
        bx = b_traces[name]
        if ax.shape != bx.shape:
            failures.append(f"{name}: shape {ax.shape} vs {bx.shape}")
            continue
        if not np.array_equal(ax, bx):
            diff = float(np.max(np.abs(ax - bx)))
            failures.append(f"{name}: max abs diff = {diff:.3e}")

    if not keep_results:
        shutil.rmtree(work, ignore_errors=True)

    assert not failures, f"{case_id}: {len(failures)} mismatches:\n" + \
                          "\n".join(failures)
