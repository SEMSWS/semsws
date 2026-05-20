"""Run CoupledSimulationFacade end-to-end with source and receiver both placed
in the fluid submesh; verify the recorded pressure trace shows an arrival
at the expected physical time (distance / vp_fluid). Exercises the
two-submesh time loop and source/receiver domain routing without engaging
the interface coupling itself.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest


TEST_DIR = Path(__file__).parent
GENERATE_SCRIPT = TEST_DIR / "generate_probe_mesh.py"


@pytest.fixture(scope="module")
def probe_exe(build_dir: Path) -> Path:
    candidates = [
        build_dir / "src" / "coupling_probe_facade",   # CMake build dir
        build_dir / "bin" / "coupling_probe_facade",   # Spack install prefix
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    pytest.skip(
        f"coupling_probe_facade not found in any of: "
        + ", ".join(str(c) for c in candidates)
    )


@pytest.fixture(scope="module")
def gmsh_available() -> None:
    try:
        import gmsh  # noqa: F401
    except ImportError:
        pytest.skip("gmsh python module not installed")


# Physical parameters shared between YAML generation and expected-arrival
# assertion. Kept in one place so drift can't accumulate.
LX, LY, LZ = 600.0, 300.0, 400.0   # fluid occupies z=0..LZ/2, solid z=LZ/2..LZ
VP_FLUID   = 1500.0
VP_SOLID   = 3000.0
VS_SOLID   = 1732.0
RHO_FLUID  = 1000.0
RHO_SOLID  = 2500.0
SOURCE_XYZ = (300.0, 150.0, 50.0)   # well inside fluid half
RECV_XYZ   = (300.0, 150.0, 150.0)  # 100 m from source, still in fluid (z < LZ/2 = 200)
FREQ       = 10.0                   # Hz (Ricker), lambda = vp/f = 150 m
DELAY      = 0.1                    # s, puts Ricker peak well inside recorded window
DT         = 2.0e-4
NT         = 1200                   # 0.24 s, several wavelengths post-arrival


def _source_to_recv_distance() -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(SOURCE_XYZ, RECV_XYZ)))


def _expected_arrival_time() -> float:
    """Ricker peak at (delay) + straight-line travel time to receiver."""
    return DELAY + _source_to_recv_distance() / VP_FLUID


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, gmsh_available, request) -> tuple[Path, Path, Path]:
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_waveform")
    mesh_file = work / "probe.msh"
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
         "--nx", "6", "--ny", "3", "--nz-half", "4"],
        check=True, capture_output=True, text=True,
    )

    # Order-4 on this mesh gives ~5 GLL per wavelength at f=10 Hz in fluid.
    yaml_text = textwrap.dedent(f"""\
        name: "Phase 1B coupled waveform (fluid-only src/recv)"
        simulation:
          dimension: 3
          order: 4
          dt: {DT}
          steps: {NT}
          cfl_factor: 0.3
          directory: "./out"
          log_interval: 1000
        mesh:
          type: external
          file: {mesh_file.name}
          max_freq: {FREQ * 2.0}
          ppw: 5.0
        material:
          type: coupled
          fluid:
            attribute: 1
            type: isotropic_acoustic
            format: constant
            vp: {VP_FLUID}
            rho: {RHO_FLUID}
          solid:
            attribute: 2
            type: isotropic_elastic
            format: constant
            vp: {VP_SOLID}
            vs: {VS_SOLID}
            rho: {RHO_SOLID}
        boundary:
          absorbing:
            sides: []
            thickness: 0.0
            alpha: 0.0
          dirichlet: []
        sources:
          list:
            - id: 1
              name: "fluid-pressure"
              type: "pressure"
              location: [{SOURCE_XYZ[0]}, {SOURCE_XYZ[1]}, {SOURCE_XYZ[2]}]
              wavelet:
                type: ricker
                frequency: {FREQ}
                delay: {DELAY}
                amplitude: 1.0e9
        receivers:
          output:
            formats:
              - ascii
            filename: "rcv"
          type: [PS]
          list:
            - name: "R01"
              location: [{RECV_XYZ[0]}, {RECV_XYZ[1]}, {RECV_XYZ[2]}]
        device:
          type: cpu
    """)
    yaml_file = work / "coupled.yaml"
    yaml_file.write_text(yaml_text)

    outdir = work / "out"
    outdir.mkdir(exist_ok=True)

    yield work, yaml_file, outdir
    if keep:
        # Print the preserved path so the user can inspect it from stdout.
        print(f"\n[--keep-results] coupling waveform workspace retained: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def _parse_kv(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _load_ascii_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column ASCII seismogram (time, amplitude)."""
    data = np.loadtxt(path)
    assert data.ndim == 2 and data.shape[1] >= 2, f"unexpected trace shape: {data.shape}"
    return data[:, 0], data[:, 1]


def _find_ascii_trace(outdir: Path) -> Path:
    """Find the PS receiver trace under the output directory.

    SEMSWS ASCII receiver output names the per-receiver file as
    `<name>_<sourceid:04d>.<type>` where <type> is the short receiver
    type token (`p` for pressure / PS). No "fluid" / "solid" suffix is
    baked into per-receiver ASCII files — the filename is driven by the
    receiver's `name` + MaterialFactory type char.
    """
    all_files = [p for p in outdir.rglob("*") if p.is_file()]
    # Pressure trace: extension ".p" (SEMSWS short-form for PS).
    press = [p for p in all_files if p.suffix == ".p"]
    press = [p for p in press if p.stat().st_size > 32]
    assert press, (
        f"no non-empty pressure receiver file; outdir contents = "
        f"{sorted(p.name for p in all_files)}"
    )
    # Prefer the R01 receiver we configured.
    r01 = [p for p in press if p.stem.startswith("R01")]
    return (r01 or press)[0]


@pytest.mark.parametrize("np_procs", [1, 2])
def test_coupled_facade_fluid_only_arrival(
    probe_exe: Path, workspace, mpi_cmd: str, np_procs: int
) -> None:
    _, yaml_file, outdir = workspace

    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(probe_exe),
               "--config", str(yaml_file), "--dim", "3", "--run"]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(probe_exe),
               "--config", str(yaml_file), "--dim", "3", "--run"]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    kv = _parse_kv(proc.stdout)
    assert kv.get("PROBE_OK") == "1"
    assert kv.get("TIME_LOOP") == "completed"
    assert int(kv["NUM_FLUID_SOURCES"]) == 1, (
        "pressure source must route to fluid"
    )
    assert int(kv["NUM_SOLID_SOURCES"]) == 0, (
        "no elastic-domain sources configured"
    )
    assert int(kv["NUM_FLUID_RECEIVERS"]) == 1, (
        "PS receiver must route to fluid"
    )
    assert int(kv["NUM_SOLID_RECEIVERS"]) == 0

    trace_path = _find_ascii_trace(outdir)
    t, p = _load_ascii_trace(trace_path)

    assert np.any(np.abs(p) > 0.0), "fluid receiver trace is identically zero"
    assert np.all(np.isfinite(p)), "trace contains NaN / inf (instability?)"

    # Peak-of-envelope arrival: absolute maximum of the pressure trace.
    peak_idx = int(np.argmax(np.abs(p)))
    t_peak = float(t[peak_idx])

    t_expected = _expected_arrival_time()

    # Tolerance: one Ricker period (1/f) around the expected peak covers
    # waveform shape + small numerical dispersion. We don't aim for a
    # tight match here — the gate is that the waveform isn't wildly
    # mistimed (e.g. arriving at t=0 or never arriving).
    one_period = 1.0 / FREQ
    assert abs(t_peak - t_expected) < one_period, (
        f"pressure peak at t={t_peak:.4f}s, expected ~{t_expected:.4f}s "
        f"(d={_source_to_recv_distance():.1f}m, vp={VP_FLUID:.0f}m/s, "
        f"delay={DELAY}s)"
    )
