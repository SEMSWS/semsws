"""
Phase 4 smoke test: Cerjan absorbing boundaries in the coupled facade
are WIRED UP and reach the per-submesh operators.

The coupled-material YAML supports `boundary.absorbing.sides/thickness/alpha`
and each sub-operator (fluid acoustic / solid elastic) must receive a
DampingConfig that is filtered to the attributes actually present on
its own submesh. This test exercises the full path and asserts:

1. Simulation with Cerjan enabled completes without abort and produces
   a finite, non-zero fluid receiver trace.
2. Cerjan-on and Cerjan-off produce DIFFERENT traces; the sponge is
   doing something, not silently no-op'ing (which would be the failure
   mode when the attr/name mapping isn't right, as happened with the
   default lumped-bdr mesh + name-based sides).

The *quantitative* physics of Cerjan absorption (how much reverb is
actually killed, how flat the transmitted pulse gets) depends heavily
on sponge thickness vs. wavelength and on α tuning; in this bounded-
domain test it's too noisy to pin down a stable tolerance. The
quantitative validation of absorbing behavior moves to the SPECFEM3D
reference comparison (Phase 5) where both codes share the same geometry
and source-time function.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest


TEST_DIR        = Path(__file__).parent
GENERATE_SCRIPT = TEST_DIR / "generate_probe_mesh.py"


LX, LY, LZ = 800.0, 400.0, 800.0
VP_FLUID, RHO_FLUID = 1500.0, 1000.0
VP_SOLID, VS_SOLID, RHO_SOLID = 3000.0, 1732.0, 2500.0
# Source/receiver centered laterally, well inside interior (300+ m from
# any side or the 100 m sponge layer).
SOLID_SOURCE_XYZ = (400.0, 200.0, 600.0)
FLUID_RECV_XYZ   = (400.0, 200.0, 200.0)
FREQ   = 10.0
DELAY  = 0.1
DT     = 2.0e-4
NT     = 3000                 # 0.60 s, covers direct + multiple reverb cycles

# Cerjan smoke config. The absolute numbers aren't physics-tuned; we
# just need the taper to be obviously non-trivial so the "traces
# differ" assertion can tell Cerjan-on and Cerjan-off apart. Source
# (z=600) and receiver (z=200) are 200 m clear of any sponge edge,
# so the first wavefront arrival is essentially untouched regardless.
SPONGE_THICKNESS = 100.0
SPONGE_ALPHA     = 1.0


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


def _write_yaml(yaml_path: Path, mesh_name: str, cerjan_on: bool) -> None:
    if cerjan_on:
        # Parent mesh has bdr attrs 11 (z=0), 12 (z=Lz), 13 (x/y sides).
        # Listing all three applies the Cerjan multiplier in a sponge
        # near the entire outer surface. The interface attribute is
        # auto-detected and is never listed here.
        sides_line = (
            "    sides: [bottom, top, left, right, front, back]\n"
            f"    thickness: {SPONGE_THICKNESS}\n"
            f"    alpha: {SPONGE_ALPHA}"
        )
    else:
        sides_line = (
            "    sides: []\n"
            "    thickness: 0.0\n"
            "    alpha: 0.0"
        )
    yaml_text = textwrap.dedent(f"""\
        name: "Phase 4 Cerjan coupled smoke (cerjan={'on' if cerjan_on else 'off'})"
        simulation:
          dimension: 3
          order: 2
          dt: {DT}
          steps: {NT}
          cfl_factor: 0.3
          directory: "./out"
          log_interval: 100000
        mesh:
          type: external
          file: {mesh_name}
          max_freq: {FREQ * 2.0}
          ppw: 3.0
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
""") + sides_line + textwrap.dedent(f"""
          dirichlet: []
        sources:
          list:
            - id: 1
              name: "solid-force-z"
              type: "force"
              location: [{SOLID_SOURCE_XYZ[0]}, {SOLID_SOURCE_XYZ[1]}, {SOLID_SOURCE_XYZ[2]}]
              direction: [0.0, 0.0, -1.0]
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
              location: [{FLUID_RECV_XYZ[0]}, {FLUID_RECV_XYZ[1]}, {FLUID_RECV_XYZ[2]}]
        device:
          type: cpu
    """)
    yaml_path.write_text(yaml_text)


def _run(exe: Path, yaml_file: Path, np_procs: int, mpi_cmd: str) -> None:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run"]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "TIME_LOOP=completed" in proc.stdout, proc.stdout


def _load_trace(outdir: Path) -> np.ndarray:
    traces = [p for p in outdir.rglob("*.p") if p.is_file() and p.stat().st_size > 32]
    assert traces, f"no pressure trace in {outdir}"
    r01 = sorted(p for p in traces if p.stem.startswith("R01"))
    return np.loadtxt((r01 or sorted(traces))[0])


@pytest.fixture(scope="module")
def cerjan_runs(tmp_path_factory, probe_exe, gmsh_available, mpi_cmd, request):
    np_procs = int(request.config.getoption("--np"))
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_cerjan")

    mesh_file = work / "probe.msh"
    # --mfem-bdrs required: YAML drives Cerjan through named sides
    # (bottom/front/right/back/left/top) that map to MFEM attributes
    # 1..6 via ParseBoundarySide; the default lumped-sides tagging
    # (11/12/13) would leave the DampingConfig targeting attrs that
    # don't exist on the parent mesh, and Cerjan would silently no-op.
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
         "--nx", "8", "--ny", "4", "--nz-half", "4",
         "--mfem-bdrs"],
        check=True, capture_output=True, text=True,
    )
    run_on  = work / "run_on";  run_on.mkdir();  (run_on  / "out").mkdir()
    run_off = work / "run_off"; run_off.mkdir(); (run_off / "out").mkdir()
    (run_on  / "probe.msh").symlink_to(mesh_file)
    (run_off / "probe.msh").symlink_to(mesh_file)
    _write_yaml(run_on  / "coupled.yaml", "probe.msh", cerjan_on=True)
    _write_yaml(run_off / "coupled.yaml", "probe.msh", cerjan_on=False)

    _run(probe_exe, run_on  / "coupled.yaml", np_procs, mpi_cmd)
    _run(probe_exe, run_off / "coupled.yaml", np_procs, mpi_cmd)

    trace_on  = _load_trace(run_on  / "out")
    trace_off = _load_trace(run_off / "out")

    yield trace_on, trace_off, work
    if keep:
        print(f"\n[--keep-results] cerjan workspace retained: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def test_cerjan_on_finite_and_nontrivial(cerjan_runs) -> None:
    trace_on, _, _ = cerjan_runs
    p = trace_on[:, 1]
    assert np.all(np.isfinite(p)), "Cerjan-on trace has NaN/inf (instability?)"
    assert np.max(np.abs(p)) > 0.0, "Cerjan-on trace is identically zero"


def test_cerjan_on_and_off_produce_different_traces(cerjan_runs) -> None:
    """If the coupled facade silently forgets to wire the DampingConfig
    into either sub-operator, the Cerjan-on run becomes identical to
    Cerjan-off. Demand a non-trivial difference so that regression fails
    loudly. We don't pin the exact magnitude because it's sensitive to
    sponge thickness vs. wavelength, which can't be cleanly tuned on a
    bounded test domain."""
    trace_on, trace_off, _ = cerjan_runs
    p_on  = trace_on [:, 1]
    p_off = trace_off[:, 1]
    diff  = p_on - p_off
    peak_off = float(np.max(np.abs(p_off)))
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    assert peak_off > 0.0, "Cerjan-off trace is identically zero"
    # Relative RMS change > 1% means the sponge is clearly active. Tight
    # enough to catch a "Cerjan is secretly a no-op" regression; loose
    # enough to be stable across sponge-tuning decisions.
    assert rms_diff > 0.01 * peak_off, (
        f"Cerjan-on trace barely differs from Cerjan-off: "
        f"rms diff = {rms_diff:.3e}, off peak = {peak_off:.3e} "
        f"(ratio = {rms_diff/peak_off:.4f}). DampingConfig likely "
        f"didn't reach one or both sub-operators."
    )
