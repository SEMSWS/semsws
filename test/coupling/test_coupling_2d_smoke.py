"""
First integration gate for the 2D fluid-solid coupled path.

`CoupledSimulationFacade<2>` is instantiated by the build system, and every
`if constexpr (Dim == 2)` branch in the facade/interface code is exercised
only when a 2D YAML actually runs end-to-end. Before this test no such
run existed — the 3D test suite validates only the <3> instantiation, so
the 2D path was silently bit-rotting territory.

Test design mirrors `test_coupling_cross_domain.py` (the 3D Phase 3a
direct-physics gate), but in 2D and with the analogous geometry:

    y = Lz   ┌─────────────┐
             │   SOLID     │   (attr=2)
             │   force src │
     Lz/2   ─┼──── Γ_fs ───┤   (auto-generated interface attr)
             │   FLUID     │   (attr=1)
             │   PS recv   │
      y = 0  └─────────────┘
             x = 0 .. Lx

Two checks, exactly like the 3D sibling:

1. With coupling DISABLED the fluid pressure trace must be identically
   zero (no source anywhere in the fluid, no initial condition).
2. With coupling ENABLED the trace must be finite and above a weak
   amplitude floor: the only way energy reaches the fluid receiver is
   through `Γ_fs` via `FluidSolidInterface<2>::ApplySolidToFluidRHS`.

Passing both sides proves every 2D branch in the coupled code path
(Setup, FluidSolidInterface<2> with H1_SegmentElement faces, 2D
operator factories, 2D source loader) is wired up and physically
meaningful. It is intentionally NOT a quantitative SPECFEM2D
comparison — the 2D vs SPECFEM2D reference is a separate, larger
benchmark that can land once the basic path is smoke-tested here.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest


TEST_DIR        = Path(__file__).parent
GENERATE_SCRIPT = TEST_DIR / "generate_probe_mesh_2d.py"


# Geometry (x = horizontal, y = vertical). Solid on top, fluid on bottom
# so the source (solid) sits OVER the receiver (fluid), matching the 3D
# sibling's convention.
LX, LZ = 1200.0, 800.0
VP_FLUID, RHO_FLUID           = 1500.0, 1000.0
VP_SOLID, VS_SOLID, RHO_SOLID = 3000.0, 1732.0, 2500.0

SOLID_SOURCE_XY = (600.0, 600.0)   # y = 600  → solid half (>Lz/2 = 400)
FLUID_RECV_XY   = (600.0, 200.0)   # y = 200  → fluid half (<Lz/2 = 400)

FREQ  = 10.0
DELAY = 0.1
# CFL in 2D is dominated by Vp_solid = 3000 and the ~100 m element size
# (NX=12, NZ_HALF=4 ⇒ h ≈ 100). dt_max ≈ 0.3*100/(4*3000) ≈ 2.5e-3; use
# 2e-4 for margin plus order-4 GLL stability.
DT = 2.0e-4
NT = 1200


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


def _write_yaml(yaml_path: Path, mesh_name: str) -> None:
    # Note: 2D force direction is 2-component. Point it toward the
    # interface (−y) so solid motion radiates downward into the fluid.
    yaml_text = textwrap.dedent(f"""\
        name: "2D coupling smoke (solid force src -> fluid PS recv)"
        simulation:
          dimension: 2
          order: 4
          dt: {DT}
          steps: {NT}
          cfl_factor: 0.3
          directory: "./out"
          log_interval: 10000
        mesh:
          type: external
          file: {mesh_name}
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
          dirichlet: [11, 12, 13]
        sources:
          list:
            - id: 1
              name: "solid-force-y"
              type: "force"
              location: [{SOLID_SOURCE_XY[0]}, {SOLID_SOURCE_XY[1]}]
              direction: [0.0, -1.0]     # toward Γ_fs at y = Lz/2
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
              location: [{FLUID_RECV_XY[0]}, {FLUID_RECV_XY[1]}]
        device:
          type: cpu
    """)
    yaml_path.write_text(yaml_text)


def _run(exe: Path, yaml_file: Path, np_procs: int, mpi_cmd: str,
         enable_coupling: bool) -> None:
    extra = [] if enable_coupling else ["--no-coupling"]
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "2", "--run", *extra]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "2", "--run", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode} "
        f"(coupling={'on' if enable_coupling else 'off'})"
        f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "TIME_LOOP=completed" in proc.stdout, proc.stdout


def _load_pressure_trace(outdir: Path) -> np.ndarray:
    traces = [p for p in outdir.rglob("*.p")
              if p.is_file() and p.stat().st_size > 32]
    assert traces, (
        f"no pressure trace in {outdir} "
        f"(dirs: {[p.name for p in outdir.rglob('*')]})"
    )
    r01 = sorted(p for p in traces if p.stem.startswith("R01"))
    chosen = (r01 or sorted(traces))[0]
    return np.loadtxt(chosen)


@pytest.fixture(scope="module")
def coupling_2d_runs(tmp_path_factory, probe_exe, gmsh_available,
                     mpi_cmd, request):
    np_procs = int(request.config.getoption("--np"))
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_2d_smoke")

    mesh_file = work / "probe.msh"
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--lz", str(LZ),
         "--nx", "12", "--nz-half", "4"],
        check=True, capture_output=True, text=True,
    )

    run_on  = work / "run_on"
    run_off = work / "run_off"
    run_on.mkdir()
    run_off.mkdir()
    for rd in (run_on, run_off):
        (rd / "probe.msh").symlink_to(mesh_file)
        (rd / "out").mkdir()
        _write_yaml(rd / "coupled.yaml", "probe.msh")

    _run(probe_exe, run_on  / "coupled.yaml", np_procs, mpi_cmd,
         enable_coupling=True)
    _run(probe_exe, run_off / "coupled.yaml", np_procs, mpi_cmd,
         enable_coupling=False)

    trace_on  = _load_pressure_trace(run_on  / "out")
    trace_off = _load_pressure_trace(run_off / "out")
    yield trace_on, trace_off, work
    if keep:
        print(f"\n[--keep-results] 2D smoke workspace retained: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def test_fluid_silent_without_coupling_2d(coupling_2d_runs) -> None:
    """Coupling OFF + no fluid source ⇒ fluid pressure trace is
    identically zero. Any non-zero sample means state is leaking across
    the 2D submesh boundary."""
    _, trace_off, _ = coupling_2d_runs
    p_off = trace_off[:, 1]
    assert np.all(p_off == 0.0), (
        f"2D fluid trace non-zero under --no-coupling: "
        f"max |p| = {float(np.max(np.abs(p_off))):.3e}"
    )


def test_fluid_activated_by_solid_source_via_coupling_2d(
    coupling_2d_runs,
) -> None:
    """Coupling ON + solid-only force source ⇒ fluid receiver records
    non-zero pressure above a weak amplitude floor. The only path from
    solid motion to fluid pressure is `FluidSolidInterface<2>`'s
    Apply*RHS kernels, so a dead 2D interface wiring falls right here."""
    trace_on, _, _ = coupling_2d_runs
    p_on = trace_on[:, 1]
    peak = float(np.max(np.abs(p_on)))

    assert np.all(np.isfinite(p_on)), "2D coupled fluid trace has NaN / inf"
    assert peak > 0.0, (
        "2D coupled fluid receiver is identically zero despite a "
        "solid-domain force source — Apply*RHS wiring is inert in 2D"
    )
    # Amplitude floor: source = 1e9, fluid-solid transmission
    # coefficient for this density/velocity contrast is O(0.1), and
    # geometric spreading over ~400 m loses another O(1) factor. A
    # coupled 2D result of at least O(1) Pa is physical; anything
    # smaller points to a packing / orientation bug in the 2D path
    # (wrong normal sign, zero Jacobian·weight, mis-indexed DOFs).
    assert peak > 1.0, (
        f"2D fluid trace peak {peak:.3e} is suspiciously small for an "
        f"amplitude-1e9 source transmitted across a single elastic-acoustic "
        f"interface; likely the 2D Apply*RHS path has a sign or indexing bug."
    )
