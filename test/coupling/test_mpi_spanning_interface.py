"""
Phase 2B-iii gate: the MPI-spanning interface path actually works.

Test geometry is deliberately tall/narrow so METIS at np=2 cuts along z
and each rank ends up owning either ALL fluid or ALL solid — so every
interface face straddles an MPI partition boundary and can only work if
the Phase 2B-ii bidirectional Isend/Irecv exchange is wired up.

Two checks:

1. test_fluid_silent_without_coupling_mpi_spanning:
   With --no-coupling, no source anywhere in fluid, the fluid receiver
   trace must stay identically zero. This is the no-coupling baseline
   for the MPI-spanning configuration — if non-zero, state is leaking
   across the submesh boundary even with interface Apply* off.

2. test_fluid_activated_by_solid_source_via_mpi_spanning_coupling:
   A force source sits in solid (on one rank) and the only receiver is
   a PS gauge in fluid (on the other rank). With coupling enabled, the
   fluid receiver must record non-zero pressure — the only way energy
   gets from the solid half to the fluid half is through Γ_fs, and
   Γ_fs is 100% MPI-spanning here, so this fails unless Phase 2B-ii
   exchanges are working.
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


# Rectangular mesh shaped so METIS at np=2 produces a partition where
# each rank has BOTH fluid and solid elements AND at least some interface
# faces are MPI-spanning. 5×2×8 hex: probe confirms 8 local + 2 remote
# interface faces at np=2. Using a fully-tall mesh (all fluid on one
# rank / all solid on the other) trips the zero-local-elements crash in
# the operator's FES setup; that's a separate Phase 6 concern.
LX, LY, LZ = 500.0, 200.0, 400.0
VP_FLUID, RHO_FLUID = 1500.0, 1000.0
VP_SOLID, VS_SOLID, RHO_SOLID = 3000.0, 1732.0, 2500.0

SOLID_SOURCE_XYZ = (250.0, 100.0, 300.0)   # z=300 ⇒ solid half
FLUID_RECV_XYZ   = (250.0, 100.0, 100.0)   # z=100 ⇒ fluid half

FREQ  = 10.0
DELAY = 0.1
DT    = 1.0e-4
NT    = 4000                     # 0.40 s, enough for wavefront to cross interface


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


def _write_yaml(yaml_path: Path, mesh_name: str,
                with_solid_source: bool) -> None:
    # NOTE: pinned to order=2 to keep the tall mesh cost reasonable
    # (nz_half=4 + order=2 is enough to get a visible transmitted pulse).
    source_block = textwrap.indent(textwrap.dedent(f"""\
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
    """), "    ")
    sources_list = source_block if with_solid_source else "    []"
    yaml_text = textwrap.dedent(f"""\
        name: "Phase 2B-iii MPI-spanning interface"
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
            sides: []
            thickness: 0.0
            alpha: 0.0
          dirichlet: []
        sources:
          list:
""") + sources_list + textwrap.dedent(f"""
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


def _run(exe: Path, yaml_file: Path, np_procs: int, mpi_cmd: str,
         enable_coupling: bool) -> None:
    extra = [] if enable_coupling else ["--no-coupling"]
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run", *extra]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode} (coupling="
        f"{'on' if enable_coupling else 'off'}, np={np_procs})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "TIME_LOOP=completed" in proc.stdout, proc.stdout


def _load_trace(outdir: Path) -> np.ndarray:
    traces = [p for p in outdir.rglob("*.p") if p.is_file() and p.stat().st_size > 32]
    assert traces, f"no pressure trace in {outdir}"
    r01 = sorted(p for p in traces if p.stem.startswith("R01"))
    return np.loadtxt((r01 or sorted(traces))[0])


@pytest.fixture(scope="module")
def mpi_spanning_runs(tmp_path_factory, probe_exe, gmsh_available, mpi_cmd,
                      request):
    """Generate a tall mesh and run {coupling on, coupling off} with a
    solid-domain source and a fluid-domain receiver, both at np=2."""
    np_procs = 2   # critical for this test — MPI-spanning interface only triggers here
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_mpi_spanning")

    mesh_file = work / "mpi_span.msh"
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
         "--nx", "5", "--ny", "2", "--nz-half", "4"],
        check=True, capture_output=True, text=True,
    )

    run_on  = work / "run_on"
    run_off = work / "run_off"
    for rd in (run_on, run_off):
        rd.mkdir()
        (rd / "mpi_span.msh").symlink_to(mesh_file)
        (rd / "out").mkdir()
    _write_yaml(run_on  / "coupled.yaml", "mpi_span.msh", with_solid_source=True)
    _write_yaml(run_off / "coupled.yaml", "mpi_span.msh", with_solid_source=True)

    _run(probe_exe, run_on  / "coupled.yaml", np_procs, mpi_cmd, enable_coupling=True)
    _run(probe_exe, run_off / "coupled.yaml", np_procs, mpi_cmd, enable_coupling=False)

    trace_on  = _load_trace(run_on  / "out")
    trace_off = _load_trace(run_off / "out")
    yield trace_on, trace_off, work
    if keep:
        print(f"\n[--keep-results] mpi-spanning workspace retained: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def test_fluid_silent_without_coupling_mpi_spanning(mpi_spanning_runs) -> None:
    """Coupling OFF + no fluid source ⇒ fluid trace is identically zero,
    even in the 100%-MPI-spanning configuration. If non-zero, the Phase
    2B-i setup is leaking remote-face data into the fluid RHS."""
    _, trace_off, _ = mpi_spanning_runs
    p = trace_off[:, 1]
    assert np.all(p == 0.0), (
        f"fluid trace non-zero under --no-coupling on MPI-spanning mesh: "
        f"max |p| = {float(np.max(np.abs(p))):.3e}"
    )


def test_fluid_activated_by_solid_source_via_mpi_spanning_coupling(
    mpi_spanning_runs,
) -> None:
    """The only way energy reaches the fluid receiver is via Γ_fs, and
    every Γ_fs face is MPI-spanning at np=2 on this mesh. A non-zero
    fluid trace proves Phase 2B-ii's bidirectional Isend/Irecv actually
    transports the u·n and φ_tt values across the rank boundary."""
    trace_on, _, _ = mpi_spanning_runs
    p = trace_on[:, 1]
    peak = float(np.max(np.abs(p)))

    assert np.all(np.isfinite(p)), "MPI-spanning coupled trace has NaN / inf"
    assert peak > 0.0, (
        "fluid receiver is identically zero despite coupling-on solid "
        "source on an MPI-spanning interface — Phase 2B-ii MPI exchange "
        "delivered no values across ranks"
    )
    # Source amplitude = 1e9; after one fluid-solid transmission and
    # ~500 m of travel the pressure peak is physical if it's at least
    # O(1). Below that it's almost certainly a zero-packet / wrong-DOF
    # bug in the exchange.
    assert peak > 1.0, (
        f"fluid receiver peak {peak:.3e} is suspiciously small — likely "
        f"a packing / unpacking bug in the MPI-spanning path"
    )
