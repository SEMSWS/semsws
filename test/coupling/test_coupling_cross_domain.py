"""
Phase 3a direct physics check: the interface Apply* path actually carries
energy between the two sub-simulations.

Setup: a force source sits inside the SOLID half-space and the only
receiver is a pressure (PS) gauge inside the FLUID half-space. Without
any interface coupling the fluid trace MUST be identically zero (the
fluid has no source and no initial condition). When coupling is enabled
the fluid trace must be finite and non-trivial because solid motion
radiating into the interface pushes the fluid potential via the Phase 3a
ApplySolidToFluidRHS / ApplyFluidToSolidRHS path.

This is a direct test of the coupling physics, not a parameter-change
invariant: if the Apply* path is a no-op the fluid stays silent and the
test fails loudly.
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


# Mesh: fluid z∈[0, 200], solid z∈[200, 400]. Source sits at z=300 (deep
# inside the solid) with a force pointing toward the interface; receiver
# at z=100 (deep inside the fluid, on the other side of Γ_fs at z=200).
LX, LY, LZ = 600.0, 300.0, 400.0
VP_FLUID, RHO_FLUID = 1500.0, 1000.0
VP_SOLID, VS_SOLID, RHO_SOLID = 3000.0, 1732.0, 2500.0
SOLID_SOURCE_XYZ = (300.0, 150.0, 300.0)
FLUID_RECV_XYZ   = (300.0, 150.0, 100.0)
FREQ  = 10.0
DELAY = 0.1
DT    = 2.0e-4
NT    = 1200


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
    yaml_text = textwrap.dedent(f"""\
        name: "Phase 3a cross-domain coupling (solid src -> fluid recv)"
        simulation:
          dimension: 3
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
              name: "solid-force-z"
              type: "force"
              location: [{SOLID_SOURCE_XYZ[0]}, {SOLID_SOURCE_XYZ[1]}, {SOLID_SOURCE_XYZ[2]}]
              direction: [0.0, 0.0, -1.0]   # toward interface at z=Lz/2
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


def _run(exe: Path, yaml_file: Path, np_procs: int, mpi_cmd: str,
         enable_coupling: bool) -> None:
    extra = [] if enable_coupling else ["--no-coupling"]
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run", *extra]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode} (coupling={'on' if enable_coupling else 'off'})"
        f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "TIME_LOOP=completed" in proc.stdout, proc.stdout


def _load_pressure_trace(outdir: Path) -> np.ndarray:
    traces = [p for p in outdir.rglob("*.p") if p.is_file() and p.stat().st_size > 32]
    assert traces, f"no pressure trace in {outdir} (dirs: {[p.name for p in outdir.rglob('*')]})"
    r01 = sorted(p for p in traces if p.stem.startswith("R01"))
    chosen = (r01 or sorted(traces))[0]
    return np.loadtxt(chosen)


@pytest.fixture(scope="module")
def cross_domain_runs(tmp_path_factory, probe_exe, gmsh_available, mpi_cmd, request):
    np_procs = int(request.config.getoption("--np"))
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_cross_domain")

    mesh_file = work / "probe.msh"
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
         "--nx", "6", "--ny", "3", "--nz-half", "4"],
        check=True, capture_output=True, text=True,
    )

    # Two runs sharing the same YAML; only the probe flag differs.
    run_on  = work / "run_on"
    run_off = work / "run_off"
    run_on.mkdir()
    run_off.mkdir()
    for rd in (run_on, run_off):
        (rd / "probe.msh").symlink_to(mesh_file)
        (rd / "out").mkdir()
        _write_yaml(rd / "coupled.yaml", "probe.msh")

    _run(probe_exe, run_on  / "coupled.yaml", np_procs, mpi_cmd, enable_coupling=True)
    _run(probe_exe, run_off / "coupled.yaml", np_procs, mpi_cmd, enable_coupling=False)

    trace_on  = _load_pressure_trace(run_on  / "out")
    trace_off = _load_pressure_trace(run_off / "out")

    yield trace_on, trace_off, work
    if keep:
        print(f"\n[--keep-results] cross-domain workspace retained: {work}")
        print(f"  on:  {run_on /'out'/'R01_0000.p'}")
        print(f"  off: {run_off/'out'/'R01_0000.p'}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def test_fluid_silent_without_coupling(cross_domain_runs) -> None:
    """No source anywhere in the fluid + SetCouplingEnabled(false) ⇒ the
    fluid receiver trace must be identically zero. Any non-zero sample is
    state leakage across the submesh boundary."""
    _, trace_off, _ = cross_domain_runs
    p_off = trace_off[:, 1]
    assert np.all(p_off == 0.0), (
        f"fluid trace non-zero under --no-coupling: "
        f"max |p| = {float(np.max(np.abs(p_off))):.3e}"
    )


def test_fluid_activated_by_solid_source_via_coupling(cross_domain_runs) -> None:
    """With coupling enabled, energy radiated from the solid force source
    must reach the fluid receiver through Γ_fs. A dead Apply* path would
    leave the fluid trace at zero even though the solid is vibrating."""
    trace_on, _, _ = cross_domain_runs
    p_on  = trace_on[:,  1]
    peak  = float(np.max(np.abs(p_on)))
    assert np.all(np.isfinite(p_on)),    "coupled fluid trace has NaN / inf"
    assert peak > 0.0, (
        "coupled fluid receiver recorded identically zero despite a "
        "solid-domain force source: interface Apply* wiring is inert"
    )
    # Weak positive-amplitude gate: the source amplitude is 1e9 and the
    # interface transmission coefficient for this density/velocity contrast
    # is O(0.1). Anything smaller than 1.0 is almost certainly noise.
    assert peak > 1.0, (
        f"fluid trace peak {peak:.3e} is suspiciously small for an "
        f"amplitude-1e9 source transmitted across a single elastic-acoustic "
        f"interface; likely Apply* is miswired (wrong normal sign, "
        f"zero jacobianw, wrong DOF list)."
    )
