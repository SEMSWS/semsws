"""
Phase 3a bidirectional gate:

- `--no-coupling` path (SetCouplingEnabled(false)): the fluid-side receiver
  trace must be **bit-identical** between two runs that differ only in the
  solid material's parameters. This locks in the escape hatch introduced
  in Phase 3a so the Phase 1B independence invariant can still be re-armed
  whenever needed (diagnostic, debugging, bisection).

- Default coupled path: flipping the solid's Vp by 2× MUST change the
  fluid receiver trace by a non-trivial amount. If it doesn't, the
  interface Apply*/Extract* path is a no-op and Phase 3a didn't actually
  wire anything.

Also implicitly exercises the Dirichlet-attr filter on the fluid side
(`attributes: [11, 12, 13]` in YAML — 12 is solid-only and must be
silently dropped, the auto-generated interface attr must never be
promoted to Dirichlet).
"""

from __future__ import annotations

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


LX, LY, LZ = 600.0, 300.0, 400.0
FLUID_VP, FLUID_RHO = 1500.0, 1000.0
SOLID_VP_A, SOLID_VP_B = 3000.0, 6000.0  # deliberate spread, 2x difference
SOLID_VS_A, SOLID_VS_B = 1732.0, 3464.0
SOLID_RHO = 2500.0
SOURCE_XYZ = (300.0, 150.0, 50.0)
RECV_XYZ   = (300.0, 150.0, 150.0)
FREQ  = 10.0
DELAY = 0.1
DT    = 2.0e-4
NT    = 1200


def _write_yaml(yaml_path: Path, mesh_name: str,
                solid_vp: float, solid_vs: float) -> None:
    yaml_text = textwrap.dedent(f"""\
        name: "Phase 1B independence (fluid recv vs solid param)"
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
            vp: {FLUID_VP}
            rho: {FLUID_RHO}
          solid:
            attribute: 2
            type: isotropic_elastic
            format: constant
            vp: {solid_vp}
            vs: {solid_vs}
            rho: {SOLID_RHO}
        boundary:
          absorbing:
            sides: []
            thickness: 0.0
            alpha: 0.0
          # All outer-surface attrs the probe mesh defines: 11=bottom,
          # 12=top, 13=sides. The fluid side only touches 11 and 13;
          # the coupled facade must silently drop 12 (solid-only) and
          # must NOT promote the auto-generated interface attr to
          # Dirichlet. If either of those safeguards fails, the two
          # runs' fluid traces will still match (no actual coupling)
          # but the setup will abort — so this YAML implicitly tests
          # the Dirichlet-attr filter on the fluid side.
          dirichlet: [11, 12, 13]
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
    yaml_path.write_text(yaml_text)


def _run(exe: Path, yaml_file: Path, np_procs: int, mpi_cmd: str,
         outdir_arg: Path) -> None:
    # Phase 3a: explicit --no-coupling so SetCouplingEnabled(false) is the
    # only path under test here. The coupled path (coupling enabled by
    # default) is covered by test_coupling_cross_domain.py.
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run",
               "--no-coupling"]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", "3", "--run",
               "--no-coupling"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"exited {proc.returncode}\ncmd: {' '.join(cmd)}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "TIME_LOOP=completed" in proc.stdout, proc.stdout
    # The probe writes via SaveReceivers into config's output.directory,
    # which is relative to the YAML dir (== cwd of the subprocess).
    assert outdir_arg.exists(), (
        f"expected output directory {outdir_arg} does not exist"
    )


def _load_trace(outdir: Path) -> np.ndarray:
    traces = [p for p in outdir.rglob("*.p") if p.stat().st_size > 32]
    assert traces, f"no pressure trace in {outdir}"
    # Deterministic pick: R01 preferred, else first sorted.
    r01 = sorted(p for p in traces if p.stem.startswith("R01"))
    chosen = (r01 or sorted(traces))[0]
    d = np.loadtxt(chosen)
    assert d.ndim == 2 and d.shape[1] >= 2, f"bad trace shape {d.shape}"
    return d


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory, probe_exe, gmsh_available, mpi_cmd, request):
    np_procs = int(request.config.getoption("--np"))
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_independence")

    mesh_file = work / "probe.msh"
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
         "--nx", "6", "--ny", "3", "--nz-half", "4"],
        check=True, capture_output=True, text=True,
    )

    run_a = work / "run_a"
    run_b = work / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    # Each run needs its own mesh copy colocated with its YAML (paths in
    # YAML are resolved relative to the YAML file's dir). Symlink is fine.
    (run_a / "probe.msh").symlink_to(mesh_file)
    (run_b / "probe.msh").symlink_to(mesh_file)

    yaml_a = run_a / "coupled.yaml"
    yaml_b = run_b / "coupled.yaml"
    _write_yaml(yaml_a, "probe.msh", SOLID_VP_A, SOLID_VS_A)
    _write_yaml(yaml_b, "probe.msh", SOLID_VP_B, SOLID_VS_B)

    out_a = run_a / "out"
    out_b = run_b / "out"
    out_a.mkdir()
    out_b.mkdir()

    _run(probe_exe, yaml_a, np_procs, mpi_cmd, out_a)
    _run(probe_exe, yaml_b, np_procs, mpi_cmd, out_b)

    trace_a = _load_trace(out_a)
    trace_b = _load_trace(out_b)

    yield trace_a, trace_b, work
    if keep:
        print(f"\n[--keep-results] independence workspace retained: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def test_fluid_trace_bit_identical_across_solid_param_change(two_runs) -> None:
    trace_a, trace_b, _ = two_runs

    # 1. Time columns identical.
    assert trace_a.shape == trace_b.shape, (
        f"trace length/shape differs: {trace_a.shape} vs {trace_b.shape}"
    )
    np.testing.assert_array_equal(trace_a[:, 0], trace_b[:, 0])

    # 2. Pressure columns bit-exact. No tolerance: any non-zero diff
    #    means the solid leaked into the fluid (= unintended coupling).
    diff = trace_a[:, 1] - trace_b[:, 1]
    max_abs_diff = float(np.max(np.abs(diff)))
    peak_signal  = float(np.max(np.abs(trace_a[:, 1])))

    assert max_abs_diff == 0.0, (
        f"fluid trace differs between solid=vp={SOLID_VP_A} and "
        f"solid=vp={SOLID_VP_B} runs under --no-coupling: "
        f"max |diff| = {max_abs_diff:.3e} "
        f"(peak signal = {peak_signal:.3e}). The SetCouplingEnabled(false) "
        f"escape hatch is broken: solid state leaked into fluid even "
        f"with interface Apply* turned off."
    )


def test_fluid_trace_non_trivial(two_runs) -> None:
    """Sanity: the trace we're asserting equality on is actually a live
    waveform, not two identically-zero buffers (which would also be
    "bit-exact" but useless)."""
    trace_a, _, _ = two_runs
    peak = float(np.max(np.abs(trace_a[:, 1])))
    assert peak > 0.0, "fluid trace is identically zero (Dirichlet may be clamping everything)"
    assert np.all(np.isfinite(trace_a[:, 1])), "NaN/inf in trace"
