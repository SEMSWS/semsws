"""
Regression gate for multi-component fluid-solid topologies.

The "N spherical fluid inclusions embedded in a solid host" layout
exercises a part of MFEM's `ParSubMesh` behavior that SEMSWS relies on
but that MFEM's documentation explicitly says *isn't* supported:

  > The attribute has to mark exactly one connected subset of the parent Mesh.
  >   — `mfem/mesh/submesh/psubmesh.hpp:65,78`

In practice, MFEM assigns the SAME auto-generated boundary attribute
(`max_bdr_attr + 1`) to every newly-exposed interface face regardless of
how many disconnected components those faces form — see:

  - `mfem/mesh/submesh/submesh_utils.cpp:318,326`
  - `mfem/mesh/submesh/submesh.cpp:182`

So in reality SEMSWS's `FluidSolidInterface::DetectInterfaceAttribute`
(which asserts `f_new.size() == 1` against the *attribute count*, not
the *component count*) PASSES for multi-inclusion meshes, and the per-
face iteration in `Setup` is topology-agnostic.

This test locks that in: generate a 2D mesh with TWO fluid rectangles
embedded in a solid host, run `semsws` end-to-end, and
assert that:

  1. The executable exits 0 (i.e. interface detection didn't trip).
  2. A `summary.txt` is written — proves the whole pipeline (Setup,
     SetupFromConfig, Run, SaveReceivers) completed.
  3. The fluid pressure receiver records a non-zero, finite trace —
     proves `Apply*RHS` actually delivered energy across at least one
     of the two disjoint fluid-solid interfaces.

If a future MFEM release ever starts enumerating interface components
into distinct attributes, test (1) fails at the `f_new.size() == 1`
check in [src/coupling/FluidSolidInterface.cpp:46-51]
(src/coupling/FluidSolidInterface.cpp#L46-L51) — exactly the early
warning this test is here to provide.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest


TEST_DIR        = Path(__file__).parent
GENERATE_SCRIPT = TEST_DIR / "generate_multi_inclusion_mesh_2d.py"


LX, LZ = 1200.0, 800.0
INCL_W = INCL_H = 200.0
# Two fluid inclusions, symmetric around x=LX/2, centered vertically.
INCLUSION_CENTERS = [(300.0, 400.0), (900.0, 400.0)]

# Material: water in the fluid inclusions, rock elsewhere.
VP_FLUID, RHO_FLUID           = 1500.0, 1000.0
VP_SOLID, VS_SOLID, RHO_SOLID = 3000.0, 1732.0, 2500.0

# Source in the solid host, midway above the inclusions so energy
# radiates down and hits both inclusions (proves the interface wiring
# handles multiple disjoint Γ_fs components in a single run).
SOURCE_XY = (LX / 2.0, 700.0)
# Receiver inside the left fluid inclusion — expect a non-zero pressure
# arrival after the solid-borne wave reaches the top of that inclusion.
FLUID_RECV_XY = (INCLUSION_CENTERS[0][0], INCLUSION_CENTERS[0][1])

FREQ, DELAY = 10.0, 0.1
DT, NT      = 2.0e-4, 1500


@pytest.fixture(scope="module")
def main_exe(build_dir: Path) -> Path:
    candidates = [
        build_dir / "src" / "semsws",   # CMake build dir
        build_dir / "bin" / "semsws",   # Spack install prefix
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    pytest.skip(
        f"semsws not found in any of: "
        + ", ".join(str(c) for c in candidates)
    )


@pytest.fixture(scope="module")
def gmsh_available() -> None:
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"gmsh python module unavailable: {e}")


def _write_yaml(yaml_path: Path, mesh_name: str) -> None:
    centers_str = ";".join(f"{cx},{cy}" for cx, cy in INCLUSION_CENTERS)
    yaml_text = textwrap.dedent(f"""\
        name: "2D multi-inclusion fluid-solid smoke (centers: {centers_str})"
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
          dirichlet: [11]
        sources:
          list:
            - id: 1
              name: "solid-force-y"
              type: "force"
              location: [{SOURCE_XY[0]}, {SOURCE_XY[1]}]
              direction: [0.0, -1.0]
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


@pytest.fixture(scope="module")
def multi_inclusion_run(tmp_path_factory, main_exe, gmsh_available,
                        mpi_cmd, request):
    """Generate the multi-inclusion mesh, write the YAML, run
    semsws once for the module, yield the workspace."""
    np_procs = int(request.config.getoption("--np"))
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_multi_inclusion_2d")

    mesh_file = work / "multi_inclusion.msh"
    centers_arg = ";".join(f"{cx},{cy}" for cx, cy in INCLUSION_CENTERS)
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--lz", str(LZ),
         "--mesh-size", "40.0",
         "--incl-w", str(INCL_W), "--incl-h", str(INCL_H),
         "--centers", centers_arg],
        check=True, capture_output=True, text=True,
    )

    yaml_file = work / "coupled.yaml"
    _write_yaml(yaml_file, mesh_file.name)
    (work / "out").mkdir(exist_ok=True)

    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(main_exe),
               "--config", str(yaml_file)]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(main_exe),
               "--config", str(yaml_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=work)
    assert proc.returncode == 0, (
        f"semsws exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    outdir = work / "out"
    yield outdir, proc
    if keep:
        print(f"\n[--keep-results] multi-inclusion workspace: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def test_summary_file_written(multi_inclusion_run) -> None:
    """End-to-end pipeline completes: summary file is written.

    A missing summary points at any one of: Setup failed, Run aborted,
    SaveReceivers threw, or the coupled facade bailed out before
    WriteSummaryToFile. On the multi-inclusion mesh the most likely
    failure is interface detection tripping `f_new.size() == 1`."""
    outdir, _ = multi_inclusion_run
    summary = outdir / "summary.txt"
    assert summary.exists(), (
        f"summary.txt not produced under {outdir}; "
        f"existing files = {sorted(p.name for p in outdir.rglob('*'))}"
    )


def test_fluid_receiver_sees_coupled_energy(multi_inclusion_run) -> None:
    """Solid-domain force source → pressure in a fluid inclusion.

    The only way energy reaches the fluid receiver is through the
    solid→fluid leg of `FluidSolidInterface::ApplyFluidToSolidRHS`
    composed with `ApplySolidToFluidRHS`. A dead interface on a
    multi-component topology silently delivers zero pressure here even
    though the run succeeds elsewhere; this test catches that."""
    outdir, _ = multi_inclusion_run
    # SEMSWS ASCII pressure files: <name>_<sourceid>.p under results dir.
    press = [p for p in outdir.rglob("*.p")
             if p.is_file() and p.stat().st_size > 32]
    r01 = sorted(p for p in press if p.stem.startswith("R01"))
    assert r01, (
        f"no R01 pressure trace under {outdir}; "
        f"existing *.p files = {sorted(p.name for p in press)}"
    )
    data = np.loadtxt(r01[0])
    assert data.ndim == 2 and data.shape[1] >= 2
    p = data[:, 1]
    assert np.all(np.isfinite(p)), "fluid pressure trace has NaN / inf"
    peak = float(np.max(np.abs(p)))
    assert peak > 0.0, (
        "fluid receiver in inclusion is identically zero — the coupling "
        "path is inert on a multi-component interface topology"
    )
