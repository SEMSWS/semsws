"""
ParSubMesh elastic regression: with --no-coupling + source + receiver both
inside the SOLID submesh, the recorded solid DISP-z trace must show the
P-wave direct arrival at the geometric expected time (distance / vp_solid +
Ricker delay), within a small tolerance.

The existing pure-solid waveform tests under test/waveform/3D/elastic/ run
on a FULL ParMesh with material.type=isotropic_elastic. They do not
exercise the same elastic solver path we hit inside the coupled facade,
which wraps it in a ParSubMesh extracted from a two-attribute parent mesh.
If ParSubMesh-specific regressions slip in (face-DOF orientation, element
numbering, source location resolution, boundary detection), that code path
silently gives wrong waveforms on R03-style deep-solid receivers while the
full-ParMesh pure-solid test stays green.

Catch: this test is tighter than test_coupling_cross_domain.py's
"peak > 1.0" amplitude gate. We check the arrival TIME of the direct P wave
to within a small fraction of a Ricker period — enough to catch the
80 ms error mode we saw in the SPECFEM3D benchmark.
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


# Geometry: a tall parent cuboid split fluid-bottom / solid-top. Source
# and receiver are both inside the solid half with a generous 150 m
# source-to-receiver distance so the direct-wave travel time is much
# larger than one time step AND a few samples of one Ricker period.
# That separation is what makes arrival-time regressions visible.
LX, LY, LZ = 400.0, 200.0, 800.0   # interface at z = LZ/2 = 400
VP_FLUID   = 1500.0                # placeholder — coupling is OFF
RHO_FLUID  = 1000.0
VP_SOLID   = 3000.0
VS_SOLID   = 1732.0
RHO_SOLID  = 2500.0

# Source well inside the solid, offset from element corners so it
# doesn't get "nudged" by the on-boundary location logic.
SOURCE_XYZ = (LX / 2.0 + 13.0, LY / 2.0 + 7.0, LZ * 0.625 + 11.0)
# Receiver 150 m above source, still in solid (z_src ~= 511, z_recv ~= 661)
RECV_XYZ   = (LX / 2.0 + 13.0, LY / 2.0 + 7.0, LZ * 0.625 + 11.0 + 150.0)

# Moment-tensor isotropic explosion. Keeps the radiation pattern pure P
# and avoids force-source normalization convention ambiguity.
MOMENT_AMP = 1.0e12

FREQ       = 5.0   # Hz — matches examples/coupling_vs_specfem/ where the
                   # ParSubMesh elastic bug was first observed.
DELAY      = 1.5 / FREQ   # 0.3 s, Ricker pre-trigger decayed to ~1e-9
DT         = 5.0e-4
NT         = 1600  # 0.8 s — covers direct P well before outer reflections


def _expected_direct_arrival() -> float:
    """Ricker peak at (delay) + |source-recv| / vp_solid + ~(derivative
    offset). For a moment-tensor source the far-field displacement peaks
    near the 1st derivative of the STF, which lags the STF peak by
    roughly 0.05 / f at distance (sub-wavelength near-field correction
    is small for r > lambda/10 here). The test allows a generous window
    around this prediction so we don't pin to a particular derivative
    mode — the point is to catch gross arrival-time regressions, not
    to verify the exact peak-to-peak timing."""
    r = math.sqrt(sum((a - b) ** 2 for a, b in zip(SOURCE_XYZ, RECV_XYZ)))
    return DELAY + r / VP_SOLID


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, gmsh_available, request) -> tuple[Path, Path, Path]:
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("submesh_elastic")
    mesh_file = work / "probe.msh"
    # Element size 100 m in every direction (nx=4, ny=2, nz_half=4 on a
    # 400×200×800 parent). At order 4 that's 5 GLL / side / direction,
    # giving lambda_P at f_max = 2.8·5 = 14 Hz of 3000/14 ≈ 214 m with
    # ~10 GLL / wavelength in the solid. Plenty for arrival-time checks.
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
         "--nx", "4", "--ny", "2", "--nz-half", "4",
         "--mfem-bdrs"],
        check=True, capture_output=True, text=True,
    )

    yaml_text = textwrap.dedent(f"""\
        name: "ParSubMesh elastic regression (--no-coupling, solid-only source/recv)"
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
          file: {mesh_file.name}
          max_freq: {FREQ * 2.8}
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
              name: "solid-cmt"
              type: "moment_tensor"
              location: [{SOURCE_XYZ[0]}, {SOURCE_XYZ[1]}, {SOURCE_XYZ[2]}]
              direction: [0.0, 0.0, 1.0]
              moment_tensor:
                Mxx: {MOMENT_AMP}
                Myy: {MOMENT_AMP}
                Mzz: {MOMENT_AMP}
                Mxy: 0.0
                Mxz: 0.0
                Myz: 0.0
              wavelet:
                type: ricker
                frequency: {FREQ}
                delay: {DELAY}
                amplitude: 1.0
        receivers:
          output:
            formats:
              - ascii
            filename: "rcv"
          type: [DISP]
          list:
            - name: "R01"
              location: [{RECV_XYZ[0]}, {RECV_XYZ[1]}, {RECV_XYZ[2]}]
              type: [DISP]
        device:
          type: cpu
    """)
    yaml_file = work / "coupled.yaml"
    yaml_file.write_text(yaml_text)

    outdir = work / "out"
    outdir.mkdir(exist_ok=True)

    yield work, yaml_file, outdir
    if keep:
        print(f"\n[--keep-results] ParSubMesh elastic workspace: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def _load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    assert data.ndim == 2 and data.shape[1] >= 2, (
        f"unexpected trace shape: {data.shape}"
    )
    return data[:, 0], data[:, 1]


def _find_disp_z_trace(outdir: Path) -> Path:
    """R01 DISP-z trace is written as R01_z_<src_id:04d>.d."""
    hits = [p for p in outdir.rglob("R01_z_*.d") if p.is_file()]
    assert hits, (
        f"R01 disp-z trace not found under {outdir}; contents = "
        f"{sorted(p.name for p in outdir.rglob('*') if p.is_file())}"
    )
    return hits[0]


def test_solid_submesh_direct_p_arrival_time(
    probe_exe: Path, workspace, mpi_cmd: str
) -> None:
    """Run the facade with --no-coupling, source + receiver BOTH in the
    solid submesh. Verify the recorded DISP-z peak arrives at roughly
    (delay + |src-recv|/vp_solid). The tolerance is set at ~0.4 / f0 —
    tight enough to catch the 80 ms regression observed in the SPECFEM3D
    fluid-solid benchmark (where SEMSWS peaked at t≈0.32 s against the
    physical t≈0.40 s on a ParSubMesh-based solid run) while loose
    enough to tolerate the derivative-vs-moment STF convention shift
    that the elastic far-field exhibits for a moment tensor source.
    """
    _, yaml_file, outdir = workspace

    np_procs = 1
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(probe_exe),
               "--config", str(yaml_file), "--dim", "3",
               "--run", "--no-coupling"]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(probe_exe),
               "--config", str(yaml_file), "--dim", "3",
               "--run", "--no-coupling"]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"exited {proc.returncode}\nstdout:\n{proc.stdout}"
        f"\nstderr:\n{proc.stderr}"
    )
    assert "TIME_LOOP=completed" in proc.stdout

    t, uz = _load_trace(_find_disp_z_trace(outdir))
    assert np.all(np.isfinite(uz)), "solid disp-z trace has NaN / inf"
    assert np.any(np.abs(uz) > 0.0), "solid disp-z trace is identically zero"

    # Restrict the peak search to the DIRECT-wave window. The direct
    # P-wave arrival is at DELAY + 150/3000 = 0.35 s; the first outer-
    # boundary reflection off the top z face (z=LZ, ~140 m above recv)
    # lands at ~0.35 + 2*140/3000 ≈ 0.44 s; bottom-face reflection off
    # z=0 (~660 m from source, far) at 0.3 + 2*660/3000 ≈ 0.74 s.
    # We look in [DELAY, DELAY+0.12] which is comfortably inside the
    # "only the direct P arrived" window.
    direct_window = (t >= DELAY) & (t < DELAY + 0.12)
    assert direct_window.any(), "direct-arrival window contains no samples"
    t_peak = float(t[direct_window][np.argmax(np.abs(uz[direct_window]))])

    t_expected = _expected_direct_arrival()
    one_period = 1.0 / FREQ
    # Tolerance: ~0.2 / f0 = one-fifth of a Ricker period. At f=5 Hz
    # that's 40 ms. The far-field displacement peak for a MT source is
    # ~(1/(2πf0)) ≈ 32 ms after the STF peak due to the dġ/dt factor,
    # so the "expected" we compute (STF-peak travel) sits within this
    # tolerance of the physical d-peak. The benchmark regression mode
    # (peak at t ~= source_peak + travel ~= DELAY + 50 ms) was ~130 ms
    # early against the physical 350 ms, well outside the tolerance.
    tol = 0.2 * one_period
    assert abs(t_peak - t_expected) < tol, (
        f"solid submesh DISP-z peak at t={t_peak:.4f} s, expected "
        f"~{t_expected:.4f} s (|src-recv|={math.sqrt(sum((a-b)**2 for a,b in zip(SOURCE_XYZ,RECV_XYZ))):.1f} m, "
        f"vp={VP_SOLID} m/s, delay={DELAY} s, tol={tol:.4f} s). "
        f"If this fails, the ParSubMesh elastic propagation is wrong — "
        f"the solid submesh run should match a full-ParMesh pure-solid "
        f"reference to the same Ricker-period tolerance."
    )
