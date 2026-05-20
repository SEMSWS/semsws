"""
CoupledSimulationFacade::SetupFromConfig splits a `material.type: coupled`
YAML into two per-submesh materials. Runs under np=1 and np=2 to cover
both the degenerate-rank case and parallel runs where METIS may or may
not align the partition with the fluid/solid interface.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


TEST_DIR = Path(__file__).parent
GENERATE_SCRIPT = TEST_DIR / "generate_probe_mesh.py"


def _parse_kv(stdout: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
    return result


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


@pytest.fixture(scope="module")
def setup_workspace(tmp_path_factory, gmsh_available) -> Path:
    work = tmp_path_factory.mktemp("coupling_facade")
    mesh_file = work / "probe.msh"
    # Wide-short aspect so METIS (which cuts along the longest axis) lands
    # orthogonal to the fluid/solid interface at z=Lz/2 — that way every
    # rank owns some fluid AND some solid elements even at np=2, avoiding
    # the zero-local-elements failure mode on a submesh FES.
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", "600", "--ly", "300", "--lz", "200",
         "--nx", "6", "--ny", "3", "--nz-half", "2"],
        check=True, capture_output=True, text=True,
    )
    # Minimal coupled YAML — boundary / sources / receivers are included to
    # satisfy YamlConfig::Validate(), but SetupFromConfig itself only touches
    # mesh + submeshes + materials in Phase 1.
    yaml_text = textwrap.dedent(f"""\
        name: "Phase 1 coupled facade smoke"
        simulation:
          dimension: 3
          order: 2
          dt: 0.001
          steps: 1
          cfl_factor: 0.3
          directory: "./out"
          log_interval: 100
        mesh:
          type: external
          file: {mesh_file.name}
          max_freq: 10.0
          ppw: 2.0
        material:
          type: coupled
          fluid:
            attribute: 1
            type: isotropic_acoustic
            format: constant
            vp: 1500.0
            rho: 1000.0
          solid:
            attribute: 2
            type: isotropic_elastic
            format: constant
            vp: 3000.0
            vs: 1732.0
            rho: 2500.0
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
              location: [300.0, 150.0, 50.0]   # z < 100 -> fluid
              wavelet:
                type: ricker
                frequency: 10.0
                delay: 0.1
                amplitude: 1.0
        receivers:
          output:
            formats:
              - ascii
            filename: "rcv"
          type: [PS]
          list:
            - name: "R01"
              location: [300.0, 150.0, 75.0]  # z < 100 -> fluid, matches PS type
        device:
          type: cpu
    """)
    yaml_file = work / "coupled.yaml"
    yaml_file.write_text(yaml_text)
    yield work, yaml_file
    shutil.rmtree(work, ignore_errors=True)


def _run_facade_probe(exe: Path, yaml_file: Path, np_procs: int, mpi_cmd: str,
                      dim: int = 3) -> dict[str, str]:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", str(dim)]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe),
               "--config", str(yaml_file), "--dim", str(dim)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                          cwd=yaml_file.parent)
    assert proc.returncode == 0, (
        f"facade probe exited {proc.returncode}\ncmd: {' '.join(cmd)}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    kv = _parse_kv(proc.stdout)
    assert kv.get("PROBE_OK") == "1", f"probe did not report OK:\n{proc.stdout}"
    return kv


@pytest.mark.parametrize("np_procs", [1, 2])
def test_coupled_facade_setup_smoke(probe_exe: Path, setup_workspace,
                                    mpi_cmd: str, np_procs: int) -> None:
    _, yaml_file = setup_workspace
    kv = _run_facade_probe(probe_exe, yaml_file, np_procs, mpi_cmd)

    # Every rank reports the same total NE counts.
    assert int(kv["PARENT_NE"]) > 0
    # probe.msh is 2x2 per face with nz_half=2 by default -> 2*2*2 = 8 fluid, 8 solid
    assert int(kv["FLUID_NE"]) + int(kv["SOLID_NE"]) == int(kv["PARENT_NE"])
    assert int(kv["FLUID_NE"]) > 0
    assert int(kv["SOLID_NE"]) > 0


@pytest.mark.parametrize("np_procs", [1, 2])
def test_coupled_facade_material_dispatch(probe_exe: Path, setup_workspace,
                                          mpi_cmd: str, np_procs: int) -> None:
    """Each sub-material must be dispatched to its domain-appropriate concrete
    MaterialType; a silently inverted coupling here would wreck downstream phases."""
    _, yaml_file = setup_workspace
    kv = _run_facade_probe(probe_exe, yaml_file, np_procs, mpi_cmd)

    assert kv["FLUID_MATERIAL"] == "isotropic_acoustic"
    assert kv["SOLID_MATERIAL"] == "isotropic_elastic"
    assert int(kv["FLUID_VDIM"]) == 1,  "fluid FES must be scalar (vdim=1)"
    assert int(kv["SOLID_VDIM"]) == 3,  "solid FES must be vector (vdim=Dim)"
    assert int(kv["FLUID_ATTR"]) == 1
    assert int(kv["SOLID_ATTR"]) == 2
