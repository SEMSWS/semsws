"""
Receiver output format consistency test.

Runs each existing waveform test config (acoustic/elastic x 2D/3D) with
receivers.output.format = [ascii, hdf5, su] and verifies the three formats
produce numerically identical waveforms.

Usage:
    pytest test/consistency/test_receiver_formats.py --build-dir ./build -v
    pytest test/consistency/test_receiver_formats.py --build-dir ./build --device hip -v
"""

from __future__ import annotations

import shutil
import struct
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


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

CASES = [
    ("2D/acoustic",          TEST_ROOT / "waveform" / "2D" / "acoustic" / "config.yaml"),
    ("2D/elastic",           TEST_ROOT / "waveform" / "2D" / "elastic" / "config.yaml"),
    ("3D/acoustic_analytic", TEST_ROOT / "waveform" / "3D" / "acoustic_analytic" / "config.yaml"),
    ("3D/elastic",           TEST_ROOT / "waveform" / "3D" / "elastic" / "config.yaml"),
]

# Reduced step count so the consistency check runs fast.
SHORT_STEPS = 500

# ASCII and HDF5 both store real_t natively (dtype detected at runtime from
# the HDF5 dataset). SU is always written as float32. Tolerances are chosen
# per real_t precision.
TOLS = {
    # rtol, atol — per dtype for the ASCII vs HDF5 comparison
    np.dtype("float64"): (1e-12, 1e-14),
    np.dtype("float32"): (1e-6,  1e-8),
}
RTOL_SU_FLOAT32 = 1e-5
ATOL_SU_FLOAT32 = 1e-6


# ---------------------------------------------------------------------------
# SU reader (SEG-Y rev 0 trace-header, no textual/binary file header).
# Format written by ReceiverArray::SaveToSU: sequence of (240-byte header,
# ns float32 samples) records.
# ---------------------------------------------------------------------------

SU_HEADER_SIZE = 240
SU_NS_OFFSET = 114   # uint16 (big or native?) — we write native-endian in C++
SU_DT_OFFSET = 116   # uint16, microseconds


def read_su_file(path: Path) -> tuple[np.ndarray, float, int, int]:
    """Return (data[num_traces, ns], dt_seconds, ns, num_traces)."""
    raw = path.read_bytes()
    # First trace header provides ns / dt; SEMSWS writes them in native byte
    # order (matches the uint16_t in SUTraceHeader struct without byteswap).
    ns = struct.unpack_from("<H", raw, SU_NS_OFFSET)[0]
    dt_us = struct.unpack_from("<H", raw, SU_DT_OFFSET)[0]
    dt = dt_us * 1e-6

    record_size = SU_HEADER_SIZE + ns * 4
    assert len(raw) % record_size == 0, (
        f"SU file size {len(raw)} not multiple of record size {record_size}"
    )
    num_traces = len(raw) // record_size

    data = np.empty((num_traces, ns), dtype=np.float32)
    for t in range(num_traces):
        off = t * record_size + SU_HEADER_SIZE
        data[t, :] = np.frombuffer(raw, dtype="<f4", count=ns, offset=off)
    return data, dt, ns, num_traces


# ---------------------------------------------------------------------------
# Helpers: read each format into a {(recv_name, type_str, comp_index): array}
# ---------------------------------------------------------------------------

# Type tokens must match ReceiverTypeToString() in include/common/Types.hpp.
ASCII_EXT_TO_TYPE = {
    ".p": "PS",
    ".d": "DISP",
    ".v": "VEL",
    ".a": "ACC",
    ".g": "GRAD",
}

# HDF5 group names and SU filename tokens are already those same short forms.

COMP_LETTERS = ("x", "y", "z")


def parse_ascii_filename(path: Path, source_id: str) -> tuple[str, str, int] | None:
    """Decode R001_0001.p / R001_x_0001.d → (name, type, comp_index)."""
    if path.suffix not in ASCII_EXT_TO_TYPE:
        return None
    rtype = ASCII_EXT_TO_TYPE[path.suffix]
    stem = path.stem  # e.g. "R001_0001" or "R001_x_0001"
    suffix = f"_{source_id}"
    if not stem.endswith(suffix):
        return None
    stem = stem[: -len(suffix)]
    # Pressure has no component token; vector types have "_x"/"_y"/"_z".
    if rtype == "PS":
        return stem, rtype, 0
    for comp_idx, letter in enumerate(COMP_LETTERS):
        tag = f"_{letter}"
        if stem.endswith(tag):
            return stem[: -len(tag)], rtype, comp_idx
    return None


def read_ascii_traces(outdir: Path, source_id: str) -> dict[tuple[str, str, int], np.ndarray]:
    """Return {(name, type, comp): data} parsed from ASCII files."""
    out = {}
    for path in sorted(outdir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_ascii_filename(path, source_id)
        if parsed is None:
            continue
        name, rtype, comp = parsed
        wf = read_sem_ascii(path)
        out[(name, rtype, comp)] = wf.data
    return out


def read_hdf5_traces(outdir: Path, base_filename: str, source_id: str
                     ) -> tuple[dict[tuple[str, str, int], np.ndarray], np.dtype]:
    """Read seis_<id>.h5 (v2.0 schema, /shots/<source_id>/receivers/...) →
    ({(name, type, comp): data}, real_dtype)."""
    import h5py  # local import so missing dep → pytest.skip
    h5_path = outdir / f"{base_filename}{source_id}.h5"
    assert h5_path.exists(), f"HDF5 output missing: {h5_path}"

    # v2.0 schema uses one canonical short form everywhere — this map is
    # now an identity for documentation purposes.
    channel_to_type = {
        "PS":   "PS",
        "DISP": "DISP",
        "VEL":  "VEL",
        "ACC":  "ACC",
    }

    out: dict[tuple[str, str, int], np.ndarray] = {}
    real_dtype: np.dtype | None = None
    with h5py.File(h5_path, "r") as f:
        # v2.0: receivers live under /shots/<source_id>/receivers/
        assert f.attrs["format_version"].decode() if isinstance(
            f.attrs["format_version"], bytes) else f.attrs["format_version"] \
            == "2.0", "expected format_version 2.0"
        rgrp = f["shots"][source_id]["receivers"]
        for recv_name in rgrp.keys():
            rg = rgrp[recv_name]
            for ch_name in rg.keys():
                if ch_name.startswith("weight_"):
                    continue
                type_name = channel_to_type.get(ch_name)
                if type_name is None:  # skip unknown channels
                    continue
                arr = np.asarray(rg[ch_name])
                if real_dtype is None:
                    real_dtype = arr.dtype
                if arr.ndim == 1:  # scalar pressure: (nt,)
                    out[(recv_name, type_name, 0)] = arr
                else:              # vector: (ncomp, nt)
                    for comp in range(arr.shape[0]):
                        out[(recv_name, type_name, comp)] = arr[comp, :]
    assert real_dtype is not None, "no receiver datasets found in HDF5"
    return out, real_dtype


def read_su_traces(outdir: Path, base_filename: str, source_id: str,
                   hdf5_order: dict[tuple[str, str, int], np.ndarray]
                   ) -> dict[tuple[str, str, int], np.ndarray]:
    """
    SU files group traces by (type, comp) into one file per combination.
    Trace ordering within each file matches the receiver insertion order used
    by ReceiverArray. We rely on the HDF5 keys to enumerate which (name, type,
    comp) tuples exist; then match SU traces by count and order.

    For acoustic cases (Pressure only, scalar) and homogeneous vector types
    we write one file per (type, comp); inside each file traces are stored in
    receiver order identical to the HDF5 dataset names ordering.
    """
    # Bucket expected receivers by (type, comp) in HDF5 iteration order.
    buckets: dict[tuple[str, int], list[str]] = {}
    for (name, rtype, comp) in hdf5_order:
        buckets.setdefault((rtype, comp), []).append(name)

    out: dict[tuple[str, str, int], np.ndarray] = {}
    for (rtype, comp), names in buckets.items():
        # rtype is already the SU/HDF5 short token (DISP/VEL/ACC/PS/GRAD).
        if rtype == "PS":
            su_path = outdir / f"{base_filename}_{rtype}_{source_id}.su"
        else:
            letter = COMP_LETTERS[comp]
            su_path = outdir / f"{base_filename}_{rtype}_{letter}_{source_id}.su"
        assert su_path.exists(), f"SU output missing: {su_path}"
        data, _dt, _ns, num_traces = read_su_file(su_path)
        assert num_traces == len(names), (
            f"SU trace count {num_traces} != expected {len(names)} for {su_path.name}"
        )
        for i, name in enumerate(names):
            out[(name, rtype, comp)] = data[i, :]
    return out


# ---------------------------------------------------------------------------
# Config mutation + simulation runner
# ---------------------------------------------------------------------------

def mutate_config(src: Path, dst_config: Path, outdir: Path, device: str) -> None:
    """Load src config, mutate for consistency test, write to dst_config."""
    with open(src) as f:
        cfg = yaml.safe_load(f)

    # Canonical mapping-list form: [{type: ascii}, {type: hdf5}, {type: su}].
    cfg["receivers"]["output"].pop("format", None)
    cfg["receivers"]["output"]["formats"] = [
        "ascii",
        "hdf5",
        "su",
    ]
    cfg["receivers"]["output"]["filename"] = "seis"

    cfg.setdefault("simulation", {})["steps"] = SHORT_STEPS

    cfg["simulation"]["directory"] = str(outdir)
    vis = cfg.get("vis") or {}
    if "wavefield" in vis:
        vis["wavefield"]["enabled"] = False

    cfg.setdefault("device", {})["type"] = device

    with open(dst_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def run_simulation(executable: Path, np_procs: int, config: Path,
                   mpi_cmd: str, cwd: Path, timeout: int = 600
                   ) -> tuple[bool, str]:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(executable), "--config", str(config)]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(executable), "--config", str(config)]
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True,
                                text=True, timeout=timeout)
        if result.returncode != 0:
            return False, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_id,config_path", CASES, ids=[c[0] for c in CASES])
def test_receiver_format_consistency(
    case_id: str,
    config_path: Path,
    executable: Path,
    np_procs: int,
    device_type: str,
    mpi_cmd: str,
    keep_results: bool,
    tmp_path_factory,
):
    if not config_path.exists():
        pytest.skip(f"config not found: {config_path}")

    work = tmp_path_factory.mktemp(f"consistency_{case_id.replace('/', '_')}")
    outdir = work / "results"
    outdir.mkdir()
    tmp_config = work / "config.yaml"
    mutate_config(config_path, tmp_config, outdir, device_type)

    ok, msg = run_simulation(executable.resolve(), np_procs, tmp_config,
                             mpi_cmd, cwd=work)
    assert ok, f"{case_id}: simulation failed:\n{msg}"

    # Filename suffix = shot_id (default 0).
    with open(config_path) as _f_cfg:
        _src_cfg = yaml.safe_load(_f_cfg).get("sources", {})
    source_id = f"{_src_cfg.get('shot_id', 0):04d}"
    ascii_traces = read_ascii_traces(outdir, source_id)
    assert ascii_traces, f"{case_id}: no ASCII receiver files found in {outdir}"

    hdf5_traces, real_dtype = read_hdf5_traces(outdir, "seis", source_id)
    rtol_ah, atol_ah = TOLS[real_dtype]
    assert hdf5_traces.keys() == ascii_traces.keys(), (
        f"{case_id}: ASCII/HDF5 receiver sets differ\n"
        f"  ASCII only: {ascii_traces.keys() - hdf5_traces.keys()}\n"
        f"  HDF5  only: {hdf5_traces.keys() - ascii_traces.keys()}"
    )

    su_traces = read_su_traces(outdir, "seis", source_id, hdf5_traces)
    assert su_traces.keys() == ascii_traces.keys(), (
        f"{case_id}: ASCII/SU receiver sets differ\n"
        f"  only ASCII: {ascii_traces.keys() - su_traces.keys()}\n"
        f"  only SU:    {su_traces.keys() - ascii_traces.keys()}"
    )

    failures: list[str] = []
    for key, a in ascii_traces.items():
        h = hdf5_traces[key]
        s = su_traces[key]
        if a.shape != h.shape:
            failures.append(f"{key}: ASCII {a.shape} vs HDF5 {h.shape} shape mismatch")
            continue
        if h.shape[-1] != s.shape[-1]:
            failures.append(f"{key}: HDF5 {h.shape} vs SU {s.shape} length mismatch")
            continue
        if not np.allclose(a, h, rtol=rtol_ah, atol=atol_ah):
            diff = float(np.max(np.abs(a - h)))
            failures.append(f"{key}: ASCII vs HDF5 max abs diff = {diff:.3e}")
        if not np.allclose(h.astype(np.float32), s,
                           rtol=RTOL_SU_FLOAT32, atol=ATOL_SU_FLOAT32):
            diff = float(np.max(np.abs(h.astype(np.float32) - s)))
            failures.append(f"{key}: HDF5 vs SU max abs diff = {diff:.3e}")

    if not keep_results:
        shutil.rmtree(work, ignore_errors=True)

    assert not failures, f"{case_id}: {len(failures)} mismatches:\n" + "\n".join(failures)
