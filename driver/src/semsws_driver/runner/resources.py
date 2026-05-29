"""Allocation detection and per-shot Slot pool.

Detects the current scheduler (SLURM, PJM, PBS, LSF, or local) and exposes a
uniform `Allocation`. `ResourcePool` carves it into `Slot` objects with
absolute cpu_ids, physical gpu_ids, and node name; GPU visibility env vars
are set automatically per slot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

log = logging.getLogger(__name__)

Scheduler = Literal["slurm", "pjm", "pbs", "lsf", "local", "manual"]


@dataclass(frozen=True)
class Allocation:
    scheduler: Scheduler
    nodes: list[str]
    cpus_per_node: int
    gpus_per_node: int
    ranks_per_node: int
    job_id: str = ""
    env_seen: dict[str, str] = field(default_factory=dict)
    env_missing_used_default: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_ranks(self) -> int:
        return self.ranks_per_node * self.n_nodes


@dataclass(frozen=True)
class Slot:
    """A unit of resources for one shot.

    * `nodes` is 1+ node names. Single-node shots have `len(nodes) == 1`,
      multi-node shots (e.g. ranks_per_shot=256 across 2x 128-core nodes)
      have len > 1.
    * `cpu_ids` and `gpu_ids` are *per-node* lists. They apply identically
      to every node in `nodes` (HPC nodes are homogeneous within an
      allocation).
    * total ranks of this slot = `len(nodes) * len(cpu_ids) / cpus_per_rank`.
    """
    index: int
    nodes: list[str]
    cpu_ids: list[int]
    gpu_ids: list[int]
    numa_nodes: Optional[list[int]] = None
    env_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def node(self) -> str:
        """Primary (first) node of this slot. Convenience for single-node
        callers; multi-node callers should iterate `slot.nodes`."""
        return self.nodes[0]


# ---------------------------------------------------------------------------
# Allocation detection
# ---------------------------------------------------------------------------

def detect_allocation(
    *,
    force: Optional[Scheduler] = None,
    cores_per_node: Optional[int] = None,
    gpus_per_node: Optional[int] = None,
    ranks_per_node: Optional[int] = None,
    nodes: Optional[Union[list[str], str]] = None,
) -> Allocation:
    """Detect the current allocation, or build a manual one.

    `force` overrides auto-detection. The numeric kwargs are used as
    fallbacks when the scheduler env does not expose a value (e.g. PJM
    doesn't publish cores_per_node) and as the only source for the
    "manual" scheduler.

    `nodes` accepts a list[str] or a string. Strings are passed through
    `_resolve_nodes`, which handles: a path to a nodefile (deduped
    line-by-line), a SLURM-compressed nodelist (`nid[1-4]`), or a
    comma-separated list.
    """
    nodes_list = _resolve_nodes(nodes) if isinstance(nodes, str) else nodes

    if force == "manual":
        return _manual_allocation(nodes_list, ranks_per_node, cores_per_node, gpus_per_node)
    if force == "slurm" or (force is None and "SLURM_JOB_ID" in os.environ):
        return _slurm_allocation(cores_per_node, gpus_per_node)
    if force == "pjm" or (force is None and "PJM_JOBID" in os.environ):
        return _pjm_allocation(cores_per_node, gpus_per_node, ranks_per_node, nodes_list)
    if force == "pbs" or (force is None and "PBS_JOBID" in os.environ):
        return _pbs_allocation(cores_per_node, gpus_per_node)
    if force == "lsf" or (force is None and "LSB_JOBID" in os.environ):
        return _lsf_allocation(cores_per_node, gpus_per_node)
    return _local_allocation(cores_per_node, gpus_per_node)


def _resolve_nodes(value: str) -> list[str]:
    """Convert a YAML `nodes:` string to a list of hostnames.

    Accepted shapes (checked in this order):
      * absolute / relative path to a nodefile  → read, dedupe lines
      * SLURM-compressed form `prefix[1-4,7]`   → expanded
      * comma-separated list `a,b,c`            → split
      * single hostname                         → wrapped in list

    Empty strings raise. Environment variables (`$VAR`, `${VAR}`) are
    expanded before path checking so the caller can pass
    `${PBS_NODEFILE}` directly.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"_resolve_nodes expects str, got {type(value).__name__}"
        )
    s = value.strip()
    if not s:
        raise ValueError("nodes string is empty")
    expanded = os.path.expandvars(os.path.expanduser(s))
    p = Path(expanded)
    if p.is_file():
        raw = [tok for tok in p.read_text().split() if tok]
        if not raw:
            raise ValueError(f"nodefile {p} is empty")
        return sorted(set(raw))
    if "[" in s:
        return _parse_slurm_nodelist(s)
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]


# ---- SLURM -----------------------------------------------------------------

def _slurm_allocation(
    cores_override: Optional[int], gpus_override: Optional[int]
) -> Allocation:
    seen: dict[str, str] = {}
    missing: list[str] = []

    def _get(name: str, *, required: bool = False) -> Optional[str]:
        v = os.environ.get(name)
        if v is not None:
            seen[name] = v
        elif required:
            raise RuntimeError(f"SLURM env required but missing: {name}")
        return v

    job_id = _get("SLURM_JOB_ID", required=True)
    nodelist = _get("SLURM_JOB_NODELIST", required=True)
    n_nodes_raw = _get("SLURM_NNODES")

    nodes = _expand_slurm_nodelist(nodelist)
    if n_nodes_raw and int(n_nodes_raw) != len(nodes):
        log.warning(
            "SLURM_NNODES=%s but nodelist expanded to %d hosts", n_nodes_raw, len(nodes)
        )

    ranks_per_node = 0
    rpn_raw = _get("SLURM_NTASKS_PER_NODE")
    if rpn_raw:
        ranks_per_node = int(rpn_raw)
    else:
        tpn_raw = _get("SLURM_TASKS_PER_NODE")
        if tpn_raw:
            ranks_per_node = int(tpn_raw.split("(")[0])
        else:
            nprocs_raw = _get("SLURM_NPROCS")
            if nprocs_raw and len(nodes) > 0:
                ranks_per_node = int(nprocs_raw) // len(nodes)
                missing.append("SLURM_NTASKS_PER_NODE")
    if ranks_per_node <= 0:
        raise RuntimeError(
            "SLURM_NTASKS_PER_NODE / SLURM_TASKS_PER_NODE / SLURM_NPROCS all "
            "missing: cannot determine ranks per node"
        )

    cpus_per_node = int(_get("SLURM_CPUS_ON_NODE") or cores_override or 0)
    if cpus_per_node <= 0:
        raise RuntimeError(
            "SLURM_CPUS_ON_NODE missing and cores_per_node not provided"
        )

    gpus_raw = _get("SLURM_GPUS_ON_NODE") or _get("SLURM_GPUS_PER_NODE")
    gpus_per_node = int(gpus_raw) if gpus_raw else (gpus_override or 0)
    if not gpus_raw and gpus_override is not None:
        missing.append("SLURM_GPUS_ON_NODE")

    return Allocation(
        scheduler="slurm",
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        gpus_per_node=gpus_per_node,
        ranks_per_node=ranks_per_node,
        job_id=job_id or "",
        env_seen=seen,
        env_missing_used_default=missing,
    )


def _expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand SLURM compressed nodelist (e.g. 'nid[001-004,008]')."""
    if shutil.which("scontrol") is not None:
        try:
            r = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return [line.strip() for line in r.stdout.splitlines() if line.strip()]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("scontrol failed (%s); falling back to pure-Python parser", e)
    return _parse_slurm_nodelist(nodelist)


def _parse_slurm_nodelist(s: str) -> list[str]:
    """Pure-Python parser for SLURM nodelist syntax 'prefix[range,...]'."""
    out: list[str] = []
    pos = 0
    n = len(s)
    while pos < n:
        # Read prefix until '[' or ',' or end
        end = pos
        while end < n and s[end] not in "[,":
            end += 1
        prefix = s[pos:end]
        pos = end
        if pos < n and s[pos] == "[":
            close = s.index("]", pos)
            ranges = s[pos + 1 : close]
            for part in ranges.split(","):
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    width = len(lo)
                    for k in range(int(lo), int(hi) + 1):
                        out.append(f"{prefix}{k:0{width}d}")
                else:
                    width = len(part)
                    out.append(f"{prefix}{int(part):0{width}d}")
            pos = close + 1
        else:
            if prefix:
                out.append(prefix)
        if pos < n and s[pos] == ",":
            pos += 1
    return out


# ---- PJM (Fugaku 系) ------------------------------------------------------

def _pjm_allocation(
    cores_override: Optional[int],
    gpus_override: Optional[int],
    ranks_override: Optional[int],
    nodes_override: Optional[list[str]],
) -> Allocation:
    seen: dict[str, str] = {}
    missing: list[str] = []

    def _get(name: str, *, required: bool = False) -> Optional[str]:
        v = os.environ.get(name)
        if v is not None:
            seen[name] = v
        elif required:
            raise RuntimeError(f"PJM env required but missing: {name}")
        return v

    job_id = _get("PJM_JOBID", required=True)
    n_nodes_raw = _get("PJM_NODE", required=True)
    n_nodes = int(n_nodes_raw)

    nodelist_env = _get("PJM_NODE_LIST")
    if nodelist_env:
        nodes = [h.strip() for h in nodelist_env.split(",") if h.strip()]
    elif nodes_override:
        nodes = list(nodes_override)
        missing.append("PJM_NODE_LIST")
    else:
        nodes = [f"pjm-node-{i}" for i in range(n_nodes)]
        missing.append("PJM_NODE_LIST")

    rpn_raw = _get("PJM_PROC_BY_NODE")
    if rpn_raw:
        ranks_per_node = int(rpn_raw)
    elif ranks_override:
        ranks_per_node = ranks_override
        missing.append("PJM_PROC_BY_NODE")
    else:
        raise RuntimeError(
            "PJM_PROC_BY_NODE missing and ranks_per_node not provided"
        )

    if cores_override is None:
        raise RuntimeError(
            "PJM scheduler: cores_per_node must be provided "
            "(no PJM env exposes it)"
        )
    cpus_per_node = cores_override

    return Allocation(
        scheduler="pjm",
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        gpus_per_node=gpus_override or 0,
        ranks_per_node=ranks_per_node,
        job_id=job_id or "",
        env_seen=seen,
        env_missing_used_default=missing,
    )


# ---- PBS / Torque ----------------------------------------------------------

def _pbs_allocation(
    cores_override: Optional[int], gpus_override: Optional[int]
) -> Allocation:
    """Build a PBS Allocation.

    Resolution priority for `ranks_per_node`:
      1. `PBS_NUM_PPN` env var (canonical, robust)
      2. Legacy: count occurrences of the first node in `PBS_NODEFILE`.
         This works when the nodefile has one line per process slot, but
         is silently wrong on NQSV/PBS-pro setups that emit one line per
         node. Triggers a DeprecationWarning when used.

    Nodes are always derived by dedupe of `PBS_NODEFILE` line tokens, so
    the file format (1-line-per-node vs 1-line-per-rank) is invisible to
    the rest of the pipeline.
    """
    seen: dict[str, str] = {}
    missing: list[str] = []

    job_id = os.environ.get("PBS_JOBID", "")
    if job_id:
        seen["PBS_JOBID"] = job_id
    nodefile_path = os.environ.get("PBS_NODEFILE")
    if not nodefile_path:
        raise RuntimeError("PBS_NODEFILE not set: cannot read PBS allocation")
    seen["PBS_NODEFILE"] = nodefile_path

    raw = Path(nodefile_path).read_text().split()
    nodes = sorted(set(raw))
    if not nodes:
        raise RuntimeError(f"PBS nodefile {nodefile_path} appears empty")

    ppn_raw = os.environ.get("PBS_NUM_PPN")
    legacy_count = raw.count(nodes[0])
    if ppn_raw:
        seen["PBS_NUM_PPN"] = ppn_raw
        ranks_per_node = int(ppn_raw)
        cpus_per_node = ranks_per_node
    else:
        ranks_per_node = legacy_count
        missing.append("PBS_NUM_PPN")
        if cores_override:
            cpus_per_node = cores_override
        else:
            cpus_per_node = os.cpu_count() or 1
        log.warning(
            "PBS_NUM_PPN not set; inferring ranks_per_node=%d from "
            "PBS_NODEFILE occurrence count. This will silently misread "
            "NQSV-style 1-line-per-node nodefiles. Export PBS_NUM_PPN "
            "or set run.ranks_per_node explicitly. (deprecated)",
            ranks_per_node,
        )
    if ranks_per_node <= 0:
        raise RuntimeError(
            "PBS allocation: ranks_per_node is 0; export PBS_NUM_PPN or "
            "set run.ranks_per_node in the YAML"
        )

    gpus_per_node = gpus_override if gpus_override is not None else _probe_gpus_local()
    if gpus_override is None:
        missing.append("(probed locally for gpus)")

    return Allocation(
        scheduler="pbs",
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        gpus_per_node=gpus_per_node,
        ranks_per_node=ranks_per_node,
        job_id=job_id,
        env_seen=seen,
        env_missing_used_default=missing,
    )


# ---- LSF -------------------------------------------------------------------

def _lsf_allocation(
    cores_override: Optional[int], gpus_override: Optional[int]
) -> Allocation:
    """Build an LSF Allocation.

    Resolution priority for `ranks_per_node`:
      1. `LSB_DJOB_NUMPROC // n_nodes` (canonical, robust)
      2. Legacy: count first node's occurrences in `LSB_DJOB_HOSTFILE`
         or `LSB_HOSTS`. Same fragility as PBS — wrong if the hostfile
         is 1-line-per-node. Triggers a DeprecationWarning when used.
    """
    seen: dict[str, str] = {}
    missing: list[str] = []

    job_id = os.environ.get("LSB_JOBID", "")
    if job_id:
        seen["LSB_JOBID"] = job_id

    hostfile = os.environ.get("LSB_DJOB_HOSTFILE", "")
    if hostfile:
        seen["LSB_DJOB_HOSTFILE"] = hostfile
    if hostfile and Path(hostfile).exists():
        raw = Path(hostfile).read_text().split()
    else:
        hosts_env = os.environ.get("LSB_HOSTS", "")
        if not hosts_env:
            raise RuntimeError(
                "LSF: LSB_DJOB_HOSTFILE or LSB_HOSTS required"
            )
        seen["LSB_HOSTS"] = hosts_env
        raw = hosts_env.split()

    nodes = sorted(set(raw))
    if not nodes:
        raise RuntimeError("LSF: empty hostfile/hosts")

    numproc_raw = os.environ.get("LSB_DJOB_NUMPROC", "")
    if numproc_raw:
        seen["LSB_DJOB_NUMPROC"] = numproc_raw
        ranks_per_node = int(numproc_raw) // len(nodes)
    else:
        ranks_per_node = raw.count(nodes[0])
        missing.append("LSB_DJOB_NUMPROC")
        log.warning(
            "LSB_DJOB_NUMPROC not set; inferring ranks_per_node=%d from "
            "hostfile occurrence count. This will silently misread "
            "1-line-per-node hostfiles. Export LSB_DJOB_NUMPROC or "
            "set run.ranks_per_node explicitly. (deprecated)",
            ranks_per_node,
        )
    if ranks_per_node <= 0:
        raise RuntimeError(
            "LSF allocation: ranks_per_node is 0; export LSB_DJOB_NUMPROC "
            "or set run.ranks_per_node in the YAML"
        )

    cpus_per_node = cores_override if cores_override else (os.cpu_count() or 1)
    if cores_override is None:
        missing.append("(LSF env exposes no cpus_per_node, used os.cpu_count)")

    gpus_per_node = gpus_override if gpus_override is not None else _probe_gpus_local()
    if gpus_override is None:
        missing.append("(probed locally for gpus)")

    return Allocation(
        scheduler="lsf",
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        gpus_per_node=gpus_per_node,
        ranks_per_node=ranks_per_node,
        job_id=job_id,
        env_seen=seen,
        env_missing_used_default=missing,
    )


# ---- local ----------------------------------------------------------------

def _local_allocation(
    cores_override: Optional[int], gpus_override: Optional[int]
) -> Allocation:
    cpus = cores_override if cores_override else (os.cpu_count() or 1)
    gpus = gpus_override if gpus_override is not None else _probe_gpus_local()
    return Allocation(
        scheduler="local",
        nodes=[socket.gethostname()],
        cpus_per_node=cpus,
        gpus_per_node=gpus,
        ranks_per_node=cpus,
        job_id="local",
        env_seen={},
        env_missing_used_default=[],
    )


def _manual_allocation(
    nodes: Optional[list[str]],
    ranks_per_node: Optional[int],
    cores_per_node: Optional[int],
    gpus_per_node: Optional[int],
) -> Allocation:
    if not nodes:
        raise RuntimeError("manual allocation requires nodes=[...]")
    if not ranks_per_node or ranks_per_node <= 0:
        raise RuntimeError("manual allocation requires ranks_per_node > 0")
    if not cores_per_node or cores_per_node <= 0:
        raise RuntimeError("manual allocation requires cores_per_node > 0")
    return Allocation(
        scheduler="manual",
        nodes=list(nodes),
        cpus_per_node=cores_per_node,
        gpus_per_node=gpus_per_node or 0,
        ranks_per_node=ranks_per_node,
        job_id="manual",
        env_seen={},
        env_missing_used_default=[],
    )


# ---------------------------------------------------------------------------
# GPU probe
# ---------------------------------------------------------------------------

def _probe_gpus_local() -> int:
    """Probe available GPUs on the local host. Returns 0 if none / unknown."""
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        v = os.environ.get(var)
        if v is not None:
            return len([x for x in v.split(",") if x.strip()])
    for cmd, count in (
        (["nvidia-smi", "-L"], lambda o: len(o.strip().splitlines())),
        (["rocm-smi", "--showid"], lambda o: sum(1 for L in o.splitlines() if "GPU[" in L)),
    ):
        try:
            r = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2)
            return count(r.stdout)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    return 0


# ---------------------------------------------------------------------------
# NUMA topology
# ---------------------------------------------------------------------------

def _build_cpu_blocks(
    *,
    cpus_per_node: int,
    cpus_per_slot: int,
    cpu_to_numa: dict[int, int],
    numa_aware: bool,
) -> list[list[int]]:
    """Split per-node CPUs into per-slot CPU id blocks.

    numa_aware=False (or no NUMA topology) → contiguous blocks
    [0..M), [M..2M), … of size `cpus_per_slot`.

    numa_aware=True → blocks are packed within a single NUMA domain.
    Each NUMA's CPUs are taken in order, and a block crosses a NUMA
    boundary only if `cpus_per_slot` exceeds one NUMA's size (in which
    case multi-NUMA blocks are allowed but warned).
    """
    if not numa_aware or not cpu_to_numa:
        n_blocks = cpus_per_node // cpus_per_slot if cpus_per_slot > 0 else 0
        return [
            list(range(j * cpus_per_slot, (j + 1) * cpus_per_slot))
            for j in range(n_blocks)
        ]

    # Group CPUs by NUMA preserving CPU-id order within each NUMA.
    by_numa: dict[int, list[int]] = {}
    for cpu_id in sorted(c for c in cpu_to_numa.keys() if c < cpus_per_node):
        by_numa.setdefault(cpu_to_numa[cpu_id], []).append(cpu_id)

    if not by_numa:
        return []

    blocks: list[list[int]] = []
    if all(len(cpus) >= cpus_per_slot for cpus in by_numa.values()):
        # Whole-shot fits in a single NUMA: pack within each NUMA.
        for numa_id in sorted(by_numa.keys()):
            cpus = by_numa[numa_id]
            n = len(cpus) // cpus_per_slot
            for j in range(n):
                blocks.append(cpus[j * cpus_per_slot:(j + 1) * cpus_per_slot])
    else:
        # Slot exceeds one NUMA: fall back to contiguous, warn.
        log.warning(
            "numa_aware=True but cpus_per_slot=%d exceeds NUMA domain size "
            "(max=%d); falling back to NUMA-crossing slots.",
            cpus_per_slot,
            max(len(c) for c in by_numa.values()),
        )
        n = cpus_per_node // cpus_per_slot
        blocks = [
            list(range(j * cpus_per_slot, (j + 1) * cpus_per_slot))
            for j in range(n)
        ]
    return blocks


def read_cpu_to_numa() -> dict[int, int]:
    """Parse `lscpu -e=CPU,NODE` to map absolute CPU id -> NUMA node id."""
    if shutil.which("lscpu") is None:
        return {}
    try:
        r = subprocess.run(
            ["lscpu", "-e=CPU,NODE"], check=True, capture_output=True, text=True, timeout=2
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    out: dict[int, int] = {}
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = int(parts[1])
    return out


# ---------------------------------------------------------------------------
# ResourcePool: slot carving + asyncio-safe acquire/release
# ---------------------------------------------------------------------------

DeviceKind = Literal["cpu", "cuda", "hip"]


class ResourcePool:
    """Carve an Allocation into per-shot slots and hand them out concurrently.

    slots_per_node = min(
        ranks_per_node // ranks_per_shot,
        cpus_per_node  // (ranks_per_shot * cpus_per_rank),
        gpus_per_node  // gpus_per_shot      # if gpus_per_shot > 0
    )
    """

    def __init__(
        self,
        *,
        allocation: Allocation,
        ranks_per_shot: int,
        gpus_per_shot: int = 0,
        cpus_per_rank: int = 1,
        device_kind: DeviceKind = "cpu",
        numa_aware: bool = False,
    ):
        if gpus_per_shot > 0 and gpus_per_shot > allocation.gpus_per_node:
            raise RuntimeError(
                f"gpus_per_shot ({gpus_per_shot}) > gpus_per_node "
                f"({allocation.gpus_per_node})"
            )

        # ---- decide single-node vs multi-node carving -----------------
        if ranks_per_shot <= allocation.ranks_per_node:
            nodes_per_shot = 1
            ranks_per_node_in_slot = ranks_per_shot
        else:
            if ranks_per_shot % allocation.ranks_per_node != 0:
                raise RuntimeError(
                    f"ranks_per_shot ({ranks_per_shot}) must be a multiple of "
                    f"ranks_per_node ({allocation.ranks_per_node}) when "
                    "spanning nodes"
                )
            nodes_per_shot = ranks_per_shot // allocation.ranks_per_node
            ranks_per_node_in_slot = allocation.ranks_per_node
            if nodes_per_shot > allocation.n_nodes:
                raise RuntimeError(
                    f"ranks_per_shot ({ranks_per_shot}) requires "
                    f"{nodes_per_shot} nodes but allocation has only "
                    f"{allocation.n_nodes}"
                )

        cpus_per_slot_per_node = ranks_per_node_in_slot * max(1, cpus_per_rank)
        cpu_to_numa = read_cpu_to_numa() if numa_aware else {}
        gpu_env_key = (
            "CUDA_VISIBLE_DEVICES" if device_kind == "cuda"
            else "ROCR_VISIBLE_DEVICES" if device_kind == "hip"
            else None
        )

        slots: list[Slot] = []
        sid = 0

        if nodes_per_shot == 1:
            # ---- single-node carving (multiple slots may fit per node) ----
            cpu_blocks_per_node = _build_cpu_blocks(
                cpus_per_node=allocation.cpus_per_node,
                cpus_per_slot=cpus_per_slot_per_node,
                cpu_to_numa=cpu_to_numa,
                numa_aware=numa_aware,
            )
            spr = allocation.ranks_per_node // ranks_per_shot
            spc = len(cpu_blocks_per_node)
            if gpus_per_shot > 0:
                spg = allocation.gpus_per_node // gpus_per_shot
                slots_per_node = min(spr, spc, spg)
            else:
                slots_per_node = min(spr, spc)
            if slots_per_node <= 0:
                raise RuntimeError(
                    "no slots fit in the allocation; check ranks_per_shot / "
                    "gpus_per_shot / cpus_per_rank vs allocation "
                    f"(numa_aware={numa_aware}, cpus_per_slot={cpus_per_slot_per_node}, "
                    f"cpus_per_node={allocation.cpus_per_node})"
                )
            for node in allocation.nodes:
                for j in range(slots_per_node):
                    cpu_ids = cpu_blocks_per_node[j]
                    gpu_ids = (
                        list(range(j * gpus_per_shot, (j + 1) * gpus_per_shot))
                        if gpus_per_shot > 0 else []
                    )
                    slots.append(self._make_slot(
                        sid, [node], cpu_ids, gpu_ids,
                        cpu_to_numa, gpu_env_key, device_kind,
                    ))
                    sid += 1
        else:
            # ---- multi-node carving (1 slot spans nodes_per_shot nodes) ---
            total_slots = allocation.n_nodes // nodes_per_shot
            if total_slots <= 0:
                raise RuntimeError(
                    "no multi-node slots fit; check ranks_per_shot vs "
                    "allocation node count"
                )
            cpu_ids = list(range(cpus_per_slot_per_node))
            gpu_ids = (list(range(gpus_per_shot)) if gpus_per_shot > 0 else [])
            for s in range(total_slots):
                slot_nodes = list(
                    allocation.nodes[s * nodes_per_shot:
                                      (s + 1) * nodes_per_shot])
                slots.append(self._make_slot(
                    sid, slot_nodes, cpu_ids, gpu_ids,
                    cpu_to_numa, gpu_env_key, device_kind,
                ))
                sid += 1

        self.allocation = allocation
        self.slots = slots
        self.ranks_per_shot = ranks_per_shot
        self.gpus_per_shot = gpus_per_shot
        self.cpus_per_rank = cpus_per_rank
        self.device_kind = device_kind
        self.numa_aware = numa_aware

        self._free: list[Slot] = list(slots)
        self._sem: Optional[asyncio.Semaphore] = None
        self._lock: Optional[asyncio.Lock] = None
        self._sem_loop: Optional[asyncio.AbstractEventLoop] = None

        log.info(
            "ResourcePool: %d slots over %d nodes "
            "(ranks/shot=%d, gpus/shot=%d, cpus/rank=%d, device=%s, scheduler=%s, numa_aware=%s)",
            len(slots), allocation.n_nodes, ranks_per_shot, gpus_per_shot,
            cpus_per_rank, device_kind, allocation.scheduler, numa_aware,
        )

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @staticmethod
    def _make_slot(
        sid: int,
        nodes: list[str],
        cpu_ids: list[int],
        gpu_ids: list[int],
        cpu_to_numa: dict,
        gpu_env_key: Optional[str],
        device_kind: str,
    ) -> Slot:
        numa_nodes: Optional[list[int]] = None
        if cpu_to_numa:
            numa_nodes = sorted(
                {cpu_to_numa[c] for c in cpu_ids if c in cpu_to_numa})
        env_overrides: dict[str, str] = {}
        if gpu_ids and gpu_env_key:
            devs = ",".join(str(g) for g in gpu_ids)
            env_overrides[gpu_env_key] = devs
            if device_kind == "hip":
                env_overrides["HIP_VISIBLE_DEVICES"] = devs
        return Slot(
            index=sid,
            nodes=list(nodes),
            cpu_ids=list(cpu_ids),
            gpu_ids=list(gpu_ids),
            numa_nodes=numa_nodes,
            env_overrides=env_overrides,
        )

    def _ensure_loop_primitives(self) -> None:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(len(self.slots))
            self._lock = asyncio.Lock()
            self._sem_loop = loop
            self._free = list(self.slots)

    async def acquire(self) -> Slot:
        self._ensure_loop_primitives()
        assert self._sem is not None and self._lock is not None
        await self._sem.acquire()
        async with self._lock:
            return self._free.pop()

    async def release(self, slot: Slot) -> None:
        assert self._sem is not None and self._lock is not None
        async with self._lock:
            self._free.append(slot)
        self._sem.release()
