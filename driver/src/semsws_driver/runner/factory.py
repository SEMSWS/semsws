"""Builds a Runner and BindingPolicy from a RunConfig."""

from __future__ import annotations

from typing import Optional

from ..core.run_config import RunConfig
from .binding import BindingPolicy
from .local import LocalRunner
from .mpi_direct import MpiDirectRunner
from .resources import Allocation, detect_allocation


def binding_policy_from(rc: RunConfig) -> BindingPolicy:
    return BindingPolicy(
        extra_launcher_args=dict(rc.binding.extra_launcher_args),
        rank_wrapper=(list(rc.binding.rank_wrapper)
                      if rc.binding.rank_wrapper is not None else None),
        extra_env=dict(rc.binding.extra_env),
    )


def _resolve_allocation(rc: RunConfig) -> Allocation:
    """Build the Allocation from the YAML run block.

    Pre-computes the allocation here (before the runner is instantiated)
    so the YAML override fields (`run.nodes` / `run.ranks_per_node` /
    `run.cpus_per_node`) are honored uniformly for every scheduler,
    independent of which runner ultimately consumes the allocation.
    """
    return detect_allocation(
        force=rc.scheduler,
        cores_per_node=rc.cpus_per_node,
        ranks_per_node=rc.ranks_per_node,
        nodes=rc.nodes,
    )


def make_runner(rc: RunConfig):
    """Return a Runner implementation suitable for `rc.scheduler`."""
    device_kind = rc.device_kind
    # GPU runs: we assume gpus_per_shot == ranks_per_shot (1 rank = 1 GPU).
    gpus_per_shot = rc.ranks_per_shot if device_kind in ("cuda", "hip") else 0

    allocation = _resolve_allocation(rc)

    if rc.scheduler == "local":
        return LocalRunner(
            semsws_binary=str(rc.binary),
            ranks_per_shot=rc.ranks_per_shot,
            cpus_per_rank=rc.cpus_per_rank,
            max_concurrent_shots=rc.shots_per_job,
            gpus_per_shot=gpus_per_shot,
            device_kind=device_kind,
            numa_aware=rc.numa_aware,
            allocation_override=allocation,
        )
    return MpiDirectRunner(
        semsws_binary=str(rc.binary),
        ranks_per_shot=rc.ranks_per_shot,
        cpus_per_rank=rc.cpus_per_rank,
        gpus_per_shot=gpus_per_shot,
        max_concurrent_shots=rc.shots_per_job,
        device_kind=device_kind,
        numa_aware=rc.numa_aware,
        force_scheduler=rc.scheduler,
        allocation_override=allocation,
    )
