"""RunConfig parsing tests."""

from __future__ import annotations

import pytest

from semsws_driver.core.run_config import RunConfig


def _base():
    return {"binary": "/path/to/semsws"}


def test_minimal_defaults():
    rc = RunConfig.from_dict(_base())
    assert str(rc.binary) == "/path/to/semsws"
    assert rc.scheduler == "local"
    assert rc.launcher is None
    assert rc.ranks_per_shot == 1
    assert rc.device_kind == "cpu"
    assert rc.shots_per_job == 1
    assert rc.binding.extra_launcher_args == {}
    assert rc.binding.rank_wrapper is None
    assert rc.binding.extra_env == {}
    assert rc.env == {}


def test_effective_launcher_defaults():
    assert RunConfig.from_dict({"binary": "/x", "scheduler": "local"}).effective_launcher() == "local"
    assert RunConfig.from_dict({"binary": "/x", "scheduler": "slurm"}).effective_launcher() == "srun"
    assert RunConfig.from_dict({"binary": "/x", "scheduler": "pjm"}).effective_launcher() == "pjm"
    assert RunConfig.from_dict({"binary": "/x", "scheduler": "pbs"}).effective_launcher() == "mpirun"
    assert RunConfig.from_dict({"binary": "/x", "scheduler": "lsf"}).effective_launcher() == "mpirun"


def test_explicit_launcher_wins():
    rc = RunConfig.from_dict({"binary": "/x", "scheduler": "slurm", "launcher": "mpirun"})
    assert rc.effective_launcher() == "mpirun"


def test_full_lumi_gpu_example():
    rc = RunConfig.from_dict({
        "binary": "/scratch/build/semsws",
        "scheduler": "slurm",
        "launcher": "srun",
        "ranks_per_shot": 4,
        "device_kind": "hip",
        "shots_per_job": 4,
        "binding": {
            "extra_launcher_args": {
                "slurm": ["--cpu-bind=cores", "--mem-bind=local"],
            },
            "rank_wrapper": ["./select_gpu.sh"],
            "extra_env": {"OMP_PROC_BIND": "close"},
        },
        "env": {"OMP_NUM_THREADS": "1", "MPICH_GPU_SUPPORT_ENABLED": "1"},
    })
    assert rc.scheduler == "slurm"
    assert rc.device_kind == "hip"
    assert rc.shots_per_job == 4
    assert rc.binding.rank_wrapper == ["./select_gpu.sh"]
    assert rc.binding.extra_launcher_args["slurm"] == [
        "--cpu-bind=cores", "--mem-bind=local"]
    assert rc.env["OMP_NUM_THREADS"] == "1"


def test_missing_binary_rejected():
    with pytest.raises(ValueError, match="binary"):
        RunConfig.from_dict({"scheduler": "local"})


def test_bad_scheduler_rejected():
    with pytest.raises(ValueError, match="scheduler"):
        RunConfig.from_dict({"binary": "/x", "scheduler": "kubernetes"})


def test_bad_device_kind_rejected():
    with pytest.raises(ValueError, match="device_kind"):
        RunConfig.from_dict({"binary": "/x", "device_kind": "tpu"})


def test_bad_launcher_rejected():
    with pytest.raises(ValueError, match="launcher"):
        RunConfig.from_dict({"binary": "/x", "launcher": "openpbs"})


def test_zero_ranks_rejected():
    with pytest.raises(ValueError, match="ranks_per_shot"):
        RunConfig.from_dict({"binary": "/x", "ranks_per_shot": 0})


def test_zero_shots_per_job_rejected():
    with pytest.raises(ValueError, match="shots_per_job"):
        RunConfig.from_dict({"binary": "/x", "shots_per_job": 0})


# ---- manual scheduler + allocation overrides -------------------------------


def test_manual_scheduler_requires_overrides():
    # nodes/ranks_per_node/cpus_per_node are mandatory for manual
    with pytest.raises(ValueError, match="manual requires"):
        RunConfig.from_dict({"binary": "/x", "scheduler": "manual"})


def test_manual_scheduler_full():
    rc = RunConfig.from_dict({
        "binary": "/x",
        "scheduler": "manual",
        "nodes": ["cn1", "cn2", "cn3", "cn4"],
        "ranks_per_node": 128,
        "cpus_per_node": 128,
        "ranks_per_shot": 16,
    })
    assert rc.scheduler == "manual"
    assert rc.nodes == ["cn1", "cn2", "cn3", "cn4"]
    assert rc.ranks_per_node == 128
    assert rc.cpus_per_node == 128
    # manual defaults its launcher to mpirun (typical inside-allocation launch)
    assert rc.effective_launcher() == "mpirun"


def test_nodes_accepts_string():
    rc = RunConfig.from_dict({
        "binary": "/x",
        "scheduler": "manual",
        "nodes": "cn[1-4]",
        "ranks_per_node": 4,
        "cpus_per_node": 4,
    })
    # _resolve_nodes is deferred to detect_allocation time; RunConfig just
    # stores the raw value.
    assert rc.nodes == "cn[1-4]"


def test_nodes_bad_type_rejected():
    with pytest.raises(ValueError, match="run.nodes"):
        RunConfig.from_dict({"binary": "/x", "nodes": 42})


def test_ranks_per_node_zero_rejected():
    with pytest.raises(ValueError, match="ranks_per_node"):
        RunConfig.from_dict({
            "binary": "/x", "scheduler": "manual",
            "nodes": ["a"], "ranks_per_node": 0, "cpus_per_node": 4,
        })


def test_cpus_per_node_zero_rejected():
    with pytest.raises(ValueError, match="cpus_per_node"):
        RunConfig.from_dict({
            "binary": "/x", "scheduler": "manual",
            "nodes": ["a"], "ranks_per_node": 4, "cpus_per_node": 0,
        })


def test_overrides_optional_for_native_schedulers():
    # PBS / SLURM etc. don't require manual overrides; they default to None
    rc = RunConfig.from_dict({"binary": "/x", "scheduler": "pbs"})
    assert rc.nodes is None
    assert rc.ranks_per_node is None
    assert rc.cpus_per_node is None
