"""Allocation detection tests with env mocking."""

from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from semsws_driver.runner.resources import (
    Allocation,
    ResourcePool,
    Slot,
    _parse_slurm_nodelist,
    detect_allocation,
    read_cpu_to_numa,
)


# ---- nodelist parser -------------------------------------------------------

def test_parse_slurm_nodelist_compressed():
    assert _parse_slurm_nodelist("nid[001-003,007]") == [
        "nid001", "nid002", "nid003", "nid007",
    ]


def test_parse_slurm_nodelist_plain():
    assert _parse_slurm_nodelist("h1,h2,h3") == ["h1", "h2", "h3"]


def test_parse_slurm_nodelist_mixed():
    assert _parse_slurm_nodelist("nid[01-02],host42") == ["nid01", "nid02", "host42"]


def test_parse_slurm_nodelist_single():
    assert _parse_slurm_nodelist("solo") == ["solo"]


# ---- SLURM detection (mocked env) -----------------------------------------

@pytest.fixture
def slurm_env(monkeypatch):
    env = {
        "SLURM_JOB_ID": "99999",
        "SLURM_NNODES": "2",
        "SLURM_JOB_NODELIST": "nid[001-002]",
        "SLURM_NTASKS_PER_NODE": "8",
        "SLURM_CPUS_ON_NODE": "64",
        "SLURM_GPUS_ON_NODE": "4",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    yield env


def test_slurm_allocation(slurm_env, monkeypatch):
    # Fake scontrol via shutil.which returning None forces Python parser
    monkeypatch.setattr("shutil.which", lambda _: None)
    a = detect_allocation()
    assert a.scheduler == "slurm"
    assert a.nodes == ["nid001", "nid002"]
    assert a.cpus_per_node == 64
    assert a.gpus_per_node == 4
    assert a.ranks_per_node == 8
    assert a.job_id == "99999"


# ---- PBS detection ---------------------------------------------------------

def test_pbs_allocation(monkeypatch, tmp_path):
    nf = tmp_path / "nodefile"
    nf.write_text("c01\nc01\nc01\nc01\nc02\nc02\nc02\nc02\n")
    monkeypatch.setenv("PBS_JOBID", "12345.head")
    monkeypatch.setenv("PBS_NODEFILE", str(nf))
    monkeypatch.setenv("PBS_NUM_PPN", "4")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PJM_JOBID", raising=False)
    a = detect_allocation()
    assert a.scheduler == "pbs"
    assert a.nodes == ["c01", "c02"]
    assert a.ranks_per_node == 4
    assert a.cpus_per_node == 4


# ---- LSF detection ---------------------------------------------------------

def test_lsf_allocation(monkeypatch, tmp_path):
    hf = tmp_path / "hostfile"
    hf.write_text("h1\nh1\nh2\nh2\n")
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "PBS_JOBID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LSB_JOBID", "777")
    monkeypatch.setenv("LSB_DJOB_HOSTFILE", str(hf))
    a = detect_allocation()
    assert a.scheduler == "lsf"
    assert a.nodes == ["h1", "h2"]
    assert a.ranks_per_node == 2


# ---- PJM detection ---------------------------------------------------------

def test_pjm_allocation(monkeypatch):
    for k in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PJM_JOBID", "fugaku.42")
    monkeypatch.setenv("PJM_NODE", "4")
    monkeypatch.setenv("PJM_PROC_BY_NODE", "12")
    a = detect_allocation(cores_per_node=48, gpus_per_node=0)
    assert a.scheduler == "pjm"
    assert len(a.nodes) == 4
    assert a.ranks_per_node == 12
    assert a.cpus_per_node == 48


def test_pjm_missing_cores_raises(monkeypatch):
    for k in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PJM_JOBID", "fugaku.42")
    monkeypatch.setenv("PJM_NODE", "4")
    monkeypatch.setenv("PJM_PROC_BY_NODE", "12")
    with pytest.raises(RuntimeError, match="cores_per_node"):
        detect_allocation()


# ---- local fallback --------------------------------------------------------

def test_local_allocation(monkeypatch):
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    a = detect_allocation()
    assert a.scheduler == "local"
    assert len(a.nodes) == 1
    assert a.cpus_per_node >= 1


# ---- ResourcePool ----------------------------------------------------------

def test_resource_pool_carving():
    alloc = Allocation(
        scheduler="manual", nodes=["n1", "n2"],
        cpus_per_node=64, gpus_per_node=4, ranks_per_node=8,
    )
    pool = ResourcePool(allocation=alloc, ranks_per_shot=8,
                         gpus_per_shot=2, cpus_per_rank=8)
    # 1 slot/node × 2 nodes
    assert pool.n_slots == 2
    assert pool.slots[0].nodes == ["n1"]
    assert pool.slots[0].node == "n1"           # back-compat property
    assert pool.slots[0].cpu_ids == list(range(0, 64))
    assert pool.slots[0].gpu_ids == [0, 1]


def test_resource_pool_multi_node_shot():
    """32 cpu/shot × 16 shot on 4 node × 128 cpu allocation:
       16 single-node slots; each shot fits on 1 node."""
    alloc = Allocation(
        scheduler="manual", nodes=["n1", "n2", "n3", "n4"],
        cpus_per_node=128, gpus_per_node=0, ranks_per_node=128,
    )
    pool = ResourcePool(allocation=alloc, ranks_per_shot=32,
                         gpus_per_shot=0, cpus_per_rank=1)
    assert pool.n_slots == 16
    # First 4 slots are on n1 with disjoint cpu ranges.
    assert pool.slots[0].nodes == ["n1"]
    assert pool.slots[0].cpu_ids == list(range(0, 32))
    assert pool.slots[1].nodes == ["n1"]
    assert pool.slots[1].cpu_ids == list(range(32, 64))
    assert pool.slots[3].cpu_ids == list(range(96, 128))
    # Slot 4 starts a new node.
    assert pool.slots[4].nodes == ["n2"]
    assert pool.slots[4].cpu_ids == list(range(0, 32))
    # Last slot is on n4 occupying cpus 96-127.
    assert pool.slots[15].nodes == ["n4"]
    assert pool.slots[15].cpu_ids == list(range(96, 128))


def test_resource_pool_cross_node_shot():
    """256 rank shot spanning 2 nodes of 128 cpus each."""
    alloc = Allocation(
        scheduler="manual", nodes=["n1", "n2", "n3", "n4"],
        cpus_per_node=128, gpus_per_node=0, ranks_per_node=128,
    )
    pool = ResourcePool(allocation=alloc, ranks_per_shot=256,
                         gpus_per_shot=0, cpus_per_rank=1)
    # 2 nodes per shot → 4 nodes / 2 = 2 slots
    assert pool.n_slots == 2
    assert pool.slots[0].nodes == ["n1", "n2"]
    assert pool.slots[1].nodes == ["n3", "n4"]
    # cpu_ids interpreted per-node, all 128 cpus per node used
    assert pool.slots[0].cpu_ids == list(range(0, 128))


def test_resource_pool_cross_node_misaligned_raises():
    alloc = Allocation(
        scheduler="manual", nodes=["n1", "n2", "n3", "n4"],
        cpus_per_node=128, gpus_per_node=0, ranks_per_node=128,
    )
    # 200 is not a multiple of 128 → reject
    with pytest.raises(RuntimeError, match="multiple of"):
        ResourcePool(allocation=alloc, ranks_per_shot=200,
                      gpus_per_shot=0, cpus_per_rank=1)


def test_resource_pool_cross_node_insufficient_nodes_raises():
    alloc = Allocation(
        scheduler="manual", nodes=["n1"],
        cpus_per_node=128, gpus_per_node=0, ranks_per_node=128,
    )
    # Want 256 ranks = 2 nodes, only 1 node available.
    with pytest.raises(RuntimeError, match="requires .* nodes"):
        ResourcePool(allocation=alloc, ranks_per_shot=256,
                      gpus_per_shot=0, cpus_per_rank=1)


def test_resource_pool_gpu_isolation_env():
    alloc = Allocation(
        scheduler="manual", nodes=["n1"],
        cpus_per_node=16, gpus_per_node=4, ranks_per_node=4,
    )
    pool = ResourcePool(allocation=alloc, ranks_per_shot=2,
                         gpus_per_shot=1, cpus_per_rank=4,
                         device_kind="hip")
    # 2 slots, each gets 1 GPU
    assert pool.n_slots == 2
    assert pool.slots[0].env_overrides["ROCR_VISIBLE_DEVICES"] == "0"
    assert pool.slots[0].env_overrides["HIP_VISIBLE_DEVICES"] == "0"
    assert pool.slots[1].env_overrides["ROCR_VISIBLE_DEVICES"] == "1"


def test_resource_pool_no_fit_raises():
    alloc = Allocation(
        scheduler="manual", nodes=["n1"],
        cpus_per_node=4, gpus_per_node=1, ranks_per_node=1,
    )
    with pytest.raises(RuntimeError):
        ResourcePool(allocation=alloc, ranks_per_shot=8,
                     gpus_per_shot=0, cpus_per_rank=1)


# ---- _resolve_nodes --------------------------------------------------------

from semsws_driver.runner.resources import _resolve_nodes


def test_resolve_nodes_list_passthrough():
    assert _resolve_nodes("a,b,c") == ["a", "b", "c"]


def test_resolve_nodes_single_hostname():
    assert _resolve_nodes("loner") == ["loner"]


def test_resolve_nodes_compressed():
    assert _resolve_nodes("nid[001-002,007]") == ["nid001", "nid002", "nid007"]


def test_resolve_nodes_path_dedupes(tmp_path):
    nf = tmp_path / "nf"
    nf.write_text("cn1\ncn1\ncn1\ncn2\ncn2\n")
    assert _resolve_nodes(str(nf)) == ["cn1", "cn2"]


def test_resolve_nodes_path_one_line_per_node(tmp_path):
    # NQSV-style nodefile: 1 line / node
    nf = tmp_path / "nf"
    nf.write_text("cn698\ncn686\ncn683\ncn688\n")
    assert _resolve_nodes(str(nf)) == ["cn683", "cn686", "cn688", "cn698"]


def test_resolve_nodes_env_var_path(monkeypatch, tmp_path):
    nf = tmp_path / "nf"
    nf.write_text("ha\nhb\n")
    monkeypatch.setenv("MY_NODEFILE", str(nf))
    assert _resolve_nodes("${MY_NODEFILE}") == ["ha", "hb"]


def test_resolve_nodes_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        _resolve_nodes("")


def test_resolve_nodes_empty_file_raises(tmp_path):
    nf = tmp_path / "empty"
    nf.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _resolve_nodes(str(nf))


# ---- manual scheduler via detect_allocation --------------------------------


def test_manual_via_detect_allocation_with_list(monkeypatch):
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    a = detect_allocation(
        force="manual",
        nodes=["cn1", "cn2"],
        ranks_per_node=8,
        cores_per_node=8,
    )
    assert a.scheduler == "manual"
    assert a.nodes == ["cn1", "cn2"]
    assert a.ranks_per_node == 8
    assert a.cpus_per_node == 8


def test_manual_via_detect_allocation_with_nodefile_string(monkeypatch, tmp_path):
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    nf = tmp_path / "nodes"
    nf.write_text("cn698\ncn686\ncn683\ncn688\n")    # NQSV-style
    a = detect_allocation(
        force="manual",
        nodes=str(nf),
        ranks_per_node=128,
        cores_per_node=128,
    )
    assert a.scheduler == "manual"
    assert a.nodes == ["cn683", "cn686", "cn688", "cn698"]
    assert a.ranks_per_node == 128
    assert a.cpus_per_node == 128


# ---- PBS env-var primary, deprecated-fallback warning ----------------------


def test_pbs_env_var_overrides_file_count(monkeypatch, tmp_path):
    """When PBS_NUM_PPN is set, it wins regardless of nodefile format.

    NQSV-style 1-line-per-node nodefile used to misread as
    ranks_per_node=1; with the new logic, PBS_NUM_PPN takes priority.
    """
    nf = tmp_path / "nodefile"
    nf.write_text("cn698\ncn686\ncn683\ncn688\n")   # 1 line / node
    monkeypatch.setenv("PBS_JOBID", "es.42")
    monkeypatch.setenv("PBS_NODEFILE", str(nf))
    monkeypatch.setenv("PBS_NUM_PPN", "128")
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    a = detect_allocation()
    assert a.scheduler == "pbs"
    assert a.nodes == ["cn683", "cn686", "cn688", "cn698"]
    assert a.ranks_per_node == 128
    assert a.cpus_per_node == 128


def test_pbs_warns_when_falling_back_to_count(monkeypatch, tmp_path, caplog):
    """No PBS_NUM_PPN → falls back to occurrence count + emits a warning."""
    nf = tmp_path / "nodefile"
    nf.write_text("h1\nh1\nh2\nh2\n")
    monkeypatch.setenv("PBS_JOBID", "x")
    monkeypatch.setenv("PBS_NODEFILE", str(nf))
    monkeypatch.delenv("PBS_NUM_PPN", raising=False)
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    with caplog.at_level("WARNING", logger="semsws_driver.runner.resources"):
        a = detect_allocation()
    assert a.ranks_per_node == 2
    assert "PBS_NUM_PPN not set" in caplog.text


def test_lsf_env_var_primary(monkeypatch, tmp_path):
    hf = tmp_path / "hf"
    hf.write_text("h1\nh2\n")          # 1 line / node
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "PBS_JOBID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LSB_JOBID", "lsf.1")
    monkeypatch.setenv("LSB_DJOB_HOSTFILE", str(hf))
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "64")     # 64 / 2 nodes = 32
    a = detect_allocation()
    assert a.scheduler == "lsf"
    assert a.nodes == ["h1", "h2"]
    assert a.ranks_per_node == 32


# ---- Unified YAML overrides honored by every scheduler ---------------------


def test_pbs_yaml_overrides_win_over_env(monkeypatch, tmp_path):
    """YAML override beats both PBS_NUM_PPN and file count."""
    nf = tmp_path / "nf"
    nf.write_text("a\na\nb\nb\n")
    monkeypatch.setenv("PBS_JOBID", "x")
    monkeypatch.setenv("PBS_NODEFILE", str(nf))
    monkeypatch.setenv("PBS_NUM_PPN", "2")
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    a = detect_allocation(
        force="pbs",
        ranks_per_node=64,             # override wins
        cores_per_node=64,             # override wins
        nodes=["override1", "override2", "override3"],
    )
    assert a.nodes == ["override1", "override2", "override3"]
    assert a.ranks_per_node == 64
    assert a.cpus_per_node == 64


def test_pbs_yaml_overrides_no_env_required(monkeypatch):
    """PBS_NODEFILE is unnecessary when YAML provides every field."""
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "LSB_JOBID",
              "PBS_NODEFILE", "PBS_NUM_PPN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PBS_JOBID", "x")
    a = detect_allocation(
        force="pbs",
        nodes=["cn1", "cn2"],
        ranks_per_node=128,
        cores_per_node=128,
    )
    assert a.scheduler == "pbs"
    assert a.nodes == ["cn1", "cn2"]
    assert a.ranks_per_node == 128


def test_slurm_yaml_overrides_win(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "nid[01-04]")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")
    a = detect_allocation(
        force="slurm",
        nodes=["forced1", "forced2"],
        ranks_per_node=32,
        cores_per_node=128,
    )
    assert a.nodes == ["forced1", "forced2"]
    assert a.ranks_per_node == 32
    assert a.cpus_per_node == 128


def test_lsf_yaml_overrides_win(monkeypatch):
    monkeypatch.setenv("LSB_JOBID", "lsf.42")
    monkeypatch.setenv("LSB_HOSTS", "h1 h1 h2 h2")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "4")
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "PBS_JOBID"):
        monkeypatch.delenv(k, raising=False)
    a = detect_allocation(
        force="lsf",
        nodes=["x", "y"],
        ranks_per_node=16,
        cores_per_node=32,
    )
    assert a.nodes == ["x", "y"]
    assert a.ranks_per_node == 16
    assert a.cpus_per_node == 32


def test_local_yaml_overrides_shape_allocation(monkeypatch):
    for k in ("SLURM_JOB_ID", "PJM_JOBID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    a = detect_allocation(
        force="local",
        nodes=["simhost1", "simhost2"],
        ranks_per_node=4,
        cores_per_node=8,
    )
    assert a.scheduler == "local"
    assert a.nodes == ["simhost1", "simhost2"]
    assert a.ranks_per_node == 4
    assert a.cpus_per_node == 8
