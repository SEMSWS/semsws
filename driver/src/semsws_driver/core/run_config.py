"""`run:` section of the user YAML, parsed into a typed RunConfig dataclass.

The driver reads `run:`, builds a Runner from it, and strips the `run:` key
from the per-shot YAML before passing it to SEMSWS C++.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

Scheduler = Literal["local", "slurm", "pjm", "pbs", "lsf"]
Launcher = Literal["local", "mpirun", "mpiexec", "srun", "pjm"]
DeviceKind = Literal["cpu", "cuda", "hip"]


@dataclass
class BindingBlock:
    """`run.binding:` sub-section."""
    extra_launcher_args: dict[str, list[str]] = field(default_factory=dict)
    rank_wrapper: Optional[list[str]] = None
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class RunConfig:
    binary: Path
    scheduler: Scheduler = "local"
    launcher: Optional[Launcher] = None
    ranks_per_shot: int = 1
    cpus_per_rank: int = 1
    device_kind: DeviceKind = "cpu"
    shots_per_job: int = 1
    numa_aware: bool = False
    binding: BindingBlock = field(default_factory=BindingBlock)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        if "binary" not in d:
            raise ValueError("run.binary is required")
        scheduler = d.get("scheduler", "local")
        if scheduler not in ("local", "slurm", "pjm", "pbs", "lsf"):
            raise ValueError(
                f"run.scheduler must be one of "
                f"local|slurm|pjm|pbs|lsf, got {scheduler!r}")
        device_kind = d.get("device_kind", "cpu")
        if device_kind not in ("cpu", "cuda", "hip"):
            raise ValueError(
                f"run.device_kind must be one of cpu|cuda|hip, "
                f"got {device_kind!r}")

        launcher = d.get("launcher")
        if launcher is not None and launcher not in (
                "local", "mpirun", "mpiexec", "srun", "pjm"):
            raise ValueError(
                f"run.launcher must be one of "
                f"local|mpirun|mpiexec|srun|pjm, got {launcher!r}")

        ranks = int(d.get("ranks_per_shot", 1))
        if ranks < 1:
            raise ValueError("run.ranks_per_shot must be >= 1")
        cpus_per_rank = int(d.get("cpus_per_rank", 1))
        if cpus_per_rank < 1:
            raise ValueError("run.cpus_per_rank must be >= 1")
        shots_per_job = int(d.get("shots_per_job", 1))
        if shots_per_job < 1:
            raise ValueError("run.shots_per_job must be >= 1")
        numa_aware = bool(d.get("numa_aware", False))

        b = d.get("binding", {}) or {}
        if not isinstance(b, dict):
            raise ValueError("run.binding must be a mapping")
        binding = BindingBlock(
            extra_launcher_args={
                k: list(v) for k, v in (b.get("extra_launcher_args") or {}).items()
            },
            rank_wrapper=(list(b["rank_wrapper"])
                          if b.get("rank_wrapper") is not None else None),
            extra_env={str(k): str(v)
                       for k, v in (b.get("extra_env") or {}).items()},
        )

        env = {str(k): str(v) for k, v in (d.get("env") or {}).items()}

        return cls(
            binary=_resolve_binary(d["binary"]),
            scheduler=scheduler,    # type: ignore[arg-type]
            launcher=launcher,      # type: ignore[arg-type]
            ranks_per_shot=ranks,
            cpus_per_rank=cpus_per_rank,
            device_kind=device_kind,    # type: ignore[arg-type]
            shots_per_job=shots_per_job,
            numa_aware=numa_aware,
            binding=binding,
            env=env,
        )

    def effective_launcher(self) -> Launcher:
        """Default launcher per scheduler when `launcher:` is omitted."""
        if self.launcher is not None:
            return self.launcher
        return {
            "local": "local",
            "slurm": "srun",
            "pjm": "pjm",
            "pbs": "mpirun",
            "lsf": "mpirun",
        }[self.scheduler]


def _resolve_binary(raw: object) -> Path:
    """Resolve `run.binary` to an absolute Path.

    Priority:
      1. $SEMSWS_BIN env var (always wins; handy for self-compiled trees)
      2. absolute path → used as-is (expanding `~`),
      3. relative path with separators → resolved against CWD,
      4. bare command name (e.g. `semsws`) → resolved via `$PATH`
         (handy with `spack load semsws` / module systems).

    Bare names that are not on PATH raise FileNotFoundError so config
    typos surface immediately instead of leaking into ToolPaths discovery.
    """
    env = os.environ.get("SEMSWS_BIN")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"$SEMSWS_BIN={env} is not a file")
        return p

    s = str(raw)
    if not s:
        raise ValueError("run.binary must be a non-empty string")
    p = Path(s).expanduser()
    has_sep = ("/" in s) or ("\\" in s)
    if has_sep:
        return p if p.is_absolute() else p.resolve()
    found = shutil.which(s)
    if found is None:
        raise FileNotFoundError(
            f"run.binary={s!r} is a bare command not found on $PATH; "
            f"either load it (e.g. `spack load semsws`), set $SEMSWS_BIN, "
            f"or write an absolute / relative path in the config"
        )
    return Path(found)
