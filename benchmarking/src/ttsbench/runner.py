"""Turn an ExperimentConfig into a stored run: one worker job per NFE value."""

from __future__ import annotations

import getpass
import logging
import platform
import re
import secrets
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from ttsbench.backends import BACKENDS
from ttsbench.config import config_hash
from ttsbench.measure.environment import git_state
from ttsbench.measure.probes import PROBES
from ttsbench.models import MODELS
from ttsbench.prompts import prompts_hash, select_prompts
from ttsbench.schema import MEASUREMENT_PROTOCOL_VERSION, ExperimentConfig, Job
from ttsbench.storage import Run, RunWriter

log = logging.getLogger(__name__)


def make_run_id(config: ExperimentConfig, chash: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", config.name.lower()).strip("-") or "run"
    return f"{stamp}_{slug}_{chash[:8]}_{secrets.token_hex(2)}"


def _validate_components(config: ExperimentConfig) -> None:
    """Fail before provisioning anything if a name is wrong."""
    MODELS.resolve(config.model.name)
    BACKENDS.resolve(config.backend.name)
    for probe in config.probes:
        PROBES.resolve(probe)


def run_experiment(config: ExperimentConfig, root: Path = Path(".")) -> Run:
    _validate_components(config)
    prompts = select_prompts(config.prompts, root)
    chash = config_hash(config)
    backend = BACKENDS.create(config.backend.name, model=config.model, **config.backend.params)
    metadata = {
        "measurement_protocol_version": MEASUREMENT_PROTOCOL_VERSION,
        "config_hash": chash,
        "prompts_hash": prompts_hash(prompts),
        "git": git_state(root),
        "client": {
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "python": platform.python_version(),
            "argv": sys.argv,
        },
        "backend": {"name": config.backend.name, **backend.describe()},
    }
    writer = RunWriter.create(make_run_id(config, chash), config, prompts, metadata)
    log.info("run %s: %d prompts x %s NFE", writer.run.path, len(prompts), list(config.sweep.nfe))

    try:
        with backend:
            for nfe in config.sweep.nfe:
                job = Job(
                    model=config.model,
                    nfe=nfe,
                    prompts=tuple(prompts),
                    measurement=config.measurement,
                    probes=config.probes,
                )
                result = backend.execute(job)
                writer.record(result)
                p50 = sorted(s.latency_ms for s in result.samples)[len(result.samples) // 2]
                log.info("nfe=%d: %d samples, median %.1f ms", nfe, len(result.samples), p50)
    except BaseException as err:
        writer.finish("failed", error=repr(err))
        raise
    writer.finish("completed")
    return writer.run
