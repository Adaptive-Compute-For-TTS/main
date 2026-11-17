from ttsbench.backends.base import Backend
from ttsbench.registry import Registry

BACKENDS: Registry[Backend] = Registry("backend")
BACKENDS.register("local", "ttsbench.backends.local:LocalBackend")
BACKENDS.register("modal", "ttsbench.backends.modal_backend:ModalBackend")
BACKENDS.register("slurm", "ttsbench.backends.slurm:SlurmBackend")
BACKENDS.register("watcloud", "ttsbench.backends.slurm:watcloud")

__all__ = ["BACKENDS", "Backend"]
