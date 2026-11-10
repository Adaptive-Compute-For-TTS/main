from __future__ import annotations

from typing import Any

from ttsbench.measure import environment
from ttsbench.measure.environment import worker_environment


class FakeVersion(str):
    """Stands in for torch.torch_version.TorchVersion."""


def _assert_builtin(value: Any) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            assert type(k) is str
            _assert_builtin(v)
    elif isinstance(value, list):
        for v in value:
            _assert_builtin(v)
    else:
        assert type(value) in (str, int, float, bool, type(None)), type(value)


def test_worker_environment_contains_only_builtin_types(monkeypatch) -> None:
    monkeypatch.setattr(environment, "_torch_info", lambda: {"torch": FakeVersion("2.8.0"), "obj": object()})
    env = worker_environment()
    _assert_builtin(env)
    assert env["torch"]["torch"] == "2.8.0"
