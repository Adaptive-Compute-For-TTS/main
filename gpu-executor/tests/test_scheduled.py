"""generate_scheduled: per-patch NFE from any scheduler, executor on every patch, hooks restored.

Same fake runtime as test_fixed_nfe.py (drives the dots.tts hooks once per patch). The real-model run is
scripts/mixed_nfe.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from adaptive_nfe import ListScheduler, generate_scheduled  # noqa: E402
from test_fixed_nfe import FakeExecutor, FakeRuntime  # noqa: E402


def test_each_patch_runs_at_its_own_nfe():
    rt, ex = FakeRuntime(num_patches=6), FakeExecutor()
    r = generate_scheduled(rt, ListScheduler([32, 8, 8, 16]), executor=ex, text="hi")
    assert r.nfe_per_patch == ex.calls == [32, 8, 8, 16, 16, 16]
    assert r.nfe_switches == 2 and r.total_dit_steps == 96
    assert rt.generate_kwargs == {"text": "hi", "num_steps": 32}


def test_scheduler_is_reset_between_calls():
    s = ListScheduler([4, 8])
    generate_scheduled(FakeRuntime(num_patches=3), s, executor=FakeExecutor())
    r = generate_scheduled(FakeRuntime(num_patches=2), s, executor=FakeExecutor())
    assert r.nfe_per_patch == [4, 8]


def test_times_patches_when_asked():
    r = generate_scheduled(FakeRuntime(num_patches=4), ListScheduler([4]), executor=FakeExecutor(), time_patches=True)
    assert len(r.patch_ms) == 4 and all(ms >= 0 for ms in r.patch_ms)


def test_restores_hooks_and_checks_executor():
    rt = FakeRuntime(fail=True)
    rt.model.nfe_scheduler, rt.model.dit_executor = "a", "b"
    with pytest.raises(RuntimeError, match="boom"):
        generate_scheduled(rt, ListScheduler([4]), executor=FakeExecutor())
    assert (rt.model.nfe_scheduler, rt.model.dit_executor) == ("a", "b")
    with pytest.raises(RuntimeError, match="Executor decoded"):
        generate_scheduled(FakeRuntime(num_patches=3, skip_executor_on=(1,)), ListScheduler([4]),
                           executor=FakeExecutor())


@pytest.mark.parametrize("bad", [[], [0], [8, -1]])
def test_list_scheduler_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ListScheduler(bad)
