"""Wrapper that lets the harness call real dots.tts the way it called the toy model."""

import time

import torch

try:
    from dots_tts.runtime import DotsTtsRuntime
except ImportError as exc:  # e.g. on a laptop without the package installed
    DotsTtsRuntime = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_MODEL = "dots-studio/dots.tts-soar"
DEFAULT_PRECISION = "bfloat16"
SMOKE_TEXT = "The quick brown fox jumps over the lazy dog."

_runtime = None
_runtime_key = None  # (model, precision) the cached runtime was built with


def get_runtime(model=DEFAULT_MODEL, precision=DEFAULT_PRECISION):
    """Build the DotsTtsRuntime on first call, then return the cached one."""
    global _runtime, _runtime_key

    if DotsTtsRuntime is None:
        raise RuntimeError(
            "dots_tts is not importable; install it to use telemetry.dots_model"
        ) from _IMPORT_ERROR

    key = (model, precision)
    if _runtime is None:
        _runtime = DotsTtsRuntime.from_pretrained(
            model, revision=None, precision=precision, optimize=False
        )
        _runtime_key = key
    elif key != _runtime_key:
        raise ValueError(
            f"runtime already loaded as {_runtime_key}; refusing to return it for {key}"
        )
    return _runtime


def run_dots(text, num_steps, seed=42, guidance_scale=1.2, speaker_scale=1.5,
             model=DEFAULT_MODEL):
    """Generate speech for `text` with `num_steps` ODE steps; return dots.tts's result dict."""
    rt = get_runtime(model)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ode, steps, cfg = rt.resolve_sampling_options(
        ode_method=None, num_steps=num_steps, guidance_scale=guidance_scale
    )
    return rt.generate(
        text=text,
        prompt_audio_path=None,
        prompt_text=None,
        ode_method=ode,
        num_steps=steps,
        guidance_scale=cfg,
        speaker_scale=speaker_scale,
    )


def audio_duration_s(r):
    """Seconds of audio in a run_dots result."""
    return r["audio"].shape[-1] / r["sample_rate"]


if __name__ == "__main__":
    if DotsTtsRuntime is None:
        raise SystemExit(f"dots_tts not installed ({_IMPORT_ERROR}); run this on the GPU box")

    t0 = time.perf_counter()
    get_runtime()
    print(f"load: {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    r = run_dots(SMOKE_TEXT, num_steps=10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"elapsed: {elapsed:.2f} s  (runtime reports time_used={r['time_used']})")
    print(f"audio:   {audio_duration_s(r):.2f} s at {r['sample_rate']} Hz")
    print(f"rtf:     {r['rtf']}")
