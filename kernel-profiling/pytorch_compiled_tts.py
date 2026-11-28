import contextlib
from typing import Any, Iterator

import torch
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils import profiling
from torch.profiler import record_function


@contextlib.contextmanager
def _dots_stage_ranges():
    """Turn dots' internal stage timers into dots/<stage>/<phase> ranges.

    Also disables their per-stage cuda sync, which would otherwise serialize the
    trace. Restored on exit.
    """
    orig_measure = profiling.InferenceProfiler.measure
    orig_sync = profiling.InferenceProfiler._sync

    @contextlib.contextmanager
    def measure(self, stage, **kw):
        with (
            record_function(f"dots/{stage}/{kw.get('phase')}"),
            orig_measure(self, stage, **kw),
        ):
            yield

    profiling.InferenceProfiler.measure = measure
    profiling.InferenceProfiler._sync = lambda self: None
    try:
        yield
    finally:
        profiling.InferenceProfiler.measure = orig_measure
        profiling.InferenceProfiler._sync = orig_sync


class TorchCompiledTTS:
    """Stock dots.tts runtime with optimize=True (torch.compile + cuda graphs)."""

    def __init__(self, model: str = "dots-studio/dots.tts-mf", vocoder_dtype=None):
        self.name = "torch-compiled"
        self.runtime = DotsTtsRuntime.from_pretrained(
            model, precision="bfloat16", optimize=True, warmup_on_optimize=False
        )
        if vocoder_dtype is not None:
            self.name += f"-vocoder-{str(vocoder_dtype).removeprefix('torch.')}"
            self.runtime.model.vocoder.decoder.register_forward_pre_hook(
                lambda module, args: (args[0].to(dtype=vocoder_dtype), *args[1:])
            )
            # hand fp32 audio back so callers see the same output as the baseline
            self.runtime.model.vocoder.decoder.register_forward_hook(
                lambda module, args, output: output.float()
            )
            self.runtime.model.vocoder.decoder.to(dtype=vocoder_dtype)
        self.runtime.run_warmup()
        self.sample_rate = self.runtime.sample_rate
        self._batch_decoder_compiled = False

    def _compile_batch_decoder(self) -> None:
        # pytorch doesnt compile the decoder stage for some reason! (only on the
        # batch .generate() mode, the .generate_stream() mode is though).
        # done lazily so the stream path keeps its own compiled step untouched.
        # TODO: dynamic=True shows ~570us cpu per cudnn_convolution call; try
        # fixed-length buckets with dynamic=False.
        if self._batch_decoder_compiled:
            return
        vocoder = self.runtime.model.vocoder
        vocoder.decoder = torch.compile(vocoder.decoder, dynamic=True)
        self._batch_decoder_compiled = True

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self._compile_batch_decoder()
        return self.runtime.generate(**kwargs)["audio"]

    def generate_stream(self, **kwargs: Any) -> Iterator[torch.Tensor]:
        yield from self.runtime.generate_stream(**kwargs)

    def profile(self, **kwargs: Any) -> torch.Tensor:
        self._compile_batch_decoder()
        with _dots_stage_ranges(), record_function("dots/generate_total"):
            return self.runtime.generate(**kwargs, profile_inference=True)["audio"]

    def profile_stream(self, **kwargs: Any) -> Iterator[torch.Tensor]:
        with _dots_stage_ranges(), record_function("dots/generate_stream_total"):
            yield from self.runtime.generate_stream(**kwargs, profile_inference=True)
