from typing import Any, Callable, Iterator, Protocol

import torch


class TTSEngine(Protocol):
    """A swappable TTS backend.

    generate/generate_stream are the clean paths the bencher times.
    profile/profile_stream run the same work with whatever instrumentation the
    engine needs (monkey patches, record_function ranges, ...) and must undo it
    on exit, so profiling never leaks into benchmark numbers.
    """

    name: str
    sample_rate: int

    def generate(self, **kwargs: Any) -> torch.Tensor: ...

    def generate_stream(self, **kwargs: Any) -> Iterator[torch.Tensor]: ...

    def profile(self, **kwargs: Any) -> torch.Tensor: ...

    def profile_stream(self, **kwargs: Any) -> Iterator[torch.Tensor]: ...


def _torch_compiled(**kwargs: Any) -> Callable[[], TTSEngine]:
    def load() -> TTSEngine:
        from pytorch_compiled_tts import TorchCompiledTTS

        return TorchCompiledTTS(**kwargs)

    return load


# imports are lazy so listing engines doesn't load every model
ENGINES: dict[str, Callable[[], TTSEngine]] = {
    "torch-compiled": _torch_compiled(),
    "torch-compiled-vocoder-bf16": _torch_compiled(vocoder_dtype=torch.bfloat16),
}


def load_engine(name: str) -> TTSEngine:
    return ENGINES[name]()
