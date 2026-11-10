from ttsbench.models.base import Requirements, SynthesisOutput, SynthesisRequest, TTSModel
from ttsbench.registry import Registry

MODELS: Registry[TTSModel] = Registry("model")
MODELS.register("sine", "ttsbench.models.sine:SineModel")
MODELS.register("dots_tts", "ttsbench.models.dots_tts:DotsTTS")
MODELS.register("dots_tts_executor", "ttsbench.models.dots_tts_executor:DotsTTSExecutor")

__all__ = ["MODELS", "Requirements", "SynthesisOutput", "SynthesisRequest", "TTSModel"]
