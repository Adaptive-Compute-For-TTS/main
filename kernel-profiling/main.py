from torch.profiler import profile, ProfilerActivity, record_function
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils import profiling
import soundfile as sf
import contextlib
import torch

runtime = DotsTtsRuntime.from_pretrained(
    "dots-studio/dots.tts-mf", precision="bfloat16", optimize=True
)


_orig = profiling.InferenceProfiler.measure


@contextlib.contextmanager
def _measure(self, stage, **kw):
    with record_function(f"dots/{stage}/{kw.get('phase')}"), _orig(self, stage, **kw):
        yield


# monkey patches
profiling.InferenceProfiler.measure = _measure
profiling.InferenceProfiler._sync = lambda self: None

kwargs = dict(text="Hi there", num_steps=4)

# testing the generate_stream mode

print("warmup")
for _ in range(3):
    for _chunk in runtime.generate_stream(**kwargs):
        pass
    torch.cuda.synchronize()

print("profiling)")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("dots/generate_stream_total"):
        for _chunk in runtime.generate_stream(**kwargs, profile_inference=True):
            pass
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

prof.export_chrome_trace("trace_stream.json")

# testing the generate() mode
# pytorch doesnt compile the decoder stage for some reason! (only on the batch .generate() mode, the .generate_stream() mode is though)
print("compiling vocoder decoder")
runtime.model.vocoder.decoder = torch.compile(
    runtime.model.vocoder.decoder, dynamic=True
)

print("warmup")
for _ in range(3):
    runtime.generate(**kwargs)
    torch.cuda.synchronize()

print("profiling")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("dots/generate_total"):
        result = runtime.generate(**kwargs, profile_inference=True)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

prof.export_chrome_trace("trace_generate.json")
