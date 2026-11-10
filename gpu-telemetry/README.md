# gpu-telemetry

A small GPU telemetry harness for the compute-cost-of-TTS work. It instruments
the **dots.tts** model and records per-run GPU statistics (hardware/runtime
fingerprint, timing, memory, utilization) so that runs can be compared and
reproduced.

This is the `gpu-telemetry/` subteam folder of the monorepo. The Python package
lives in `gpu-telemetry/telemetry/`; run everything from `gpu-telemetry/` so the
package imports resolve and `outputs/` lands here.

There are two drivers that share the same measurement modules and run-record
format:

- **`run_tts_eval`**: real dots.tts (`dots-studio/dots.tts-soar`) over a set
  of benchmark prompts, one measured generation per prompt.
- **`run_sweep`**: a toy flow-matching model (`toy_model.py`) swept over NFE
  values. It needs no model download, so it is used to develop and check the
  harness itself.

## Status

- Done: fingerprint, timing, peak memory, NVML sampler, profiler, run-records,
  and the NFE sweep on the toy model.
- Done: `dots_model.py` wrapper and the `run_tts_eval` driver for real dots.tts.
- Next: first `run_tts_eval` run on the GPU box with the benchmark prompts file.

## Setup

All pins are in `requirements.txt`: `torch==2.11.0`, `torchaudio==2.11.0`,
`nvidia-ml-py`, `numpy`, `dots.tts==0.3.1`.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
```

Gotchas that will bite you if skipped:

- **`torch` and `torchaudio` minor versions MUST match** (`2.11.x` with
  `2.11.x`). A mismatched pair loads but fails at runtime inside torchaudio's C++
  ops. If you bump one, bump the other to the same minor.
- **Install from the `cu130` wheel index**
  (`--index-url https://download.pytorch.org/whl/cu130`). The default PyPI wheels
  are built against a different CUDA and will not match the cluster's driver.
  The `--index-url` applies to the whole `requirements.txt` install.
- **Install from the cluster login node, not a compute node.** Compute nodes
  have flaky outbound internet and the large torch/torchaudio downloads will
  stall or fail there. Build the venv on the login node, then use it from the
  compute node.

`nvidia-ml-py` provides the `pynvml` module. Both it and a GPU are optional:
on a CPU-only machine every module still runs and fills GPU fields with `"n/a"`.

`run_tts_eval` also needs the `dots_tts` package (it must provide
`dots_tts.runtime.DotsTtsRuntime`) and a CUDA GPU that can hold the 2B model in
bf16. Without `dots_tts` the rest of the harness still imports and runs;
only the dots.tts entry points fail, with a clear message.

## Running the dots.tts eval

On a machine with dots.tts installed, from `gpu-telemetry/`:

```
python -m telemetry.run_tts_eval --prompts prompts.json --num-steps 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--prompts` | required | JSON file of benchmark prompts (format below) |
| `--num-steps` | `10` | ODE steps per generation; recorded as `nfe` |
| `--model` | `dots-studio/dots.tts-soar` | dots.tts checkpoint |
| `--out-dir` | `outputs/` | where run-records and the CSV are written |

The prompts file is a JSON **list**; every entry needs these four keys:

```json
[
  {"id": "short_001", "text": "Hello there.", "length_bucket": "short", "category": "greeting"}
]
```

What the eval does:

1. Loads the dots.tts runtime once (`get_runtime()` caches it for the process).
2. Runs one warmup generation on the first prompt and discards it, so one-time
   costs (cuBLAS/cuDNN setup, kernel loading, allocator growth, lazy init)
   don't land on the first measured prompt.
3. For each prompt: reset peak memory, start the GPU sampler, time **one**
   `run_dots()` call with the CUDA-event timer, stop the sampler, read peak memory.
4. Saves one run-record per prompt and appends a row to the summary CSV,
   flushed after every prompt so a crash keeps finished rows.

Each prompt gets one generation rather than repeats plus a median, as in the
sweep: a real generation lasts seconds, which already outweighs timer noise and
gives the sampler many readings. With a fixed seed (42) a repeat would just
redo the same work. The variation this eval is about is between prompts.

Peak VRAM includes the model weights, which are the same for every prompt, so
differences between rows are per-prompt activation memory.

To smoke-test the wrapper on its own (one short sentence at 10 steps):

```
python -m telemetry.dots_model
```

## Running a sweep (toy model)

Run from `gpu-telemetry/` (the `-m` form is required so package imports resolve):

```
python -m telemetry.run_sweep
```

For each NFE value this times the model, measures peak GPU memory and samples
GPU utilization/power/clock/temperature, then writes one run-record per NFE
plus a CSV summary to `outputs/`.

| Flag | Default | Meaning |
|---|---|---|
| `--nfe N [N ...]` | `4 8 16 32 64` | NFE values to sweep |
| `--warmups` | `3` | untimed runs before measuring each NFE |
| `--repeats` | `10` | minimum timed runs per NFE |
| `--min-duration-s` | `1.0` | keep repeating until the timed section lasts at least this long; `0` disables |
| `--sample-interval-ms` | `50.0` | GPU sampling period |
| `--profile` | off | also run one `torch.profiler` pass per NFE (kernel launches, op counts) |

A timed section ends once **both** `--repeats` runs are done **and**
`--min-duration-s` has passed, so fast configurations run more times than slow
ones. The minimum duration exists so the GPU sampler sees enough of the run to
be meaningful (NVML's own utilization window is up to 1 s).

`--profile` is off by default because the profiler instruments every op and
kernel launch, which slows the run and inflates exactly the launch overhead it
counts. Use it as an occasional characterization pass (e.g. once per model or
hardware change), not on every sweep. Its numbers come from a separate run
after the measured ones, so it never affects timing, memory or GPU stats.

Each module also has a small demo: `python -m telemetry.<module>`.

## Modules

All live in `telemetry/`.

| Module | What it does |
|---|---|
| `fingerprint.py` | `get_fingerprint()`: Python/torch/CUDA versions, GPU name, total memory, driver version |
| `record.py` | `RunRecord`: one run's config + fingerprint + metrics, saved atomically as JSON |
| `timing.py` | `time_callable()`: warm-ups, then timed repeats (CUDA events on GPU, `perf_counter` on CPU) |
| `memory.py` | `reset()` / `read_peak()`: peak allocated and reserved CUDA memory |
| `nvml_sampler.py` | `GpuSampler`: background thread polling NVML; summarizes mean/peak per metric |
| `profiler.py` | `profile_callable()`: one `torch.profiler` run; kernel-launch and op counts, top ops by time |
| `dots_model.py` | `get_runtime()` (load-once dots.tts runtime), `run_dots()` (seeded generation), `audio_duration_s()` |
| `run_tts_eval.py` | the dots.tts eval: one measured generation per benchmark prompt |
| `run_sweep.py` | the toy pipeline: runs the measurements for each NFE and writes the outputs |
| `toy_model.py` | stand-in model: an Euler flow-matching loop whose cost scales with NFE |

What each measurement covers, per NFE in the sweep (the eval has no repeats,
so all of its measurements cover the one generation):

- **memory**: warm-ups and timed repeats. One-time allocations are real memory the run needs.
- **timing** and **gpu**: timed repeats only. Warm-ups (clock ramp-up, one-time init) would skew means.
- **profile** (with `--profile`): one extra run after all of the above, under the profiler.

## Output files

All go to `outputs/` under `gpu-telemetry/` (git-ignored) unless `--out-dir` is given.

- `run_<run_id>.json`: one run-record per sweep NFE or per eval prompt (schema below)
- `sweep_<sweep_id>.csv`: one row per NFE, flattened from the run-records
- `tts_eval_<eval_id>.csv`: one row per prompt (columns below)

## Run-record schema (`schema_version: 1`)

Example (from a CPU-only machine, so GPU fields are `"n/a"`):

```json
{
  "run_id": "20261018_220406_3bd9",
  "created_at": "2026-10-18T22:04:06-05:00",
  "schema_version": 1,
  "config": {
    "model": "toy",
    "nfe": 4,
    "sweep_id": "20261018_220406"
  },
  "fingerprint": {
    "python_version": "3.14.3",
    "torch_version": "2.14.1+cpu",
    "torch_cuda_version": "n/a",
    "device": "cpu",
    "gpu_name": "n/a",
    "gpu_total_memory_mb": "n/a",
    "driver_version": "n/a"
  },
  "metrics": {
    "timing": {
      "steps": 4,
      "median_ms": 47.81229980289936,
      "min_ms": 44.33709988370538,
      "max_ms": 50.7171000353992,
      "per_step_ms": 11.95307495072484,
      "device": "cpu",
      "timer": "perf_counter",
      "warmups": 3,
      "repeats": 3,
      "min_repeats": 3,
      "min_duration_s": 0.0,
      "measured_s": 0.14291389985010028
    },
    "memory": {
      "peak_allocated_mb": "n/a",
      "peak_reserved_mb": "n/a"
    },
    "gpu": {
      "util_mean_pct": "n/a",
      "util_peak_pct": "n/a",
      "power_mean_w": "n/a",
      "power_peak_w": "n/a",
      "sm_clock_mean_mhz": "n/a",
      "sm_clock_peak_mhz": "n/a",
      "temp_mean_c": "n/a",
      "temp_peak_c": "n/a",
      "num_samples": 0,
      "interval_ms": 50.0
    }
  }
}
```

### Conventions

- **Missing values are the string `"n/a"`**, never `null` or `0`. This happens
  when there is no GPU, or the GPU doesn't report a metric (e.g. power on some
  laptop GPUs). pandas `read_csv` treats `"n/a"` as NaN by default.
- **Units are in the key name**: `_ms`, `_s`, `_mb` (MiB, 2^20 bytes), `_pct`, `_w`, `_mhz`, `_c`.
- **Times are fixed US Eastern Standard Time (UTC-05:00)**, no daylight
  saving, so timestamps never jump or repeat.
- Stored values are full precision; round only for display.

### Top level

| Field | Type | Meaning |
|---|---|---|
| `run_id` | str | `YYYYMMDD_HHMMSS_xxxx` (creation time + 4 random hex chars); also the file name |
| `created_at` | str | ISO 8601 with offset, e.g. `2026-10-18T22:04:06-05:00` |
| `schema_version` | int | version of this format |
| `config` | object | what was run |
| `fingerprint` | object | where it ran |
| `metrics` | object | what was measured: `timing`, `memory`, `gpu`; `profile` only when the sweep ran with `--profile`; `audio` only in eval records |

### `config`

Sweep records:

| Field | Type | Meaning |
|---|---|---|
| `model` | str | `"toy"` |
| `nfe` | int | number of function evaluations (sampler steps) |
| `sweep_id` | str | `YYYYMMDD_HHMMSS` of the sweep this run belongs to; shared by all its runs |

Eval records:

| Field | Type | Meaning |
|---|---|---|
| `model` | str | dots.tts checkpoint, e.g. `dots-studio/dots.tts-soar` |
| `nfe` | int | `--num-steps` passed to dots.tts |
| `prompt_id`, `text`, `length_bucket`, `category` | str | copied from the prompts file |
| `eval_id` | str | `YYYYMMDD_HHMMSS` of the eval this run belongs to; shared by all its runs |

### `fingerprint`

| Field | Type | Meaning |
|---|---|---|
| `python_version` | str | e.g. `3.14.3` |
| `torch_version` | str | e.g. `2.14.1+cu130` |
| `torch_cuda_version` | str | CUDA version torch was *built* against |
| `device` | str | `cuda:<index>` or `cpu` |
| `gpu_name` | str | e.g. `NVIDIA GeForce RTX 4090` |
| `gpu_total_memory_mb` | int | total device memory |
| `driver_version` | str | NVIDIA driver version, read via NVML |

### `metrics.timing`

| Field | Type | Meaning |
|---|---|---|
| `steps` | int | steps per run (equals `config.nfe`) |
| `median_ms` | float | median wall time of **one** run |
| `min_ms`, `max_ms` | float | fastest and slowest single run |
| `per_step_ms` | float | `median_ms / steps` |
| `device` | str | `cuda` or `cpu` |
| `timer` | str | `cuda_event` (GPU) or `perf_counter` (CPU) |
| `warmups` | int | untimed runs before measuring |
| `repeats` | int | timed runs that **actually ran**; the sample size behind the stats above |
| `min_repeats` | int | requested minimum (`--repeats`) |
| `min_duration_s` | float | requested minimum timed duration |
| `measured_s` | float | wall time of the whole timed section, including gaps between runs |

`repeats` varies between NFEs when `min_duration_s` is set. That does not
change what `median_ms` means: every sample is one complete run of the same
work, so the median is time per run regardless of how many samples there are;
more repeats only make it more precise. `measured_s` is not
`median_ms * repeats`. It includes per-run overhead and is the window the GPU
sampler observed.

Eval records always have `warmups: 0` and `repeats: 1` (the one discarded
warmup happens once, before the first prompt), and add two fields:

| Field | Type | Meaning |
|---|---|---|
| `latency_s` | float | the measured generation time (`median_ms / 1000`) |
| `model_time_used_s` | float | dots.tts's own `time_used` for the same call; may cover a narrower span than `latency_s` |

### `metrics.memory`

| Field | Type | Meaning |
|---|---|---|
| `peak_allocated_mb` | float | high-water mark of memory held by live tensors; what the model needs |
| `peak_reserved_mb` | float | high-water mark of memory PyTorch's caching allocator held from the driver (≥ allocated; closer to `nvidia-smi` and what determines OOM) |

Peak reserved starts from whatever the allocator had cached before the run, so
it can carry over between NFEs in one sweep. Peak allocated does not.

### `metrics.gpu`

Summary of NVML samples taken every `interval_ms` during the timed repeats.

| Field | Type | Meaning |
|---|---|---|
| `util_mean_pct`, `util_peak_pct` | float, int | % of time at least one kernel was running (NVML definition; not SM occupancy) |
| `power_mean_w`, `power_peak_w` | float | board power draw |
| `sm_clock_mean_mhz`, `sm_clock_peak_mhz` | float, int | SM clock |
| `temp_mean_c`, `temp_peak_c` | float, int | GPU core temperature |
| `num_samples` | int | samples taken; `0` with no GPU. Check this before trusting the means |
| `interval_ms` | float | sampling period |

### `metrics.audio` (eval records only)

| Field | Type | Meaning |
|---|---|---|
| `audio_s` | float | length of the generated audio: samples / `sample_rate` |
| `rtf` | float | real-time factor as reported by dots.tts |
| `sample_rate` | int | Hz |

### `metrics.profile` (only with `--profile`)

One run of the model under `torch.profiler`. Counts are deterministic for a
given NFE, so one run is enough. Its times include profiler overhead; use
`metrics.timing` for timing.

| Field | Type | Meaning |
|---|---|---|
| `steps` | int | steps per run (equals `config.nfe`) |
| `device` | str | `cuda` or `cpu` |
| `warmups` | int | unprofiled runs first (`0` in sweeps: the model is already warm) |
| `cpu_op_calls` | int | top-level `aten::` ops the Python code called (nested ops like `aten::matmul` → `aten::mm` count once) |
| `distinct_cpu_ops` | int | distinct names among those |
| `kernel_launches` | int | GPU kernels launched (excludes memcpy/memset) |
| `distinct_kernels` | int | distinct kernel names |
| `gpu_kernel_time_ms` | float | summed GPU execution time of those kernels |
| `top_ops` | list | up to 5 `aten::` ops ranked by self time (GPU time on CUDA, CPU time otherwise); each `{name, calls, self_cpu_time_ms, self_gpu_time_ms}` |

The three kernel fields and `self_gpu_time_ms` are `"n/a"` on CPU, and also on
a GPU if CUDA tracing is unavailable (no kernels recorded), rather than a
misleading `0`. For the toy model, `cpu_op_calls` is `4 × NFE + 7`, and
`kernel_launches` should be about 4 per step.

`self_` times are each op's own cost, excluding ops it called, so they never
count the same work twice. `calls` counts every invocation, nested ones
included.

## Sweep CSV columns

One row per NFE, in this order:

`nfe, median_ms, per_step_ms, min_ms, max_ms, repeats, measured_s,
peak_allocated_mb, peak_reserved_mb, util_mean_pct, util_peak_pct,
power_mean_w, power_peak_w, sm_clock_mean_mhz, sm_clock_peak_mhz, temp_mean_c,
temp_peak_c, gpu_num_samples, run_id`

With `--profile`, four more columns come just before `run_id`:
`kernel_launches, distinct_kernels, gpu_kernel_time_ms, cpu_op_calls`
(`top_ops` is a list, so it is only in the JSON).

Each column has the same meaning as the run-record field of the same name;
`gpu_num_samples` is `metrics.gpu.num_samples`. Use `run_id` to find the full
JSON record for a row.

## Eval CSV columns

One row per prompt, in this order:

`run_id, prompt_id, length_bucket, nfe, latency_s, audio_s, rtf, peak_vram_mb`

`latency_s` is `metrics.timing.latency_s`, `audio_s` and `rtf` come from
`metrics.audio`, and `peak_vram_mb` is `metrics.memory.peak_allocated_mb`.
