# benchmarking (ttsbench)

Benchmarking and evaluation harness for dots.tts adaptive-compute work. Every experiment runs the
same prompts, measures latency the same way, and stores the config, environment, raw samples and
audio together, so any two runs can be compared directly.

Owner: Afreed. Moved here from [AfreedHassan/adaptive-compute](https://github.com/AfreedHassan/adaptive-compute)
at `7eb72ec`; history stays in that repo.

## Where it sits

Not part of the request path (React -> FastAPI -> GPU worker). It is the measurement side of the
**Fixed-NFE Euler Executor** box: it answers "how fast is this path at NFE N, on this GPU, and did
the audio change?" with numbers that hold up across machines (Modal H100, WATcloud SLURM), instead
of the rough laptop timings in `gpu-executor/README.md`.

- `gpu-executor/` decides *what* runs; `benchmarking/` measures it, through the `dots_tts_executor`
  model (stock, scheduler or executor mode, eager or optimized).
- `gpu-telemetry/` can plug in as a probe (`measure/probes.py`): per-sample GPU stats land in `Sample.telemetry`.
- `quality-evals/` scorers can run on a run's audio through `ttsbench manifest` + `ttsbench eval`.

```bash
uv sync --extra modal            # add --extra publish for push
uv run ttsbench run configs/smoke-sine.yaml            # CPU stand-in model, a few seconds
uv run ttsbench run configs/dots-example.yaml          # dots.tts on a Modal H100
uv run ttsbench run configs/dots-watcloud.yaml         # dots.tts on WATcloud (SLURM)
uv run ttsbench run configs/dots-example.yaml --set sweep.nfe=[8,16] --set backend.params.gpu=A100-80GB
uv run ttsbench ls
uv run ttsbench compare runs/<baseline> runs/<candidate>   # speedup per NFE + audio equality
```

## Layout

| Path | What it is |
| --- | --- |
| `src/ttsbench/schema.py` | Data contracts: config, `Job`, `Sample`, `JobResult` |
| `src/ttsbench/worker.py` | The measured loop. Runs inside the execution environment |
| `src/ttsbench/measure/` | Timing, environment capture, telemetry probes |
| `src/ttsbench/models/` | Model adapters (`dots_tts`, `sine` stand-in) |
| `src/ttsbench/backends/` | Where jobs run (`local`, `modal`, `slurm` / `watcloud`) |
| `src/ttsbench/storage.py` | Run directory format |
| `src/ttsbench/publish.py`, `sql/schema.sql` | Upload runs to S3 + index them in Postgres |
| `benchmarks/prompts/` | Versioned prompt sets (JSONL). `v0` is a placeholder until the canonical set lands |
| `configs/` | Experiment configs |

## Plugging things in

Each extension point is a registry keyed by the name used in configs.

- **Backend** (where it runs): subclass `backends.base.Backend` and implement `execute(job) -> JobResult`
  by getting the job to `ttsbench.worker.execute_job` and returning the result. Provision in
  `__enter__` and release in `__exit__`. Register it in `backends/__init__.py`. Backends never time
  anything, so a slow transport can't skew results.
- **Model** (what runs): subclass `models.base.TTSModel` (`load`, `synthesize`, `requirements`). Use
  `timer.stage("name")` to attribute time to stages. Stage timers use CUDA events and never force a sync.
- **Telemetry probe** (per-sample GPU stats): subclass `measure.probes.Probe`, register it, and list
  it under `probes:`. Its output is stored in `Sample.telemetry`.
- **Quality metrics** (WER, speaker similarity): not part of this harness. They read a run's
  `audio/` and `samples.jsonl` and can write `<run>/metrics/<name>.jsonl` next to them.

## WATcloud (SLURM)

The `watcloud` backend is the generic `slurm` backend with WATcloud defaults (`host: wato-login1`,
`partition: compute`). It works over SSH, so `ssh wato-login1` has to work non-interactively first
(set up your key and the bastion `ProxyJump` in `~/.ssh/config`, as described in the WATcloud SSH docs).

Each run does the following:
1. Uploads `src/ttsbench` and `benchmarks/` to `~/ttsbench` on the cluster.
2. Submits one `sbatch` job per NFE value.
3. Polls `sacct` until the job finishes, then reads `result.json` back.
4. Cancels any in-flight job if the client crashes or is interrupted.

- **Python environment:** built inside the first job on the GPU node with `uv`, from the model's
  pinned requirements. It is cached under `~/ttsbench/envs/<hash>`. Use `python_env: existing` with
  `python: /path/to/python` to use your own environment instead.
- **Whole GPU by default.** `gres: gpu:1` requests a whole GPU, which keeps latency clean. Shared GPUs
  (`shard:<MiB>`) work, but the run records `exclusive_gpu: false`, and their latency is not comparable
  across runs.
- **One node for the whole run.** After the first job, later jobs are pinned to the same node (`--nodelist`).
  Each job reloads the model, so per-NFE warmup still applies.
- **Checkpoint cache:** set `hf_home` to a directory with room for the ~9 GB checkpoint if your home quota is small.

## How latency is measured

These rules are versioned as `MEASUREMENT_PROTOCOL_VERSION` (`schema.py`). Change the version
whenever a rule changes. `compare` warns when two runs used different versions.

1. Timing happens in the worker process, next to the GPU. Network and RPC time is never included.
2. The model is loaded once per container (load time is recorded separately as `model_load_ms`).
   Each NFE value then gets `warmup` discarded runs before measuring, so compile, autotune and
   CUDA-graph cost for that shape stays out of the numbers.
3. `latency_ms` is host wall time (`perf_counter_ns`), with `cuda.synchronize()` immediately before
   the clock starts and immediately after `synthesize` returns. `device_ms` brackets the same region
   with CUDA events.
4. Copying the output audio to host memory is timed separately (`to_host_ms`) and is not part of `latency_ms`.
5. GC is collected, then disabled, during each timed region. Seeds are set outside it.
6. RNG is seeded per sample. With `vary_seed: false`, every repeat must produce bit-identical
   audio. `deterministic_fraction` in the summary reports how many cells did.
7. All NFE values in a run go to the same hardware: one Modal container (`max_containers=1`), or one SLURM node (`--nodelist`).
8. Summaries report p50/p90/p95/p99, mean, std and CV. Groups with CV > 5% are flagged `noisy`.
   Use p50 for comparisons.

## Run directory

```
runs/<UTC time>_<name>_<config hash[:8]>_<rand>/
  manifest.json    resolved config, config_hash, prompts_hash, git sha/dirty, client, backend,
                   protocol version, status (running|completed|failed)
  prompts.jsonl    exact prompts used
  workers.jsonl    per-NFE worker environment (GPU, driver, torch/CUDA versions), model load and warmup times
  samples.jsonl    one row per measured synthesis (latency, device time, stages, RTF, audio hash, telemetry)
  audio/<prompt>/nfeNNN_rNN.wav   float32 WAV
  summary.json / summary.csv
```

Rows are appended after each NFE value finishes. If a run crashes, everything up to that point is kept.

## Evals on Modal

`ttsbench eval` runs a self-contained eval bundle (a folder with `requirements.txt` and an entrypoint) in a
Modal container. It runs the command from the bundle root and copies its output folder back, along with an
`eval_run.json` record (command, exit code, wall time, container environment). `{out}` in the command is the
remote output folder. `--data` mounts an extra folder at `/data`. To score a run's audio, write its manifest first:

```bash
uv run ttsbench manifest runs/<run>        # -> runs/<run>/eval_manifest.csv (clip_id, audio_path, reference_text, ...)
uv run ttsbench eval path/to/wer_cer_sample_eval --data runs/<run> --out evals/<run> -- \
    python sample_eval/run_eval.py --manifest /data/eval_manifest.csv --out {out} --run-id <run>
```

## Publishing (S3 + Postgres)

`runs/` is gitignored. Git tracks code, configs and prompt sets. Run artifacts go to object storage,
and Postgres holds a queryable index (`runs`, `run_summaries`, `samples`) for the website and cross-run queries.

```bash
cp .env.example .env   # fill in TTSBENCH_ARTIFACTS_URI and TTSBENCH_DATABASE_URL
uv run ttsbench push runs/<run_id>
uv run ttsbench run configs/dots-example.yaml --push
```

Publishing uploads the run directory first, then upserts the index in one transaction. Re-pushing is
safe. Only `completed` runs are published unless `--allow-incomplete` is passed. The schema is applied
on every push (`create ... if not exists`). Set `TTSBENCH_TEST_DATABASE_URL` to run the DB integration test.
Neither a bucket nor a database is provisioned yet.
