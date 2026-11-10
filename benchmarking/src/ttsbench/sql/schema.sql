-- ttsbench run index. Artifacts (audio, raw jsonl) live in object storage; this is the queryable index.
-- Idempotent: safe to run repeatedly.

create table if not exists runs (
    run_id            text primary key,
    name              text not null,
    status            text not null,
    model             text not null,
    backend           text not null,
    gpu               text,
    nfe               integer[] not null,
    config_hash       text not null,
    prompts_hash      text not null,
    protocol_version  integer not null,
    git_sha           text,
    git_dirty         boolean,
    started_at        timestamptz,
    finished_at       timestamptz,
    artifact_uri      text not null,
    manifest          jsonb not null,
    published_at      timestamptz not null default now()
);
create index if not exists runs_name_idx on runs (name);
create index if not exists runs_config_hash_idx on runs (config_hash);

create table if not exists run_summaries (
    run_id                  text not null references runs on delete cascade,
    nfe                     integer not null,
    length                  text not null,
    n                       integer not null,
    latency_p50_ms          double precision not null,
    latency_p90_ms          double precision not null,
    latency_p95_ms          double precision not null,
    latency_mean_ms         double precision not null,
    latency_std_ms          double precision not null,
    within_prompt_cv        double precision not null,
    device_p50_ms           double precision,
    rtf_p50                 double precision not null,
    audio_seconds_mean      double precision not null,
    deterministic_fraction  double precision,
    noisy                   boolean not null,
    stats                   jsonb not null,
    primary key (run_id, nfe, length)
);

create table if not exists samples (
    run_id         text not null references runs on delete cascade,
    prompt_id      text not null,
    nfe            integer not null,
    repeat         integer not null,
    seed           integer not null,
    latency_ms     double precision not null,
    device_ms      double precision,
    to_host_ms     double precision not null,
    audio_seconds  double precision not null,
    rtf            double precision not null,
    audio_sha256   text not null,
    audio_uri      text,
    stages_ms      jsonb not null,
    telemetry      jsonb not null,
    primary key (run_id, prompt_id, nfe, repeat)
);
