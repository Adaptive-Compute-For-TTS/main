import type { Run } from "@/lib/run-history";

type RecordValue = Record<string, unknown>;
function record(value: unknown): RecordValue {
  return value && typeof value === "object" && !Array.isArray(value) ? value as RecordValue : {};
}
function number(value: unknown, unit = "", digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toLocaleString(undefined, {maximumFractionDigits: digits})}${unit}` : "Not available";
}
function percentage(value: unknown) { return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Not available"; }

export function MetricGroup({ value, empty }: { value: unknown; empty: string }) {
  const entries = Object.entries(record(value));
  if (!entries.length) return <p className="detail-empty">{empty}</p>;
  return <dl className="detail-metrics">{entries.map(([key, item]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{item === null || item === undefined || item === "n/a" ? "Not available" : typeof item === "boolean" ? item ? "Yes" : "No" : typeof item === "object" ? JSON.stringify(item) : String(item)}</dd></div>)}</dl>;
}
function Messages({ notes, errors }: {notes?: unknown; errors?: unknown}) {
  return <>{Array.isArray(notes) && notes.map((note, i) => <p className="detail-empty diagnostic-note" key={`note-${i}`}>{String(note)}</p>)}
    {Array.isArray(errors) && errors.length > 0 && <ul className="diagnostic-errors" role="status">{errors.map((error, i) => <li key={i}>{String(error)}</li>)}</ul>}</>;
}
const FLAG_LABELS: Record<string, string> = {
  long_gap: "Long internal pause", leading_silence: "Excessive leading silence", abrupt_end: "Possible abrupt ending",
  clipped: "Clipped samples", too_loud: "Peak level exceeds threshold", silent: "Silent audio",
};

export function RunDiagnostics({run}: {run: Run}) {
  const validation = record(run.result?.validation);
  const checks = record(validation.checks);
  const flags = Array.isArray(validation.flags) ? validation.flags.map(String) : [];
  const telemetry = record(run.result?.telemetry);
  const fingerprint = record(telemetry.fingerprint);
  const timing = record(telemetry.timing);
  const memory = record(telemetry.memory);
  const gpu = record(telemetry.gpu);
  const quality = record(run.result?.quality);
  const pending = run.status === "queued" || run.status === "running";
  return <>
    <section className="detail-section"><h4>WAV validation</h4>
      {Object.keys(validation).length ? <>
        <p className="diagnostic-outcome">{validation.ok === true ? flags.length ? "Passed with cautions" : "Passed" : "Failed"}</p>
        {flags.length > 0 && <ul className="diagnostic-flags">{flags.map(flag => <li key={flag}>{FLAG_LABELS[flag] || flag.replaceAll("_", " ")}</li>)}</ul>}
        <MetricGroup empty="No signal measurements were recorded." value={Object.keys(checks).length ? {
          "Duration": number(checks.duration_s, " s"), "Sample rate": number(checks.sample_rate, " Hz", 0),
          "Channels": checks.channels, "Peak level": number(checks.peak, "", 4), "RMS level": number(checks.rms, "", 4),
          "Clipped samples": checks.clipped_samples, "Non-finite samples": checks.non_finite_samples,
          "Leading silence": number(checks.leading_silence_s, " s"), "Trailing silence": number(checks.trailing_silence_s, " s"),
          "Longest internal pause": number(checks.longest_gap_s, " s"),
        } : null} />
        <p className="detail-empty diagnostic-note">Validation checks the audio signal and file. Speech accuracy is evaluated separately below.</p>
      </> : <p className="detail-empty">{pending ? "Validation runs after audio generation." : "No WAV validation was recorded for this run."}</p>}
    </section>
    <section className="detail-section"><h4>Telemetry</h4>
      {Object.keys(telemetry).length ? <>
        <MetricGroup empty="No runtime information was recorded." value={Object.keys(fingerprint).length ? {
          "Execution device": fingerprint.device, "GPU": fingerprint.gpu_name, "Python": fingerprint.python_version,
          "PyTorch": fingerprint.torch_version, "PyTorch CUDA build": fingerprint.torch_cuda_version,
          "Measured generation time": number(timing.wall_time_s, " s"), "Timer": timing.timer,
        } : telemetry} />
        <p className="detail-empty diagnostic-note">Generation measurements exclude WAV validation and quality evaluation.</p>
        {Object.keys(memory).length > 0 && <div className="diagnostic-subgroup"><MetricGroup empty="" value={{"Peak allocated GPU memory": number(memory.peak_allocated_mb, " MiB"), "Peak reserved GPU memory": number(memory.peak_reserved_mb, " MiB")}} /></div>}
        {typeof gpu.num_samples === "number" && gpu.num_samples > 0 && <div className="diagnostic-subgroup"><MetricGroup empty="" value={{
          "GPU samples": gpu.num_samples, "Mean GPU utilization": number(gpu.util_mean_pct, "%"), "Peak GPU utilization": number(gpu.util_peak_pct, "%"),
          "Mean power": number(gpu.power_mean_w, " W"), "Peak power": number(gpu.power_peak_w, " W"),
          "Mean temperature": number(gpu.temp_mean_c, " °C"), "Peak temperature": number(gpu.temp_peak_c, " °C"),
          "Mean SM clock": number(gpu.sm_clock_mean_mhz, " MHz"), "Peak SM clock": number(gpu.sm_clock_peak_mhz, " MHz"),
        }} /></div>}
        {typeof telemetry.gpu_note === "string" && <p className="detail-empty diagnostic-note">{telemetry.gpu_note}</p>}
        <Messages errors={telemetry.errors} />
      </> : <p className="detail-empty">{pending ? "Runtime telemetry is collected during generation." : "Telemetry was not recorded for this run. New runs collect runtime measurements."}</p>}
    </section>
    <section className="detail-section"><h4>Quality evaluation</h4>
      {Object.keys(quality).length ? <>
        <p className="diagnostic-outcome">{quality.status === "skipped" ? "Skipped" : quality.scored ? quality.status === "partial" ? "Partially available" : "Evaluated" : "Evaluation unavailable"}</p>
        <MetricGroup empty="No quality measurements were recorded." value={{"Word error rate (WER)": percentage(quality.wer), "Character error rate (CER)": percentage(quality.cer), "Speaker similarity": number(quality.sim_ref, "", 4), "Evaluation time": number(quality.evaluation_time_s, " s")}} />
        {typeof quality.transcript === "string" && <div className="diagnostic-subgroup"><h5>Recognized speech</h5><p className="detail-prompt">{quality.transcript || "No speech was recognized."}</p></div>}
        {typeof quality.asr_model === "string" && <p className="detail-empty diagnostic-note">Transcription model: {quality.asr_model}, English. WER and CER use the team’s normalized, intro-adjusted scores, capped at 100%. Lower is better.</p>}
        <Messages notes={quality.notes} errors={quality.errors} />
      </> : <p className="detail-empty">{pending ? "Quality evaluation runs after WAV validation. First use may take longer while the transcription model loads." : "Quality evaluation was not recorded for this run. Submit a new run to collect scores."}</p>}
    </section>
  </>;
}
