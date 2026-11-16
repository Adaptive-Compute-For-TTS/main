"use client";
import { useEffect, useState } from "react";
import { cacheRunAudio } from "@/lib/run-audio";
import { hasRun, type Run } from "@/lib/run-history";
import { MetricGroup, RunDiagnostics } from "./RunDiagnostics";
function exportDetails(run: Run) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(run, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url; link.download = `run-${run.id}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function RunDetail({ run, refresh, remove, refreshing, removing, onAudioSaved }: {run: Run; onAudioSaved: (id: string) => void; refresh: () => void; remove: () => void; refreshing: boolean; removing: boolean}) {
  const [audio, setAudio] = useState<{url?: string; error?: string}>({});
  const [attempt, setAttempt] = useState(0);
  const [playbackError, setPlaybackError] = useState(false);
  useEffect(() => {
    if (run.status !== "completed" || !run.audio_url) return;
    let cancelled = false;
    let url: string | undefined;
    void cacheRunAudio(run.id, () => hasRun(run.id)).then(blob => {
      if (cancelled) return;
      url = URL.createObjectURL(blob);
      setAudio({url});
      setPlaybackError(false);
      onAudioSaved(run.id);
    }).catch(error => { if (!cancelled) setAudio({error: error instanceof Error ? error.message : "Could not save audio on this device."}); });
    return () => { cancelled = true; if (url) URL.revokeObjectURL(url); };
  }, [run.id, run.status, run.audio_url, attempt, onAudioSaved]);
  const result = run.result;
  const performance = result ? {
    ...(typeof result.audio_s === "number" ? {"Audio duration": `${result.audio_s.toFixed(2)} s`} : {}),
    ...(typeof result.time_s === "number" ? {"Generation time": `${result.time_s.toFixed(2)} s`} : {}),
    ...(typeof result.rtf === "number" ? {"Real-time factor": result.rtf.toFixed(3)} : {}),
    ...(typeof result.sample_rate === "number" ? {"Sample rate": `${result.sample_rate.toLocaleString()} Hz`} : {}),
    ...(typeof result.peak_vram_mb === "number" ? {"Peak GPU memory": `${result.peak_vram_mb.toFixed(1)} MB`} : {}),
  } : null;
  return <>
    <header className="detail-heading"><div><span className="detail-status">NFE {run.nfe} · {run.status}</span><h3>Run details</h3></div><button className="detail-text-button" type="button" disabled={refreshing} onClick={refresh}>{refreshing ? "Refreshing…" : "Refresh metrics"}</button></header>
    <p className="detail-id">{run.id}</p>
    <time className="detail-date" dateTime={run.created_at}>{new Date(run.created_at).toLocaleString()}</time>
    <section className="detail-section"><h4>Text</h4><p className="detail-prompt">{run.prompt}</p></section>
    <section className="detail-section"><h4>Audio</h4>
      {audio.url ? <><audio key={audio.url} controls preload="metadata" src={audio.url} onError={() => setPlaybackError(true)}>Your browser does not support audio playback.</audio>
        <div className="detail-audio-footer"><span>Saved on this device</span><a href={audio.url} download={`run-${run.id}.wav`}>Download WAV ↗</a></div>
        {playbackError && <p role="alert" className="request-error">This browser could not play the audio. Download the WAV to play it on your device.</p>}</> :
        audio.error ? <div role="status"><p className="request-error">{audio.error}</p><button className="detail-text-button" type="button" onClick={() => { setAudio({}); setAttempt(attempt + 1); }}>Retry audio save</button></div> :
        <p className="detail-empty">{run.status === "completed" && run.audio_url ? "Saving audio to this device…" : (run.status === "failed" || run.status === "completed") ? "No audio was reported for this run." : "Audio will appear here when generation finishes."}</p>}
      {run.error_message && <p className="request-error">{run.error_message}</p>}
    </section>
    <section className="detail-section"><h4>Performance</h4><MetricGroup value={performance} empty="Performance metrics have not been reported for this run." /></section>
    <RunDiagnostics run={run} />
    <section className="detail-section"><details><summary>All saved metrics</summary><pre>{JSON.stringify(result || {}, null, 2)}</pre></details></section>
    <footer className="detail-footer"><button className="detail-text-button" type="button" onClick={() => exportDetails(run)}>Export run details ↗</button><button className="run-remove" type="button" disabled={removing} onClick={remove}>{removing ? "Removing…" : "Remove from this device"}</button></footer>
  </>;
}
