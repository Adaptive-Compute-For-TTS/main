"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { forgetRun, hasRun, rememberRun, useRunHistory, type Run } from "@/lib/run-history";
import { cacheRunAudio, deleteStoredAudio, requestPersistentStorage } from "@/lib/run-audio";
import { RunDetail } from "./RunDetail";

export function RunHistory() {
  const { runs, persistent } = useRunHistory();
  const trigger = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const detail = useRef<HTMLElement>(null);
  const [open, setOpen] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [removingId, setRemovingId] = useState<string | null>(null);
  const removing = useRef(false);
  const [removalNotice, setRemovalNotice] = useState("");
  const [savedAudio, setSavedAudio] = useState<Record<string, boolean>>({});
  const [audioErrors, setAudioErrors] = useState<Record<string, string>>({});
  const active = runs.filter(run => run.status === "queued" || run.status === "running").map(run => run.id).join(",");
  const completed = runs.filter(run => run.status === "completed" && run.audio_url).map(run => run.id).join(",");
  const selected = runs.find(run => run.id === selectedId);
  const audioSaved = useCallback((id: string) => {
    if (!hasRun(id)) return;
    setSavedAudio(previous => previous[id] ? previous : ({...previous, [id]: true}));
    setAudioErrors(previous => {
      if (!previous[id]) return previous;
      const next = {...previous}; delete next[id]; return next;
    });
  }, []);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let polling = false;
    async function refresh() {
      if (polling) return;
      polling = true;
      try {
        await Promise.all(active.split(",").map(async id => {
          const response = await fetch(`/api/runs?id=${id}`, { signal: controller.signal, cache: "no-store" });
          if (!response.ok) throw new Error("Could not refresh runs. Retrying shortly.");
          await rememberRun(await response.json(), true);
        }));
        setRefreshError("");
      } catch { if (!controller.signal.aborted) setRefreshError("Could not refresh runs. Retrying shortly."); }
      finally { polling = false; }
    }
    void refresh();
    const timer = setInterval(() => void refresh(), 2000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [active]);

  useEffect(() => {
    if (!completed) return;
    let cancelled = false;
    async function saveCompleted() {
      for (const id of completed.split(",")) {
        if (cancelled) return;
        try {
          await cacheRunAudio(id, () => hasRun(id));
          if (!cancelled) audioSaved(id);
        } catch (error) {
          if (!cancelled && hasRun(id)) setAudioErrors(previous => ({...previous, [id]: error instanceof Error ? error.message : "Audio is not saved yet."}));
        }
      }
    }
    void saveCompleted();
    return () => { cancelled = true; };
  }, [completed, audioSaved]);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) element.showModal();
    if (!open && element.open) element.close();
    if (!open) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = overflow; };
  }, [open]);

  useEffect(() => { if (selectedId) detail.current?.focus({preventScroll: true}); }, [selectedId]);

  function closeRuns() { setOpen(false); trigger.current?.focus(); }
  async function refreshDetails(run: Run) {
    setRefreshing(true);
    try {
      const response = await fetch(`/api/runs?id=${run.id}`, {cache: "no-store"});
      if (!response.ok) throw new Error();
      await rememberRun(await response.json(), true);
      setRefreshError("");
    } catch { setRefreshError("The backend is unavailable. Showing the details saved on this device."); }
    finally { setRefreshing(false); }
  }
  function selectRun(run: Run) { setSelectedId(run.id); void refreshDetails(run); }
  async function removeRun(run: Run) {
    if (removing.current) return;
    removing.current = true;
    setRemovingId(run.id);
    setRemovalNotice("");
    const index = runs.findIndex(item => item.id === run.id);
    const nextId = runs[index + 1]?.id ?? runs[index - 1]?.id;
    try {
      if (run.status === "completed" && run.audio_url) await deleteStoredAudio(run.id);
      await forgetRun(run.id);
      setSelectedId(previous => previous === run.id ? null : previous);
      setRemovalNotice("Run removed from this device.");
      requestAnimationFrame(() => {
        const next = Array.from(dialog.current?.querySelectorAll<HTMLButtonElement>(".run-item") || []).find(button => button.dataset.runId === nextId);
        (next ?? dialog.current?.querySelector<HTMLButtonElement>(".runs-close"))?.focus();
      });
    } catch { setRemovalNotice("The local copy could not be removed. Close other app tabs and try again."); }
    finally { removing.current = false; setRemovingId(null); }
  }
  return <>
    <button type="button" ref={trigger} className="history-toggle" aria-haspopup="dialog" aria-expanded={open} aria-controls="run-history-dialog" onClick={() => { setOpen(true); void requestPersistentStorage(); if (selected) void refreshDetails(selected); }}>Your runs <span>{runs.length}</span></button>
    <dialog id="run-history-dialog" ref={dialog} className="runs-dialog" aria-labelledby="runs-title" onCancel={() => setOpen(false)} onClose={() => { setOpen(false); trigger.current?.focus(); }} onClick={event => { if (event.target === event.currentTarget) closeRuns(); }}>
      <header className="runs-header"><div><h2 id="runs-title">Your runs</h2><p>{runs.length} {runs.length === 1 ? "run" : "runs"} on this device</p></div><button type="button" autoFocus onClick={closeRuns} className="runs-close" aria-label="Close runs">×</button></header>
      {!persistent && <p role="status" className="runs-notice">Run details cannot be persisted in this browser. Export them to keep a file copy.</p>}
      {refreshError && <p role="status" className="runs-notice">{refreshError}</p>}
      {removalNotice && <p role="status" className="runs-notice">{removalNotice}</p>}
      <div className={`runs-content${selected ? " has-selection" : ""}`}>
        <section className="runs-navigation" aria-label="Saved runs">
          <div className="runs-table-intro"><span aria-hidden="true" /><p>{active ? "Active runs refresh every 2 seconds." : "Select a run to open its audio and metrics."}</p></div>
          {!runs.length ? <p className="runs-empty">Your submitted runs will appear here.</p> : <div className="runs-table-scroll" tabIndex={0} role="region" aria-label="Saved runs table">
            <table className="runs-table">
              <caption className="sr-only">Runs saved on this device. Open a run to see its audio and metrics.</caption>
              <thead><tr><th scope="col">Text / Run</th><th scope="col">NFE</th><th scope="col">Current progress</th><th scope="col">Created</th><th scope="col">Local audio</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>{runs.map(run => <tr key={run.id} onClick={() => selectRun(run)}>
                <td><button className="run-item" type="button" data-run-id={run.id} onClick={event => { event.stopPropagation(); selectRun(run); }} aria-label={`Open run: ${run.prompt.slice(0, 120)}`}><span className="run-item-prompt">{run.prompt}</span><span className="run-item-id">{run.id.slice(0, 8)} ↗</span></button></td>
                <td className="run-nfe">{run.nfe}</td>
                <td><span role="status" aria-atomic="true" className={`run-progress progress-${run.status}`}><span aria-hidden="true" className="run-progress-dot" />{{queued: "Queued", running: "Processing", completed: "Completed", failed: "Failed"}[run.status]}</span></td>
                <td><time dateTime={run.created_at}>{new Date(run.created_at).toLocaleString()}</time></td>
                <td className="run-storage">{run.status === "completed" && run.audio_url ? savedAudio[run.id] ? "Saved" : audioErrors[run.id] ? "Needs saving" : "Saving…" : "—"}</td>
                <td><button className="run-remove" type="button" disabled={removingId !== null} aria-label={`Remove run: ${run.prompt.slice(0, 120)} (${run.id.slice(0, 8)})`} onClick={event => { event.stopPropagation(); void removeRun(run); }}>{removingId === run.id ? "Removing…" : "Remove"}</button></td>
              </tr>)}</tbody>
            </table>
          </div>}
        </section>
        <section className="run-detail" ref={detail} tabIndex={-1} aria-label="Selected run details">
          <button type="button" className="runs-back detail-text-button" onClick={() => { setSelectedId(null); requestAnimationFrame(() => Array.from(dialog.current?.querySelectorAll<HTMLButtonElement>(".run-item") || []).find(button => button.dataset.runId === selectedId)?.focus()); }}>← All runs</button>
          {selected ? <RunDetail key={selected.id} run={selected} onAudioSaved={audioSaved} refresh={() => void refreshDetails(selected)} refreshing={refreshing} remove={() => void removeRun(selected)} removing={removingId !== null} /> : <div className="detail-empty-state"><h3>A place for every run.</h3><p>Select a saved run to see its audio, performance, and telemetry.</p></div>}
        </section>
      </div>
    </dialog>
  </>;
}
