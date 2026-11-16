"use client";
import { useSyncExternalStore } from "react";

export type Run = {
  id: string; prompt: string; nfe: number;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string; updated_at?: string; started_at?: string | null; completed_at?: string | null;
  result?: Record<string, unknown> | null; error_message?: string | null; audio_url?: string | null;
};
const KEY = "adaptive-compute.runs.v1";
const EVENT = "adaptive-compute-history";
let fallback = "[]";
let storageFailed = false;

export function normalizeRun(value: unknown): Run | null {
  if (!value || typeof value !== "object") return null;
  const run = value as Run;
  if (typeof run.id !== "string" || !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(run.id) || typeof run.prompt !== "string" ||
      typeof run.nfe !== "number" || !["queued", "running", "completed", "failed"].includes(run.status) || typeof run.created_at !== "string") return null;
  return { id: run.id, prompt: run.prompt, nfe: run.nfe, status: run.status, created_at: run.created_at,
    updated_at: typeof run.updated_at === "string" ? run.updated_at : undefined,
    started_at: typeof run.started_at === "string" ? run.started_at : null,
    completed_at: typeof run.completed_at === "string" ? run.completed_at : null,
    result: run.result && typeof run.result === "object" && !Array.isArray(run.result) ? run.result : null,
    error_message: typeof run.error_message === "string" ? run.error_message : null,
    audio_url: typeof run.audio_url === "string" ? run.audio_url : null };
}
export function parseRuns(value: string): Run[] {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map(normalizeRun).filter((run): run is Run => Boolean(run)) : [];
  } catch { return []; }
}
export function historySnapshot() {
  try { return storageFailed ? fallback : localStorage.getItem(KEY) || "[]"; }
  catch { storageFailed = true; return fallback; }
}
function subscribe(listener: () => void) {
  window.addEventListener("storage", listener);
  window.addEventListener(EVENT, listener);
  return () => { window.removeEventListener("storage", listener); window.removeEventListener(EVENT, listener); };
}
function save(runs: Run[]) {
  fallback = JSON.stringify(runs);
  try { localStorage.setItem(KEY, fallback); storageFailed = false; }
  catch { storageFailed = true; }
  window.dispatchEvent(new Event(EVENT));
}
// Web Locks serialize the read/modify/write across every tab on this origin.
// Older browsers still serialize operations within this tab.
let mutations = Promise.resolve();
function mutateHistory(change: () => void): Promise<void> {
  if (typeof navigator !== "undefined" && navigator.locks) return navigator.locks.request(KEY, change);
  const next = mutations.then(change);
  mutations = next.catch(() => {});
  return next;
}
export function rememberRun(value: unknown, existingOnly = false): Promise<void> {
  const run = normalizeRun(value);
  if (!run) return Promise.resolve();
  return mutateHistory(() => {
    const runs = parseRuns(historySnapshot());
    const existing = runs.find(item => item.id === run.id);
    if (existingOnly && !existing) return;
    const rank = { queued: 0, running: 1, completed: 2, failed: 2 };
    if (existing?.updated_at && run.updated_at && Date.parse(run.updated_at) < Date.parse(existing.updated_at)) return;
    if (existing && rank[run.status] < rank[existing.status]) return;
    if (existing && JSON.stringify(existing) === JSON.stringify(run)) return;
    save(existing ? runs.map(item => item.id === run.id ? run : item) : [run, ...runs]);
  });
}
export function forgetRun(id: string): Promise<void> {
  return mutateHistory(() => save(parseRuns(historySnapshot()).filter(run => run.id !== id)));
}
export function useRunHistory() {
  const value = useSyncExternalStore(subscribe, historySnapshot, () => "[]");
  return { runs: parseRuns(value), persistent: !storageFailed };
}

export function hasRun(id: string) { return parseRuns(historySnapshot()).some(run => run.id === id); }
