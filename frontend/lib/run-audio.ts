"use client";

const DATABASE = "adaptive-compute.device-files";
const STORE = "audio";
const pending = new Map<string, { controller: AbortController; promise: Promise<Blob> }>();

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") { reject(new Error("Device audio storage is unavailable in this browser.")); return; }
    const request = indexedDB.open(DATABASE, 1);
    let blocked = false;
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE, { keyPath: "id" });
    };
    request.onerror = () => reject(new Error("Could not open device audio storage."));
    request.onblocked = () => { blocked = true; reject(new Error("Close other app tabs and try saving audio again.")); };
    request.onsuccess = () => { if (blocked) request.result.close(); else resolve(request.result); };
  });
}

export async function getStoredAudio(id: string): Promise<Blob | null> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE, "readonly");
    const request = transaction.objectStore(STORE).get(id);
    let blob: Blob | null = null;
    request.onsuccess = () => { blob = request.result?.blob instanceof Blob ? request.result.blob : null; };
    transaction.oncomplete = () => { db.close(); resolve(blob); };
    transaction.onabort = () => { db.close(); reject(new Error("Could not read audio saved on this device.")); };
    transaction.onerror = () => { /* onabort reports the transaction failure */ };
  });
}

async function storeAudio(id: string, blob: Blob) {
  const db = await openDatabase();
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite", { durability: "strict" });
    transaction.objectStore(STORE).put({ id, blob });
    transaction.oncomplete = () => { db.close(); resolve(); };
    transaction.onabort = () => { db.close(); reject(new Error("Audio could not be saved. Check available device storage and retry.")); };
    transaction.onerror = () => { /* onabort reports the transaction failure */ };
  });
}

export function cacheRunAudio(id: string, keep: () => boolean = () => true): Promise<Blob> {
  const existing = pending.get(id);
  if (existing) return existing.promise;
  const controller = new AbortController();
  const promise = (async () => {
    const stored = await getStoredAudio(id);
    if (stored) return stored;
    const response = await fetch(`/api/runs?id=${encodeURIComponent(id)}&audio=1`, { signal: controller.signal });
    if (!response.ok) throw new Error("Could not download this run’s audio. Keep the backend running and retry.");
    const blob = await response.blob();
    const bytes = new Uint8Array(await blob.slice(0, 12).arrayBuffer());
    const text = new TextDecoder().decode(bytes);
    if (text.slice(0, 4) !== "RIFF" || text.slice(8, 12) !== "WAVE") throw new Error("The service did not return a valid WAV file.");
    if (controller.signal.aborted || !keep()) throw new Error("Local audio save was cancelled.");
    await storeAudio(id, blob);
    return blob;
  })().finally(() => { pending.delete(id); });
  pending.set(id, { controller, promise });
  return promise;
}

export async function deleteStoredAudio(id: string) {
  const saving = pending.get(id);
  if (saving) {
    saving.controller.abort();
    await saving.promise.catch(() => {});
  }
  const db = await openDatabase();
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite");
    transaction.objectStore(STORE).delete(id);
    transaction.oncomplete = () => { db.close(); resolve(); };
    transaction.onabort = () => { db.close(); reject(new Error("Could not remove the local audio copy.")); };
  });
}

export async function requestPersistentStorage(): Promise<boolean> {
  try { return await navigator.storage?.persist?.() || false; } catch { return false; }
}
