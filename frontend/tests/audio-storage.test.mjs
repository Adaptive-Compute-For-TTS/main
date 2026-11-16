import 'fake-indexeddb/auto';
import test from 'node:test';
import assert from 'node:assert/strict';
import { cacheRunAudio, getStoredAudio, deleteStoredAudio } from '../lib/run-audio.ts';

function wav() {
 const data = new Uint8Array(44);
 data.set(new TextEncoder().encode('RIFF'),0);
 data.set(new TextEncoder().encode('WAVE'),8);
 return data;
}

test('saves the actual WAV blob durably and reads it back with the server offline', async t => {
 const id = 'offline-audio-test';
 let requests = 0;
 t.mock.method(globalThis,'fetch', async () => { requests++; return new Response(wav(), {headers:{'Content-Type':'audio/wav'}}); });
 const blob = await cacheRunAudio(id);
 assert.equal(blob.size,44);
 assert.deepEqual(new Uint8Array(await (await getStoredAudio(id)).arrayBuffer()),wav());
 t.mock.method(globalThis,'fetch', async () => { throw new Error('Backend offline'); });
 const restored = await cacheRunAudio(id);
 assert.equal(restored.size,44);
 assert.equal(requests,1);
 await deleteStoredAudio(id);
 assert.equal(await getStoredAudio(id),null);
});

test('deduplicates concurrent audio downloads and never caches a server error page', async t => {
 let requests = 0;
 t.mock.method(globalThis,'fetch', async () => { requests++; return new Response(wav()); });
 await Promise.all([cacheRunAudio('shared-download'),cacheRunAudio('shared-download')]);
 assert.equal(requests,1);
 await deleteStoredAudio('shared-download');
 t.mock.method(globalThis,'fetch', async () => new Response('<html>error</html>'));
 await assert.rejects(cacheRunAudio('not-a-wav'), /valid WAV/);
 assert.equal(await getStoredAudio('not-a-wav'),null);
});

test('a removed receipt cannot be recreated as an orphan audio file', async t => {
 t.mock.method(globalThis,'fetch', async () => new Response(wav()));
 await assert.rejects(cacheRunAudio('removed-run',()=>false), /cancelled/);
 assert.equal(await getStoredAudio('removed-run'),null);
});

test('removal waits for an in-flight save to settle before deleting the blob', async t => {
 let release;
 const waiting = new Promise(resolve => { release = resolve; });
 let started;
 const downloading = new Promise(resolve => { started = resolve; });
 t.mock.method(globalThis,'fetch', async () => { started(); await waiting; return new Response(wav()); });
 const saving = cacheRunAudio('remove-during-download');
 await downloading;
 const removed = deleteStoredAudio('remove-during-download');
 release();
 await assert.rejects(saving,/cancelled/);
 await removed;
 assert.equal(await getStoredAudio('remove-during-download'),null);
});
