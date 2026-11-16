import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeRun, parseRuns } from '../lib/run-history.ts';
const run = {id:'d379c590-04de-4c75-b4f5-6b3ab7665ed7',prompt:'hello',nfe:20,status:'queued',created_at:'2026-11-09T12:00:00Z'};
test('restores valid run receipts and rejects corrupt browser history', () => {
 assert.equal(parseRuns('invalid-json').length, 0);
 assert.equal(parseRuns('{}').length, 0);
 assert.equal(parseRuns(JSON.stringify([null,{},run]))[0].id,run.id);
 assert.equal(normalizeRun({...run,status:'made-up'}),null);
 assert.equal(normalizeRun({...run,id:'../private'}),null);
 assert.equal(parseRuns(JSON.stringify(Array(101).fill(run))).length,101);
});

test('keeps full performance, telemetry and future metrics in a device receipt', () => {
 const source = {...run,updated_at:'2026-11-09T12:01:00Z',started_at:'2026-11-09T12:00:01Z',completed_at:'2026-11-09T12:00:30Z',status:'completed',result:{time_s:29,rtf:0.8,telemetry:{temperature:70},quality:{wer:0.1},future_metrics:{a:[1,2]}}};
 const restored = parseRuns(JSON.stringify([source]))[0];
 assert.deepEqual(restored.result,source.result);
 assert.equal(restored.completed_at,source.completed_at);
});

test('serializes two tabs so new receipts survive and removed runs stay removed', async t => {
 const storage = new Map();
 function install(name,value) {
  const original = Object.getOwnPropertyDescriptor(globalThis,name);
  Object.defineProperty(globalThis,name,{configurable:true,value});
  t.after(()=> { if (original) Object.defineProperty(globalThis,name,original); else delete globalThis[name]; });
 }
 install('localStorage', {
  getItem: key => storage.get(key) ?? null,
  setItem: (key,value) => storage.set(key,value),
 });
 install('window', new EventTarget());
 let tail = Promise.resolve();
 let requests = 0;
 install('navigator', {locks:{request: (_name,change) => {
  requests++;
  const next = tail.then(change);
  tail = next.catch(()=>{});
  return next;
 }}});
 const tabA = await import('../lib/run-history.ts?tab=a');
 const tabB = await import('../lib/run-history.ts?tab=b');
 const other = {...run,id:'ad81deac-c621-4f9e-a2c8-b480f5fbcc6f'};
 await Promise.all([tabA.rememberRun(run),tabB.rememberRun(other)]);
 assert.equal(parseRuns(tabA.historySnapshot()).length,2);
 await Promise.all([
  tabA.forgetRun(run.id),
  tabB.rememberRun({...run,status:'completed'},true),
  tabA.rememberRun({...other,status:'completed',updated_at:'2026-11-09T12:02:00Z'},true),
  tabB.rememberRun({...other,status:'running',updated_at:'2026-11-09T12:01:00Z'},true),
 ]);
 const saved = parseRuns(tabA.historySnapshot());
 assert.equal(saved.length,1);
 assert.equal(saved[0].id,other.id);
 assert.equal(saved[0].status,'completed');
 assert.equal(requests,6);
});
