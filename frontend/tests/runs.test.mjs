import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import test from "node:test";
import { GET, POST } from "../app/api/runs/route.ts";

function configure(t, value) {
  const previous = process.env.FASTAPI_BASE_URL;
  if (value === undefined) delete process.env.FASTAPI_BASE_URL;
  else process.env.FASTAPI_BASE_URL = value;
  t.after(() => {
    if (previous === undefined) delete process.env.FASTAPI_BASE_URL;
    else process.env.FASTAPI_BASE_URL = previous;
  });
}

function request(body = { prompt: "Hello world.", nfe: 20 }) {
  return new Request("http://localhost/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function serve(t, handler) {
  const server = createServer(handler);
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  return { server, base: `http://127.0.0.1:${server.address().port}` };
}

test("forwards only entered text and chosen NFE to the configured base path and preserves 202", async (t) => {
  let received;
  const { base } = await serve(t, async (incoming, response) => {
    const chunks = [];
    for await (const chunk of incoming) chunks.push(chunk);
    received = {
      method: incoming.method,
      url: incoming.url,
      contentType: incoming.headers["content-type"],
      body: JSON.parse(Buffer.concat(chunks).toString()),
    };
    response.writeHead(202, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ id: "d379c590-04de-4c75-b4f5-6b3ab7665ed7", status: "queued" }));
  });
  configure(t, `${base}/speech/v1/`);

  const response = await POST(request({ prompt: "  A new line.\nAnother sentence.  ", nfe: 32, ignored: true }));

  assert.deepEqual(received, {
    method: "POST", url: "/speech/v1/runs", contentType: "application/json",
    body: { prompt: "  A new line.\nAnother sentence.  ", nfe: 32 },
  });
  assert.equal(response.status, 202);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.deepEqual(await response.json(), { id: "d379c590-04de-4c75-b4f5-6b3ab7665ed7", status: "queued" });
});

test("rejects invalid input before contacting the backend", async (t) => {
  let calls = 0;
  const { base } = await serve(t, (_incoming, response) => { calls++; response.end(); });
  configure(t, base);
  for (const body of [null, [], {}, { prompt: " \n ", nfe: 4 }, { prompt: 123, nfe: 20 },
    { prompt: "Hello", nfe: 8 }, { prompt: "Hello", nfe: "20" }]) {
    const response = await POST(request(body));
    assert.equal(response.status, 400);
    assert.equal(typeof (await response.json()).error, "string");
  }
  const malformed = await POST(new Request("http://localhost/api/runs", { method: "POST", body: "{" }));
  assert.equal(malformed.status, 400);
  assert.equal(calls, 0);
});

test("missing or invalid server configuration returns an actionable 503", async (t) => {
  configure(t, undefined);
  const missing = await POST(request());
  assert.equal(missing.status, 503);
  assert.deepEqual(await missing.json(), {
    error: "The speech service is not connected yet. Please try again later.",
  });
  for (const value of ["not a URL", "file:///tmp/speech", "https://user:secret@example.com", "https://example.com?token=secret"]) {
    process.env.FASTAPI_BASE_URL = value;
    const response = await POST(request());
    assert.equal(response.status, 503);
    assert.doesNotMatch(JSON.stringify(await response.json()), /secret|file:|example\.com/);
  }
});

test("preserves FastAPI validation details and status", async (t) => {
  const details = { detail: [{ loc: ["body", "prompt"], msg: "Prompt is too long", type: "value_error" }] };
  const { base } = await serve(t, (_incoming, response) => {
    response.writeHead(422, { "Content-Type": "application/json" });
    response.end(JSON.stringify(details));
  });
  configure(t, base);
  const response = await POST(request());
  assert.equal(response.status, 422);
  assert.deepEqual(await response.json(), details);
});

test("turns non-JSON upstream errors into a safe message while preserving the status", async (t) => {
  const { base } = await serve(t, (_incoming, response) => {
    response.writeHead(500, { "Content-Type": "text/html" });
    response.end("<html>Server internals: secret-stack-trace</html>");
  });
  configure(t, base);
  const response = await POST(request());
  assert.equal(response.status, 500);
  const body = await response.json();
  assert.match(body.error, /speech service/i);
  assert.doesNotMatch(JSON.stringify(body), /html|secret-stack-trace/);
});

test("does not report a non-JSON successful upstream response as a successful run", async (t) => {
  const { base } = await serve(t, (_incoming, response) => {
    response.writeHead(200, { "Content-Type": "text/html" });
    response.end("<html>Backend sign-in</html>");
  });
  configure(t, base);
  const response = await POST(request());
  assert.equal(response.status, 502);
  assert.match((await response.json()).error, /invalid response/i);
});

test("an unreachable upstream returns 502 without leaking connection details", async (t) => {
  const { server, base } = await serve(t, (_incoming, response) => response.end());
  await new Promise((resolve) => server.close(resolve));
  configure(t, base);
  const response = await POST(request());
  assert.equal(response.status, 502);
  assert.match((await response.json()).error, /could not be reached/i);
});

test("an upstream that does not respond times out with 504", async (t) => {
  const { base } = await serve(t, () => {});
  configure(t, base);
  const timeout = AbortSignal.timeout.bind(AbortSignal);
  t.mock.method(AbortSignal, "timeout", () => timeout(25));
  const response = await POST(request());
  assert.equal(response.status, 504);
  assert.match((await response.json()).error, /too long/i);
});

test("reads the persisted run through the same-origin receipt URL", async (t) => {
  const id = "d379c590-04de-4c75-b4f5-6b3ab7665ed7";
  let received;
  const { base } = await serve(t, (incoming, response) => {
    received = { method: incoming.method, path: incoming.url };
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ id, prompt: "Saved text", nfe: 20, status: "queued" }));
  });
  configure(t, base);
  const response = await GET(new Request(`http://localhost/api/runs?id=${id}`));
  assert.equal(response.status, 200);
  assert.deepEqual(received, { method: "GET", path: `/runs/${id}` });
  assert.equal((await response.json()).id, id);
  assert.equal((await GET(new Request("http://localhost/api/runs?id=../../health"))).status, 400);
});

test("streams audio bytes and forwards seek ranges without JSON decoding", async (t) => {
  const id = "d379c590-04de-4c75-b4f5-6b3ab7665ed7";
  let received;
  const { base } = await serve(t, (incoming, response) => {
    received = { url: incoming.url, range: incoming.headers.range };
    response.writeHead(206, { "Content-Type": "audio/wav", "Content-Range": "bytes 0-3/100", "Accept-Ranges": "bytes" });
    response.end(Buffer.from("RIFF"));
  });
  configure(t, base);
  const response = await GET(new Request(`http://localhost/api/runs?id=${id}&audio=1`, { headers: {Range:"bytes=0-3"} }));
  assert.equal(response.status, 206);
  assert.equal(response.headers.get("content-type"), "audio/wav");
  assert.equal(response.headers.get("content-range"), "bytes 0-3/100");
  assert.equal(await response.text(), "RIFF");
  assert.deepEqual(received, {url:`/runs/${id}/audio`,range:"bytes=0-3"});
});
