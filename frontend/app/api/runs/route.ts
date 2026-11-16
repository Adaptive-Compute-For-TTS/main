const SERVICE_UNAVAILABLE = "The speech service is not connected yet. Please try again later.";
const UPSTREAM_TIMEOUT_MS = 30_000;
const RUN_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function json(body: unknown, status: number) {
  return Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
}

async function forward(path: string, body?: { prompt: string; nfe: number }, range?: string | null) {
  const configuredBase = process.env.FASTAPI_BASE_URL?.trim();
  if (!configuredBase) return json({ error: SERVICE_UNAVAILABLE }, 503);

  let endpoint: URL;
  try {
    endpoint = new URL(configuredBase);
    if (
      !["http:", "https:"].includes(endpoint.protocol) ||
      endpoint.username || endpoint.password || endpoint.search || endpoint.hash
    ) {
      throw new Error("Invalid speech service URL");
    }
    endpoint.pathname = `${endpoint.pathname.replace(/\/+$/, "")}/${path}`;
  } catch {
    return json({ error: "The speech service is not configured correctly. Please try again later." }, 503);
  }

  const timeout = AbortSignal.timeout(UPSTREAM_TIMEOUT_MS);
  try {
    const upstream = await fetch(endpoint, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json", Accept: "application/json" } : { Accept: path.endsWith("/audio") ? "audio/wav" : "application/json", ...(range ? { Range: range } : {}) },
      ...(body ? { body: JSON.stringify(body) } : {}),
      cache: "no-store",
      signal: timeout,
    });

    if (upstream.status === 204 || upstream.status === 205) {
      return new Response(null, { status: upstream.status, headers: { "Cache-Control": "no-store" } });
    }

    if (path.endsWith("/audio") && upstream.ok) {
      const headers = new Headers({ "Cache-Control": "no-store" });
      for (const key of ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"]) {
        const value = upstream.headers.get(key);
        if (value) headers.set(key, value);
      }
      return new Response(upstream.body, { status: upstream.status, headers });
    }
    const responseText = await upstream.text();
    let result: unknown;
    try {
      result = JSON.parse(responseText);
    } catch {
      return json({ error: upstream.ok
        ? "The speech service returned an invalid response. Please try again."
        : "The speech service could not accept this request. Please try again."
      }, upstream.status >= 400 ? upstream.status : 502);
    }

    return json(result, upstream.status);
  } catch {
    return timeout.aborted
      ? json({ error: "The speech service took too long to respond. Please try again." }, 504)
      : json({ error: "The speech service could not be reached. Please try again." }, 502);
  }
}

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Send your text and NFE budget as JSON." }, 400);
  }

  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return json({ error: "Send your text and NFE budget as JSON." }, 400);
  }

  const { prompt, nfe } = body as Record<string, unknown>;
  if (typeof prompt !== "string" || !prompt.trim()) {
    return json({ error: "Enter some text before submitting." }, 400);
  }
  if (nfe !== 4 && nfe !== 20 && nfe !== 32) {
    return json({ error: "Choose an NFE budget of 4, 20, or 32." }, 400);
  }

  return forward("runs", { prompt, nfe });
}

export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id");
  if (id !== null && !RUN_ID.test(id)) return json({ error: "Choose a valid run ID." }, 400);
  const audio = new URL(request.url).searchParams.get("audio") === "1";
  if (audio && !id) return json({ error: "Choose a valid run ID." }, 400);
  return forward(id ? `runs/${id}${audio ? "/audio" : ""}` : "runs", undefined, request.headers.get("range"));
}
