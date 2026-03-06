/**
 * Storage KV API — T-WS-11.
 *
 * Simple key-value store backed by an in-memory Map.
 */

function jsonResponse(data: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(data), {
    status: init?.status ?? 200,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
}

function errorResponse(status: number, detail: string): Response {
  return jsonResponse({ detail }, { status });
}

/**
 * Create a storage handler with its own backing Map.
 * Returns a function that handles /api/storage/{key} requests.
 */
export function createStorageHandler(): (request: Request) => Promise<Response> {
  const store = new Map<string, unknown>();
  return (request: Request) => handleStorageRequest(request, store);
}

/**
 * Handle storage KV API requests.
 * Routes: GET/PUT/DELETE /api/storage/{key}
 */
export async function handleStorageRequest(
  request: Request,
  store: Map<string, unknown>
): Promise<Response> {
  const url = new URL(request.url);
  const pathname = url.pathname.replace(/\/+$/, "");
  const prefix = "/api/storage/";

  if (!pathname.startsWith(prefix)) {
    return errorResponse(404, "Not found");
  }

  const key = decodeURIComponent(pathname.slice(prefix.length));
  if (!key) {
    return errorResponse(400, "Key is required");
  }

  if (request.method === "GET") {
    if (!store.has(key)) {
      return errorResponse(404, "Key not found");
    }
    return jsonResponse({ key, value: store.get(key) });
  }

  if (request.method === "PUT") {
    let body: Record<string, unknown>;
    try {
      body = (await request.json()) as Record<string, unknown>;
    } catch {
      return errorResponse(400, "Invalid JSON body");
    }
    if (!body || !("value" in body)) {
      return errorResponse(400, "Body must contain a 'value' field");
    }
    store.set(key, body.value);
    return jsonResponse({ key, value: body.value });
  }

  if (request.method === "DELETE") {
    if (!store.has(key)) {
      return errorResponse(404, "Key not found");
    }
    store.delete(key);
    return new Response(null, { status: 204 });
  }

  return errorResponse(405, "Method not allowed");
}
