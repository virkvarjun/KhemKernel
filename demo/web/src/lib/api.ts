// Thin client for the local python inference backend (demo/server.py).
// When the backend is absent (e.g. the static GitHub Pages build) health()
// returns unavailable and the UI falls back to precomputed examples.

export interface Health {
  ok: boolean;
  step: number;
  opsin: boolean;
}
export interface Smiles2Iupac {
  ok: boolean;
  name: string;
  trace: string;
  verified: boolean;
  opsin_smiles: string | null;
  decode: string;
  error?: string;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return (await r.json()) as T;
}

export async function health(): Promise<Health | null> {
  try {
    const r = await fetch("/api/health", { signal: AbortSignal.timeout(2500) });
    if (!r.ok) return null;
    return (await r.json()) as Health;
  } catch {
    return null;
  }
}

export function smiles2iupac(smiles: string) {
  return postJSON<Smiles2Iupac>("/api/smiles2iupac", { smiles });
}

export function iupac2smiles(name: string) {
  return postJSON<{ ok: boolean; smiles?: string; error?: string }>(
    "/api/iupac2smiles",
    { name },
  );
}
