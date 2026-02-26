import { type NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  let res: Response;
  try {
    const form = await req.formData();
    res = await fetch(`${BACKEND}/ingest`, { method: "POST", body: form });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend unreachable";
    return NextResponse.json({ detail: message }, { status: 502 });
  }

  const text = await res.text();
  try {
    return NextResponse.json(JSON.parse(text), { status: res.status });
  } catch {
    return NextResponse.json(
      { detail: `Backend returned non-JSON response (status ${res.status})` },
      { status: 502 }
    );
  }
}
