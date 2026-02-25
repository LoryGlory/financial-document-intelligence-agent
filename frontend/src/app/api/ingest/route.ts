import { type NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const res = await fetch(`${BACKEND}/ingest`, { method: "POST", body: form });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
