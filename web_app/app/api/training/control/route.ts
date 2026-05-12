import { NextResponse } from 'next/server';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';

export async function POST() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const errorMessage = payload?.detail || payload?.error || 'Training control request failed';
      return NextResponse.json({ error: errorMessage }, { status: response.status });
    }

    return NextResponse.json(payload, { status: response.status });
  } catch (error: any) {
    return NextResponse.json(
      { error: error?.message || 'Failed to reach backend training endpoint' },
      { status: 500 },
    );
  }
}
