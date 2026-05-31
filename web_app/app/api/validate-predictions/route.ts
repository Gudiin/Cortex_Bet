import { NextResponse } from 'next/server';

const API_BASE_URL = 'http://127.0.0.1:8000';

export async function POST() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/validate-predictions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data?.detail || data?.error || 'Validation failed' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to connect to backend' },
      { status: 500 }
    );
  }
}
