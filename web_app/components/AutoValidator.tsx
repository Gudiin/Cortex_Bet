"use client";

import { useEffect } from 'react';

export default function AutoValidator() {
  useEffect(() => {
    // Validate immediately on mount (catch any leftover PENDING from overnight)
    runValidation();

    // Run validation every 5 minutes
    const interval = setInterval(runValidation, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  return null;
}

async function runValidation() {
  try {
    // 1. Validate user bets (user_bets table)
    const betsRes = await fetch('/api/validate-bets', { method: 'POST' });
    const betsData = await betsRes.json();
    if (betsData.success && betsData.validated_count > 0) {
      console.log(`✅ Auto-validated ${betsData.validated_count} user bets`);
    }
  } catch (err) {
    console.error('User bet validation failed:', err);
  }

  try {
    // 2. Validate system predictions (predictions table → GREEN/RED)
    // This fixes predictions stuck as PENDING after a match finishes
    const predRes = await fetch('/api/validate-predictions', { method: 'POST' });
    const predData = await predRes.json();
    if (predData.success) {
      console.log('✅ System predictions validated (GREEN/RED updated)');
    }
  } catch (err) {
    console.error('Prediction validation failed:', err);
  }
}
