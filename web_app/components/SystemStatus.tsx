"use client";

import { useState, useEffect } from 'react';

interface StatusData {
  status: string;
  last_updated: string;
  live_matches: number;
}
interface OpsStatus {
  last_training_at?: string;
  last_training_pid?: number;
  last_scanner_at?: string;
  last_scanner_action?: string;
  last_scanner_pid?: number;
  last_update_at?: string;
  last_update_range?: string;
  last_update_pid?: number;
}

interface SystemStatusProps {
  mode?: 'compact' | 'full';
}

export default function SystemStatus({ mode = 'full' }: SystemStatusProps) {
  const [data, setData] = useState<StatusData | null>(null);
  const [scannerActive, setScannerActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);
  const [training, setTraining] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [opsStatus, setOpsStatus] = useState<OpsStatus | null>(null);
  const [feedback, setFeedback] = useState<string>('');

  const fetchStatus = async () => {
    try {
      // 1. Get System Data Status
      const res = await fetch('/api/system-status');
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
      
      // 2. Get Scanner Process Status
      const procRes = await fetch('/api/scanner/control');
      if (procRes.ok) {
         const procJson = await procRes.json();
         setScannerActive(procJson.active);
      }

      const opsRes = await fetch('/api/ops-status');
      if (opsRes.ok) {
        const opsJson = await opsRes.json();
        setOpsStatus(opsJson);
      }

    } catch (error) {
      console.error('Status fetch error:', error);
    } finally {
      setLoading(false);
    }
  };

  const triggerTraining = async () => {
    setTraining(true);
    try {
      const res = await fetch('/api/training/control', { method: 'POST' });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.error || 'Falha ao iniciar treino');
      }
      const payload = await res.json();
      setFeedback(`✅ Treino iniciado (PID ${payload?.pid ?? '-'})`);
      setTimeout(fetchStatus, 800);
    } catch (error) {
      console.error('Training trigger error:', error);
    } finally {
      setTraining(false);
    }
  };

  const toggleScanner = async () => {
    setToggling(true);
    try {
      const action = scannerActive ? 'stop' : 'start';
      const res = await fetch('/api/scanner/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      
      if (res.ok) {
        const json = await res.json();
        // Optimistic update or wait for next poll? Better wait or check immediately.
        // Let's check immediately
        setTimeout(fetchStatus, 1000); 
        setScannerActive(action === 'start');
        setFeedback(`✅ Scanner ${action === 'start' ? 'iniciado' : 'parado'}${json?.pid ? ` (PID ${json.pid})` : ''}`);
      }
    } catch (error) {
      console.error('Toggle error:', error);
    } finally {
      setToggling(false);
    }
  };

  const triggerDateRangeUpdate = async () => {
    if (!startDate || !endDate) {
      console.error('Selecione data inicial e final');
      return;
    }
    setUpdating(true);
    try {
      const res = await fetch('/api/update/all-leagues/date-range', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_date: startDate, end_date: endDate }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.error || 'Falha ao iniciar atualização por intervalo');
      }
      const payload = await res.json();
      setFeedback(`✅ Atualização iniciada (PID ${payload?.pid ?? '-'})`);
      setTimeout(fetchStatus, 800);
    } catch (error) {
      console.error('Date-range update error:', error);
    } finally {
      setUpdating(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Check every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) return null;

  // Calculate if status is "fresh" (< 5 mins)
  const lastUpdate = data ? new Date(data.last_updated) : new Date();
  const now = new Date();
  // Simple diff (assuming same timezone context or forgiving logic)
  const diffMinutes = data ? (now.getTime() - new Date(data.last_updated.replace(" ", "T")).getTime()) / 60000 : 0;
  
  let statusColor = "bg-emerald-500";
  let statusText = "Online";
  
  if (!data || data.status === 'error') {
    statusColor = "bg-red-500";
    statusText = "Error";
  } else if (diffMinutes > 10) {
    statusColor = "bg-yellow-500";
    statusText = "Stalled"; 
  } else if (!scannerActive && diffMinutes > 2) {
    statusColor = "bg-slate-500";
    statusText = "Offline";
  }

  const isCompact = mode === 'compact';

  return (
    <div className="flex items-center gap-4">
      {/* Scanner Toggle */}
      <div className="flex items-center gap-2 bg-slate-900/50 px-2 py-1 rounded-full border border-slate-800">
        <span className="text-[10px] uppercase font-bold text-slate-400 pl-1">Scanner</span>
        <button
          onClick={toggleScanner}
          disabled={toggling}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900 ${
            scannerActive ? 'bg-blue-600' : 'bg-slate-700'
          }`}
        >
          <span
            className={`${
              scannerActive ? 'translate-x-4.5' : 'translate-x-0.5'
            } inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200`}
            style={{ transform: scannerActive ? 'translateX(18px)' : 'translateX(2px)' }}
          />
        </button>
      </div>

      {!isCompact && <button
        onClick={triggerTraining}
        disabled={training}
        className="px-3 py-1.5 text-[10px] uppercase font-bold tracking-wide rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-300 hover:bg-violet-500/20 disabled:opacity-50"
        title="Executa o treino do modelo (equivalente à opção 2 do CLI)"
      >
        {training ? 'Treinando...' : 'Treinar IA'}
      </button>}

      {!isCompact && <div className="flex items-center gap-2 bg-slate-900/50 px-2 py-1 rounded-full border border-slate-800">
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          className="bg-slate-800 text-slate-200 text-[10px] rounded px-1 py-0.5 border border-slate-700"
          title="Data inicial"
        />
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          className="bg-slate-800 text-slate-200 text-[10px] rounded px-1 py-0.5 border border-slate-700"
          title="Data final"
        />
        <button
          onClick={triggerDateRangeUpdate}
          disabled={updating}
          className="px-2 py-1 text-[10px] uppercase font-bold rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 disabled:opacity-50"
          title="Atualizar todas as ligas por intervalo de datas"
        >
          {updating ? 'Atualizando...' : 'Atualizar Ligas'}
        </button>
      </div>}

      {/* System Status Indicator */}
      <div className="flex items-center gap-3 px-3 py-1.5 bg-slate-900/50 rounded-full border border-slate-800">
        <div className="relative flex h-2.5 w-2.5">
          {scannerActive && <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${statusColor}`}></span>}
          <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${statusColor}`}></span>
        </div>
        <div className="flex flex-col leading-none">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
            System {statusText}
          </span>
          {data && (
            <span className="text-[10px] text-slate-500 font-mono">
              Last: {data.last_updated.split(' ')[1] || data.last_updated}
            </span>
          )}
        </div>
        {data && data.live_matches > 0 && (
          <div className="ml-2 px-1.5 py-0.5 bg-red-500/10 border border-red-500/20 rounded text-[10px] font-bold text-red-400 animate-pulse">
            {data.live_matches} LIVE
          </div>
        )}
      </div>
      {!isCompact && <div className="flex flex-col gap-1 text-[10px] text-slate-400 bg-slate-900/40 border border-slate-800 rounded-lg px-2 py-1">
        <span>Últ. treino: {opsStatus?.last_training_at ? new Date(opsStatus.last_training_at).toLocaleString() : 'N/A'}</span>
        <span>Últ. scanner: {opsStatus?.last_scanner_at ? `${opsStatus.last_scanner_action || ''} em ${new Date(opsStatus.last_scanner_at).toLocaleString()}` : 'N/A'}</span>
        <span>Últ. atualização: {opsStatus?.last_update_at ? `${opsStatus.last_update_range || ''} em ${new Date(opsStatus.last_update_at).toLocaleString()}` : 'N/A'}</span>
        {feedback && <span className="text-emerald-300">{feedback}</span>}
      </div>}
    </div>
  );
}
