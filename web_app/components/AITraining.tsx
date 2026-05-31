"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Brain,
  Play,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
  CalendarCheck,
  Cpu,
  BarChart3,
  AlertTriangle,
  Zap,
  Calendar,
} from "lucide-react";

interface TrainingStatus {
  last_trained_at: string | null;
  last_started_at: string | null;
  last_scanner_run: string | null;
  status: string;
  oof_metrics: Record<string, number | string>;
  config: Record<string, number | string>;
  training_in_progress: boolean;
  next_auto_retrain: { date: string; days_left: number } | null;
  model_files: string[];
}

function formatDate(iso: string | null): string {
  if (!iso) return "Nunca";
  try {
    const d = new Date(iso);
    return d.toLocaleString("pt-BR", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function timeAgo(iso: string | null): string {
  if (!iso) return "";
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const days = Math.floor(diff / 86400000);
    const hours = Math.floor((diff % 86400000) / 3600000);
    const mins = Math.floor((diff % 3600000) / 60000);
    if (days > 0) return `há ${days}d ${hours}h`;
    if (hours > 0) return `há ${hours}h ${mins}min`;
    return `há ${mins}min`;
  } catch {
    return "";
  }
}

function StatusPill({ status }: { status: string }) {
  if (status === "success")
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-900/40 text-emerald-400 border border-emerald-500/30">
        <CheckCircle2 size={11} /> Concluído
      </span>
    );
  if (status === "running")
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-blue-900/40 text-blue-400 border border-blue-500/30 animate-pulse">
        <Loader2 size={11} className="animate-spin" /> Treinando...
      </span>
    );
  if (status === "error")
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-red-900/40 text-red-400 border border-red-500/30">
        <XCircle size={11} /> Erro
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-slate-800 text-slate-400 border border-slate-700">
      <AlertTriangle size={11} /> Não treinado
    </span>
  );
}

export default function AITraining() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [feedbackType, setFeedbackType] = useState<"success" | "error" | "info">("info");

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/training");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStatus(data);
    } catch (err: any) {
      console.error("Training status error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    // Poll every 10s while training is in progress
    const interval = setInterval(() => {
      fetchStatus();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleTrainNow = async () => {
    setStarting(true);
    setFeedback(null);
    try {
      const res = await fetch("/api/training", { method: "POST" });
      const data = await res.json();
      if (data.success) {
        setFeedback("🧬 Treino iniciado! O modelo será atualizado em background. Esta página atualiza automaticamente.");
        setFeedbackType("success");
        fetchStatus();
      } else {
        setFeedback(data.message || data.error || "Não foi possível iniciar o treino.");
        setFeedbackType(data.training_in_progress ? "info" : "error");
      }
    } catch (err: any) {
      setFeedback(err.message || "Erro ao conectar ao backend.");
      setFeedbackType("error");
    } finally {
      setStarting(false);
    }
  };

  const isTraining = status?.training_in_progress || status?.status === "running";

  const feedbackColors = {
    success: "bg-emerald-900/30 border-emerald-500/40 text-emerald-300",
    error: "bg-red-900/30 border-red-500/40 text-red-300",
    info: "bg-blue-900/30 border-blue-500/40 text-blue-300",
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Page Header */}
      <div className="flex items-center gap-3 mb-2">
        <div className="p-2 bg-purple-900/30 rounded-xl border border-purple-500/30">
          <Brain size={22} className="text-purple-400" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-white">AI Training Center</h2>
          <p className="text-slate-500 text-sm">
            Gerencie o treino do modelo científico Joint Corners
          </p>
        </div>
        <button
          onClick={fetchStatus}
          className="ml-auto p-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 transition-all"
          title="Atualizar status"
        >
          <RefreshCw size={15} />
        </button>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-24">
          <Loader2 size={32} className="animate-spin text-purple-500" />
        </div>
      )}

      {!loading && (
        <>
          {/* Feedback Banner */}
          {feedback && (
            <div
              className={`rounded-xl border p-4 text-sm font-medium ${feedbackColors[feedbackType]}`}
            >
              {feedback}
            </div>
          )}

          {/* Main Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

            {/* Card: Último Treino */}
            <div className="bg-slate-800/60 rounded-2xl border border-slate-700/50 p-5 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock size={15} className="text-slate-400" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                    Último Treino
                  </span>
                </div>
                {status && <StatusPill status={status.status} />}
              </div>

              <div>
                <p className="text-2xl font-bold text-white">
                  {formatDate(status?.last_trained_at ?? null)}
                </p>
                {status?.last_trained_at && (
                  <p className="text-slate-500 text-xs mt-1">
                    {timeAgo(status.last_trained_at)}
                  </p>
                )}
              </div>

              {/* Config info */}
              {status?.config && Object.keys(status.config).length > 0 && (
                <div className="pt-2 border-t border-slate-700/50 flex flex-wrap gap-3">
                  {Object.entries(status.config).map(([k, v]) => (
                    <span key={k} className="text-xs text-slate-400 bg-slate-900/50 px-2 py-0.5 rounded">
                      <span className="text-slate-500">{k}: </span>
                      <span className="text-slate-300 font-mono">{String(v)}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Card: Próximo Retrain Automático */}
            <div className="bg-slate-800/60 rounded-2xl border border-slate-700/50 p-5 flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <CalendarCheck size={15} className="text-indigo-400" />
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                  Retrain Automático (15 dias)
                </span>
              </div>

              {status?.next_auto_retrain ? (
                <>
                  <div>
                    <p className="text-2xl font-bold text-white">
                      {status.next_auto_retrain.date}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {status.next_auto_retrain.days_left === 0
                        ? "🔥 Retrain disponível hoje!"
                        : `Em ${status.next_auto_retrain.days_left} dia${status.next_auto_retrain.days_left !== 1 ? "s" : ""}`}
                    </p>
                  </div>
                  {/* Progress bar */}
                  <div>
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>Ciclo de 15 dias</span>
                      <span>{Math.max(0, 15 - (status.next_auto_retrain.days_left))}/15d</span>
                    </div>
                    <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-600 to-purple-500 rounded-full transition-all"
                        style={{
                          width: `${Math.min(100, ((15 - Math.max(0, status.next_auto_retrain.days_left)) / 15) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                </>
              ) : (
                <p className="text-slate-500 text-sm">
                  Treine o modelo pela primeira vez para ativar o ciclo automático.
                </p>
              )}

              {/* Last scanner run */}
              <div className="pt-2 border-t border-slate-700/50">
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <Zap size={11} className="text-yellow-400" />
                  <span>Último Scanner Automático:</span>
                  <span className="text-slate-300 ml-1">
                    {status?.last_scanner_run ? formatDate(status.last_scanner_run) : "Nunca"}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Card: OOF Metrics */}
          {status?.oof_metrics && Object.keys(status.oof_metrics).length > 0 && (
            <div className="bg-slate-800/60 rounded-2xl border border-slate-700/50 p-5">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 size={15} className="text-blue-400" />
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                  Métricas Walk-Forward (OOF)
                </span>
                <span className="text-[10px] bg-blue-900/30 text-blue-300 px-1.5 py-0.5 rounded border border-blue-500/30 ml-auto">
                  Joint Model v1
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {Object.entries(status.oof_metrics).map(([k, v]) => (
                  <div key={k} className="bg-slate-900/50 rounded-lg p-3 text-center">
                    <p className="text-xs text-slate-500 mb-1 truncate" title={k}>{k}</p>
                    <p className="text-sm font-bold font-mono text-white">
                      {typeof v === "number" ? v.toFixed(4) : String(v)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Card: Model Files */}
          {status?.model_files && status.model_files.length > 0 && (
            <div className="bg-slate-800/60 rounded-2xl border border-slate-700/50 p-5">
              <div className="flex items-center gap-2 mb-4">
                <Cpu size={15} className="text-emerald-400" />
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                  Arquivos de Modelo
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {status.model_files.map((f) => (
                  <span
                    key={f}
                    className="text-xs font-mono bg-emerald-900/20 text-emerald-300 border border-emerald-500/20 px-2.5 py-1 rounded-lg"
                  >
                    📦 {f}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Train Button Card */}
          <div className="bg-gradient-to-br from-purple-900/20 to-indigo-900/20 rounded-2xl border border-purple-500/30 p-6">
            <div className="flex flex-col md:flex-row items-start md:items-center gap-5">
              <div className="flex-1">
                <h3 className="text-white font-bold text-base mb-1 flex items-center gap-2">
                  <Brain size={16} className="text-purple-400" />
                  Treino Multimercado Científico
                  <span className="text-[10px] bg-purple-900/50 text-purple-300 px-1.5 py-0.5 rounded border border-purple-500/30">
                    Joint Model
                  </span>
                </h3>
                <p className="text-slate-400 text-sm">
                  Equivalente ao <span className="text-white font-mono">cli.py → opção 2</span> (TREINO MULTIMERCADO CIENTÍFICO).
                  Treina 4 targets <span className="font-mono text-blue-300">[h1H, a1H, h2H, a2H]</span> com
                  walk-forward temporal e calibração por família.
                  Configuração: <span className="font-mono text-slate-300">5 folds · seed=42 · 10.000 MC</span>
                </p>

                {isTraining && (
                  <div className="mt-3 flex items-center gap-2 text-blue-400 text-sm">
                    <Loader2 size={14} className="animate-spin" />
                    <span>Treinamento em andamento... Esta página atualiza a cada 10s.</span>
                  </div>
                )}
              </div>

              <button
                onClick={handleTrainNow}
                disabled={isTraining || starting}
                className={`flex-shrink-0 flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-sm transition-all shadow-lg ${
                  isTraining || starting
                    ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-purple-900/30 hover:shadow-purple-500/20 active:scale-95"
                }`}
              >
                {isTraining || starting ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    {isTraining ? "Treinando..." : "Iniciando..."}
                  </>
                ) : (
                  <>
                    <Play size={16} />
                    🧬 Treinar Agora
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Automation Info Banner */}
          <div className="bg-slate-800/40 rounded-2xl border border-slate-700/40 p-4 flex gap-3">
            <div className="flex-shrink-0 p-2 bg-yellow-900/20 rounded-lg border border-yellow-500/20 self-start">
              <Calendar size={15} className="text-yellow-400" />
            </div>
            <div className="text-xs text-slate-400 space-y-1">
              <p className="font-semibold text-slate-300">⚙️ Automações Ativas (enquanto o servidor estiver rodando)</p>
              <p>
                <span className="text-yellow-400 font-semibold">📡 Scanner Diário:</span>{" "}
                Executa automaticamente 1x por dia para analisar os jogos do dia.
              </p>
              <p>
                <span className="text-purple-400 font-semibold">🧬 Retrain a cada 15 dias:</span>{" "}
                O modelo é retreinado automaticamente ao completar 15 dias do último treino.
              </p>
              <p className="text-slate-500">
                ℹ️ Ambas as rotinas rodam em background e verificam a cada hora se é necessário executar.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
