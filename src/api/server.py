import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from src.analysis.performance_calculator import get_performance_data
from src.analysis.bet_validator import validate_pending_bets
from src.analysis.manager_ai import ManagerAI
from src.analysis.unified_scanner import scan_opportunities_core
from src.database.db_manager import DBManager
from src.monitoring.model_health import get_model_health_snapshot
from src.web.bankroll_api import (
    auth_user,
    delete_bet,
    get_bet_history,
    get_current_balance,
    get_leaderboard,
    get_public_feed,
    get_stats,
    manage_funds,
    place_bet,
)

from web_app.lib.dashboard_data import DashboardDataProvider


from contextlib import asynccontextmanager

# Global Provider Instance
provider = None

# Training state
_training_in_progress = False
_training_lock = threading.Lock()


class AuthRequest(BaseModel):
    """Request model for user authentication."""

    username: str
    password: str


class TransactionRequest(BaseModel):
    """Request model for bankroll deposit and withdrawal operations."""

    userId: int
    type: str
    amount: float


class ScannerRunRequest(BaseModel):
    """Request model for one-shot scanner execution."""

    date: str = "today"


class ScannerControlRequest(BaseModel):
    """Request model for scanner loop process control."""

    action: str


def _open_db_cursor() -> tuple[DBManager, Any, Any]:
    """Create DBManager, connection, and cursor for API operations."""
    db = DBManager()
    conn = db.connect()
    cursor = conn.cursor()
    return db, conn, cursor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCANNER_PID_FILE = PROJECT_ROOT / "web_app" / ".scanner.pid"
SCANNER_SCRIPT = PROJECT_ROOT / "scripts" / "quick_scan.py"
TRAINING_LOG_FILE = PROJECT_ROOT / "data" / "training_log.json"


# ---------------------------------------------------------------------------
# Training log helpers
# ---------------------------------------------------------------------------

def _read_training_log() -> dict:
    """Load training log from JSON file; return empty dict if missing."""
    try:
        if TRAINING_LOG_FILE.exists():
            return json.loads(TRAINING_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_training_log(data: dict) -> None:
    """Persist training log dict to JSON file."""
    TRAINING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_joint_training_sync() -> dict:
    """Run JointTrainer with default params (no interactive input).
    
    Called from a background thread so the API remains responsive.
    Returns a summary dict saved to the training log.
    """
    global _training_in_progress
    brt = timezone(timedelta(hours=-3))
    started_at = datetime.now(tz=brt).isoformat()
    result = {"started_at": started_at, "status": "running"}
    _write_training_log({**_read_training_log(), **result, "last_started_at": started_at})

    try:
        from src.database.db_manager import DBManager as _DBManager
        from src.features.feature_store import FeatureStore
        from src.training.joint_trainer import JointTrainer

        db = _DBManager()
        try:
            df_history = db.get_historical_data()
            if df_history is None or len(df_history) < 200:
                result["status"] = "error"
                result["error"] = "Histórico insuficiente (< 200 jogos)"
                return result

            feature_store = FeatureStore(db)
            trainer = JointTrainer(n_splits=5, n_simulations=10_000, random_state=42)
            report = trainer.run(df_history, feature_store)

            finished_at = datetime.now(tz=brt).isoformat()
            oof = report.get("oof_metrics", {})
            result = {
                "status": "success",
                "last_trained_at": finished_at,
                "last_started_at": started_at,
                "oof_metrics": {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in oof.items()
                },
                "config": {"n_splits": 5, "random_state": 42, "n_simulations": 10_000},
            }
        finally:
            db.close()
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        _write_training_log(result)
        with _training_lock:
            _training_in_progress = False

    return result


async def _auto_scheduler_loop() -> None:
    """Background coroutine: runs scanner daily + retrains AI every 15 days."""
    brt = timezone(timedelta(hours=-3))
    print("🕐 Auto-scheduler started (daily scanner + 15-day AI retrain)")

    while True:
        try:
            now = datetime.now(tz=brt)
            log = _read_training_log()

            # ── Daily scanner ──────────────────────────────────────────────
            last_scan_str = log.get("last_scanner_run")
            should_scan = True
            if last_scan_str:
                try:
                    last_scan_dt = datetime.fromisoformat(last_scan_str)
                    # Run scanner once per calendar day (BRT)
                    if last_scan_dt.date() >= now.date():
                        should_scan = False
                except Exception:
                    pass

            if should_scan:
                print(f"📡 Auto-scanner: running for {now.strftime('%Y-%m-%d')}...")
                try:
                    db = DBManager()
                    try:
                        results = await asyncio.to_thread(
                            scan_opportunities_core,
                            date_str=now.strftime("%Y-%m-%d"),
                            db=db,
                            manager=None,
                            verbose=False,
                        )
                        processed = len(results or [])
                        print(f"📡 Auto-scanner: {processed} matches processed.")
                    finally:
                        db.close()

                    log = _read_training_log()
                    log["last_scanner_run"] = now.isoformat()
                    _write_training_log(log)

                    # Validate finished predictions after auto-scan
                    try:
                        db2 = DBManager()
                        try:
                            await asyncio.to_thread(db2.check_predictions)
                            print("📡 Auto-scanner: predictions validated (GREEN/RED).")
                        finally:
                            db2.close()
                    except Exception as val_exc:
                        print(f"📡 Auto-scanner: validation error: {val_exc}")
                except Exception as exc:
                    print(f"📡 Auto-scanner error: {exc}")

            # ── 15-day AI retrain ──────────────────────────────────────────
            global _training_in_progress
            last_train_str = log.get("last_trained_at")
            should_retrain = False
            if not last_train_str:
                # Never trained — schedule after 1 day to avoid immediate heavy load at startup
                should_retrain = False
            else:
                try:
                    last_train_dt = datetime.fromisoformat(last_train_str)
                    days_since = (now - last_train_dt).days
                    if days_since >= 15:
                        should_retrain = True
                except Exception:
                    pass

            if should_retrain:
                with _training_lock:
                    if not _training_in_progress:
                        _training_in_progress = True
                        print("🧬 Auto-retrain: 15-day interval reached — starting training in background...")
                        threading.Thread(
                            target=_run_joint_training_sync, daemon=True
                        ).start()

        except Exception as exc:
            print(f"⚠️ Auto-scheduler error: {exc}")

        # Check every hour
        await asyncio.sleep(3600)


def _is_pid_running(pid: int) -> bool:
    """Check if a process PID is alive in current OS."""
    if pid <= 0:
        return False

    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "").lower()
        return "no tasks are running" not in output and str(pid) in output

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_scanner_pid() -> int | None:
    """Read scanner PID from pidfile when available and valid."""
    if not SCANNER_PID_FILE.exists():
        return None

    try:
        return int(SCANNER_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _remove_scanner_pid_file() -> None:
    """Delete scanner pidfile if it exists."""
    if SCANNER_PID_FILE.exists():
        SCANNER_PID_FILE.unlink()


def _start_scanner_loop_process() -> int:
    """Start quick scanner loop process and persist its PID."""
    python_executable = sys.executable

    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        process = subprocess.Popen(
            [python_executable, str(SCANNER_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    else:
        process = subprocess.Popen(
            [python_executable, str(SCANNER_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    SCANNER_PID_FILE.write_text(str(process.pid), encoding="utf-8")
    return int(process.pid)


def _stop_scanner_loop_process(pid: int) -> None:
    """Stop scanner loop process identified by PID."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    try:
        os.kill(pid, 15)
    except OSError:
        pass


def _resolve_scan_date(date_value: str) -> str:
    """Normalize scanner date aliases to YYYY-MM-DD values."""
    from datetime import datetime, timedelta

    normalized = (date_value or "today").strip().lower()
    if normalized == "today":
        return datetime.now().strftime("%Y-%m-%d")
    if normalized == "tomorrow":
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    return date_value

@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider
    print("Initializing Cortex Bet Data Provider...")
    try:
        from web_app.lib.dashboard_data import DashboardDataProvider
        # Initialize
        provider = DashboardDataProvider() 
        print("Data Provider Ready! Models Loaded.")
    except Exception as e:
        print(f"Failed to initialize provider: {e}")
        import traceback
        traceback.print_exc()

    # Start the background auto-scheduler
    scheduler_task = asyncio.create_task(_auto_scheduler_loop())
    
    yield

    # Cleanup: cancel the scheduler
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Cortex Bet API", version="1.0.0", lifespan=lifespan)

# Enable CORS for Next.js (Port 3000) and Streamlit (8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Return health status for the official HTTP backend."""
    return {"status": "ok", "provider_loaded": provider is not None}

@app.get("/api/predictions")
async def get_predictions(
    date: str = 'today',
    league: str = 'all',
    status: str = 'all',
    top7_only: bool = False,
    sort_by: str = 'confidence'
):
    """Return pre-live predictions from the canonical dashboard provider."""
    if not provider:
         raise HTTPException(status_code=503, detail="Provider not initialized")
    
    try:
        # The provider logic is synchronous, identifying N+1 bottleneck
        # In a real async microservice, we'd run this in a threadpool if it blocks,
        # but for now, just removing the process-spawn overhead is the big win.
        data = provider.get_predictions_with_reasoning(
            date_str=date,
            league=league,
            status=status,
            top7_only=top7_only,
            sort_by=sort_by
        )
        return data
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth")
async def post_auth(request: AuthRequest) -> Dict[str, Any]:
    """Authenticate user credentials using the canonical bankroll domain service."""
    db, conn, cursor = _open_db_cursor()
    try:
        result = auth_user(cursor, request.username, request.password)
        if result.get("error"):
            raise HTTPException(status_code=401, detail=result["error"])
        return result
    finally:
        conn.close()
        db.close()


@app.get("/api/feed")
async def get_feed(limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
    """Return public social betting feed from backend domain services."""
    db, conn, cursor = _open_db_cursor()
    try:
        return {"feed": get_public_feed(cursor, limit)}
    finally:
        conn.close()
        db.close()


@app.get("/api/leaderboard")
async def get_leaderboard_data() -> Dict[str, Any]:
    """Return ranking summary across all users based on validated bets."""
    db, conn, cursor = _open_db_cursor()
    try:
        return {"leaderboard": get_leaderboard(cursor)}
    finally:
        conn.close()
        db.close()


@app.get("/api/performance")
async def get_performance(
    from_date: str | None = None,
    to_date: str | None = None,
) -> Dict[str, Any]:
    """Return performance analytics for model evaluation and monitoring."""
    try:
        return get_performance_data(from_date, to_date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/model-health")
async def get_model_health() -> Dict[str, Any]:
    """Return online calibration/drift alerts and active champion metadata."""
    try:
        return get_model_health_snapshot()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/bankroll")
async def get_bankroll(type: str = "all", user_id: int = 1) -> Dict[str, Any]:
    """Return bankroll balance, history, and user stats from canonical services."""
    db, conn, cursor = _open_db_cursor()
    try:
        if type == "balance":
            return {"balance": get_current_balance(cursor, user_id)}
        if type == "history":
            return {"bets": get_bet_history(cursor, user_id)}

        return {
            "balance": get_current_balance(cursor, user_id),
            "bets": get_bet_history(cursor, user_id),
            "stats": get_stats(cursor, user_id),
        }
    finally:
        conn.close()
        db.close()


@app.post("/api/bankroll")
async def post_bankroll(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle bet placement and bankroll transactions through HTTP backend."""
    db, conn, cursor = _open_db_cursor()
    try:
        if payload.get("action") == "transaction":
            request = TransactionRequest(**payload)
            tx_type = request.type.upper()
            result = manage_funds(cursor, conn, request.userId, request.amount, tx_type)
        else:
            if "userId" not in payload:
                payload["userId"] = 1
            result = place_bet(cursor, conn, payload, int(payload["userId"]))

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    finally:
        conn.close()
        db.close()


@app.delete("/api/bankroll")
async def delete_bankroll_bet(id: int, user_id: int = 1) -> Dict[str, Any]:
    """Delete an open bet and process refund logic through canonical backend."""
    db, conn, cursor = _open_db_cursor()
    try:
        result = delete_bet(cursor, conn, id, user_id)
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    finally:
        conn.close()
        db.close()


@app.post("/api/scanner")
async def run_scanner(request: ScannerRunRequest) -> Dict[str, Any]:
    """Run one scanner cycle without spawning subprocess in frontend routes."""
    target_date = _resolve_scan_date(request.date)

    db = DBManager()
    manager = None
    try:
        try:
            manager = ManagerAI(db)
        except Exception:
            manager = None

        results = await asyncio.to_thread(
            scan_opportunities_core,
            date_str=target_date,
            db=db,
            manager=manager,
            verbose=False,
        )
        processed = len(results or [])

        # Automatically validate finished predictions after scanning
        try:
            await asyncio.to_thread(db.check_predictions)
        except Exception as val_exc:
            print(f"⚠️ check_predictions after scan error: {val_exc}")

        return {
            "success": True,
            "message": "Scanner completed successfully",
            "matchesProcessed": processed,
            "matches_processed": processed,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


@app.get("/api/scanner/control")
async def get_scanner_control_status() -> Dict[str, Any]:
    """Return scanner loop status based on PID state."""
    pid = _read_scanner_pid()
    if not pid:
        return {"active": False}

    active = _is_pid_running(pid)
    if not active:
        _remove_scanner_pid_file()
        return {"active": False}

    return {"active": True, "pid": pid}


@app.post("/api/scanner/control")
async def post_scanner_control(request: ScannerControlRequest) -> Dict[str, Any]:
    """Start/stop scanner loop and keep response contract for UI controls."""
    action = request.action.strip().lower()
    if action not in {"start", "stop", "status"}:
        raise HTTPException(status_code=400, detail="Invalid action")

    if action == "status":
        return await get_scanner_control_status()

    pid = _read_scanner_pid()
    is_active = bool(pid and _is_pid_running(pid))

    if action == "start":
        if is_active:
            return {"message": "Scanner already running", "status": "running", "pid": pid}

        new_pid = _start_scanner_loop_process()
        return {"message": "Scanner started", "status": "started", "pid": new_pid}

    if pid:
        _stop_scanner_loop_process(pid)
    _remove_scanner_pid_file()
    return {"message": "Scanner stopped", "status": "stopped"}


@app.get("/api/system-status")
async def get_system_status() -> Dict[str, Any]:
    """Expose canonical dashboard system status via official HTTP backend."""
    current_provider = provider or DashboardDataProvider()
    data = current_provider.get_system_status()
    if data.get("status") == "error":
        raise HTTPException(status_code=500, detail=data.get("error", "Unknown error"))
    return data


@app.post("/api/validate-bets")
async def post_validate_bets() -> Dict[str, Any]:
    """Validate pending bets and return processing summary."""
    try:
        validated_count = await asyncio.to_thread(validate_pending_bets)
        return {
            "success": True,
            "validated_count": int(validated_count),
            "message": "Validation complete",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/validate-predictions")
async def post_validate_predictions() -> Dict[str, Any]:
    """Run check_predictions() to mark finished match predictions as GREEN/RED."""
    db = DBManager()
    try:
        await asyncio.to_thread(db.check_predictions)
        return {
            "success": True,
            "message": "Predictions validated (GREEN/RED updated).",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


@app.get("/api/training/status")
async def get_training_status() -> Dict[str, Any]:
    """Return last training timestamp, metrics, and whether training is running."""
    log = _read_training_log()
    brt = timezone(timedelta(hours=-3))
    now = datetime.now(tz=brt)

    next_retrain = None
    last_train_str = log.get("last_trained_at")
    if last_train_str:
        try:
            last_train_dt = datetime.fromisoformat(last_train_str)
            next_dt = last_train_dt + timedelta(days=15)
            days_left = max(0, (next_dt.date() - now.date()).days)
            next_retrain = {
                "date": next_dt.strftime("%Y-%m-%d"),
                "days_left": days_left,
            }
        except Exception:
            pass

    # Check if model files exist
    models_dir = PROJECT_ROOT / "models"
    model_files = [
        f.name for f in models_dir.glob("*.joblib")
    ] if models_dir.exists() else []

    return {
        "last_trained_at": log.get("last_trained_at"),
        "last_started_at": log.get("last_started_at"),
        "last_scanner_run": log.get("last_scanner_run"),
        "status": log.get("status", "never_trained"),
        "oof_metrics": log.get("oof_metrics", {}),
        "config": log.get("config", {}),
        "training_in_progress": _training_in_progress,
        "next_auto_retrain": next_retrain,
        "model_files": model_files,
    }


@app.post("/api/training/run")
async def post_training_run() -> Dict[str, Any]:
    """Trigger joint model training in a background thread (non-blocking)."""
    global _training_in_progress
    with _training_lock:
        if _training_in_progress:
            return {
                "success": False,
                "message": "Treino já em andamento. Aguarde a conclusão.",
                "training_in_progress": True,
            }
        _training_in_progress = True

    threading.Thread(target=_run_joint_training_sync, daemon=True).start()
    return {
        "success": True,
        "message": "Treino iniciado em background. Verifique o status em /api/training/status.",
        "training_in_progress": True,
    }


if __name__ == "__main__":
    # Run slightly different config for direct execution
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
