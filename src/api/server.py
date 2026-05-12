import asyncio
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict
import json
from datetime import datetime, timezone

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
from src.data.updater import update_all_leagues_by_date_range_values
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


class UpdateRangeRequest(BaseModel):
    start_date: str
    end_date: str


def _open_db_cursor() -> tuple[DBManager, Any, Any]:
    """Create DBManager, connection, and cursor for API operations."""
    db = DBManager()
    conn = db.connect()
    cursor = conn.cursor()
    return db, conn, cursor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCANNER_PID_FILE = PROJECT_ROOT / "web_app" / ".scanner.pid"
SCANNER_SCRIPT = PROJECT_ROOT / "scripts" / "quick_scan.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_model.py"
OPS_STATUS_FILE = PROJECT_ROOT / "data" / "ops_status.json"


def _load_ops_status() -> Dict[str, Any]:
    if not OPS_STATUS_FILE.exists():
        return {}
    try:
        return json.loads(OPS_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ops_status(patch: Dict[str, Any]) -> None:
    current = _load_ops_status()
    current.update(patch)
    OPS_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPS_STATUS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


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
    
    yield
    # Cleanup code can go here if needed

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
        _save_ops_status(
            {
                "last_scanner_action": "start",
                "last_scanner_pid": int(new_pid),
                "last_scanner_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"message": "Scanner started", "status": "started", "pid": new_pid}

    if pid:
        _stop_scanner_loop_process(pid)
    _save_ops_status(
        {
            "last_scanner_action": "stop",
            "last_scanner_pid": int(pid) if pid else None,
            "last_scanner_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _remove_scanner_pid_file()
    return {"message": "Scanner stopped", "status": "stopped"}


@app.post("/api/training/control")
async def post_training_control() -> Dict[str, Any]:
    """Trigger model training (equivalent to CLI option 2) in background."""
    python_executable = sys.executable
    try:
        process = subprocess.Popen(
            [python_executable, str(TRAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),
        )
        _save_ops_status(
            {
                "last_training_pid": int(process.pid),
                "last_training_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"message": "Training started", "status": "started", "pid": int(process.pid)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/update/all-leagues/date-range")
async def post_update_all_leagues_date_range(request: UpdateRangeRequest) -> Dict[str, Any]:
    """Run all-leagues update in background for a provided date range."""
    try:
        from datetime import datetime
        datetime.strptime(request.start_date, "%Y-%m-%d")
        datetime.strptime(request.end_date, "%Y-%m-%d")
        if request.start_date > request.end_date:
            raise HTTPException(status_code=400, detail="start_date must be <= end_date")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD") from exc

    python_executable = sys.executable
    inline = (
        "from src.data.updater import update_all_leagues_by_date_range_values;"
        f"update_all_leagues_by_date_range_values('{request.start_date}','{request.end_date}')"
    )
    try:
        process = subprocess.Popen(
            [python_executable, "-c", inline],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),
        )
        _save_ops_status(
            {
                "last_update_pid": int(process.pid),
                "last_update_at": datetime.now(timezone.utc).isoformat(),
                "last_update_range": f"{request.start_date} -> {request.end_date}",
            }
        )
        return {"message": "All-leagues update started", "status": "started", "pid": int(process.pid)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/ops-status")
async def get_ops_status() -> Dict[str, Any]:
    """Return latest operational actions for scanner/training/updater."""
    return _load_ops_status()


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

if __name__ == "__main__":
    # Run slightly different config for direct execution
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
