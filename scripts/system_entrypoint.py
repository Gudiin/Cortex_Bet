"""Operational entrypoint for starting the official local stack.

This module centralizes stack orchestration for API, scanner, and web frontend.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import List


API_SCRIPT = "src/api/server.py"
SCANNER_SCRIPT = "scripts/quick_scan.py"
WEB_WORKDIR = "web_app"
WEB_COMMAND = ["npm", "run", "dev"]
SCANNER_PID_FILE = os.path.join(WEB_WORKDIR, ".scanner.pid")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_local_ip() -> str:
    """Return the machine local IP used to expose local services in LAN."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip_address = sock.getsockname()[0]
        sock.close()
        return ip_address
    except Exception:
        return "127.0.0.1"


def write_scanner_pid(pid: int) -> None:
    """Persist the scanner PID for web controls that track scanner state."""
    with open(SCANNER_PID_FILE, "w", encoding="utf-8") as file_handle:
        file_handle.write(str(pid))


def start_stack_processes(python_executable: str) -> List[subprocess.Popen]:
    """Start API, scanner, and web frontend subprocesses for local stack execution."""
    processes: List[subprocess.Popen] = []

    print("Starting API server on port 8000...")
    api_process = subprocess.Popen(
        [python_executable, "-m", "src.api.server"],
        cwd=str(PROJECT_ROOT),
    )
    processes.append(api_process)
    print(f"API PID: {api_process.pid}")

    time.sleep(5)

    print("Starting quick scanner...")
    scanner_process = subprocess.Popen(
        [python_executable, SCANNER_SCRIPT],
        cwd=str(PROJECT_ROOT),
    )
    processes.append(scanner_process)
    write_scanner_pid(scanner_process.pid)

    print("Starting web frontend on port 8501...")
    web_process = subprocess.Popen(WEB_COMMAND, cwd=WEB_WORKDIR, shell=True)
    processes.append(web_process)

    return processes


def stop_processes(processes: List[subprocess.Popen]) -> None:
    """Terminate all processes created by stack orchestration."""
    for process in processes:
        try:
            process.terminate()
        except Exception:
            pass


def run_system_stack(python_executable: str | None = None) -> None:
    """Run the official local stack orchestration loop until interrupted."""
    executable = python_executable or sys.executable

    print("Starting local stack...")
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")

    processes: List[subprocess.Popen] = []

    try:
        processes = start_stack_processes(executable)
        api_process = processes[0]

        print("Services online:")
        print(f"- API: http://{local_ip}:8000")
        print(f"- Web: http://{local_ip}:8501")
        print("Press CTRL+C to stop services.")

        while True:
            time.sleep(1)
            if api_process.poll() is not None:
                print("API process exited. Stopping stack.")
                break

    except KeyboardInterrupt:
        print("Stopping local stack...")
    finally:
        stop_processes(processes)
        print("Stack shutdown complete.")
