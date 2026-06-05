from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
SCAN_DIR = ROOT / "scan_results"
LOG_DIR.mkdir(exist_ok=True)
SCAN_DIR.mkdir(exist_ok=True)


def run_command(args: list[str], timeout: int | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    out = proc.stdout or ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "_".join(a.replace("/", "-") for a in args[:2]) or "command"
    (LOG_DIR / f"{stamp}_{safe_name}.log").write_text(out)
    return proc.returncode, out


def run_scan_chunks(universe: str = "custom", chunk_count: int = 15, second_chance: bool = True) -> list[tuple[int, str, str]]:
    SCAN_DIR.mkdir(exist_ok=True)
    for old in SCAN_DIR.glob("chunk_*.json"):
        old.unlink()
    results: list[tuple[int, str, str]] = []
    for i in range(chunk_count):
        output = SCAN_DIR / f"chunk_{i}.json"
        args = [
            "main.py", "trade-screen",
            "--universe", universe,
            "--chunk-index", str(i),
            "--chunk-count", str(chunk_count),
            "--recommend-only",
            "--output", str(output),
        ]
        if second_chance:
            args.append("--second-chance")
        code, out = run_command(args)
        if not output.exists():
            output.write_text(json.dumps({"created_at": datetime.now().isoformat(), "candidates": [], "failures": []}, indent=2))
        results.append((code, str(output), out))
    return results


def execute_top_trades(max_slots: int = 5, dry_run: bool = False, second_chance: bool = True) -> tuple[int, str]:
    args = ["main.py", "execute-ranked-trades", "--input-dir", str(SCAN_DIR), "--max-trade-slots", str(max_slots)]
    if second_chance:
        args.append("--second-chance")
    if dry_run:
        args.append("--dry-run")
    return run_command(args)


def monitor_positions(dry_run: bool = False) -> tuple[int, str]:
    args = ["main.py", "monitor-positions"]
    if dry_run:
        args.append("--dry-run")
    return run_command(args)


def latest_logs(limit: int = 5) -> list[Path]:
    return sorted(LOG_DIR.glob("*.log"), reverse=True)[:limit]


def load_scan_results() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(SCAN_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for c in payload.get("candidates", []) or []:
            c = dict(c)
            c.setdefault("source_file", path.name)
            rows.append(c)
    rows.sort(key=lambda r: int(float(r.get("conviction_score") or 0)), reverse=True)
    return rows
