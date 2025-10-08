"""Utilities for working with dashboard log files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def extract_log_timestamp(line: str) -> Optional[datetime]:
    """Best-effort extraction of a timestamp from a log line."""
    try:
        if not line:
            return None
        stripped = line.strip()
        if not stripped:
            return None
        first = stripped.split()[0]
        if "T" in first:
            candidate = first.rstrip("|")
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                pass
        parts = stripped.split()
        if len(parts) >= 2:
            candidate = f"{parts[0]} {parts[1].rstrip('|')}"
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                try:
                    return datetime.strptime(candidate, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None
    except Exception:
        return None
    return None


def get_last_log_time(path: str) -> str:
    """Return HH:MM:SS for the most recent entry in the given log file."""
    log_path = Path(path)
    if not log_path.exists():
        return "--"
    try:
        with log_path.open('rb') as fh:
            fh.seek(0, 2)  # Seek to end
            size = fh.tell()
            if size <= 0:
                return "--"
            block = min(size, 8192)
            fh.seek(-block, 2)
            data = fh.read().decode('utf-8', errors='ignore')
        for raw_line in reversed([ln for ln in data.splitlines() if ln.strip()]):
            ts = extract_log_timestamp(raw_line)
            if ts:
                return ts.strftime('%H:%M:%S')
    except Exception:
        return "--"
    return "--"


__all__ = ['extract_log_timestamp', 'get_last_log_time']
