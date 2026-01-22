"""
Data Snapshot Manager
=====================

Creates versioned snapshots of training data for reproducibility.
Generates manifests with metadata (hash, timestamp, file counts, etc.)
and allows loading specific snapshots for reproducible training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataManifest:
    """Manifest for a data snapshot."""

    snapshot_id: str
    created_at: str
    description: str
    duckdb_hash: str
    duckdb_size_bytes: int
    table_counts: Dict[str, int]
    symbol_count: int
    date_range: Dict[str, str]  # {"start": "2024-01-01", "end": "2024-12-31"}
    git_commit: Optional[str] = None
    extra_metadata: Dict = field(default_factory=dict)


class DataSnapshotManager:
    """
    Manage versioned snapshots of market data for reproducible training.

    Usage:
        manager = DataSnapshotManager()

        # Create a snapshot
        snapshot_id = manager.create_snapshot(
            description="Training data for LSTM v2",
            symbols=["AAPL", "MSFT", "GOOGL"]
        )

        # List available snapshots
        snapshots = manager.list_snapshots()

        # Load a specific snapshot
        df = manager.load_snapshot(snapshot_id, table="ohlcv")
    """

    def __init__(
        self,
        snapshots_dir: str = "data/snapshots",
        duckdb_path: str = "data/market_data.duckdb",
    ) -> None:
        self.snapshots_dir = Path(snapshots_dir)
        self.duckdb_path = Path(duckdb_path)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        description: str = "",
        symbols: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new data snapshot.

        Parameters
        ----------
        description : str
            Human-readable description of the snapshot.
        symbols : Optional[List[str]]
            Subset of symbols to include. If None, includes all.
        tables : Optional[List[str]]
            Subset of tables to include. If None, includes all data tables.

        Returns
        -------
        str
            The snapshot ID (timestamp-based).
        """
        snapshot_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.snapshots_dir / snapshot_id
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Default tables to snapshot
        if tables is None:
            tables = ["ohlcv", "features", "news_sentiment", "fundamentals"]

        # Get table row counts and date ranges
        table_counts = {}
        date_range = {"start": None, "end": None}
        symbol_set = set()

        with duckdb.connect(str(self.duckdb_path), read_only=True) as conn:
            # Get list of actual tables
            existing_tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            existing_table_names = {t[0] for t in existing_tables}

            for table in tables:
                if table not in existing_table_names:
                    logger.warning("Table %s does not exist, skipping.", table)
                    continue

                # Build query with optional symbol filter
                if symbols:
                    placeholders = ",".join(["?"] * len(symbols))
                    where_clause = f"WHERE symbol IN ({placeholders})"
                    params = symbols
                else:
                    where_clause = ""
                    params = []

                # Export table to parquet
                try:
                    df = conn.execute(
                        f"SELECT * FROM {table} {where_clause}", params
                    ).fetch_df()

                    if df.empty:
                        logger.info("Table %s is empty, skipping.", table)
                        continue

                    parquet_path = snapshot_path / f"{table}.parquet"
                    df.to_parquet(parquet_path, index=False, compression="snappy")
                    table_counts[table] = len(df)

                    # Collect symbols
                    if "symbol" in df.columns:
                        symbol_set.update(df["symbol"].unique())

                    # Update date range
                    ts_col = "ts" if "ts" in df.columns else "date" if "date" in df.columns else None
                    if ts_col:
                        df_ts = pd.to_datetime(df[ts_col])
                        if date_range["start"] is None or df_ts.min() < pd.Timestamp(date_range["start"]):
                            date_range["start"] = str(df_ts.min().date())
                        if date_range["end"] is None or df_ts.max() > pd.Timestamp(date_range["end"]):
                            date_range["end"] = str(df_ts.max().date())

                    logger.info(
                        "Exported %s: %d rows to %s", table, len(df), parquet_path
                    )

                except Exception as exc:
                    logger.error("Failed to export table %s: %s", table, exc)
                    continue

        # Compute hash of the snapshot directory
        snapshot_hash = self._compute_directory_hash(snapshot_path)
        snapshot_size = self._get_directory_size(snapshot_path)

        # Get git commit if available
        git_commit = self._get_git_commit()

        # Create manifest
        manifest = DataManifest(
            snapshot_id=snapshot_id,
            created_at=datetime.utcnow().isoformat(),
            description=description,
            duckdb_hash=snapshot_hash,
            duckdb_size_bytes=snapshot_size,
            table_counts=table_counts,
            symbol_count=len(symbol_set),
            date_range=date_range,
            git_commit=git_commit,
            extra_metadata={
                "source_db": str(self.duckdb_path),
                "symbols_filter": symbols,
                "tables_filter": tables,
            },
        )

        # Save manifest
        manifest_path = snapshot_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2, default=str)

        logger.info(
            "Created snapshot %s: %d tables, %d symbols, %d bytes",
            snapshot_id,
            len(table_counts),
            len(symbol_set),
            snapshot_size,
        )

        return snapshot_id

    def list_snapshots(self) -> List[DataManifest]:
        """List all available snapshots."""
        snapshots = []
        for snapshot_dir in sorted(self.snapshots_dir.iterdir(), reverse=True):
            if not snapshot_dir.is_dir():
                continue
            manifest_path = snapshot_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                snapshots.append(DataManifest(**data))
        return snapshots

    def get_manifest(self, snapshot_id: str) -> Optional[DataManifest]:
        """Get the manifest for a specific snapshot."""
        manifest_path = self.snapshots_dir / snapshot_id / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DataManifest(**data)

    def load_snapshot(
        self, snapshot_id: str, table: str
    ) -> Optional[pd.DataFrame]:
        """
        Load a specific table from a snapshot.

        Parameters
        ----------
        snapshot_id : str
            The snapshot ID to load.
        table : str
            The table name to load.

        Returns
        -------
        Optional[pd.DataFrame]
            The loaded DataFrame, or None if not found.
        """
        parquet_path = self.snapshots_dir / snapshot_id / f"{table}.parquet"
        if not parquet_path.exists():
            logger.warning("Table %s not found in snapshot %s", table, snapshot_id)
            return None

        df = pd.read_parquet(parquet_path)
        logger.info(
            "Loaded %s from snapshot %s: %d rows", table, snapshot_id, len(df)
        )
        return df

    def load_all_tables(self, snapshot_id: str) -> Dict[str, pd.DataFrame]:
        """Load all tables from a snapshot."""
        snapshot_path = self.snapshots_dir / snapshot_id
        if not snapshot_path.exists():
            logger.error("Snapshot %s not found", snapshot_id)
            return {}

        tables = {}
        for parquet_file in snapshot_path.glob("*.parquet"):
            table_name = parquet_file.stem
            tables[table_name] = pd.read_parquet(parquet_file)
            logger.info(
                "Loaded %s: %d rows", table_name, len(tables[table_name])
            )
        return tables

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        snapshot_path = self.snapshots_dir / snapshot_id
        if not snapshot_path.exists():
            logger.warning("Snapshot %s not found", snapshot_id)
            return False

        shutil.rmtree(snapshot_path)
        logger.info("Deleted snapshot %s", snapshot_id)
        return True

    def verify_snapshot(self, snapshot_id: str) -> bool:
        """Verify a snapshot's integrity by checking the hash."""
        manifest = self.get_manifest(snapshot_id)
        if not manifest:
            logger.error("Manifest not found for snapshot %s", snapshot_id)
            return False

        snapshot_path = self.snapshots_dir / snapshot_id
        current_hash = self._compute_directory_hash(snapshot_path)

        if current_hash != manifest.duckdb_hash:
            logger.error(
                "Hash mismatch for snapshot %s: expected %s, got %s",
                snapshot_id,
                manifest.duckdb_hash,
                current_hash,
            )
            return False

        logger.info("Snapshot %s verified successfully", snapshot_id)
        return True

    def _compute_directory_hash(self, directory: Path) -> str:
        """Compute a hash of all parquet files in a directory."""
        hasher = hashlib.sha256()
        for parquet_file in sorted(directory.glob("*.parquet")):
            with open(parquet_file, "rb") as f:
                hasher.update(f.read())
        return hasher.hexdigest()[:16]

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of all files in a directory."""
        total = 0
        for f in directory.glob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def _get_git_commit(self) -> Optional[str]:
        """Get the current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None


def create_training_snapshot(
    description: str = "Training data snapshot",
    symbols: Optional[List[str]] = None,
) -> str:
    """Convenience function to create a training data snapshot."""
    manager = DataSnapshotManager()
    return manager.create_snapshot(description=description, symbols=symbols)


__all__ = [
    "DataSnapshotManager",
    "DataManifest",
    "create_training_snapshot",
]
