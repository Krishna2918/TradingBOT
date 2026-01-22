"""
Training Integration Module
===========================

Integrates training pipelines with:
- Data snapshots for reproducibility
- Model registry for versioning and promotion
- Evaluation reports for checkpoint comparison

Usage:
    from src.ai.training_integration import TrainingIntegration

    integration = TrainingIntegration()

    # Before training: create data snapshot
    snapshot_id = integration.prepare_training_data(symbols=["AAPL", "MSFT", ...])

    # After training: register model
    model_id = integration.register_trained_model(
        model_path="models/lstm_best.pt",
        model_type="lstm",
        metrics={"test_accuracy": 0.42, "val_accuracy": 0.40},
        snapshot_id=snapshot_id,
    )

    # Evaluate and promote best model
    integration.evaluate_and_promote(model_name="lstm_attention")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Evaluation report for model checkpoints."""

    report_id: str
    created_at: str
    model_name: str
    checkpoints: List[Dict[str, Any]]
    best_checkpoint: str
    best_metrics: Dict[str, float]
    promotion_threshold: Dict[str, float]
    promoted: bool
    notes: str = ""


class TrainingIntegration:
    """
    Integrates training with data snapshots, model registry, and evaluation.

    Ensures reproducibility by:
    1. Creating data snapshots before training
    2. Linking trained models to snapshots
    3. Recording metrics and code versions
    4. Evaluating checkpoints and promoting best models
    """

    def __init__(
        self,
        snapshots_dir: str = "data/snapshots",
        models_dir: str = "models",
        reports_dir: str = "reports/evaluations",
        duckdb_path: str = "data/market_data.duckdb",
    ) -> None:
        self.snapshots_dir = Path(snapshots_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.duckdb_path = Path(duckdb_path)

        # Create directories
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize data snapshot manager and model registry."""
        try:
            from src.data_pipeline.data_snapshot import DataSnapshotManager
            self.snapshot_manager = DataSnapshotManager(
                snapshots_dir=str(self.snapshots_dir),
                duckdb_path=str(self.duckdb_path),
            )
        except ImportError:
            logger.warning("DataSnapshotManager not available")
            self.snapshot_manager = None

        try:
            from src.ai.model_registry import ModelRegistry
            self.model_registry = ModelRegistry(
                storage_path=str(self.models_dir / "registry")
            )
        except ImportError:
            logger.warning("ModelRegistry not available")
            self.model_registry = None

    def prepare_training_data(
        self,
        symbols: Optional[List[str]] = None,
        description: str = "Training data snapshot",
    ) -> Optional[str]:
        """
        Create a data snapshot before training for reproducibility.

        Parameters
        ----------
        symbols : Optional[List[str]]
            Symbols to include. If None, includes all.
        description : str
            Description of the snapshot.

        Returns
        -------
        Optional[str]
            Snapshot ID, or None if snapshot creation failed.
        """
        if not self.snapshot_manager:
            logger.warning("Snapshot manager not available, skipping snapshot creation")
            return None

        try:
            snapshot_id = self.snapshot_manager.create_snapshot(
                description=description,
                symbols=symbols,
            )
            logger.info("Created training data snapshot: %s", snapshot_id)
            return snapshot_id
        except Exception as exc:
            logger.error("Failed to create snapshot: %s", exc)
            return None

    def register_trained_model(
        self,
        model_path: str,
        model_type: str,
        model_name: str,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        snapshot_id: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Register a trained model in the registry.

        Parameters
        ----------
        model_path : str
            Path to the trained model file.
        model_type : str
            Type of model (lstm, transformer, xgboost, etc.).
        model_name : str
            Name for the model (e.g., "lstm_attention").
        metrics : Dict[str, float]
            Training/validation metrics.
        hyperparameters : Optional[Dict[str, Any]]
            Model hyperparameters.
        snapshot_id : Optional[str]
            ID of the data snapshot used for training.
        description : str
            Model description.
        tags : Optional[List[str]]
            Tags for filtering.

        Returns
        -------
        Optional[str]
            Model ID, or None if registration failed.
        """
        if not self.model_registry:
            logger.warning("Model registry not available, skipping registration")
            return None

        try:
            from src.ai.model_registry import ModelMetadata

            # Get git commit
            git_commit = self._get_git_commit()

            # Calculate data hash if snapshot exists
            data_hash = ""
            if snapshot_id and self.snapshot_manager:
                manifest = self.snapshot_manager.get_manifest(snapshot_id)
                if manifest:
                    data_hash = manifest.duckdb_hash

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_name,
                model_type=model_type,
                version=self._get_next_version(model_name),
                training_date=datetime.now(),
                performance_metrics=metrics,
                hyperparameters=hyperparameters or {},
                feature_importance=None,
                training_data_hash=data_hash,
                model_size_mb=0.0,
                inference_latency_ms=0.0,
                training_time_seconds=0.0,
                validation_split=0.2,
                test_split=0.1,
                feature_columns=[],
                target_column="target",
                data_snapshot_id=snapshot_id,
                git_commit=git_commit,
                description=description,
                tags=tags or [],
            )

            # Register model
            model_path_str = self.model_registry.register_model(
                model=None,  # Just metadata, model loaded separately
                metadata=metadata,
                overwrite=False,
            )

            model_key = f"{model_name}_{metadata.version}"
            logger.info("Registered model: %s", model_key)
            return model_key

        except Exception as exc:
            logger.error("Failed to register model: %s", exc)
            return None

    def evaluate_checkpoints(
        self,
        model_name: str,
        checkpoints_dir: str = "checkpoints",
        metric: str = "val_accuracy",
        minimize: bool = False,
    ) -> EvaluationReport:
        """
        Evaluate all checkpoints for a model and identify the best one.

        Parameters
        ----------
        model_name : str
            Name of the model.
        checkpoints_dir : str
            Directory containing checkpoints.
        metric : str
            Metric to use for comparison.
        minimize : bool
            If True, lower is better.

        Returns
        -------
        EvaluationReport
            Evaluation report with rankings.
        """
        checkpoints_path = Path(checkpoints_dir)
        checkpoints = []

        # Scan checkpoints directory
        for checkpoint_file in checkpoints_path.glob(f"*{model_name}*.pt"):
            try:
                # Try to load checkpoint metadata
                meta_file = checkpoint_file.with_suffix(".json")
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {"file": str(checkpoint_file)}

                checkpoints.append({
                    "file": str(checkpoint_file),
                    "name": checkpoint_file.stem,
                    "metrics": meta.get("metrics", {}),
                    "epoch": meta.get("epoch", 0),
                    "timestamp": meta.get("timestamp", ""),
                })
            except Exception as exc:
                logger.warning("Failed to load checkpoint %s: %s", checkpoint_file, exc)

        if not checkpoints:
            logger.warning("No checkpoints found for %s", model_name)
            return EvaluationReport(
                report_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                created_at=datetime.utcnow().isoformat(),
                model_name=model_name,
                checkpoints=[],
                best_checkpoint="",
                best_metrics={},
                promotion_threshold={metric: 0.40},
                promoted=False,
                notes="No checkpoints found",
            )

        # Sort by metric
        valid_checkpoints = [c for c in checkpoints if metric in c.get("metrics", {})]

        if valid_checkpoints:
            if minimize:
                best = min(valid_checkpoints, key=lambda c: c["metrics"][metric])
            else:
                best = max(valid_checkpoints, key=lambda c: c["metrics"][metric])
        else:
            best = checkpoints[0] if checkpoints else {}
            logger.warning("No checkpoints have metric '%s', using first checkpoint", metric)

        # Create report
        report = EvaluationReport(
            report_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            created_at=datetime.utcnow().isoformat(),
            model_name=model_name,
            checkpoints=checkpoints,
            best_checkpoint=best.get("file", ""),
            best_metrics=best.get("metrics", {}),
            promotion_threshold={metric: 0.40},  # Default threshold
            promoted=False,
            notes=f"Evaluated {len(checkpoints)} checkpoints",
        )

        # Save report
        report_path = self.reports_dir / f"{model_name}_{report.report_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info("Saved evaluation report to %s", report_path)
        return report

    def evaluate_and_promote(
        self,
        model_name: str,
        checkpoints_dir: str = "checkpoints",
        metric: str = "val_accuracy",
        threshold: float = 0.40,
        minimize: bool = False,
    ) -> bool:
        """
        Evaluate checkpoints and promote the best model if it meets threshold.

        Parameters
        ----------
        model_name : str
            Name of the model.
        checkpoints_dir : str
            Directory containing checkpoints.
        metric : str
            Metric to use for comparison.
        threshold : float
            Minimum metric value required for promotion.
        minimize : bool
            If True, lower is better (threshold is maximum).

        Returns
        -------
        bool
            True if a model was promoted.
        """
        report = self.evaluate_checkpoints(
            model_name=model_name,
            checkpoints_dir=checkpoints_dir,
            metric=metric,
            minimize=minimize,
        )

        if not report.best_checkpoint:
            logger.warning("No best checkpoint found, cannot promote")
            return False

        best_value = report.best_metrics.get(metric, 0)

        # Check threshold
        if minimize:
            meets_threshold = best_value <= threshold
        else:
            meets_threshold = best_value >= threshold

        if not meets_threshold:
            logger.warning(
                "Best checkpoint (%s=%s) does not meet threshold (%s)",
                metric, best_value, threshold
            )
            return False

        # Register and promote
        if self.model_registry:
            # First register the model
            model_id = self.register_trained_model(
                model_path=report.best_checkpoint,
                model_type="lstm",  # TODO: detect type
                model_name=model_name,
                metrics=report.best_metrics,
                description=f"Auto-promoted from evaluation {report.report_id}",
                tags=["auto-promoted"],
            )

            if model_id:
                # Get version from model_id
                parts = model_id.rsplit("_v", 1)
                if len(parts) == 2:
                    version = parts[1]
                    self.model_registry.promote_model(model_name, version)
                    report.promoted = True

                    # Update report
                    report_path = self.reports_dir / f"{model_name}_{report.report_id}.json"
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(asdict(report), f, indent=2, default=str)

                    logger.info("Promoted model %s to production", model_id)
                    return True

        return False

    def get_reproducibility_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get full reproducibility information for a model.

        Returns data snapshot, code version, hyperparameters, etc.
        """
        info = {
            "model_id": model_id,
            "reproducible": False,
            "data_snapshot": None,
            "git_commit": None,
            "hyperparameters": {},
            "metrics": {},
        }

        if not self.model_registry:
            return info

        # Parse model_id
        parts = model_id.rsplit("_v", 1)
        if len(parts) != 2:
            return info

        model_name, version = parts

        try:
            model_key = f"{model_name}_{version}"
            if model_key in self.model_registry.metadata_db:
                metadata = self.model_registry.metadata_db[model_key]
                info["data_snapshot"] = metadata.data_snapshot_id
                info["git_commit"] = metadata.git_commit
                info["hyperparameters"] = metadata.hyperparameters
                info["metrics"] = metadata.performance_metrics
                info["reproducible"] = bool(metadata.data_snapshot_id and metadata.git_commit)
        except Exception as exc:
            logger.error("Failed to get reproducibility info: %s", exc)

        return info

    def _get_next_version(self, model_name: str) -> str:
        """Get the next version string for a model."""
        if not self.model_registry:
            return "v1"

        versions = []
        for model_key in self.model_registry.metadata_db:
            if model_key.startswith(f"{model_name}_"):
                try:
                    v = model_key.rsplit("_", 1)[1]
                    versions.append(v)
                except (ValueError, IndexError):
                    pass

        if not versions:
            return "v1"

        # Find max version number
        max_v = 0
        for v in versions:
            try:
                num = int(v.lstrip("v"))
                max_v = max(max_v, num)
            except ValueError:
                pass

        return f"v{max_v + 1}"

    def _get_git_commit(self) -> Optional[str]:
        """Get the current git commit hash."""
        try:
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


# Convenience functions
def create_training_run(
    model_name: str,
    symbols: Optional[List[str]] = None,
    description: str = "",
) -> Dict[str, str]:
    """
    Create a new training run with data snapshot.

    Returns dict with snapshot_id and run_id.
    """
    integration = TrainingIntegration()

    snapshot_id = integration.prepare_training_data(
        symbols=symbols,
        description=f"Training run for {model_name}: {description}",
    )

    return {
        "snapshot_id": snapshot_id,
        "run_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
    }


def finalize_training_run(
    run_info: Dict[str, str],
    model_path: str,
    model_type: str,
    metrics: Dict[str, float],
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Finalize a training run by registering the model.

    Returns model_id.
    """
    integration = TrainingIntegration()

    return integration.register_trained_model(
        model_path=model_path,
        model_type=model_type,
        model_name=run_info["model_name"],
        metrics=metrics,
        hyperparameters=hyperparameters,
        snapshot_id=run_info.get("snapshot_id"),
        description=f"Training run {run_info['run_id']}",
    )


__all__ = [
    "TrainingIntegration",
    "EvaluationReport",
    "create_training_run",
    "finalize_training_run",
]
