"""
Model Registry System for Multi-Model Training

This module provides centralized model storage, versioning, and metadata management
for different model types including LSTM, Transformer, XGBoost, and LightGBM models.
"""

import json
import pickle
import logging
import hashlib
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np

# For model serialization
import torch
import joblib

# Import architecture definitions for model reconstruction
from src.ai.architectures import get_architecture, infer_architecture_config

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Comprehensive metadata for trained models"""
    model_id: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    training_data_hash: str
    model_size_mb: float
    inference_latency_ms: float

    # Training details
    training_time_seconds: float
    validation_split: float
    test_split: float

    # Data information
    feature_columns: List[str]
    target_column: str
    sequence_length: Optional[int] = None  # For sequence models

    # Model architecture details
    architecture_config: Dict[str, Any] = None

    # Deployment information
    deployment_status: str = "trained"  # trained, validated, deployed, retired
    deployment_date: Optional[datetime] = None
    is_production: bool = False  # True if this is the promoted production model

    # Performance tracking
    production_metrics: Optional[Dict[str, float]] = None

    # Data snapshot linkage for reproducibility
    data_snapshot_id: Optional[str] = None
    git_commit: Optional[str] = None
    description: str = ""
    tags: List[str] = None

    # Model file path (for models stored outside registry directory)
    model_path: Optional[str] = None
    model_hash: Optional[str] = None

    # Registry hygiene: evaluation report, calibration config, feature manifest
    evaluation_report: Optional[str] = None
    calibration_config: Optional[str] = None
    feature_manifest_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if isinstance(data['training_date'], datetime):
            data['training_date'] = data['training_date'].isoformat()
        if data.get('deployment_date') and isinstance(data['deployment_date'], datetime):
            data['deployment_date'] = data['deployment_date'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary (JSON deserialization)"""
        # Convert ISO strings back to datetime objects
        if isinstance(data['training_date'], str):
            data['training_date'] = datetime.fromisoformat(data['training_date'])
        if data.get('deployment_date') and isinstance(data['deployment_date'], str):
            data['deployment_date'] = datetime.fromisoformat(data['deployment_date'])
        return cls(**data)

class ModelRegistry:
    """
    Centralized model storage and versioning system
    
    Features:
    - Model artifact storage with versioning
    - Comprehensive metadata tracking
    - Model comparison and performance analysis
    - Deployment status management
    - Automatic cleanup of old versions
    """
    
    def __init__(self, storage_path: str = "models/registry"):
        """
        Initialize Model Registry
        
        Args:
            storage_path: Base path for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Registry database (JSON file)
        self.registry_file = self.storage_path / "registry.json"
        self.metadata_db: Dict[str, ModelMetadata] = {}
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Initialized ModelRegistry at {self.storage_path}")
    
    def register_model(self, 
                      model: Any, 
                      metadata: ModelMetadata,
                      overwrite: bool = False) -> str:
        """
        Register a trained model with metadata
        
        Args:
            model: Trained model object
            metadata: Model metadata
            overwrite: Whether to overwrite existing model
            
        Returns:
            Path to saved model
        """
        model_key = f"{metadata.model_id}_{metadata.version}"
        
        # Check if model already exists
        if model_key in self.metadata_db and not overwrite:
            raise ValueError(f"Model {model_key} already exists. Use overwrite=True to replace.")
        
        # Create model directory
        model_dir = self.storage_path / metadata.model_type / metadata.model_id / metadata.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_path = self._save_model_artifacts(model, model_dir, metadata)

        # Calculate model size
        metadata.model_size_mb = self._calculate_model_size(model_dir)

        # Compute and store model hash for integrity verification
        if model_path.exists():
            metadata.model_hash = self._compute_file_hash(model_path)
            metadata.model_path = str(model_path)
            logger.info(f"Computed model hash: {metadata.model_hash}")

        # Update registry
        self.metadata_db[model_key] = metadata
        self._save_registry()
        
        logger.info(f"Registered model {model_key} at {model_path}")
        return str(model_path)
    
    def load_model(self, 
                   model_id: str, 
                   version: str = "latest") -> Tuple[Any, ModelMetadata]:
        """
        Load model and metadata
        
        Args:
            model_id: Model identifier
            version: Model version ("latest" for most recent)
            
        Returns:
            Tuple of (model, metadata)
        """
        if version == "latest":
            version = self._get_latest_version(model_id)
        
        model_key = f"{model_id}_{version}"
        
        if model_key not in self.metadata_db:
            raise ValueError(f"Model {model_key} not found in registry")
        
        metadata = self.metadata_db[model_key]
        model_dir = self.storage_path / metadata.model_type / model_id / version
        
        # Load model artifacts
        model = self._load_model_artifacts(model_dir, metadata)
        
        logger.info(f"Loaded model {model_key}")
        return model, metadata
    
    def list_models(self, 
                   model_type: Optional[str] = None,
                   deployment_status: Optional[str] = None) -> pd.DataFrame:
        """
        List all registered models
        
        Args:
            model_type: Filter by model type
            deployment_status: Filter by deployment status
            
        Returns:
            DataFrame with model information
        """
        models_data = []
        
        for model_key, metadata in self.metadata_db.items():
            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue
            if deployment_status and metadata.deployment_status != deployment_status:
                continue
            
            # Extract key metrics
            test_accuracy = metadata.performance_metrics.get('test_accuracy', 0.0)
            val_accuracy = metadata.performance_metrics.get('val_accuracy', 0.0)
            
            models_data.append({
                'model_id': metadata.model_id,
                'model_type': metadata.model_type,
                'version': metadata.version,
                'training_date': metadata.training_date,
                'test_accuracy': test_accuracy,
                'val_accuracy': val_accuracy,
                'training_time_seconds': metadata.training_time_seconds,
                'model_size_mb': metadata.model_size_mb,
                'deployment_status': metadata.deployment_status
            })
        
        return pd.DataFrame(models_data)
    
    def compare_models(self, 
                      model_ids: List[str],
                      metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple models across specified metrics
        
        Args:
            model_ids: List of model IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['test_accuracy', 'val_accuracy', 'training_time_seconds']
        
        comparison_data = []
        
        for model_id in model_ids:
            # Get latest version
            latest_version = self._get_latest_version(model_id)
            model_key = f"{model_id}_{latest_version}"
            
            if model_key not in self.metadata_db:
                logger.warning(f"Model {model_id} not found, skipping")
                continue
            
            metadata = self.metadata_db[model_key]
            
            row = {
                'model_id': model_id,
                'model_type': metadata.model_type,
                'version': latest_version
            }
            
            # Add requested metrics
            for metric in metrics:
                if metric in ['training_time_seconds', 'model_size_mb', 'inference_latency_ms']:
                    row[metric] = getattr(metadata, metric, None)
                else:
                    row[metric] = metadata.performance_metrics.get(metric, None)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_model_history(self, model_id: str) -> pd.DataFrame:
        """
        Get version history for a specific model
        
        Args:
            model_id: Model identifier
            
        Returns:
            DataFrame with version history
        """
        history_data = []
        
        for model_key, metadata in self.metadata_db.items():
            if metadata.model_id == model_id:
                test_accuracy = metadata.performance_metrics.get('test_accuracy', 0.0)
                
                history_data.append({
                    'version': metadata.version,
                    'training_date': metadata.training_date,
                    'test_accuracy': test_accuracy,
                    'training_time_seconds': metadata.training_time_seconds,
                    'deployment_status': metadata.deployment_status
                })
        
        df = pd.DataFrame(history_data)
        if not df.empty:
            df = df.sort_values('training_date')
        
        return df
    
    def update_deployment_status(self,
                               model_id: str,
                               version: str,
                               status: str,
                               production_metrics: Optional[Dict[str, float]] = None):
        """
        Update deployment status and production metrics

        Args:
            model_id: Model identifier
            version: Model version
            status: New deployment status
            production_metrics: Production performance metrics
        """
        model_key = f"{model_id}_{version}"

        if model_key not in self.metadata_db:
            raise ValueError(f"Model {model_key} not found")

        metadata = self.metadata_db[model_key]
        metadata.deployment_status = status

        if status == "deployed":
            metadata.deployment_date = datetime.now()

        if production_metrics:
            metadata.production_metrics = production_metrics

        self._save_registry()
        logger.info(f"Updated deployment status for {model_key} to {status}")

    def promote_model(self, model_id: str, version: str) -> bool:
        """
        Promote a model to production.

        Demotes any other production model with the same model_id.

        Args:
            model_id: Model identifier
            version: Model version to promote

        Returns:
            True if successful
        """
        model_key = f"{model_id}_{version}"

        if model_key not in self.metadata_db:
            logger.error(f"Model {model_key} not found in registry")
            return False

        # Demote other production models with the same model_id
        for other_key, other_metadata in self.metadata_db.items():
            if other_metadata.model_id == model_id and other_metadata.is_production:
                other_metadata.is_production = False
                other_metadata.deployment_status = "validated"
                logger.info(f"Demoted {other_key} from production")

        # Promote this model
        metadata = self.metadata_db[model_key]
        metadata.is_production = True
        metadata.deployment_status = "deployed"
        metadata.deployment_date = datetime.now()
        self._save_registry()

        logger.info(f"Promoted {model_key} to production")
        return True

    def get_production_model(self, model_id: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """
        Get the current production model for a given model_id.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, metadata) or None if no production model
        """
        # First check if there's a designated production model key
        if hasattr(self, "_production_model_key") and self._production_model_key:
            if self._production_model_key in self.metadata_db:
                metadata = self.metadata_db[self._production_model_key]

                # Try loading from model_path first (for models stored outside registry)
                if metadata.model_path:
                    model_path = Path(metadata.model_path)
                    if model_path.exists():
                        model = self._load_model_from_path(model_path, metadata)
                        logger.info(f"Loaded production model from {model_path}")
                        return model, metadata
                    else:
                        logger.warning(
                            f"Production model path {model_path} does not exist"
                        )

                # Fallback: try standard directory structure
                model_dir = self.storage_path / metadata.model_type / metadata.model_id / metadata.version
                if model_dir.exists():
                    model = self._load_model_artifacts(model_dir, metadata)
                    logger.info(f"Loaded production model {self._production_model_key}")
                    return model, metadata

                logger.warning(
                    f"Production model {self._production_model_key} not found at "
                    f"{metadata.model_path} or {model_dir}"
                )

        # Fallback: search by model_id and is_production flag
        for model_key, metadata in self.metadata_db.items():
            if metadata.model_id == model_id and metadata.is_production:
                # Try model_path first
                if metadata.model_path:
                    model_path = Path(metadata.model_path)
                    if model_path.exists():
                        model = self._load_model_from_path(model_path, metadata)
                        logger.info(f"Loaded production model from {model_path}")
                        return model, metadata

                # Standard directory structure
                model_dir = self.storage_path / metadata.model_type / model_id / metadata.version
                if model_dir.exists():
                    model = self._load_model_artifacts(model_dir, metadata)
                    logger.info(f"Loaded production model {model_key}")
                    return model, metadata

        logger.warning(f"No production model found for {model_id}")
        return None

    def _load_model_from_path(
        self,
        model_path: Path,
        metadata: ModelMetadata,
        strict_hash: bool = True
    ) -> Any:
        """Load model directly from a path (for models not in registry directory structure).

        Args:
            model_path: Path to model file
            metadata: Model metadata with expected hash
            strict_hash: If True, raise error on hash mismatch. If False, only warn.
        """
        try:
            # Validate hash if available
            if metadata.model_hash:
                actual_hash = self._compute_file_hash(model_path)
                if actual_hash != metadata.model_hash:
                    msg = (
                        f"Model integrity check FAILED for {model_path}: "
                        f"expected hash {metadata.model_hash}, got {actual_hash}. "
                        f"Model file may have been modified or corrupted."
                    )
                    if strict_hash:
                        logger.error(msg)
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
                else:
                    logger.info(f"Model integrity verified (hash: {actual_hash})")

            if model_path.suffix in [".pth", ".pt"]:
                loaded_data = torch.load(model_path, map_location="cpu")

                # Handle checkpoint format (dict with model_state_dict)
                if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                    state_dict = loaded_data['model_state_dict']
                    # Extract config from checkpoint if not in metadata
                    if not metadata.architecture_config and 'config' in loaded_data:
                        checkpoint_config = loaded_data['config']
                        inferred = infer_architecture_config(state_dict)
                        metadata.architecture_config = {
                            'input_dim': inferred.get('input_dim', 58),
                            'hidden_size': checkpoint_config.get('hidden_size', 256),
                            'num_layers': checkpoint_config.get('num_layers', 3),
                            'dropout': checkpoint_config.get('dropout', 0.3),
                            'num_classes': loaded_data.get('num_classes', checkpoint_config.get('num_classes', 3)),
                        }
                        logger.info(f"Extracted architecture config from checkpoint: {metadata.architecture_config}")
                    model_dir = model_path.parent
                    model = self._reconstruct_model(state_dict, metadata, model_dir)
                    return model

                # Check if it's a raw state dict (dict of tensors)
                elif isinstance(loaded_data, dict) and all(isinstance(v, torch.Tensor) for v in loaded_data.values()):
                    # It's a raw state dict - reconstruct the model
                    model_dir = model_path.parent
                    model = self._reconstruct_model(loaded_data, metadata, model_dir)
                    return model
                else:
                    # It's a full model object
                    return loaded_data

            elif model_path.suffix == ".pkl":
                return joblib.load(model_path)
            else:
                with open(model_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file (first 16 chars)."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def validate_features(
        self,
        model_key: str,
        input_features: List[str],
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate input features against model's expected feature columns.

        Args:
            model_key: Registry key for the model (e.g., "lstm_production_v1")
            input_features: List of feature column names in the input data
            strict: If True, raise error on mismatch. If False, return validation report.

        Returns:
            Dict with validation results: present, missing, extra, coverage_pct, is_valid
        """
        if model_key not in self.metadata_db:
            raise ValueError(f"Model {model_key} not found in registry")

        metadata = self.metadata_db[model_key]
        expected = set(metadata.feature_columns) if metadata.feature_columns else set()
        provided = set(input_features)

        missing = expected - provided
        extra = provided - expected
        present = expected & provided
        coverage = (len(present) / len(expected) * 100) if expected else 100.0

        result = {
            "model_key": model_key,
            "expected_count": len(expected),
            "provided_count": len(provided),
            "present_count": len(present),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "coverage_pct": round(coverage, 1),
            "missing_features": sorted(missing),
            "is_valid": len(missing) == 0,
        }

        if missing:
            msg = (
                f"Feature validation for {model_key}: "
                f"{len(missing)} missing features ({coverage:.1f}% coverage). "
                f"Missing: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
            if strict:
                logger.error(msg)
                raise ValueError(msg)
            elif coverage < 80:
                logger.warning(msg + " - Model predictions may be unreliable!")
            else:
                logger.warning(msg)
        else:
            logger.info(
                f"Feature validation passed for {model_key}: "
                f"{len(present)}/{len(expected)} features present (100% coverage)"
            )

        return result

    def copy_model_to_registry(self, model_key: str) -> Path:
        """
        Copy a model from external path into the registry directory structure.

        Args:
            model_key: Registry key for the model

        Returns:
            New path to the model within registry tree
        """
        if model_key not in self.metadata_db:
            raise ValueError(f"Model {model_key} not found in registry")

        metadata = self.metadata_db[model_key]

        if not metadata.model_path:
            raise ValueError(f"Model {model_key} has no external model_path")

        source_path = Path(metadata.model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source model not found: {source_path}")

        # Create registry directory structure
        model_dir = (
            self.storage_path
            / metadata.model_type
            / metadata.model_id
            / metadata.version
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dest_path = model_dir / f"model{source_path.suffix}"
        shutil.copy2(source_path, dest_path)

        # Verify copy integrity
        source_hash = self._compute_file_hash(source_path)
        dest_hash = self._compute_file_hash(dest_path)
        if source_hash != dest_hash:
            raise RuntimeError(f"Copy verification failed: hash mismatch")

        # Update metadata to point to new location (relative to storage_path)
        old_path = metadata.model_path
        metadata.model_path = str(dest_path)
        metadata.model_hash = dest_hash
        self._save_registry()

        logger.info(
            f"Copied model {model_key} from {old_path} to {dest_path} "
            f"(hash: {dest_hash})"
        )
        return dest_path

    def get_best_model(
        self,
        model_id: str,
        metric: str = "test_accuracy",
        minimize: bool = False
    ) -> Optional[Tuple[str, ModelMetadata]]:
        """
        Get the best model version by a specific metric.

        Args:
            model_id: Model identifier
            metric: Metric name to compare
            minimize: If True, lower is better

        Returns:
            Tuple of (version, metadata) or None if not found
        """
        matching = [
            (key, meta) for key, meta in self.metadata_db.items()
            if meta.model_id == model_id and metric in meta.performance_metrics
        ]

        if not matching:
            return None

        if minimize:
            best_key, best_meta = min(matching, key=lambda x: x[1].performance_metrics[metric])
        else:
            best_key, best_meta = max(matching, key=lambda x: x[1].performance_metrics[metric])

        return best_meta.version, best_meta
    
    def delete_model(self, model_id: str, version: str):
        """
        Delete a model and its artifacts
        
        Args:
            model_id: Model identifier
            version: Model version
        """
        model_key = f"{model_id}_{version}"
        
        if model_key not in self.metadata_db:
            raise ValueError(f"Model {model_key} not found")
        
        metadata = self.metadata_db[model_key]
        
        # Delete model directory
        model_dir = self.storage_path / metadata.model_type / model_id / version
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.metadata_db[model_key]
        self._save_registry()
        
        logger.info(f"Deleted model {model_key}")
    
    def cleanup_old_versions(self, model_id: str, keep_versions: int = 5):
        """
        Clean up old versions of a model, keeping only the most recent
        
        Args:
            model_id: Model identifier
            keep_versions: Number of versions to keep
        """
        # Get all versions for this model
        versions = []
        for model_key, metadata in self.metadata_db.items():
            if metadata.model_id == model_id:
                versions.append((metadata.version, metadata.training_date))
        
        # Sort by training date (newest first)
        versions.sort(key=lambda x: x[1], reverse=True)
        
        # Delete old versions
        for version, _ in versions[keep_versions:]:
            try:
                self.delete_model(model_id, version)
                logger.info(f"Cleaned up old version {model_id}_{version}")
            except Exception as e:
                logger.error(f"Failed to delete {model_id}_{version}: {e}")
    
    def _save_model_artifacts(self, 
                            model: Any, 
                            model_dir: Path, 
                            metadata: ModelMetadata) -> Path:
        """Save model artifacts to disk"""
        
        if metadata.model_type in ['lstm', 'transformer']:
            # PyTorch models
            model_path = model_dir / "model.pth"
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                # For placeholder models, save as pickle
                with open(model_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            
            # Save architecture config
            if metadata.architecture_config:
                config_path = model_dir / "architecture.json"
                with open(config_path, 'w') as f:
                    json.dump(metadata.architecture_config, f, indent=2)
        
        elif metadata.model_type in ['xgboost', 'lightgbm']:
            # Tree-based models
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
        
        else:
            # Generic pickle serialization
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return model_path
    
    def _load_model_artifacts(self, model_dir: Path, metadata: ModelMetadata) -> Any:
        """Load model artifacts from disk with hash verification and architecture reconstruction."""

        # Determine model path based on type
        if metadata.model_type in ['lstm', 'transformer']:
            model_path = model_dir / "model.pth"
        elif metadata.model_type in ['xgboost', 'lightgbm']:
            model_path = model_dir / "model.pkl"
        else:
            model_path = model_dir / "model.pkl"

        # Verify model integrity before loading
        if metadata.model_hash and model_path.exists():
            actual_hash = self._compute_file_hash(model_path)
            if actual_hash != metadata.model_hash:
                raise ValueError(
                    f"Model integrity check FAILED for {model_path}: "
                    f"expected {metadata.model_hash}, got {actual_hash}"
                )
            logger.info(f"Model integrity verified (hash: {actual_hash})")

        # Load based on model type
        if metadata.model_type in ['lstm', 'transformer']:
            loaded_data = torch.load(model_path, map_location='cpu')

            # Handle checkpoint format (dict with model_state_dict) vs raw state dict
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                state_dict = loaded_data['model_state_dict']
                # Extract config from checkpoint if not in metadata
                if not metadata.architecture_config and 'config' in loaded_data:
                    checkpoint_config = loaded_data['config']
                    # Infer input_dim from state dict
                    inferred = infer_architecture_config(state_dict)
                    metadata.architecture_config = {
                        'input_dim': inferred.get('input_dim', 58),
                        'hidden_size': checkpoint_config.get('hidden_size', 256),
                        'num_layers': checkpoint_config.get('num_layers', 3),
                        'dropout': checkpoint_config.get('dropout', 0.3),
                        'num_classes': loaded_data.get('num_classes', checkpoint_config.get('num_classes', 3)),
                    }
                    logger.info(f"Extracted architecture config from checkpoint: {metadata.architecture_config}")
            else:
                state_dict = loaded_data

            # Reconstruct model architecture
            model = self._reconstruct_model(state_dict, metadata, model_dir)
            return model

        elif metadata.model_type in ['xgboost', 'lightgbm']:
            return joblib.load(model_path)

        else:
            with open(model_path, 'rb') as f:
                return pickle.load(f)

    def _reconstruct_model(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: ModelMetadata,
        model_dir: Optional[Path] = None
    ) -> torch.nn.Module:
        """
        Reconstruct a PyTorch model from state dict and architecture config.

        Args:
            state_dict: Model state dictionary
            metadata: Model metadata with architecture_config
            model_dir: Optional directory containing architecture.json

        Returns:
            Reconstructed PyTorch model with loaded weights
        """
        # Try to get architecture config from multiple sources
        architecture_config = None

        # 1. Check metadata.architecture_config
        if metadata.architecture_config:
            architecture_config = metadata.architecture_config
            logger.info("Using architecture_config from metadata")

        # 2. Check architecture.json file
        if architecture_config is None and model_dir:
            config_path = model_dir / "architecture.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    architecture_config = json.load(f)
                logger.info(f"Loaded architecture config from {config_path}")

        # 3. Infer from state dict
        if architecture_config is None:
            architecture_config = infer_architecture_config(state_dict)
            logger.info(f"Inferred architecture config from state dict: {architecture_config}")

        # Get model type for architecture selection
        model_type = metadata.model_type

        # Create model instance
        try:
            model = get_architecture(model_type, architecture_config)
            logger.info(f"Created {model_type} model with config: input_dim={architecture_config.get('input_dim')}, "
                       f"hidden_size={architecture_config.get('hidden_size')}, num_classes={architecture_config.get('num_classes')}")
        except Exception as e:
            logger.error(f"Failed to create model architecture: {e}")
            raise ValueError(f"Could not reconstruct model architecture: {e}")

        # Load state dict into model
        try:
            # Handle potential key mismatches (module. prefix from DataParallel, etc.)
            model_state = model.state_dict()
            filtered_state = {}

            for key, value in state_dict.items():
                # Remove 'module.' prefix if present (from DataParallel)
                clean_key = key.replace('module.', '')

                if clean_key in model_state:
                    if model_state[clean_key].shape == value.shape:
                        filtered_state[clean_key] = value
                    else:
                        logger.warning(f"Shape mismatch for {clean_key}: "
                                     f"model={model_state[clean_key].shape}, state_dict={value.shape}")
                else:
                    logger.debug(f"Key {clean_key} not in model architecture, skipping")

            # Load filtered state dict
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)

            if missing:
                logger.warning(f"Missing keys when loading state dict: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                logger.warning(f"Unexpected keys in state dict: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

            logger.info(f"Successfully loaded state dict into model ({len(filtered_state)} parameters)")

        except Exception as e:
            logger.error(f"Failed to load state dict: {e}")
            raise ValueError(f"Could not load state dict into model: {e}")

        # Set model to eval mode
        model.eval()

        return model
    
    def _calculate_model_size(self, model_dir: Path) -> float:
        """Calculate total size of model artifacts in MB"""
        total_size = 0
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _get_latest_version(self, model_id: str) -> str:
        """Get the latest version for a model ID"""
        versions = []
        for model_key, metadata in self.metadata_db.items():
            if metadata.model_id == model_id:
                versions.append((metadata.version, metadata.training_date))
        
        if not versions:
            raise ValueError(f"No versions found for model {model_id}")
        
        # Sort by training date and return latest version
        versions.sort(key=lambda x: x[1], reverse=True)
        return versions[0][0]
    
    def _load_registry(self):
        """Load registry from JSON file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)

                # Handle new format with "models" key
                if "models" in registry_data:
                    models_data = registry_data["models"]
                    self._production_model_key = registry_data.get("production_model")
                else:
                    # Legacy format: direct model_key -> metadata mapping
                    models_data = registry_data
                    self._production_model_key = None

                # Convert to ModelMetadata objects
                for model_key, metadata_dict in models_data.items():
                    try:
                        # Handle fields that may be missing in simpler registry format
                        defaults = {
                            "feature_importance": None,
                            "model_size_mb": metadata_dict.get("model_size_mb", 0.0),
                            "inference_latency_ms": 0.0,
                            "training_time_seconds": 0.0,
                            "validation_split": 0.2,
                            "test_split": 0.1,
                            "feature_columns": metadata_dict.get("feature_columns", []),
                            "target_column": metadata_dict.get("target_column", "target"),
                            "training_date": metadata_dict.get("training_date", datetime.now().isoformat()),
                        }
                        # Merge defaults with actual data
                        for key, default_val in defaults.items():
                            if key not in metadata_dict:
                                metadata_dict[key] = default_val

                        self.metadata_db[model_key] = ModelMetadata.from_dict(metadata_dict)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {model_key}: {e}")

                logger.info(f"Loaded {len(self.metadata_db)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.metadata_db = {}
        else:
            logger.info("No existing registry found, starting fresh")
    
    def _save_registry(self):
        """Save registry to JSON file, preserving production_model key."""
        try:
            # Build models dict
            models_data = {}
            for model_key, metadata in self.metadata_db.items():
                models_data[model_key] = metadata.to_dict()

            # Use the new format with production_model key at top level
            registry_data = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "production_model": getattr(self, "_production_model_key", None),
                "models": models_data,
            }

            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)

            logger.debug(f"Saved registry with {len(self.metadata_db)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry"""
        if not self.metadata_db:
            return {"total_models": 0}
        
        stats = {
            "total_models": len(self.metadata_db),
            "models_by_type": {},
            "models_by_status": {},
            "total_storage_mb": 0,
            "avg_model_size_mb": 0
        }
        
        # Count by type and status
        for metadata in self.metadata_db.values():
            # By type
            model_type = metadata.model_type
            stats["models_by_type"][model_type] = stats["models_by_type"].get(model_type, 0) + 1
            
            # By status
            status = metadata.deployment_status
            stats["models_by_status"][status] = stats["models_by_status"].get(status, 0) + 1
            
            # Storage
            stats["total_storage_mb"] += metadata.model_size_mb
        
        # Calculate averages
        if stats["total_models"] > 0:
            stats["avg_model_size_mb"] = stats["total_storage_mb"] / stats["total_models"]
        
        return stats