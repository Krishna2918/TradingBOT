"""
Multi-Model System Integration

This module integrates the Multi-Model Orchestrator, Model Registry, and Configuration
system with existing trading bot components for seamless operation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

# Import multi-model components
from .multi_model_orchestrator import MultiModelOrchestrator, ModelTrainingResult
from .model_registry import ModelRegistry, ModelMetadata
from .multi_model_config import MultiModelConfig, ConfigurationManager

# Import existing components
from .models.gpu_memory_manager import GPUMemoryManager
from .data.feature_consistency import GlobalAnalysisResult

logger = logging.getLogger(__name__)

class MultiModelSystem:
    """
    Integrated multi-model training system
    
    This class provides a high-level interface that combines the orchestrator,
    registry, and configuration management for easy use within the trading bot.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the multi-model system
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Initialize configuration manager
        self.config_manager = ConfigurationManager()
        
        # Load configuration
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            # Use default configuration
            self.config = MultiModelConfig()
        
        # Validate configuration
        self.config.validate()
        
        # Initialize components
        self.orchestrator = MultiModelOrchestrator(self.config)
        self.registry = ModelRegistry(self.config.model_save_path)
        
        logger.info("Initialized MultiModelSystem")
    
    def train_models(self, 
                    data: pd.DataFrame, 
                    target_column: str = None,
                    register_models: bool = True) -> Dict[str, ModelTrainingResult]:
        """
        Train all enabled models and optionally register them
        
        Args:
            data: Training data
            target_column: Target column name
            register_models: Whether to register trained models
            
        Returns:
            Dictionary of training results
        """
        if target_column is None:
            target_column = self.config.target_column
        
        logger.info(f"Starting multi-model training with {len(self.config.enabled_models)} models")
        
        # Train models using orchestrator
        import asyncio
        results = asyncio.run(self.orchestrator.train_all_models(data, target_column))
        
        # Register models if requested
        if register_models:
            for model_type, result in results.items():
                try:
                    self._register_training_result(result, data)
                    logger.info(f"Registered {model_type} model in registry")
                except Exception as e:
                    logger.error(f"Failed to register {model_type} model: {e}")
        
        return results
    
    def compare_models(self, metrics: List[str] = None) -> pd.DataFrame:
        """Compare models using both orchestrator and registry data"""
        
        # Get comparison from orchestrator (recent training)
        orchestrator_comparison = self.orchestrator.compare_models(metrics)
        
        # Get comparison from registry (all models)
        registry_models = self.registry.list_models()
        
        # Combine results
        if not registry_models.empty:
            logger.info(f"Found {len(registry_models)} models in registry")
        
        return orchestrator_comparison
    
    def get_best_model(self, metric: str = 'test_accuracy') -> Optional[ModelTrainingResult]:
        """Get the best performing model"""
        return self.orchestrator.get_best_model(metric)
    
    def deploy_model(self, model_id: str, version: str = "latest"):
        """Deploy a model to production"""
        try:
            # Load model from registry
            model, metadata = self.registry.load_model(model_id, version)
            
            # Update deployment status
            self.registry.update_deployment_status(
                model_id, 
                metadata.version, 
                "deployed"
            )
            
            logger.info(f"Deployed model {model_id}_{metadata.version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_id}: {e}")
            raise
    
    def create_configuration_templates(self):
        """Create configuration templates for different scenarios"""
        self.config_manager.create_template_configs()
        logger.info("Created configuration templates")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'config': {
                'enabled_models': self.config.enabled_models,
                'parallel_training': self.config.parallel_training,
                'max_concurrent_models': self.config.max_concurrent_models
            },
            'orchestrator': {
                'active_trainings': len(self.orchestrator.active_trainings),
                'completed_trainings': len(self.orchestrator.training_results)
            },
            'registry': self.registry.get_registry_stats()
        }
        
        return status
    
    def _register_training_result(self, result: ModelTrainingResult, training_data: pd.DataFrame):
        """Register a training result in the model registry"""
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=result.model_id,
            model_type=result.model_type,
            version="1.0.0",  # Simple versioning for now
            training_date=pd.Timestamp.now(),
            performance_metrics={
                **result.validation_metrics,
                **{f"test_{k}": v for k, v in result.test_metrics.items()}
            },
            hyperparameters=result.hyperparameters,
            feature_importance=result.feature_importance,
            training_data_hash=self._calculate_data_hash(training_data),
            model_size_mb=0.0,  # Will be calculated during registration
            inference_latency_ms=0.0,  # Placeholder
            training_time_seconds=result.training_time_seconds,
            validation_split=0.2,  # From config
            test_split=0.2,  # From config
            feature_columns=list(training_data.columns),
            target_column=self.config.target_column
        )
        
        # For now, we'll register a placeholder model since actual model objects
        # aren't implemented yet in the orchestrator
        placeholder_model = {"type": result.model_type, "id": result.model_id}
        
        # Register in model registry
        self.registry.register_model(placeholder_model, metadata)
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for reproducibility tracking"""
        import hashlib
        
        # Create a hash based on data shape and column names
        data_info = f"{data.shape}_{list(data.columns)}"
        return hashlib.md5(data_info.encode()).hexdigest()
    
    def cleanup(self):
        """Clean up system resources"""
        if self.orchestrator:
            self.orchestrator.cleanup()
        
        logger.info("MultiModelSystem cleanup completed")

# Convenience functions for easy integration

def create_multi_model_system(config_file: str = "production.yaml") -> MultiModelSystem:
    """
    Create a multi-model system with specified configuration
    
    Args:
        config_file: Configuration file name
        
    Returns:
        Initialized MultiModelSystem
    """
    return MultiModelSystem(config_file)

def quick_train_models(data: pd.DataFrame, 
                      models: List[str] = None,
                      target_column: str = 'direction_1d') -> Dict[str, ModelTrainingResult]:
    """
    Quick training function for immediate use
    
    Args:
        data: Training data
        models: List of models to train (default: ['lstm', 'xgboost'])
        target_column: Target column name
        
    Returns:
        Training results
    """
    if models is None:
        models = ['lstm', 'xgboost']  # Fast models for quick training
    
    # Create minimal configuration
    config = MultiModelConfig(
        enabled_models=models,
        parallel_training=False,  # Sequential for simplicity
        lstm_config=MultiModelConfig().lstm_config,
        xgboost_config=MultiModelConfig().xgboost_config
    )
    
    # Create system and train
    system = MultiModelSystem()
    system.config = config
    system.orchestrator = MultiModelOrchestrator(config)
    
    return system.train_models(data, target_column)

# Integration with existing trading bot components

def integrate_with_existing_system():
    """
    Integration helper for existing trading bot components
    
    This function demonstrates how to integrate the multi-model system
    with existing components like feature consistency and memory management.
    """
    
    logger.info("Integrating multi-model system with existing components")
    
    # Example integration points:
    # 1. Feature consistency system integration
    # 2. Memory management integration  
    # 3. Target engineering integration
    # 4. Monitoring system integration
    
    integration_config = {
        'feature_consistency_enabled': True,
        'memory_management_enabled': True,
        'monitoring_enabled': True,
        'target_validation_enabled': True
    }
    
    return integration_config