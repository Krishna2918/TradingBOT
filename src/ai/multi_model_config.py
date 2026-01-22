"""
Unified Configuration System for Multi-Model Training

This module provides comprehensive configuration management for different model types
including LSTM, Transformer, XGBoost, LightGBM with validation and template generation.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class BaseModelConfig(ABC):
    """Base configuration class for all model types"""
    
    # Common training parameters
    random_seed: int = 42
    validation_split: float = 0.2
    test_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Data parameters
    target_column: str = 'direction_1d'
    feature_columns: List[str] = field(default_factory=list)
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration parameters"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class LSTMConfig(BaseModelConfig):
    """Configuration for LSTM models"""
    
    # Architecture parameters
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Sequence parameters
    sequence_length: int = 252  # Trading days in a year
    
    # Optimization parameters
    optimizer: str = 'adam'  # adam, sgd, rmsprop
    scheduler: str = 'plateau'  # plateau, step, cosine
    
    # Memory optimization
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    
    def validate(self) -> bool:
        """Validate LSTM configuration"""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        
        return True

@dataclass
class TransformerConfig(BaseModelConfig):
    """Configuration for Transformer models"""
    
    # Architecture parameters
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    
    # Sequence parameters
    max_seq_length: int = 512
    sequence_length: int = 252
    
    # Attention parameters
    attention_dropout: float = 0.1
    use_causal_mask: bool = True
    
    # Optimization parameters
    optimizer: str = 'adamw'
    scheduler: str = 'warmup_cosine'
    warmup_steps: int = 1000
    
    # Memory optimization
    gradient_accumulation_steps: int = 2
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    def validate(self) -> bool:
        """Validate Transformer configuration"""
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.nhead <= 0:
            raise ValueError("nhead must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        
        return True

@dataclass
class XGBoostConfig(BaseModelConfig):
    """Configuration for XGBoost models"""
    
    # Core parameters
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: int = 1
    
    # Training parameters
    objective: str = 'multi:softprob'
    eval_metric: List[str] = field(default_factory=lambda: ['mlogloss', 'merror'])
    early_stopping_rounds: int = 50
    
    # Cross-validation
    cv_folds: int = 5
    stratified: bool = True
    
    # Performance
    n_jobs: int = -1
    tree_method: str = 'hist'  # hist, gpu_hist, exact
    
    # Class imbalance handling
    scale_pos_weight: Optional[float] = None
    
    def validate(self) -> bool:
        """Validate XGBoost configuration"""
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be between 0 and 1")
        if not 0 < self.colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be between 0 and 1")
        if self.cv_folds <= 1:
            raise ValueError("cv_folds must be greater than 1")
        
        return True

@dataclass
class LightGBMConfig(BaseModelConfig):
    """Configuration for LightGBM models"""
    
    # Core parameters
    num_leaves: int = 31
    learning_rate: float = 0.1
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    min_child_samples: int = 20
    min_child_weight: float = 1e-3
    
    # Training parameters
    objective: str = 'multiclass'
    metric: List[str] = field(default_factory=lambda: ['multi_logloss', 'multi_error'])
    early_stopping_rounds: int = 50
    
    # LightGBM specific
    boosting_type: str = 'gbdt'  # gbdt, dart, goss
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    
    # Performance
    n_jobs: int = -1
    device_type: str = 'cpu'  # cpu, gpu
    
    # Categorical features
    categorical_features: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate LightGBM configuration"""
        if self.num_leaves <= 1:
            raise ValueError("num_leaves must be greater than 1")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be between 0 and 1")
        if not 0 < self.colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be between 0 and 1")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction must be between 0 and 1")
        
        return True

@dataclass
class EnsembleConfig(BaseModelConfig):
    """Configuration for ensemble methods"""
    
    # Ensemble type
    ensemble_method: str = 'voting'  # voting, stacking, blending
    
    # Voting ensemble parameters
    voting_type: str = 'soft'  # hard, soft
    weights: Optional[List[float]] = None
    
    # Stacking ensemble parameters
    meta_learner: str = 'logistic_regression'  # logistic_regression, random_forest, xgboost
    use_probabilities: bool = True
    cv_folds: int = 5
    
    # Blending parameters
    blend_method: str = 'weighted_average'  # simple_average, weighted_average, rank_average
    
    # Model selection
    base_models: List[str] = field(default_factory=lambda: ['lstm', 'transformer', 'xgboost', 'lightgbm'])
    diversity_threshold: float = 0.1  # Minimum diversity required between base models
    
    def validate(self) -> bool:
        """Validate ensemble configuration"""
        valid_methods = ['voting', 'stacking', 'blending']
        if self.ensemble_method not in valid_methods:
            raise ValueError(f"ensemble_method must be one of {valid_methods}")
        
        if self.voting_type not in ['hard', 'soft']:
            raise ValueError("voting_type must be 'hard' or 'soft'")
        
        if self.weights and len(self.weights) != len(self.base_models):
            raise ValueError("weights length must match base_models length")
        
        if self.cv_folds <= 1:
            raise ValueError("cv_folds must be greater than 1")
        
        if not 0 <= self.diversity_threshold <= 1:
            raise ValueError("diversity_threshold must be between 0 and 1")
        
        return True

@dataclass
class HyperparameterOptimizationConfig:
    """Configuration for hyperparameter optimization"""
    
    # Optimization framework
    framework: str = 'optuna'  # optuna, hyperopt, skopt
    
    # Search parameters
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    
    # Search strategy
    sampler: str = 'tpe'  # tpe, random, grid, cmaes
    pruner: str = 'median'  # median, successive_halving, hyperband
    
    # Optimization objective
    objective_metric: str = 'val_accuracy'
    direction: str = 'maximize'  # maximize, minimize
    
    # Search spaces (model-specific)
    search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Parallel optimization
    n_jobs: int = 1
    
    def validate(self) -> bool:
        """Validate hyperparameter optimization configuration"""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        
        if self.direction not in ['maximize', 'minimize']:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        
        return True

@dataclass
class MultiModelConfig:
    """Unified configuration for multi-model training system"""
    
    # Model configurations
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    xgboost_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm_config: LightGBMConfig = field(default_factory=LightGBMConfig)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Hyperparameter optimization
    hyperopt_config: HyperparameterOptimizationConfig = field(default_factory=HyperparameterOptimizationConfig)
    
    # Training orchestration
    enabled_models: List[str] = field(default_factory=lambda: ['lstm', 'transformer', 'xgboost', 'lightgbm'])
    parallel_training: bool = True
    max_concurrent_models: int = 4
    
    # Resource allocation
    resource_allocation: Dict[str, float] = field(default_factory=lambda: {
        'lstm': 0.4,
        'transformer': 0.4,
        'xgboost': 0.1,
        'lightgbm': 0.1
    })
    
    # Data configuration
    data_path: str = "TrainingData/processed"
    target_column: str = 'direction_1d'
    
    # Validation configuration
    use_walk_forward_validation: bool = True
    validation_windows: int = 5
    
    # Output configuration
    model_save_path: str = "models/multi_model"
    results_save_path: str = "results/multi_model"
    
    # Integration settings
    memory_manager_enabled: bool = True
    feature_consistency_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate all configuration components"""
        # Validate individual model configs
        if 'lstm' in self.enabled_models:
            self.lstm_config.validate()
        if 'transformer' in self.enabled_models:
            self.transformer_config.validate()
        if 'xgboost' in self.enabled_models:
            self.xgboost_config.validate()
        if 'lightgbm' in self.enabled_models:
            self.lightgbm_config.validate()
        
        # Validate ensemble config if ensemble is enabled
        if any(model in self.enabled_models for model in ['voting', 'stacking', 'blending']):
            self.ensemble_config.validate()
        
        # Validate hyperopt config
        self.hyperopt_config.validate()
        
        # Validate orchestration settings
        if not self.enabled_models:
            raise ValueError("At least one model must be enabled")
        
        if sum(self.resource_allocation.values()) > 1.0:
            raise ValueError("Resource allocation cannot exceed 100%")
        
        if self.max_concurrent_models < 1:
            raise ValueError("max_concurrent_models must be at least 1")
        
        if self.validation_windows < 1:
            raise ValueError("validation_windows must be at least 1")
        
        return True
    
    def get_model_config(self, model_type: str) -> BaseModelConfig:
        """Get configuration for specific model type"""
        config_map = {
            'lstm': self.lstm_config,
            'transformer': self.transformer_config,
            'xgboost': self.xgboost_config,
            'lightgbm': self.lightgbm_config,
            'ensemble': self.ensemble_config
        }
        
        if model_type not in config_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return config_map[model_type]

class ConfigurationManager:
    """
    Configuration manager for loading, saving, and validating configurations
    """
    
    def __init__(self, config_dir: str = "config/multi_model"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ConfigurationManager at {self.config_dir}")
    
    def save_config(self, config: MultiModelConfig, filename: str = "config.yaml") -> Path:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            filename: Output filename
            
        Returns:
            Path to saved configuration file
        """
        config_path = self.config_dir / filename
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Save as YAML for readability
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            # Save as JSON
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        return config_path
    
    def load_config(self, filename: str = "config.yaml") -> MultiModelConfig:
        """
        Load configuration from file
        
        Args:
            filename: Configuration filename
            
        Returns:
            Loaded configuration
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        
        # Convert to MultiModelConfig
        config = self._dict_to_config(config_dict)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def create_template_configs(self):
        """Create template configuration files for different scenarios"""
        
        # Development configuration (fast training)
        dev_config = MultiModelConfig(
            enabled_models=['lstm', 'xgboost'],
            lstm_config=LSTMConfig(
                hidden_size=64,
                num_layers=1,
                epochs=10,
                batch_size=32
            ),
            xgboost_config=XGBoostConfig(
                n_estimators=100,
                max_depth=3
            ),
            parallel_training=False,
            hyperopt_config=HyperparameterOptimizationConfig(
                n_trials=10
            )
        )
        self.save_config(dev_config, "development.yaml")
        
        # Production configuration (comprehensive training)
        prod_config = MultiModelConfig(
            enabled_models=['lstm', 'transformer', 'xgboost', 'lightgbm'],
            parallel_training=True,
            max_concurrent_models=4,
            hyperopt_config=HyperparameterOptimizationConfig(
                n_trials=200,
                timeout_seconds=3600 * 24  # 24 hours
            )
        )
        self.save_config(prod_config, "production.yaml")
        
        # Testing configuration (minimal resources)
        test_config = MultiModelConfig(
            enabled_models=['xgboost'],
            xgboost_config=XGBoostConfig(
                n_estimators=10,
                max_depth=2
            ),
            parallel_training=False,
            hyperopt_config=HyperparameterOptimizationConfig(
                n_trials=5
            )
        )
        self.save_config(test_config, "testing.yaml")
        
        logger.info("Created template configuration files")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MultiModelConfig:
        """Convert dictionary to MultiModelConfig object"""
        
        # Extract nested configurations
        lstm_config = LSTMConfig(**config_dict.get('lstm_config', {}))
        transformer_config = TransformerConfig(**config_dict.get('transformer_config', {}))
        xgboost_config = XGBoostConfig(**config_dict.get('xgboost_config', {}))
        lightgbm_config = LightGBMConfig(**config_dict.get('lightgbm_config', {}))
        ensemble_config = EnsembleConfig(**config_dict.get('ensemble_config', {}))
        hyperopt_config = HyperparameterOptimizationConfig(**config_dict.get('hyperopt_config', {}))
        
        # Create main config
        main_config_dict = {k: v for k, v in config_dict.items() 
                           if k not in ['lstm_config', 'transformer_config', 'xgboost_config', 
                                       'lightgbm_config', 'ensemble_config', 'hyperopt_config']}
        
        config = MultiModelConfig(
            lstm_config=lstm_config,
            transformer_config=transformer_config,
            xgboost_config=xgboost_config,
            lightgbm_config=lightgbm_config,
            ensemble_config=ensemble_config,
            hyperopt_config=hyperopt_config,
            **main_config_dict
        )
        
        return config
    
    def validate_config_file(self, filename: str) -> bool:
        """
        Validate a configuration file
        
        Args:
            filename: Configuration filename
            
        Returns:
            True if valid, raises exception if invalid
        """
        try:
            config = self.load_config(filename)
            config.validate()
            logger.info(f"Configuration file {filename} is valid")
            return True
        except Exception as e:
            logger.error(f"Configuration file {filename} is invalid: {e}")
            raise