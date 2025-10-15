"""
Meta-Learning Module

This module implements meta-learning algorithms for the AI trading system,
including Model-Agnostic Meta-Learning (MAML), Reptile, and gradient-based
meta-learning for rapid adaptation to new market conditions.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import random
import os
import pickle

from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

@dataclass
class MetaTask:
    """Represents a meta-learning task."""
    task_id: str
    market_condition: str
    time_period: str
    training_data: List[Dict[str, Any]]
    validation_data: List[Dict[str, Any]]
    test_data: List[Dict[str, Any]]
    task_type: str  # "classification", "regression", "trading"

@dataclass
class MetaLearningResult:
    """Result of meta-learning training."""
    task_id: str
    initial_loss: float
    final_loss: float
    adaptation_steps: int
    convergence_time: float
    performance_metrics: Dict[str, float]

class MetaTradingNetwork(nn.Module):
    """Meta-learning network for trading decisions."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 3):
        super(MetaTradingNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # Buy/Sell classification
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Price prediction
        )
        
        self.trading_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)  # Trading actions
        )
    
    def forward(self, x, task_type="trading"):
        """Forward pass through the network."""
        features = self.feature_extractor(x)
        
        if task_type == "classification":
            return self.classification_head(features)
        elif task_type == "regression":
            return self.regression_head(features)
        else:  # trading
            return self.trading_head(features)

class MAML:
    """Model-Agnostic Meta-Learning implementation."""
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Training tracking
        self.training_step = 0
        self.meta_losses = []
        
        logger.info("MAML initialized")
    
    def meta_update(self, tasks: List[MetaTask], inner_steps: int = 5) -> Dict[str, Any]:
        """Perform one meta-update step."""
        meta_loss = 0.0
        task_results = []
        
        # Sample a batch of tasks
        batch_tasks = random.sample(tasks, min(len(tasks), 4))
        
        for task in batch_tasks:
            # Clone model for task-specific adaptation
            task_model = self._clone_model()
            task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)
            
            # Inner loop: adapt to task
            initial_loss = None
            final_loss = None
            
            for step in range(inner_steps):
                # Forward pass on training data
                train_loss = self._compute_task_loss(task_model, task.training_data, task.task_type)
                
                if step == 0:
                    initial_loss = train_loss.item()
                
                # Backward pass
                task_optimizer.zero_grad()
                train_loss.backward()
                task_optimizer.step()
            
            # Compute final loss on validation data
            val_loss = self._compute_task_loss(task_model, task.validation_data, task.task_type)
            final_loss = val_loss.item()
            
            # Accumulate meta-loss
            meta_loss += val_loss
            
            # Store task results
            task_results.append(MetaLearningResult(
                task_id=task.task_id,
                initial_loss=initial_loss,
                final_loss=final_loss,
                adaptation_steps=inner_steps,
                convergence_time=0.0,  # Placeholder
                performance_metrics={}
            ))
        
        # Meta-update: update base model parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        self.training_step += 1
        self.meta_losses.append(meta_loss.item())
        
        return {
            "meta_loss": meta_loss.item(),
            "task_results": task_results,
            "training_step": self.training_step
        }
    
    def _clone_model(self) -> nn.Module:
        """Create a clone of the model for task-specific adaptation."""
        cloned_model = MetaTradingNetwork(
            self.model.input_size,
            self.model.hidden_size,
            self.model.output_size
        )
        cloned_model.load_state_dict(self.model.state_dict())
        return cloned_model
    
    def _compute_task_loss(self, model: nn.Module, data: List[Dict[str, Any]], task_type: str) -> torch.Tensor:
        """Compute loss for a specific task."""
        if not data:
            return torch.tensor(0.0, requires_grad=True)
        
        # Convert data to tensors
        inputs = []
        targets = []
        
        for sample in data:
            inputs.append(sample["features"])
            targets.append(sample["target"])
        
        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)
        
        # Forward pass
        outputs = model(inputs, task_type)
        
        # Compute loss based on task type
        if task_type == "classification":
            loss_fn = nn.CrossEntropyLoss()
            targets = targets.long()
        elif task_type == "regression":
            loss_fn = nn.MSELoss()
        else:  # trading
            loss_fn = nn.CrossEntropyLoss()
            targets = targets.long()
        
        return loss_fn(outputs, targets)
    
    def save_model(self, filepath: str):
        """Save the meta-learned model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'training_step': self.training_step,
            'meta_losses': self.meta_losses
        }, filepath)
        logger.info(f"MAML model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a meta-learned model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            self.training_step = checkpoint['training_step']
            self.meta_losses = checkpoint['meta_losses']
            logger.info(f"MAML model loaded from {filepath}")
        else:
            logger.warning(f"MAML model file {filepath} not found")

class Reptile:
    """Reptile meta-learning algorithm."""
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Training tracking
        self.training_step = 0
        self.meta_losses = []
        
        logger.info("Reptile initialized")
    
    def meta_update(self, tasks: List[MetaTask], inner_steps: int = 5) -> Dict[str, Any]:
        """Perform one Reptile meta-update step."""
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        task_results = []
        
        # Sample a batch of tasks
        batch_tasks = random.sample(tasks, min(len(tasks), 4))
        
        for task in batch_tasks:
            # Clone model for task-specific adaptation
            task_model = self._clone_model()
            task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)
            
            # Inner loop: adapt to task
            initial_loss = None
            final_loss = None
            
            for step in range(inner_steps):
                # Forward pass on training data
                train_loss = self._compute_task_loss(task_model, task.training_data, task.task_type)
                
                if step == 0:
                    initial_loss = train_loss.item()
                
                # Backward pass
                task_optimizer.zero_grad()
                train_loss.backward()
                task_optimizer.step()
            
            # Compute final loss on validation data
            val_loss = self._compute_task_loss(task_model, task.validation_data, task.task_type)
            final_loss = val_loss.item()
            
            # Store task results
            task_results.append(MetaLearningResult(
                task_id=task.task_id,
                initial_loss=initial_loss,
                final_loss=final_loss,
                adaptation_steps=inner_steps,
                convergence_time=0.0,
                performance_metrics={}
            ))
        
        # Reptile update: move parameters towards task-adapted parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in initial_params:
                    # Compute parameter update
                    param_update = param - initial_params[name]
                    # Apply meta-learning rate
                    param.data = initial_params[name] + self.meta_lr * param_update
        
        self.training_step += 1
        
        return {
            "meta_loss": 0.0,  # Reptile doesn't compute explicit meta-loss
            "task_results": task_results,
            "training_step": self.training_step
        }
    
    def _clone_model(self) -> nn.Module:
        """Create a clone of the model for task-specific adaptation."""
        cloned_model = MetaTradingNetwork(
            self.model.input_size,
            self.model.hidden_size,
            self.model.output_size
        )
        cloned_model.load_state_dict(self.model.state_dict())
        return cloned_model
    
    def _compute_task_loss(self, model: nn.Module, data: List[Dict[str, Any]], task_type: str) -> torch.Tensor:
        """Compute loss for a specific task."""
        if not data:
            return torch.tensor(0.0, requires_grad=True)
        
        # Convert data to tensors
        inputs = []
        targets = []
        
        for sample in data:
            inputs.append(sample["features"])
            targets.append(sample["target"])
        
        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)
        
        # Forward pass
        outputs = model(inputs, task_type)
        
        # Compute loss based on task type
        if task_type == "classification":
            loss_fn = nn.CrossEntropyLoss()
            targets = targets.long()
        elif task_type == "regression":
            loss_fn = nn.MSELoss()
        else:  # trading
            loss_fn = nn.CrossEntropyLoss()
            targets = targets.long()
        
        return loss_fn(outputs, targets)
    
    def save_model(self, filepath: str):
        """Save the meta-learned model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'training_step': self.training_step,
            'meta_losses': self.meta_losses
        }, filepath)
        logger.info(f"Reptile model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a meta-learned model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            self.training_step = checkpoint['training_step']
            self.meta_losses = checkpoint['meta_losses']
            logger.info(f"Reptile model saved to {filepath}")
        else:
            logger.warning(f"Reptile model file {filepath} not found")

class MetaLearningTaskGenerator:
    """Generates meta-learning tasks from market data."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.task_counter = 0
        
        logger.info(f"Meta-learning task generator initialized for {mode} mode")
    
    def generate_trading_tasks(self, num_tasks: int = 10) -> List[MetaTask]:
        """Generate trading tasks from market data."""
        tasks = []
        
        for i in range(num_tasks):
            task_id = f"trading_task_{self.task_counter}"
            self.task_counter += 1
            
            # Generate synthetic market data for the task
            market_condition = random.choice(["bull", "bear", "sideways", "volatile"])
            time_period = random.choice(["intraday", "daily", "weekly"])
            
            # Generate training data
            training_data = self._generate_task_data(market_condition, time_period, 100)
            
            # Generate validation data
            validation_data = self._generate_task_data(market_condition, time_period, 50)
            
            # Generate test data
            test_data = self._generate_task_data(market_condition, time_period, 50)
            
            task = MetaTask(
                task_id=task_id,
                market_condition=market_condition,
                time_period=time_period,
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                task_type="trading"
            )
            
            tasks.append(task)
        
        logger.info(f"Generated {num_tasks} trading tasks")
        return tasks
    
    def generate_classification_tasks(self, num_tasks: int = 10) -> List[MetaTask]:
        """Generate classification tasks from market data."""
        tasks = []
        
        for i in range(num_tasks):
            task_id = f"classification_task_{self.task_counter}"
            self.task_counter += 1
            
            # Generate synthetic market data for the task
            market_condition = random.choice(["bull", "bear", "sideways", "volatile"])
            time_period = random.choice(["intraday", "daily", "weekly"])
            
            # Generate training data
            training_data = self._generate_classification_data(market_condition, time_period, 100)
            
            # Generate validation data
            validation_data = self._generate_classification_data(market_condition, time_period, 50)
            
            # Generate test data
            test_data = self._generate_classification_data(market_condition, time_period, 50)
            
            task = MetaTask(
                task_id=task_id,
                market_condition=market_condition,
                time_period=time_period,
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                task_type="classification"
            )
            
            tasks.append(task)
        
        logger.info(f"Generated {num_tasks} classification tasks")
        return tasks
    
    def generate_regression_tasks(self, num_tasks: int = 10) -> List[MetaTask]:
        """Generate regression tasks from market data."""
        tasks = []
        
        for i in range(num_tasks):
            task_id = f"regression_task_{self.task_counter}"
            self.task_counter += 1
            
            # Generate synthetic market data for the task
            market_condition = random.choice(["bull", "bear", "sideways", "volatile"])
            time_period = random.choice(["intraday", "daily", "weekly"])
            
            # Generate training data
            training_data = self._generate_regression_data(market_condition, time_period, 100)
            
            # Generate validation data
            validation_data = self._generate_regression_data(market_condition, time_period, 50)
            
            # Generate test data
            test_data = self._generate_regression_data(market_condition, time_period, 50)
            
            task = MetaTask(
                task_id=task_id,
                market_condition=market_condition,
                time_period=time_period,
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                task_type="regression"
            )
            
            tasks.append(task)
        
        logger.info(f"Generated {num_tasks} regression tasks")
        return tasks
    
    def _generate_task_data(self, market_condition: str, time_period: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic trading task data."""
        data = []
        
        for i in range(num_samples):
            # Generate market features
            features = self._generate_market_features(market_condition, time_period)
            
            # Generate trading action target
            target = self._generate_trading_target(features, market_condition)
            
            data.append({
                "features": features,
                "target": target
            })
        
        return data
    
    def _generate_classification_data(self, market_condition: str, time_period: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic classification task data."""
        data = []
        
        for i in range(num_samples):
            # Generate market features
            features = self._generate_market_features(market_condition, time_period)
            
            # Generate classification target (0: Sell, 1: Buy)
            target = self._generate_classification_target(features, market_condition)
            
            data.append({
                "features": features,
                "target": target
            })
        
        return data
    
    def _generate_regression_data(self, market_condition: str, time_period: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic regression task data."""
        data = []
        
        for i in range(num_samples):
            # Generate market features
            features = self._generate_market_features(market_condition, time_period)
            
            # Generate regression target (price change)
            target = self._generate_regression_target(features, market_condition)
            
            data.append({
                "features": features,
                "target": target
            })
        
        return data
    
    def _generate_market_features(self, market_condition: str, time_period: str) -> List[float]:
        """Generate synthetic market features."""
        features = []
        
        # Base features
        features.extend([
            random.uniform(0, 1),  # price
            random.uniform(0, 1),  # volume
            random.uniform(0, 1),  # volatility
            random.uniform(0, 1),  # rsi
            random.uniform(0, 1),  # macd
            random.uniform(0, 1),  # sma_20
            random.uniform(0, 1),  # sma_50
            random.uniform(0, 1),  # bollinger_position
            random.uniform(0, 1),  # sentiment
            random.uniform(0, 1),  # news_impact
        ])
        
        # Market condition specific features
        if market_condition == "bull":
            features.extend([1.0, 0.0, 0.0, 0.0])  # bull, bear, sideways, volatile
        elif market_condition == "bear":
            features.extend([0.0, 1.0, 0.0, 0.0])
        elif market_condition == "sideways":
            features.extend([0.0, 0.0, 1.0, 0.0])
        else:  # volatile
            features.extend([0.0, 0.0, 0.0, 1.0])
        
        # Time period specific features
        if time_period == "intraday":
            features.extend([1.0, 0.0, 0.0])  # intraday, daily, weekly
        elif time_period == "daily":
            features.extend([0.0, 1.0, 0.0])
        else:  # weekly
            features.extend([0.0, 0.0, 1.0])
        
        # Pad to fixed size
        while len(features) < 50:
            features.append(0.0)
        
        return features[:50]
    
    def _generate_trading_target(self, features: List[float], market_condition: str) -> int:
        """Generate trading action target."""
        # Simple rule-based target generation
        if market_condition == "bull":
            return random.choices([0, 1, 2], weights=[0.2, 0.6, 0.2])[0]  # HOLD, BUY, SELL
        elif market_condition == "bear":
            return random.choices([0, 1, 2], weights=[0.2, 0.2, 0.6])[0]
        else:
            return random.choices([0, 1, 2], weights=[0.6, 0.2, 0.2])[0]
    
    def _generate_classification_target(self, features: List[float], market_condition: str) -> int:
        """Generate classification target."""
        # Simple rule-based target generation
        if market_condition == "bull":
            return random.choices([0, 1], weights=[0.3, 0.7])[0]  # Sell, Buy
        elif market_condition == "bear":
            return random.choices([0, 1], weights=[0.7, 0.3])[0]
        else:
            return random.choices([0, 1], weights=[0.5, 0.5])[0]
    
    def _generate_regression_target(self, features: List[float], market_condition: str) -> float:
        """Generate regression target."""
        # Simple rule-based target generation
        if market_condition == "bull":
            return random.uniform(0.0, 0.05)  # Positive returns
        elif market_condition == "bear":
            return random.uniform(-0.05, 0.0)  # Negative returns
        else:
            return random.uniform(-0.02, 0.02)  # Small returns

class MetaLearningTrainer:
    """Trainer for meta-learning algorithms."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.task_generator = MetaLearningTaskGenerator(mode)
        
        # Initialize models
        self.input_size = 50
        self.hidden_size = 128
        self.output_size = 3
        
        self.maml_model = MetaTradingNetwork(self.input_size, self.hidden_size, self.output_size)
        self.reptile_model = MetaTradingNetwork(self.input_size, self.hidden_size, self.output_size)
        
        # Initialize meta-learning algorithms
        self.maml = MAML(self.maml_model)
        self.reptile = Reptile(self.reptile_model)
        
        # Training configuration
        self.meta_epochs = 100
        self.inner_steps = 5
        
        logger.info(f"Meta-learning trainer initialized for {mode} mode")
    
    def train_maml(self, meta_epochs: int = None) -> Dict[str, Any]:
        """Train MAML model."""
        if meta_epochs is None:
            meta_epochs = self.meta_epochs
        
        logger.info(f"Starting MAML training for {meta_epochs} meta-epochs")
        
        # Generate tasks
        tasks = self.task_generator.generate_trading_tasks(20)
        
        meta_losses = []
        task_results = []
        
        for epoch in range(meta_epochs):
            # Perform meta-update
            result = self.maml.meta_update(tasks, self.inner_steps)
            
            meta_losses.append(result["meta_loss"])
            task_results.extend(result["task_results"])
            
            if epoch % 10 == 0:
                avg_loss = np.mean(meta_losses[-10:])
                logger.info(f"MAML Epoch {epoch}, Avg Meta Loss: {avg_loss:.4f}")
        
        # Save trained model
        model_path = f"models/maml_model_{self.mode}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.maml.save_model(model_path)
        
        results = {
            "meta_epochs": meta_epochs,
            "final_meta_loss": meta_losses[-1] if meta_losses else 0.0,
            "avg_meta_loss": np.mean(meta_losses),
            "task_results": task_results,
            "model_path": model_path
        }
        
        logger.info(f"MAML training completed. Final meta loss: {results['final_meta_loss']:.4f}")
        return results
    
    def train_reptile(self, meta_epochs: int = None) -> Dict[str, Any]:
        """Train Reptile model."""
        if meta_epochs is None:
            meta_epochs = self.meta_epochs
        
        logger.info(f"Starting Reptile training for {meta_epochs} meta-epochs")
        
        # Generate tasks
        tasks = self.task_generator.generate_trading_tasks(20)
        
        task_results = []
        
        for epoch in range(meta_epochs):
            # Perform meta-update
            result = self.reptile.meta_update(tasks, self.inner_steps)
            
            task_results.extend(result["task_results"])
            
            if epoch % 10 == 0:
                logger.info(f"Reptile Epoch {epoch}")
        
        # Save trained model
        model_path = f"models/reptile_model_{self.mode}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.reptile.save_model(model_path)
        
        results = {
            "meta_epochs": meta_epochs,
            "task_results": task_results,
            "model_path": model_path
        }
        
        logger.info(f"Reptile training completed")
        return results
    
    def evaluate_meta_learning(self, algorithm: str = "maml", num_tasks: int = 10) -> Dict[str, Any]:
        """Evaluate meta-learning performance."""
        logger.info(f"Evaluating {algorithm} meta-learning performance")
        
        # Generate test tasks
        test_tasks = self.task_generator.generate_trading_tasks(num_tasks)
        
        if algorithm == "maml":
            model = self.maml_model
        elif algorithm == "reptile":
            model = self.reptile_model
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        task_performances = []
        
        for task in test_tasks:
            # Clone model for task adaptation
            task_model = MetaTradingNetwork(self.input_size, self.hidden_size, self.output_size)
            task_model.load_state_dict(model.state_dict())
            task_optimizer = optim.SGD(task_model.parameters(), lr=0.01)
            
            # Adapt to task
            for step in range(self.inner_steps):
                train_loss = self._compute_task_loss(task_model, task.training_data, task.task_type)
                task_optimizer.zero_grad()
                train_loss.backward()
                task_optimizer.step()
            
            # Evaluate on test data
            test_loss = self._compute_task_loss(task_model, task.test_data, task.task_type)
            
            task_performances.append({
                "task_id": task.task_id,
                "market_condition": task.market_condition,
                "test_loss": test_loss.item()
            })
        
        # Calculate average performance
        avg_test_loss = np.mean([perf["test_loss"] for perf in task_performances])
        
        results = {
            "algorithm": algorithm,
            "num_tasks": num_tasks,
            "avg_test_loss": avg_test_loss,
            "task_performances": task_performances
        }
        
        logger.info(f"{algorithm} evaluation completed. Avg test loss: {avg_test_loss:.4f}")
        return results
    
    def _compute_task_loss(self, model: nn.Module, data: List[Dict[str, Any]], task_type: str) -> torch.Tensor:
        """Compute loss for a specific task."""
        if not data:
            return torch.tensor(0.0, requires_grad=True)
        
        # Convert data to tensors
        inputs = []
        targets = []
        
        for sample in data:
            inputs.append(sample["features"])
            targets.append(sample["target"])
        
        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)
        
        # Forward pass
        outputs = model(inputs, task_type)
        
        # Compute loss based on task type
        if task_type == "classification":
            loss_fn = nn.CrossEntropyLoss()
            targets = targets.long()
        elif task_type == "regression":
            loss_fn = nn.MSELoss()
        else:  # trading
            loss_fn = nn.CrossEntropyLoss()
            targets = targets.long()
        
        return loss_fn(outputs, targets)
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get meta-learning system status."""
        return {
            "mode": self.mode,
            "maml_training_step": self.maml.training_step,
            "reptile_training_step": self.reptile.training_step,
            "maml_meta_losses": len(self.maml.meta_losses),
            "reptile_meta_losses": len(self.reptile.meta_losses),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size
        }

# Global meta-learning trainer instance
_meta_learning_trainer: Optional[MetaLearningTrainer] = None

def get_meta_learning_trainer(mode: str = None) -> MetaLearningTrainer:
    """Get the global meta-learning trainer instance."""
    global _meta_learning_trainer
    if _meta_learning_trainer is None:
        if mode is None:
            mode = get_current_mode()
        _meta_learning_trainer = MetaLearningTrainer(mode)
    return _meta_learning_trainer

def train_maml(meta_epochs: int = 100, mode: str = None) -> Dict[str, Any]:
    """Train MAML model."""
    return get_meta_learning_trainer(mode).train_maml(meta_epochs)

def train_reptile(meta_epochs: int = 100, mode: str = None) -> Dict[str, Any]:
    """Train Reptile model."""
    return get_meta_learning_trainer(mode).train_reptile(meta_epochs)

def evaluate_meta_learning(algorithm: str = "maml", num_tasks: int = 10, mode: str = None) -> Dict[str, Any]:
    """Evaluate meta-learning performance."""
    return get_meta_learning_trainer(mode).evaluate_meta_learning(algorithm, num_tasks)

def get_meta_learning_status(mode: str = None) -> Dict[str, Any]:
    """Get meta-learning status."""
    return get_meta_learning_trainer(mode).get_meta_learning_status()
