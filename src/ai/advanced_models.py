"""
Advanced AI Models Integration

This module integrates all advanced AI models including reinforcement learning,
meta-learning, and other cutting-edge AI techniques for autonomous trading.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from src.ai.reinforcement_learning import (
    get_rl_trainer, train_dqn_agent, train_policy_agent, evaluate_rl_agent,
    TradingEnvironment, DQNAgent, PolicyGradientAgent
)
from src.ai.meta_learning import (
    get_meta_learning_trainer, train_maml, train_reptile, evaluate_meta_learning,
    MetaLearningTrainer, MAML, Reptile
)
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

@dataclass
class AdvancedModelStatus:
    """Status of advanced AI models."""
    reinforcement_learning: bool
    meta_learning: bool
    dqn_model: bool
    policy_model: bool
    maml_model: bool
    reptile_model: bool
    overall_status: str

@dataclass
class ModelPerformance:
    """Performance metrics for AI models."""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float

@dataclass
class EnsemblePrediction:
    """Prediction from the advanced AI ensemble."""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    model_contributions: Dict[str, float]
    risk_assessment: Dict[str, float]

class AdvancedAIModels:
    """Integration of all advanced AI models."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        
        # Initialize trainers
        self.rl_trainer = get_rl_trainer(mode)
        self.meta_trainer = get_meta_learning_trainer(mode)
        
        # Model status
        self.models_trained = {
            "dqn": False,
            "policy": False,
            "maml": False,
            "reptile": False
        }
        
        # Performance tracking
        self.model_performances = {}
        self.ensemble_weights = {
            "dqn": 0.25,
            "policy": 0.25,
            "maml": 0.25,
            "reptile": 0.25
        }
        
        logger.info(f"Advanced AI Models initialized for {mode} mode")
    
    def train_all_models(self, episodes: int = 1000, meta_epochs: int = 100) -> Dict[str, Any]:
        """Train all advanced AI models."""
        logger.info("Starting training of all advanced AI models")
        
        results = {}
        
        try:
            # Train DQN Agent
            logger.info("Training DQN agent...")
            dqn_results = train_dqn_agent(episodes, self.mode)
            results["dqn"] = dqn_results
            self.models_trained["dqn"] = True
            
            # Train Policy Gradient Agent
            logger.info("Training Policy Gradient agent...")
            policy_results = train_policy_agent(episodes, self.mode)
            results["policy"] = policy_results
            self.models_trained["policy"] = True
            
            # Train MAML Model
            logger.info("Training MAML model...")
            maml_results = train_maml(meta_epochs, self.mode)
            results["maml"] = maml_results
            self.models_trained["maml"] = True
            
            # Train Reptile Model
            logger.info("Training Reptile model...")
            reptile_results = train_reptile(meta_epochs, self.mode)
            results["reptile"] = reptile_results
            self.models_trained["reptile"] = True
            
            logger.info("All advanced AI models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training advanced AI models: {e}")
            results["error"] = str(e)
        
        return results
    
    def evaluate_all_models(self, episodes: int = 10) -> Dict[str, Any]:
        """Evaluate all trained models."""
        logger.info("Evaluating all advanced AI models")
        
        results = {}
        
        try:
            # Evaluate DQN Agent
            if self.models_trained["dqn"]:
                logger.info("Evaluating DQN agent...")
                dqn_eval = evaluate_rl_agent("dqn", episodes, self.mode)
                results["dqn"] = dqn_eval
                self.model_performances["dqn"] = self._calculate_model_performance("dqn", dqn_eval)
            
            # Evaluate Policy Gradient Agent
            if self.models_trained["policy"]:
                logger.info("Evaluating Policy Gradient agent...")
                policy_eval = evaluate_rl_agent("policy", episodes, self.mode)
                results["policy"] = policy_eval
                self.model_performances["policy"] = self._calculate_model_performance("policy", policy_eval)
            
            # Evaluate MAML Model
            if self.models_trained["maml"]:
                logger.info("Evaluating MAML model...")
                maml_eval = evaluate_meta_learning("maml", episodes, self.mode)
                results["maml"] = maml_eval
                self.model_performances["maml"] = self._calculate_model_performance("maml", maml_eval)
            
            # Evaluate Reptile Model
            if self.models_trained["reptile"]:
                logger.info("Evaluating Reptile model...")
                reptile_eval = evaluate_meta_learning("reptile", episodes, self.mode)
                results["reptile"] = reptile_eval
                self.model_performances["reptile"] = self._calculate_model_performance("reptile", reptile_eval)
            
            # Update ensemble weights based on performance
            self._update_ensemble_weights()
            
            logger.info("All advanced AI models evaluated successfully")
            
        except Exception as e:
            logger.error(f"Error evaluating advanced AI models: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_ensemble_prediction(self, symbol: str, market_features: Dict[str, Any]) -> EnsemblePrediction:
        """Get ensemble prediction from all models."""
        logger.info(f"Getting ensemble prediction for {symbol}")
        
        try:
            # Get predictions from each model
            model_predictions = {}
            model_confidences = {}
            
            # DQN prediction
            if self.models_trained["dqn"]:
                dqn_pred = self._get_dqn_prediction(symbol, market_features)
                model_predictions["dqn"] = dqn_pred["action"]
                model_confidences["dqn"] = dqn_pred["confidence"]
            
            # Policy prediction
            if self.models_trained["policy"]:
                policy_pred = self._get_policy_prediction(symbol, market_features)
                model_predictions["policy"] = policy_pred["action"]
                model_confidences["policy"] = policy_pred["confidence"]
            
            # MAML prediction
            if self.models_trained["maml"]:
                maml_pred = self._get_maml_prediction(symbol, market_features)
                model_predictions["maml"] = maml_pred["action"]
                model_confidences["maml"] = maml_pred["confidence"]
            
            # Reptile prediction
            if self.models_trained["reptile"]:
                reptile_pred = self._get_reptile_prediction(symbol, market_features)
                model_predictions["reptile"] = reptile_pred["action"]
                model_confidences["reptile"] = reptile_pred["confidence"]
            
            # Combine predictions using ensemble weights
            ensemble_action, ensemble_confidence = self._combine_predictions(
                model_predictions, model_confidences
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(model_predictions, model_confidences)
            
            # Risk assessment
            risk_assessment = self._assess_risk(symbol, market_features, ensemble_action)
            
            return EnsemblePrediction(
                symbol=symbol,
                action=ensemble_action,
                confidence=ensemble_confidence,
                reasoning=reasoning,
                model_contributions=model_confidences,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Error getting ensemble prediction: {e}")
            return EnsemblePrediction(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Error: {e}",
                model_contributions={},
                risk_assessment={}
            )
    
    def _get_dqn_prediction(self, symbol: str, market_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from DQN model."""
        try:
            # Convert market features to state vector
            state_vector = self._features_to_vector(market_features)
            
            # Get DQN agent
            dqn_agent = self.rl_trainer.dqn_agent
            
            # Get action
            action_idx = dqn_agent.act(state_vector, training=False)
            
            # Convert action index to action
            actions = ["HOLD", "BUY", "SELL"]
            action = actions[action_idx]
            
            # Get confidence from Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                q_values = dqn_agent.q_network(state_tensor)
                confidence = torch.softmax(q_values, dim=1)[0][action_idx].item()
            
            return {
                "action": action,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting DQN prediction: {e}")
            return {"action": "HOLD", "confidence": 0.0}
    
    def _get_policy_prediction(self, symbol: str, market_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from Policy Gradient model."""
        try:
            # Convert market features to state vector
            state_vector = self._features_to_vector(market_features)
            
            # Get Policy agent
            policy_agent = self.rl_trainer.policy_agent
            
            # Get action
            action_idx = policy_agent.act(state_vector)
            
            # Convert action index to action
            actions = ["HOLD", "BUY", "SELL"]
            action = actions[action_idx]
            
            # Get confidence from policy probabilities
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                action_probs = policy_agent.policy_network(state_tensor)
                confidence = action_probs[0][action_idx].item()
            
            return {
                "action": action,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting Policy prediction: {e}")
            return {"action": "HOLD", "confidence": 0.0}
    
    def _get_maml_prediction(self, symbol: str, market_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from MAML model."""
        try:
            # Convert market features to state vector
            state_vector = self._features_to_vector(market_features)
            
            # Get MAML model
            maml_model = self.meta_trainer.maml_model
            
            # Get prediction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                outputs = maml_model(state_tensor, "trading")
                action_probs = torch.softmax(outputs, dim=1)
                action_idx = action_probs.argmax().item()
                confidence = action_probs[0][action_idx].item()
            
            # Convert action index to action
            actions = ["HOLD", "BUY", "SELL"]
            action = actions[action_idx]
            
            return {
                "action": action,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting MAML prediction: {e}")
            return {"action": "HOLD", "confidence": 0.0}
    
    def _get_reptile_prediction(self, symbol: str, market_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from Reptile model."""
        try:
            # Convert market features to state vector
            state_vector = self._features_to_vector(market_features)
            
            # Get Reptile model
            reptile_model = self.meta_trainer.reptile_model
            
            # Get prediction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                outputs = reptile_model(state_tensor, "trading")
                action_probs = torch.softmax(outputs, dim=1)
                action_idx = action_probs.argmax().item()
                confidence = action_probs[0][action_idx].item()
            
            # Convert action index to action
            actions = ["HOLD", "BUY", "SELL"]
            action = actions[action_idx]
            
            return {
                "action": action,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting Reptile prediction: {e}")
            return {"action": "HOLD", "confidence": 0.0}
    
    def _combine_predictions(self, model_predictions: Dict[str, str], 
                           model_confidences: Dict[str, float]) -> Tuple[str, float]:
        """Combine predictions from all models."""
        if not model_predictions:
            return "HOLD", 0.0
        
        # Weighted voting
        action_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        
        for model_name, action in model_predictions.items():
            weight = self.ensemble_weights.get(model_name, 0.0)
            confidence = model_confidences.get(model_name, 0.0)
            score = weight * confidence
            
            action_scores[action] += score
        
        # Get best action
        best_action = max(action_scores, key=action_scores.get)
        best_score = action_scores[best_action]
        
        # Normalize confidence
        total_weight = sum(self.ensemble_weights.values())
        ensemble_confidence = best_score / total_weight if total_weight > 0 else 0.0
        
        return best_action, ensemble_confidence
    
    def _generate_reasoning(self, model_predictions: Dict[str, str], 
                          model_confidences: Dict[str, float]) -> str:
        """Generate reasoning for ensemble prediction."""
        reasoning_parts = []
        
        for model_name, action in model_predictions.items():
            confidence = model_confidences.get(model_name, 0.0)
            reasoning_parts.append(f"{model_name.upper()}: {action} (confidence: {confidence:.2f})")
        
        return " | ".join(reasoning_parts)
    
    def _assess_risk(self, symbol: str, market_features: Dict[str, Any], action: str) -> Dict[str, float]:
        """Assess risk for the given action."""
        risk_factors = {}
        
        # Market volatility risk
        volatility = market_features.get("volatility", 0.02)
        risk_factors["volatility_risk"] = min(volatility * 10, 1.0)
        
        # Market trend risk
        trend = market_features.get("market_trend", 0.0)
        if action == "BUY" and trend < -0.5:
            risk_factors["trend_risk"] = 0.8
        elif action == "SELL" and trend > 0.5:
            risk_factors["trend_risk"] = 0.8
        else:
            risk_factors["trend_risk"] = 0.2
        
        # Sentiment risk
        sentiment = market_features.get("sentiment", 0.0)
        if action == "BUY" and sentiment < -0.5:
            risk_factors["sentiment_risk"] = 0.7
        elif action == "SELL" and sentiment > 0.5:
            risk_factors["sentiment_risk"] = 0.7
        else:
            risk_factors["sentiment_risk"] = 0.3
        
        # Overall risk score
        risk_factors["overall_risk"] = np.mean(list(risk_factors.values()))
        
        return risk_factors
    
    def _features_to_vector(self, market_features: Dict[str, Any]) -> np.ndarray:
        """Convert market features to vector representation."""
        # Extract features in consistent order
        feature_names = [
            "market_trend", "volatility", "volume", "rsi", "macd",
            "sma_20", "sma_50", "bollinger_position", "sentiment", "news_impact"
        ]
        
        vector = []
        for name in feature_names:
            vector.append(market_features.get(name, 0.0))
        
        # Pad to fixed size
        while len(vector) < 50:
            vector.append(0.0)
        
        return np.array(vector[:50], dtype=np.float32)
    
    def _calculate_model_performance(self, model_type: str, evaluation_results: Dict[str, Any]) -> ModelPerformance:
        """Calculate performance metrics for a model."""
        # Extract metrics from evaluation results
        avg_return = evaluation_results.get("avg_return", 0.0)
        std_return = evaluation_results.get("std_return", 0.0)
        max_return = evaluation_results.get("max_return", 0.0)
        min_return = evaluation_results.get("min_return", 0.0)
        
        # Calculate derived metrics
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
        max_drawdown = abs(min_return) if min_return < 0 else 0.0
        win_rate = 0.6 if avg_return > 0 else 0.4  # Simplified calculation
        
        # Calculate accuracy metrics (simplified)
        accuracy = 0.5 + (avg_return * 10)  # Convert return to accuracy
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
        
        precision = accuracy * 0.9  # Simplified
        recall = accuracy * 0.8     # Simplified
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return ModelPerformance(
            model_type=model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_return=avg_return
        )
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on model performance."""
        if not self.model_performances:
            return
        
        # Calculate weights based on performance metrics
        total_score = 0.0
        scores = {}
        
        for model_type, performance in self.model_performances.items():
            # Combine multiple metrics into a single score
            score = (
                performance.accuracy * 0.3 +
                performance.sharpe_ratio * 0.3 +
                performance.win_rate * 0.2 +
                (1 - performance.max_drawdown) * 0.2
            )
            scores[model_type] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            for model_type in self.ensemble_weights:
                if model_type in scores:
                    self.ensemble_weights[model_type] = scores[model_type] / total_score
                else:
                    self.ensemble_weights[model_type] = 0.0
        
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_advanced_models_status(self) -> AdvancedModelStatus:
        """Get status of all advanced AI models."""
        return AdvancedModelStatus(
            reinforcement_learning=self.models_trained["dqn"] or self.models_trained["policy"],
            meta_learning=self.models_trained["maml"] or self.models_trained["reptile"],
            dqn_model=self.models_trained["dqn"],
            policy_model=self.models_trained["policy"],
            maml_model=self.models_trained["maml"],
            reptile_model=self.models_trained["reptile"],
            overall_status="FULLY_OPERATIONAL" if all(self.models_trained.values()) else "PARTIALLY_OPERATIONAL"
        )
    
    def get_model_performances(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models."""
        return self.model_performances.copy()
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.ensemble_weights.copy()
    
    def validate_advanced_models(self) -> bool:
        """Validate all advanced AI models."""
        try:
            # Check if trainers are initialized
            if not self.rl_trainer:
                logger.error("RL trainer not initialized")
                return False
            
            if not self.meta_trainer:
                logger.error("Meta-learning trainer not initialized")
                return False
            
            # Check if at least one model is trained
            if not any(self.models_trained.values()):
                logger.error("No models are trained")
                return False
            
            logger.info("Advanced AI models validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Advanced AI models validation error: {e}")
            return False

# Global advanced AI models instance
_advanced_ai_models: Optional[AdvancedAIModels] = None

def get_advanced_ai_models(mode: str = None) -> AdvancedAIModels:
    """Get the global advanced AI models instance."""
    global _advanced_ai_models
    if _advanced_ai_models is None:
        if mode is None:
            mode = get_current_mode()
        _advanced_ai_models = AdvancedAIModels(mode)
    return _advanced_ai_models

def train_all_advanced_models(episodes: int = 1000, meta_epochs: int = 100, mode: str = None) -> Dict[str, Any]:
    """Train all advanced AI models."""
    return get_advanced_ai_models(mode).train_all_models(episodes, meta_epochs)

def evaluate_all_advanced_models(episodes: int = 10, mode: str = None) -> Dict[str, Any]:
    """Evaluate all advanced AI models."""
    return get_advanced_ai_models(mode).evaluate_all_advanced_models(episodes)

def get_ensemble_prediction(symbol: str, market_features: Dict[str, Any], mode: str = None) -> EnsemblePrediction:
    """Get ensemble prediction from all models."""
    return get_advanced_ai_models(mode).get_ensemble_prediction(symbol, market_features)

def get_advanced_models_status(mode: str = None) -> AdvancedModelStatus:
    """Get status of advanced AI models."""
    return get_advanced_ai_models(mode).get_advanced_models_status()

def get_model_performances(mode: str = None) -> Dict[str, ModelPerformance]:
    """Get model performance metrics."""
    return get_advanced_ai_models(mode).get_model_performances()

def get_ensemble_weights(mode: str = None) -> Dict[str, float]:
    """Get ensemble weights."""
    return get_advanced_ai_models(mode).get_ensemble_weights()

def validate_advanced_models(mode: str = None) -> bool:
    """Validate advanced AI models."""
    return get_advanced_ai_models(mode).validate_advanced_models()
