"""
Advanced Model Integration for Unified AI System

This module provides unified interfaces for integrating all advanced AI models
including deep learning, time series, reinforcement learning, and NLP components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from collections import defaultdict, deque
import warnings

# Import all advanced model components
try:
    from ..deep_learning.model_manager import DeepLearningManager
    from ..time_series.time_series_manager import TimeSeriesManager
    from ..reinforcement_learning.rl_manager import ReinforcementLearningManager
    from ..natural_language_processing.nlp_manager import NaturalLanguageProcessingManager
    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    ADVANCED_MODELS_AVAILABLE = False
    warnings.warn(f"Advanced models not available: {e}")

# Import existing system components
try:
    from ..multi_model import MultiModelAI
    from ..enhanced_ensemble import EnhancedEnsemble
    from ..adaptive_weights import AdaptiveWeights
    from ..regime_detection import RegimeDetector
    EXISTING_MODELS_AVAILABLE = True
except ImportError as e:
    EXISTING_MODELS_AVAILABLE = False
    warnings.warn(f"Existing models not available: {e}")

logger = logging.getLogger(__name__)

class AdvancedModelIntegration:
    """
    Unified interface for integrating all advanced AI models.
    """
    
    def __init__(
        self,
        model_name: str = "advanced_model_integration",
        enable_deep_learning: bool = True,
        enable_time_series: bool = True,
        enable_reinforcement_learning: bool = True,
        enable_nlp: bool = True,
        max_concurrent_models: int = 4
    ):
        """
        Initialize advanced model integration.
        
        Args:
            model_name: Name for the integration system
            enable_deep_learning: Whether to enable deep learning models
            enable_time_series: Whether to enable time series models
            enable_reinforcement_learning: Whether to enable RL models
            enable_nlp: Whether to enable NLP models
            max_concurrent_models: Maximum number of concurrent model executions
        """
        self.model_name = model_name
        self.enable_deep_learning = enable_deep_learning
        self.enable_time_series = enable_time_series
        self.enable_reinforcement_learning = enable_reinforcement_learning
        self.enable_nlp = enable_nlp
        self.max_concurrent_models = max_concurrent_models
        
        # Initialize model managers
        self.model_managers = {}
        
        if self.enable_deep_learning and ADVANCED_MODELS_AVAILABLE:
            try:
                self.model_managers['deep_learning'] = DeepLearningManager()
                logger.info("Deep Learning Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Deep Learning Manager: {e}")
                self.enable_deep_learning = False
        
        if self.enable_time_series and ADVANCED_MODELS_AVAILABLE:
            try:
                self.model_managers['time_series'] = TimeSeriesManager()
                logger.info("Time Series Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Time Series Manager: {e}")
                self.enable_time_series = False
        
        if self.enable_reinforcement_learning and ADVANCED_MODELS_AVAILABLE:
            try:
                self.model_managers['reinforcement_learning'] = ReinforcementLearningManager()
                logger.info("Reinforcement Learning Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RL Manager: {e}")
                self.enable_reinforcement_learning = False
        
        if self.enable_nlp and ADVANCED_MODELS_AVAILABLE:
            try:
                self.model_managers['nlp'] = NaturalLanguageProcessingManager()
                logger.info("NLP Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NLP Manager: {e}")
                self.enable_nlp = False
        
        # Initialize existing system components
        self.existing_models = {}
        
        if EXISTING_MODELS_AVAILABLE:
            try:
                self.existing_models['multi_model'] = MultiModelAI()
                self.existing_models['enhanced_ensemble'] = EnhancedEnsemble()
                self.existing_models['adaptive_weights'] = AdaptiveWeights()
                self.existing_models['regime_detector'] = RegimeDetector()
                logger.info("Existing system components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize existing components: {e}")
        
        # Model execution history
        self.execution_history = deque(maxlen=1000)
        
        # Performance metrics
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_execution_time': 0.0,
            'model_usage_counts': defaultdict(int),
            'last_execution_time': None
        }
        
        logger.info(f"Advanced Model Integration initialized: {model_name}")
    
    async def predict_async(
        self,
        data: Dict[str, Any],
        model_types: Optional[List[str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Asynchronously predict using multiple model types.
        
        Args:
            data: Input data for prediction
            model_types: List of model types to use
            timeout: Timeout for predictions
            
        Returns:
            Dictionary of predictions from different model types
        """
        if model_types is None:
            model_types = list(self.model_managers.keys())
        
        # Create prediction tasks
        tasks = []
        for model_type in model_types:
            if model_type in self.model_managers:
                task = self._create_prediction_task(model_type, data, timeout)
                tasks.append(task)
        
        # Execute predictions concurrently
        results = {}
        if tasks:
            try:
                completed_tasks = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
                
                for i, result in enumerate(completed_tasks):
                    model_type = model_types[i]
                    if isinstance(result, Exception):
                        logger.warning(f"Prediction failed for {model_type}: {result}")
                        results[model_type] = {'error': str(result)}
                    else:
                        results[model_type] = result
                        
            except asyncio.TimeoutError:
                logger.warning(f"Prediction timeout after {timeout} seconds")
                for model_type in model_types:
                    if model_type not in results:
                        results[model_type] = {'error': 'timeout'}
        
        # Update performance metrics
        self._update_performance_metrics(results)
        
        return results
    
    def _create_prediction_task(
        self,
        model_type: str,
        data: Dict[str, Any],
        timeout: float
    ) -> asyncio.Task:
        """
        Create an async prediction task for a specific model type.
        
        Args:
            model_type: Type of model to use
            data: Input data
            timeout: Task timeout
            
        Returns:
            Async task for prediction
        """
        async def prediction_task():
            try:
                manager = self.model_managers[model_type]
                
                if model_type == 'deep_learning':
                    return await self._deep_learning_prediction(manager, data)
                elif model_type == 'time_series':
                    return await self._time_series_prediction(manager, data)
                elif model_type == 'reinforcement_learning':
                    return await self._rl_prediction(manager, data)
                elif model_type == 'nlp':
                    return await self._nlp_prediction(manager, data)
                else:
                    return {'error': f'Unknown model type: {model_type}'}
                    
            except Exception as e:
                logger.error(f"Error in {model_type} prediction task: {e}")
                return {'error': str(e)}
        
        return asyncio.create_task(prediction_task())
    
    async def _deep_learning_prediction(
        self,
        manager: 'DeepLearningManager',
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute deep learning prediction."""
        try:
            # Extract market data
            market_data = data.get('market_data')
            if market_data is None:
                return {'error': 'No market data provided'}
            
            # Convert to DataFrame if needed
            if isinstance(market_data, dict):
                market_data = pd.DataFrame([market_data])
            elif not isinstance(market_data, pd.DataFrame):
                return {'error': 'Invalid market data format'}
            
            # Get predictions from different deep learning models
            predictions = {}
            
            # LSTM prediction
            lstm_pred = manager.predict_price_lstm(market_data)
            if lstm_pred:
                predictions['lstm'] = lstm_pred
            
            # CNN-LSTM prediction
            cnn_lstm_pred = manager.predict_trend_cnn_lstm(market_data)
            if cnn_lstm_pred:
                predictions['cnn_lstm'] = cnn_lstm_pred
            
            # Transformer prediction
            transformer_pred = manager.predict_price_transformer(market_data)
            if transformer_pred:
                predictions['transformer'] = transformer_pred
            
            return {
                'model_type': 'deep_learning',
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Deep learning prediction error: {e}")
            return {'error': str(e)}
    
    async def _time_series_prediction(
        self,
        manager: 'TimeSeriesManager',
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute time series prediction."""
        try:
            # Extract time series data
            ts_data = data.get('time_series_data')
            if ts_data is None:
                return {'error': 'No time series data provided'}
            
            # Convert to DataFrame if needed
            if isinstance(ts_data, dict):
                ts_data = pd.DataFrame([ts_data])
            elif not isinstance(ts_data, pd.DataFrame):
                return {'error': 'Invalid time series data format'}
            
            # Get predictions from different time series models
            predictions = {}
            
            # ARIMA-GARCH prediction
            arima_pred = manager.predict_arima_garch(ts_data)
            if arima_pred:
                predictions['arima_garch'] = arima_pred
            
            # Prophet prediction
            prophet_pred = manager.predict_prophet(ts_data)
            if prophet_pred:
                predictions['prophet'] = prophet_pred
            
            # VAR prediction
            var_pred = manager.predict_var(ts_data)
            if var_pred:
                predictions['var'] = var_pred
            
            return {
                'model_type': 'time_series',
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Time series prediction error: {e}")
            return {'error': str(e)}
    
    async def _rl_prediction(
        self,
        manager: 'ReinforcementLearningManager',
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute reinforcement learning prediction."""
        try:
            # Extract market state
            market_state = data.get('market_state')
            if market_state is None:
                return {'error': 'No market state provided'}
            
            # Get predictions from different RL agents
            predictions = {}
            
            # DQN prediction
            dqn_pred = manager.predict_dqn(market_state)
            if dqn_pred:
                predictions['dqn'] = dqn_pred
            
            # PPO prediction
            ppo_pred = manager.predict_ppo(market_state)
            if ppo_pred:
                predictions['ppo'] = ppo_pred
            
            # SAC prediction
            sac_pred = manager.predict_sac(market_state)
            if sac_pred:
                predictions['sac'] = sac_pred
            
            return {
                'model_type': 'reinforcement_learning',
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"RL prediction error: {e}")
            return {'error': str(e)}
    
    async def _nlp_prediction(
        self,
        manager: 'NaturalLanguageProcessingManager',
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute NLP prediction."""
        try:
            # Extract text data
            text_data = data.get('text_data')
            if text_data is None:
                return {'error': 'No text data provided'}
            
            # Get predictions from different NLP models
            predictions = {}
            
            # Sentiment analysis
            if isinstance(text_data, str):
                sentiment_pred = manager.analyze_sentiment(text_data, use_financial=True)
                predictions['sentiment'] = sentiment_pred
            
            # News classification
            if isinstance(text_data, str):
                classification_pred = manager.classify_news(text_data)
                predictions['classification'] = classification_pred
            
            # Batch processing for multiple texts
            if isinstance(text_data, list):
                batch_sentiment = []
                batch_classification = []
                
                for text in text_data:
                    sentiment = manager.analyze_sentiment(text, use_financial=True)
                    classification = manager.classify_news(text)
                    batch_sentiment.append(sentiment)
                    batch_classification.append(classification)
                
                predictions['batch_sentiment'] = batch_sentiment
                predictions['batch_classification'] = batch_classification
            
            return {
                'model_type': 'nlp',
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"NLP prediction error: {e}")
            return {'error': str(e)}
    
    def predict_sync(
        self,
        data: Dict[str, Any],
        model_types: Optional[List[str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Synchronously predict using multiple model types.
        
        Args:
            data: Input data for prediction
            model_types: List of model types to use
            timeout: Timeout for predictions
            
        Returns:
            Dictionary of predictions from different model types
        """
        from datetime import datetime
        import time
        
        start_time = time.time()
        
        # Run async prediction in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.predict_async(data, model_types, timeout)
            )
            
            # Add timestamp and execution time
            result['timestamp'] = datetime.now().isoformat()
            result['execution_time'] = time.time() - start_time
            result['model_name'] = self.model_name
            result['models_used'] = list(result.keys()) if result else []
            
            return result
        finally:
            loop.close()
    
    def ensemble_predict(
        self,
        data: Dict[str, Any],
        ensemble_method: str = "weighted_average",
        model_types: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create ensemble prediction from multiple model types.
        
        Args:
            data: Input data for prediction
            ensemble_method: Method for combining predictions
            model_types: List of model types to use
            weights: Weights for different model types
            
        Returns:
            Ensemble prediction results
        """
        from datetime import datetime
        
        # Get individual predictions
        individual_predictions = self.predict_sync(data, model_types)
        
        # Filter out errors - handle both dict and non-dict values
        valid_predictions = {}
        for k, v in individual_predictions.items():
            if isinstance(v, dict) and 'error' not in v:
                valid_predictions[k] = v
            elif not isinstance(v, dict) and v is not None:
                # Convert non-dict values to dict format
                valid_predictions[k] = {'prediction': v, 'confidence': 0.5}
        
        # Handle case when no models are enabled
        if not valid_predictions or len(valid_predictions) == 0:
            return {
                'timestamp': datetime.now().isoformat(),
                'ensemble_method': ensemble_method,
                'individual_predictions': [],
                'final_prediction': None,
                'confidence': 0.0,
                'model_count': 0,
                'note': 'No models enabled for prediction'
            }
        
        # Create ensemble prediction
        if ensemble_method == "weighted_average":
            final_pred = self._weighted_average_ensemble(valid_predictions, weights)
        elif ensemble_method == "majority_vote":
            final_pred = self._majority_vote_ensemble(valid_predictions)
        elif ensemble_method == "stacking":
            final_pred = self._stacking_ensemble(valid_predictions)
        else:
            final_pred = None
        
        # Calculate ensemble confidence
        ensemble_confidence = 0.0
        if final_pred is not None:
            # Simple confidence calculation based on number of valid predictions
            ensemble_confidence = min(0.9, len(valid_predictions) / 5.0)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'ensemble_method': ensemble_method,
            'individual_predictions': list(valid_predictions.values()),
            'final_prediction': final_pred,
            'confidence': ensemble_confidence,
            'model_count': len(valid_predictions),
            'model_weights': weights
        }
    
    def _weighted_average_ensemble(
        self,
        predictions: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Create weighted average ensemble prediction."""
        if weights is None:
            # Equal weights
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Extract numeric predictions
        numeric_predictions = {}
        for model_type, prediction in predictions.items():
            if 'predictions' in prediction:
                # Extract first numeric prediction
                for sub_model, sub_pred in prediction['predictions'].items():
                    if isinstance(sub_pred, (int, float)):
                        numeric_predictions[f"{model_type}_{sub_model}"] = sub_pred
                        break
        
        if not numeric_predictions:
            return {'error': 'No numeric predictions found'}
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred_key, pred_value in numeric_predictions.items():
            # Find corresponding weight
            weight = 0.0
            for model_type in weights:
                if model_type in pred_key:
                    weight = weights[model_type]
                    break
            
            weighted_sum += pred_value * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_value = weighted_sum / total_weight
        else:
            ensemble_value = np.mean(list(numeric_predictions.values()))
        
        return {
            'ensemble_value': ensemble_value,
            'weights_used': weights,
            'individual_values': numeric_predictions
        }
    
    def _majority_vote_ensemble(
        self,
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create majority vote ensemble prediction."""
        # Extract categorical predictions
        categorical_predictions = {}
        for model_type, prediction in predictions.items():
            if 'predictions' in prediction:
                for sub_model, sub_pred in prediction['predictions'].items():
                    if isinstance(sub_pred, str):
                        categorical_predictions[f"{model_type}_{sub_model}"] = sub_pred
                        break
        
        if not categorical_predictions:
            return {'error': 'No categorical predictions found'}
        
        # Count votes
        vote_counts = defaultdict(int)
        for pred in categorical_predictions.values():
            vote_counts[pred] += 1
        
        # Find majority vote
        majority_prediction = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_prediction]
        
        return {
            'majority_prediction': majority_prediction,
            'vote_counts': dict(vote_counts),
            'majority_count': majority_count,
            'total_votes': len(categorical_predictions),
            'confidence': majority_count / len(categorical_predictions)
        }
    
    def _stacking_ensemble(
        self,
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create stacking ensemble prediction."""
        # For now, use simple averaging as stacking
        # In a full implementation, this would use a meta-learner
        return self._weighted_average_ensemble(predictions)
    
    def _update_performance_metrics(self, results: Dict[str, Any]) -> None:
        """Update performance metrics based on prediction results."""
        self.performance_metrics['total_predictions'] += 1
        self.performance_metrics['last_execution_time'] = datetime.now()
        
        successful = 0
        failed = 0
        
        for model_type, result in results.items():
            if 'error' in result:
                failed += 1
            else:
                successful += 1
                self.performance_metrics['model_usage_counts'][model_type] += 1
        
        self.performance_metrics['successful_predictions'] += successful
        self.performance_metrics['failed_predictions'] += failed
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            **self.performance_metrics,
            'success_rate': (
                self.performance_metrics['successful_predictions'] / 
                max(self.performance_metrics['total_predictions'], 1)
            ),
            'available_models': list(self.model_managers.keys()),
            'enabled_features': {
                'deep_learning': self.enable_deep_learning,
                'time_series': self.enable_time_series,
                'reinforcement_learning': self.enable_reinforcement_learning,
                'nlp': self.enable_nlp
            }
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            'integration_name': self.model_name,
            'available_managers': list(self.model_managers.keys()),
            'manager_status': {}
        }
        
        for manager_name, manager in self.model_managers.items():
            try:
                if hasattr(manager, 'get_model_status'):
                    status['manager_status'][manager_name] = manager.get_model_status()
                else:
                    status['manager_status'][manager_name] = 'available'
            except Exception as e:
                status['manager_status'][manager_name] = f'error: {str(e)}'
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models."""
        health_status = {
            'overall_health': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'component_health': {}
        }
        
        # Check each model manager
        for manager_name, manager in self.model_managers.items():
            try:
                # Simple health check - try to access manager
                if hasattr(manager, 'health_check'):
                    health_status['component_health'][manager_name] = manager.health_check()
                else:
                    health_status['component_health'][manager_name] = 'healthy'
            except Exception as e:
                health_status['component_health'][manager_name] = f'unhealthy: {str(e)}'
                health_status['overall_health'] = 'degraded'
        
        # Check existing models
        for model_name, model in self.existing_models.items():
            try:
                if hasattr(model, 'health_check'):
                    health_status['component_health'][model_name] = model.health_check()
                else:
                    health_status['component_health'][model_name] = 'healthy'
            except Exception as e:
                health_status['component_health'][model_name] = f'unhealthy: {str(e)}'
                health_status['overall_health'] = 'degraded'
        
        return health_status


class ModelEnsemble:
    """
    Advanced ensemble system for combining predictions from multiple models.
    """
    
    def __init__(
        self,
        ensemble_name: str = "model_ensemble",
        ensemble_methods: List[str] = None
    ):
        """
        Initialize model ensemble.
        
        Args:
            ensemble_name: Name for the ensemble
            ensemble_methods: List of ensemble methods to use
        """
        self.ensemble_name = ensemble_name
        self.ensemble_methods = ensemble_methods or [
            'weighted_average',
            'majority_vote',
            'stacking',
            'bayesian_model_averaging'
        ]
        
        # Ensemble history
        self.ensemble_history = deque(maxlen=1000)
        
        # Performance tracking
        self.ensemble_performance = {
            'total_ensembles': 0,
            'successful_ensembles': 0,
            'method_performance': defaultdict(list)
        }
        
        logger.info(f"Model Ensemble initialized: {ensemble_name}")
    
    def create_ensemble(
        self,
        predictions: Dict[str, Any],
        method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
        meta_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create ensemble prediction from individual model predictions.
        
        Args:
            predictions: Dictionary of individual model predictions
            method: Ensemble method to use
            weights: Weights for different models
            meta_features: Additional features for meta-learning
            
        Returns:
            Ensemble prediction results
        """
        start_time = datetime.now()
        
        try:
            if method == "weighted_average":
                result = self._weighted_average_ensemble(predictions, weights)
            elif method == "majority_vote":
                result = self._majority_vote_ensemble(predictions)
            elif method == "stacking":
                result = self._stacking_ensemble(predictions, meta_features)
            elif method == "bayesian_model_averaging":
                result = self._bayesian_model_averaging(predictions)
            else:
                result = {'error': f'Unknown ensemble method: {method}'}
            
            # Add metadata
            result['ensemble_method'] = method
            result['timestamp'] = datetime.now().isoformat()
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Store in history
            self.ensemble_history.append(result)
            
            # Update performance
            self.ensemble_performance['total_ensembles'] += 1
            if 'error' not in result:
                self.ensemble_performance['successful_ensembles'] += 1
                self.ensemble_performance['method_performance'][method].append(
                    result['processing_time']
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble creation error: {e}")
            return {
                'error': str(e),
                'ensemble_method': method,
                'timestamp': datetime.now().isoformat()
            }
    
    def _weighted_average_ensemble(
        self,
        predictions: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Create weighted average ensemble."""
        # Implementation similar to AdvancedModelIntegration
        if weights is None:
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Extract and combine numeric predictions
        ensemble_value = 0.0
        total_weight = 0.0
        individual_values = {}
        
        for model_name, prediction in predictions.items():
            if 'error' in prediction:
                continue
            
            # Extract numeric value from prediction
            numeric_value = self._extract_numeric_value(prediction)
            if numeric_value is not None:
                weight = weights.get(model_name, 0.0)
                ensemble_value += numeric_value * weight
                total_weight += weight
                individual_values[model_name] = numeric_value
        
        if total_weight > 0:
            final_value = ensemble_value / total_weight
        else:
            final_value = np.mean(list(individual_values.values())) if individual_values else 0.0
        
        return {
            'ensemble_value': final_value,
            'weights_used': weights,
            'individual_values': individual_values,
            'confidence': min(total_weight, 1.0)
        }
    
    def _majority_vote_ensemble(
        self,
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create majority vote ensemble."""
        # Implementation similar to AdvancedModelIntegration
        vote_counts = defaultdict(int)
        individual_votes = {}
        
        for model_name, prediction in predictions.items():
            if 'error' in prediction:
                continue
            
            # Extract categorical value from prediction
            categorical_value = self._extract_categorical_value(prediction)
            if categorical_value is not None:
                vote_counts[categorical_value] += 1
                individual_votes[model_name] = categorical_value
        
        if not vote_counts:
            return {'error': 'No valid categorical predictions found'}
        
        # Find majority vote
        majority_prediction = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_prediction]
        total_votes = sum(vote_counts.values())
        
        return {
            'majority_prediction': majority_prediction,
            'vote_counts': dict(vote_counts),
            'individual_votes': individual_votes,
            'confidence': majority_count / total_votes,
            'total_votes': total_votes
        }
    
    def _stacking_ensemble(
        self,
        predictions: Dict[str, Any],
        meta_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create stacking ensemble with meta-learner."""
        # For now, use weighted average as a simple stacking approach
        # In a full implementation, this would train a meta-learner
        return self._weighted_average_ensemble(predictions)
    
    def _bayesian_model_averaging(
        self,
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Bayesian model averaging ensemble."""
        # Simple implementation - use equal weights for now
        # In a full implementation, this would use Bayesian inference
        return self._weighted_average_ensemble(predictions)
    
    def _extract_numeric_value(self, prediction: Dict[str, Any]) -> Optional[float]:
        """Extract numeric value from prediction dictionary."""
        # Try different common keys
        for key in ['value', 'prediction', 'score', 'probability', 'ensemble_value']:
            if key in prediction and isinstance(prediction[key], (int, float)):
                return float(prediction[key])
        
        # Try nested predictions
        if 'predictions' in prediction:
            for sub_pred in prediction['predictions'].values():
                if isinstance(sub_pred, (int, float)):
                    return float(sub_pred)
        
        return None
    
    def _extract_categorical_value(self, prediction: Dict[str, Any]) -> Optional[str]:
        """Extract categorical value from prediction dictionary."""
        # Try different common keys
        for key in ['label', 'category', 'prediction', 'class', 'majority_prediction']:
            if key in prediction and isinstance(prediction[key], str):
                return prediction[key]
        
        # Try nested predictions
        if 'predictions' in prediction:
            for sub_pred in prediction['predictions'].values():
                if isinstance(sub_pred, str):
                    return sub_pred
        
        return None
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble performance statistics."""
        return {
            'ensemble_name': self.ensemble_name,
            'total_ensembles': self.ensemble_performance['total_ensembles'],
            'successful_ensembles': self.ensemble_performance['successful_ensembles'],
            'success_rate': (
                self.ensemble_performance['successful_ensembles'] / 
                max(self.ensemble_performance['total_ensembles'], 1)
            ),
            'method_performance': {
                method: {
                    'count': len(times),
                    'avg_time': np.mean(times) if times else 0.0,
                    'std_time': np.std(times) if times else 0.0
                }
                for method, times in self.ensemble_performance['method_performance'].items()
            },
            'history_size': len(self.ensemble_history)
        }

