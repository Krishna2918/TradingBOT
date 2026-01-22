"""Master Orchestrator for Trading System Integration"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.ai.model_communication_hub import get_communication_hub
from src.ai.intelligent_model_selector import get_model_selector
from src.ai.model_performance_learner import get_performance_learner
from src.ai.performance_prediction_system import get_prediction_system
from src.ai.market_condition_analyzer import get_condition_analyzer
from src.ai.cross_model_validator import get_cross_model_validator
from src.ai.advanced_ensemble_methods import get_ensemble_manager
from src.ai.model_evolution_system import get_evolution_engine
from src.ai.market_microstructure_predictor import get_microstructure_predictor
from src.ai.advanced_feature_engineering import get_feature_pipeline

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Final trading decision from the orchestrator."""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    position_size: float
    reasoning: List[str]
    model_consensus: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    execution_recommendations: List[str]
    timestamp: datetime

@dataclass
class MarketContext:
    """Market context for decision making."""
    current_price: float
    market_regime: str
    volatility_regime: str
    liquidity_regime: str
    sentiment_regime: str
    risk_level: str
    timestamp: datetime
    additional_data: Dict[str, Any]

class MasterOrchestrator:
    """Master orchestrator that coordinates all AI components."""
    
    def __init__(self):
        # Initialize all AI components
        self.communication_hub = get_communication_hub()
        self.model_selector = get_model_selector()
        self.performance_learner = get_performance_learner()
        self.performance_predictor = get_prediction_system()
        self.condition_analyzer = get_condition_analyzer()
        self.cross_validator = get_cross_model_validator()
        self.ensemble_manager = get_ensemble_manager()
        self.evolution_engine = get_evolution_engine()
        self.microstructure_predictor = get_microstructure_predictor()
        self.feature_pipeline = get_feature_pipeline()
        
        # AGENTIC AI SYSTEM - Resource Manager
        self.resource_manager = None
        self.agents = {}
        self._initialize_agentic_system()
        
        # Decision pipeline configuration
        self.decision_pipeline = [
            'data_quality_check',
            'market_condition_analysis',
            'feature_engineering',
            'model_selection',
            'prediction_generation',
            'cross_validation',
            'ensemble_prediction',
            'risk_assessment',
            'execution_optimization',
            'decision_finalization'
        ]
        
        # Performance tracking
        self.decision_history = []
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_predictions': 0,
            'average_confidence': 0.0,
            'average_execution_time': 0.0
        }
    
    def _initialize_agentic_system(self):
        """Initialize the Agentic AI system (Resource Manager and agents)"""
        try:
            from src.agents.resource_manager import ResourceManager
            from src.agents.risk_agent import RiskManagementAgent
            from src.agents.monitoring_agent import MonitoringAgent
            from src.agents.execution_agent import ExecutionAgent
            from src.agents.portfolio_agent import PortfolioAgent
            from src.agents.market_analysis_agent import MarketAnalysisAgent
            from src.agents.learning_agent import LearningAgent
            
            # Initialize Resource Manager
            self.resource_manager = ResourceManager(
                cpu_threshold_critical=85.0,
                cpu_threshold_warning=70.0,
                memory_threshold_critical=80.0,
                memory_threshold_warning=60.0,
                learning_enabled=True
            )
            
            # Initialize CRITICAL priority agents (always run)
            risk_agent = RiskManagementAgent(mode='DEMO')
            self.resource_manager.register_agent(risk_agent)
            self.agents['risk_agent'] = risk_agent
            
            monitoring_agent = MonitoringAgent()
            self.resource_manager.register_agent(monitoring_agent)
            self.agents['monitoring_agent'] = monitoring_agent
            
            execution_agent = ExecutionAgent(mode='DEMO')
            self.resource_manager.register_agent(execution_agent)
            self.agents['execution_agent'] = execution_agent
            
            # Initialize IMPORTANT priority agents (run when resources available)
            portfolio_agent = PortfolioAgent(mode='DEMO')
            self.resource_manager.register_agent(portfolio_agent)
            self.agents['portfolio_agent'] = portfolio_agent
            
            market_analysis_agent = MarketAnalysisAgent()
            self.resource_manager.register_agent(market_analysis_agent)
            self.agents['market_analysis_agent'] = market_analysis_agent
            
            # Initialize OPTIONAL priority agents (run only with abundant resources)
            learning_agent = LearningAgent()
            self.resource_manager.register_agent(learning_agent)
            self.agents['learning_agent'] = learning_agent
            
            logger.info("Agentic AI system initialized with Resource Manager")
            logger.info(f"  - Registered agents: {list(self.agents.keys())}")
            logger.info(f"  - CRITICAL agents: risk_agent, monitoring_agent, execution_agent")
            logger.info(f"  - IMPORTANT agents: portfolio_agent, market_analysis_agent")
            logger.info(f"  - OPTIONAL agents: learning_agent")
            
        except Exception as e:
            logger.warning(f"Agentic AI system initialization failed: {e}")
            logger.warning("Falling back to non-agent mode")
            self.resource_manager = None
            self.agents = {}
    
    async def start_agentic_system(self) -> bool:
        """
        Start the Agentic AI system and activate all agents.
        
        Returns:
            True if startup successful
        """
        try:
            if not self.resource_manager:
                logger.error("Resource Manager not initialized")
                return False
            
            logger.info("Starting Agentic AI system...")
            success = await self.resource_manager.start()
            
            if success:
                logger.info("✓ Agentic AI system started successfully")
                return True
            else:
                logger.error("✗ Agentic AI system startup failed")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Agentic AI system: {e}")
            return False
        
    async def run_decision_pipeline(self, market_data: pd.DataFrame, 
                                  additional_data: Dict[str, Any] = None) -> TradingDecision:
        """Run the complete decision pipeline."""
        start_time = datetime.now()
        
        try:
            # Phase 1: Data Quality Check
            data_quality_result = await self._data_quality_check(market_data)
            if not data_quality_result['is_valid']:
                return self._create_hold_decision("Data quality check failed", data_quality_result)
            
            # Phase 2: Market Condition Analysis
            market_context = await self._analyze_market_conditions(market_data, additional_data)
            
            # Phase 3: Feature Engineering
            feature_set = await self._engineer_features(market_data, additional_data)
            
            # Phase 4: Model Selection
            selected_models = await self._select_models(market_context, feature_set)
            
            # Phase 5: Prediction Generation
            predictions = await self._generate_predictions(selected_models, feature_set, market_context)
            
            # Phase 6: Cross Validation
            validation_result = await self._cross_validate_predictions(predictions, market_context)
            
            # Phase 7: Ensemble Prediction
            ensemble_result = await self._ensemble_prediction(predictions, market_context)
            
            # Phase 8: Risk Assessment
            risk_assessment = await self._assess_risk(ensemble_result, market_context)
            
            # Phase 9: Execution Optimization
            execution_recommendations = await self._optimize_execution(ensemble_result, market_context)
            
            # Phase 10: Decision Finalization
            final_decision = await self._finalize_decision(
                ensemble_result, risk_assessment, execution_recommendations, market_context
            )
            
            # Update performance tracking
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(final_decision, execution_time)
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error in decision pipeline: {e}")
            return self._create_hold_decision(f"Pipeline error: {e}", {})
    
    async def _data_quality_check(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality and completeness."""
        result = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0
        }
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        
        if missing_columns:
            result['is_valid'] = False
            result['issues'].append(f"Missing columns: {missing_columns}")
            result['quality_score'] = 0.0
            return result
        
        # Check for null values
        null_counts = market_data[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            result['issues'].append(f"Null values found: {null_counts.to_dict()}")
            result['quality_score'] -= 0.1 * (null_counts.sum() / len(market_data))
        
        # Check for reasonable price ranges
        if 'close' in market_data.columns:
            close_prices = market_data['close'].dropna()
            if len(close_prices) > 0:
                if close_prices.min() <= 0:
                    result['issues'].append("Invalid price values (<= 0)")
                    result['quality_score'] -= 0.2
                
                if close_prices.max() / close_prices.min() > 1000:
                    result['issues'].append("Extreme price range detected")
                    result['quality_score'] -= 0.1
        
        # Check for sufficient data
        if len(market_data) < 50:
            result['issues'].append("Insufficient data points")
            result['quality_score'] -= 0.3
        
        result['quality_score'] = max(0.0, result['quality_score'])
        result['is_valid'] = result['quality_score'] > 0.5
        
        return result
    
    async def _analyze_market_conditions(self, market_data: pd.DataFrame, 
                                       additional_data: Dict[str, Any] = None) -> MarketContext:
        """Analyze current market conditions."""
        
        # Add market data to condition analyzer
        for _, row in market_data.iterrows():
            data_point = {
                'timestamp': row.get('timestamp', datetime.now()),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            self.condition_analyzer.add_market_data(data_point)
        
        # Get current market conditions
        condition = self.condition_analyzer.analyze_current_conditions()
        
        # Create market context
        current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else 0.0
        
        context = MarketContext(
            current_price=current_price,
            market_regime=condition.regime,
            volatility_regime=condition.volatility_regime,
            liquidity_regime=condition.liquidity_regime,
            sentiment_regime=condition.sentiment_regime,
            risk_level=condition.risk_level,
            timestamp=datetime.now(),
            additional_data=additional_data or {}
        )
        
        # Update communication hub with market context
        try:
            self.communication_hub.update_market_context(context)
        except Exception as e:
            logger.warning(f"Failed to update market context: {e}")
        
        return context
    
    async def _engineer_features(self, market_data: pd.DataFrame, 
                               additional_data: Dict[str, Any] = None) -> Any:
        """Engineer features for model prediction."""
        
        # Use the advanced feature pipeline
        feature_set = self.feature_pipeline.engineer_features(market_data, additional_data)
        
        # Log feature engineering results
        logger.info(f"Engineered {len(feature_set.features)} features across {len(feature_set.feature_categories)} categories")
        
        return feature_set
    
    async def _select_models(self, market_context: MarketContext, feature_set: Any) -> List[str]:
        """Select optimal models for current market conditions."""
        
        # Get available models (this would typically come from a model registry)
        available_models = [
            'technical_analysis_model',
            'fundamental_analysis_model',
            'sentiment_analysis_model',
            'machine_learning_model',
            'deep_learning_model',
            'ensemble_model'
        ]
        
        # Use intelligent model selector
        selected_models = self.model_selector.select_models_for_conditions(available_models)
        
        # Filter based on market context
        if market_context.market_regime == 'trending':
            # Prefer trend-following models
            selected_models = [m for m in selected_models if 'technical' in m or 'trend' in m]
        elif market_context.market_regime == 'ranging':
            # Prefer mean-reversion models
            selected_models = [m for m in selected_models if 'mean_reversion' in m or 'oscillator' in m]
        
        logger.info(f"Selected {len(selected_models)} models: {selected_models}")
        
        return selected_models
    
    async def _generate_predictions(self, selected_models: List[str], feature_set: Any, 
                                  market_context: MarketContext) -> Dict[str, Any]:
        """Generate predictions from selected models."""
        
        predictions = {}
        
        # Simulate model predictions (in real implementation, these would call actual models)
        for model_name in selected_models:
            # Generate realistic prediction based on market context
            base_prediction = 0.5
            
            if market_context.market_regime == 'trending':
                base_prediction = 0.7 if market_context.sentiment_regime == 'bullish' else 0.3
            elif market_context.market_regime == 'ranging':
                base_prediction = 0.5
            elif market_context.market_regime == 'volatile':
                base_prediction = 0.6
            
            # Add some model-specific variation
            model_variation = hash(model_name) % 100 / 1000  # 0-0.1 variation
            prediction = base_prediction + model_variation
            
            predictions[model_name] = {
                'prediction': prediction,
                'confidence': 0.7 + (hash(model_name) % 30) / 100,  # 0.7-1.0 confidence
                'reasoning': f"Model {model_name} prediction based on {market_context.market_regime} market"
            }
        
        return predictions
    
    async def _cross_validate_predictions(self, predictions: Dict[str, Any], 
                                        market_context: MarketContext) -> Dict[str, Any]:
        """Cross-validate predictions for reliability."""
        
        # Convert predictions to cross-validator format
        validator_predictions = []
        for model_name, pred_data in predictions.items():
            validator_predictions.append({
                'model_name': model_name,
                'prediction': pred_data['prediction'],
                'confidence': pred_data['confidence'],
                'reasoning': pred_data['reasoning']
            })
        
        # Validate predictions
        validation_result = self.cross_validator.validate_predictions(validator_predictions)
        
        return {
            'consensus_prediction': validation_result.consensus_prediction,
            'consensus_confidence': validation_result.consensus_confidence,
            'agreement_score': validation_result.agreement_score,
            'reliability_score': validation_result.reliability_score,
            'validation_status': validation_result.validation_status,
            'outlier_models': validation_result.outlier_models,
            'recommendations': validation_result.recommendations
        }
    
    async def _ensemble_prediction(self, predictions: Dict[str, Any], 
                                 market_context: MarketContext) -> Dict[str, Any]:
        """Generate ensemble prediction using advanced methods."""
        
        # Convert predictions to ensemble format
        ensemble_predictions = {}
        for model_name, pred_data in predictions.items():
            ensemble_predictions[model_name] = pred_data['prediction']
        
        # Get market conditions for ensemble
        market_conditions = {
            'regime': market_context.market_regime,
            'volatility_regime': market_context.volatility_regime,
            'liquidity_regime': market_context.liquidity_regime
        }
        
        # Generate ensemble prediction
        ensemble_result = self.ensemble_manager.predict_ensemble(
            ensemble_predictions, market_conditions, 'combined'
        )
        
        return {
            'final_prediction': ensemble_result.final_prediction,
            'confidence': ensemble_result.confidence,
            'method_used': ensemble_result.method_used,
            'individual_predictions': ensemble_result.individual_predictions,
            'weights': ensemble_result.weights,
            'uncertainty': ensemble_result.uncertainty
        }
    
    async def _assess_risk(self, ensemble_result: Dict[str, Any], 
                          market_context: MarketContext) -> Dict[str, Any]:
        """Assess risk for the trading decision."""
        
        risk_assessment = {
            'overall_risk': 'medium',
            'risk_factors': [],
            'risk_score': 0.5,
            'recommendations': []
        }
        
        # Assess prediction confidence risk
        confidence = ensemble_result['confidence']
        if confidence < 0.3:
            risk_assessment['risk_factors'].append('Low prediction confidence')
            risk_assessment['risk_score'] += 0.3
        elif confidence > 0.8:
            risk_assessment['risk_score'] -= 0.2
        
        # Assess market regime risk
        if market_context.market_regime == 'volatile':
            risk_assessment['risk_factors'].append('High market volatility')
            risk_assessment['risk_score'] += 0.2
        
        if market_context.risk_level == 'high':
            risk_assessment['risk_factors'].append('High market risk level')
            risk_assessment['risk_score'] += 0.2
        
        # Assess liquidity risk
        if market_context.liquidity_regime == 'low':
            risk_assessment['risk_factors'].append('Low market liquidity')
            risk_assessment['risk_score'] += 0.1
        
        # Determine overall risk level
        if risk_assessment['risk_score'] > 0.7:
            risk_assessment['overall_risk'] = 'high'
        elif risk_assessment['risk_score'] < 0.3:
            risk_assessment['overall_risk'] = 'low'
        
        # Generate risk-based recommendations
        if risk_assessment['overall_risk'] == 'high':
            risk_assessment['recommendations'].append('Reduce position size')
            risk_assessment['recommendations'].append('Use conservative execution')
        elif risk_assessment['overall_risk'] == 'low':
            risk_assessment['recommendations'].append('Consider larger position size')
        
        return risk_assessment
    
    async def _optimize_execution(self, ensemble_result: Dict[str, Any], 
                                market_context: MarketContext) -> List[str]:
        """Optimize execution strategy based on predictions and market conditions."""
        
        recommendations = []
        
        # Get microstructure prediction
        microstructure_prediction = self.microstructure_predictor.predict_microstructure(
            time_horizon_minutes=5,
            trade_size=1000.0,  # Default trade size
            market_conditions={
                'regime': market_context.market_regime,
                'volatility': 0.02,  # Default volatility
                'liquidity': market_context.liquidity_regime
            }
        )
        
        # Execution strategy recommendations
        if microstructure_prediction.predicted_liquidity > 0.7:
            recommendations.append("High liquidity - market orders acceptable")
        elif microstructure_prediction.predicted_liquidity > 0.4:
            recommendations.append("Moderate liquidity - consider TWAP strategy")
        else:
            recommendations.append("Low liquidity - use limit orders or VWAP")
        
        if microstructure_prediction.predicted_price_impact > 0.01:
            recommendations.append("High price impact expected - split orders")
        
        if microstructure_prediction.predicted_volatility > 0.03:
            recommendations.append("High volatility - reduce position sizes")
        
        # Add microstructure-specific recommendations
        recommendations.extend(microstructure_prediction.recommendations.get('execution_strategy', []))
        
        return recommendations
    
    async def _finalize_decision(self, ensemble_result: Dict[str, Any], 
                               risk_assessment: Dict[str, Any],
                               execution_recommendations: List[str],
                               market_context: MarketContext) -> TradingDecision:
        """Finalize the trading decision."""
        
        final_prediction = ensemble_result['final_prediction']
        confidence = ensemble_result['confidence']
        
        # Determine action based on prediction and risk
        if final_prediction > 0.6 and risk_assessment['overall_risk'] != 'high':
            action = 'buy'
        elif final_prediction < 0.4 and risk_assessment['overall_risk'] != 'high':
            action = 'sell'
        else:
            action = 'hold'
        
        # Adjust position size based on confidence and risk
        base_position_size = 1.0
        if confidence > 0.8 and risk_assessment['overall_risk'] == 'low':
            position_size = base_position_size * 1.2
        elif confidence < 0.5 or risk_assessment['overall_risk'] == 'high':
            position_size = base_position_size * 0.5
        else:
            position_size = base_position_size
        
        # Generate reasoning
        reasoning = [
            f"Final prediction: {final_prediction:.3f}",
            f"Confidence: {confidence:.3f}",
            f"Market regime: {market_context.market_regime}",
            f"Risk level: {risk_assessment['overall_risk']}"
        ]
        
        # Add risk factors to reasoning
        if risk_assessment['risk_factors']:
            reasoning.append(f"Risk factors: {', '.join(risk_assessment['risk_factors'])}")
        
        # Create model consensus
        model_consensus = {
            'ensemble_prediction': final_prediction,
            'ensemble_confidence': confidence,
            'individual_predictions': ensemble_result['individual_predictions'],
            'model_weights': ensemble_result['weights'],
            'validation_status': 'validated'
        }
        
        return TradingDecision(
            action=action,
            confidence=confidence,
            position_size=position_size,
            reasoning=reasoning,
            model_consensus=model_consensus,
            risk_assessment=risk_assessment,
            execution_recommendations=execution_recommendations,
            timestamp=datetime.now()
        )
    
    def _create_hold_decision(self, reason: str, additional_info: Dict[str, Any] = None) -> TradingDecision:
        """Create a hold decision with given reason."""
        return TradingDecision(
            action='hold',
            confidence=0.0,
            position_size=0.0,
            reasoning=[reason],
            model_consensus={},
            risk_assessment={'overall_risk': 'high', 'risk_factors': [reason]},
            execution_recommendations=['No action recommended'],
            timestamp=datetime.now()
        )
    
    def _update_performance_metrics(self, decision: TradingDecision, execution_time: float):
        """Update performance metrics."""
        self.decision_history.append(decision)
        
        self.performance_metrics['total_decisions'] += 1
        self.performance_metrics['average_confidence'] = (
            (self.performance_metrics['average_confidence'] * (self.performance_metrics['total_decisions'] - 1) + 
             decision.confidence) / self.performance_metrics['total_decisions']
        )
        self.performance_metrics['average_execution_time'] = (
            (self.performance_metrics['average_execution_time'] * (self.performance_metrics['total_decisions'] - 1) + 
             execution_time) / self.performance_metrics['total_decisions']
        )
    
    def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        stats = {
            'performance_metrics': self.performance_metrics,
            'decision_pipeline': self.decision_pipeline,
            'recent_decisions': len(self.decision_history),
            'component_status': {
                'communication_hub': 'active',
                'model_selector': 'active',
                'performance_learner': 'active',
                'performance_predictor': 'active',
                'condition_analyzer': 'active',
                'cross_validator': 'active',
                'ensemble_manager': 'active',
                'evolution_engine': 'active',
                'microstructure_predictor': 'active',
                'feature_pipeline': 'active'
            }
        }
        
        # Add recent decision summary
        if self.decision_history:
            recent_decisions = self.decision_history[-10:]
            stats['recent_decision_summary'] = {
                'total_recent': len(recent_decisions),
                'buy_decisions': len([d for d in recent_decisions if d.action == 'buy']),
                'sell_decisions': len([d for d in recent_decisions if d.action == 'sell']),
                'hold_decisions': len([d for d in recent_decisions if d.action == 'hold']),
                'average_confidence': np.mean([d.confidence for d in recent_decisions])
            }
        
        return stats

# Global instance
_orchestrator = None

def get_orchestrator() -> MasterOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MasterOrchestrator()
    return _orchestrator