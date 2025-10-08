"""
Model Attribution Dashboard
Provides detailed attribution of trading decisions to specific AI models and components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of AI models"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    META_ENSEMBLE = "meta_ensemble"
    PPO_AGENT = "ppo_agent"
    DQN_AGENT = "dqn_agent"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    REGIME_DETECTOR = "regime_detector"
    VOLATILITY_DETECTOR = "volatility_detector"
    NEWS_ANALYZER = "news_analyzer"
    CHATGPT = "chatgpt"
    LOCAL_REASONER = "local_reasoner"

class AttributionType(Enum):
    """Types of model attribution"""
    SIGNAL_WEIGHT = "signal_weight"
    CONFIDENCE_SCORE = "confidence_score"
    DECISION_INFLUENCE = "decision_influence"
    PERFORMANCE_CONTRIBUTION = "performance_contribution"
    RISK_CONTRIBUTION = "risk_contribution"

@dataclass
class ModelAttribution:
    """Model attribution record"""
    model_id: str
    model_type: ModelType
    attribution_type: AttributionType
    value: float
    weight: float
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class DecisionAttribution:
    """Complete decision attribution"""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str
    final_decision: str
    model_attributions: List[ModelAttribution]
    total_confidence: float
    decision_factors: Dict[str, Any]

class ModelAttributionDashboard:
    """Dashboard for model attribution analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.attribution_history = []
        self.model_performance = {}
        self.decision_history = []
        
        # Initialize model tracking
        self._initialize_model_tracking()
        
        logger.info("Model Attribution Dashboard initialized")
    
    def _initialize_model_tracking(self):
        """Initialize model performance tracking"""
        try:
            # Initialize performance tracking for each model type
            for model_type in ModelType:
                self.model_performance[model_type.value] = {
                    'total_decisions': 0,
                    'correct_decisions': 0,
                    'accuracy': 0.0,
                    'average_confidence': 0.0,
                    'total_attribution_weight': 0.0,
                    'performance_score': 0.0,
                    'last_updated': datetime.now()
                }
            
            logger.info("Model performance tracking initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model tracking: {e}")
    
    def record_decision_attribution(self, decision_attribution: DecisionAttribution):
        """Record a decision attribution"""
        try:
            # Store decision attribution
            self.decision_history.append(decision_attribution)
            
            # Update model performance
            self._update_model_performance(decision_attribution)
            
            # Store individual attributions
            for attribution in decision_attribution.model_attributions:
                self.attribution_history.append(attribution)
            
            # Keep only last 10000 records
            if len(self.decision_history) > 10000:
                self.decision_history = self.decision_history[-10000:]
            
            if len(self.attribution_history) > 10000:
                self.attribution_history = self.attribution_history[-10000:]
            
            logger.debug(f"Recorded decision attribution: {decision_attribution.decision_id}")
            
        except Exception as e:
            logger.error(f"Error recording decision attribution: {e}")
    
    def _update_model_performance(self, decision_attribution: DecisionAttribution):
        """Update model performance metrics"""
        try:
            for attribution in decision_attribution.model_attributions:
                model_type = attribution.model_type.value
                
                if model_type in self.model_performance:
                    # Update decision count
                    self.model_performance[model_type]['total_decisions'] += 1
                    
                    # Update attribution weight
                    self.model_performance[model_type]['total_attribution_weight'] += attribution.weight
                    
                    # Update average confidence
                    current_avg = self.model_performance[model_type]['average_confidence']
                    total_decisions = self.model_performance[model_type]['total_decisions']
                    new_avg = ((current_avg * (total_decisions - 1)) + attribution.confidence) / total_decisions
                    self.model_performance[model_type]['average_confidence'] = new_avg
                    
                    # Update last updated timestamp
                    self.model_performance[model_type]['last_updated'] = datetime.now()
            
            # Calculate performance scores
            self._calculate_performance_scores()
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def _calculate_performance_scores(self):
        """Calculate performance scores for all models"""
        try:
            for model_type, performance in self.model_performance.items():
                if performance['total_decisions'] > 0:
                    # Calculate accuracy (placeholder - would need actual outcome data)
                    accuracy = performance['correct_decisions'] / performance['total_decisions']
                    performance['accuracy'] = accuracy
                    
                    # Calculate performance score (weighted combination)
                    performance_score = (
                        accuracy * 0.4 +  # 40% accuracy
                        performance['average_confidence'] * 0.3 +  # 30% confidence
                        min(performance['total_attribution_weight'] / 100, 1.0) * 0.3  # 30% usage
                    )
                    performance['performance_score'] = performance_score
            
        except Exception as e:
            logger.error(f"Error calculating performance scores: {e}")
    
    def get_model_attribution_summary(self, time_period_hours: int = 24) -> Dict:
        """Get model attribution summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            
            # Filter recent attributions
            recent_attributions = [
                attr for attr in self.attribution_history 
                if attr.timestamp >= cutoff_time
            ]
            
            if not recent_attributions:
                return {}
            
            # Group by model type
            model_summary = {}
            for attribution in recent_attributions:
                model_type = attribution.model_type.value
                
                if model_type not in model_summary:
                    model_summary[model_type] = {
                        'total_attributions': 0,
                        'average_weight': 0.0,
                        'average_confidence': 0.0,
                        'total_value': 0.0,
                        'attribution_types': {}
                    }
                
                summary = model_summary[model_type]
                summary['total_attributions'] += 1
                summary['total_value'] += attribution.value
                
                # Update attribution types
                attr_type = attribution.attribution_type.value
                if attr_type not in summary['attribution_types']:
                    summary['attribution_types'][attr_type] = 0
                summary['attribution_types'][attr_type] += 1
            
            # Calculate averages
            for model_type, summary in model_summary.items():
                if summary['total_attributions'] > 0:
                    summary['average_weight'] = summary['total_value'] / summary['total_attributions']
                    summary['average_confidence'] = np.mean([
                        attr.confidence for attr in recent_attributions 
                        if attr.model_type.value == model_type
                    ])
            
            return {
                'time_period_hours': time_period_hours,
                'total_attributions': len(recent_attributions),
                'model_summary': model_summary,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting model attribution summary: {e}")
            return {}
    
    def get_decision_attribution_breakdown(self, decision_id: str) -> Dict:
        """Get detailed attribution breakdown for a specific decision"""
        try:
            # Find decision
            decision = None
            for d in self.decision_history:
                if d.decision_id == decision_id:
                    decision = d
                    break
            
            if not decision:
                return {}
            
            # Create breakdown
            breakdown = {
                'decision_id': decision.decision_id,
                'timestamp': decision.timestamp,
                'symbol': decision.symbol,
                'action': decision.action,
                'final_decision': decision.final_decision,
                'total_confidence': decision.total_confidence,
                'model_breakdown': [],
                'attribution_summary': {}
            }
            
            # Process model attributions
            for attribution in decision.model_attributions:
                model_breakdown = {
                    'model_id': attribution.model_id,
                    'model_type': attribution.model_type.value,
                    'attribution_type': attribution.attribution_type.value,
                    'value': attribution.value,
                    'weight': attribution.weight,
                    'confidence': attribution.confidence,
                    'context': attribution.context
                }
                breakdown['model_breakdown'].append(model_breakdown)
            
            # Create attribution summary
            attribution_summary = {}
            for attribution in decision.model_attributions:
                model_type = attribution.model_type.value
                if model_type not in attribution_summary:
                    attribution_summary[model_type] = {
                        'total_weight': 0.0,
                        'average_confidence': 0.0,
                        'attribution_count': 0
                    }
                
                summary = attribution_summary[model_type]
                summary['total_weight'] += attribution.weight
                summary['attribution_count'] += 1
            
            # Calculate averages
            for model_type, summary in attribution_summary.items():
                if summary['attribution_count'] > 0:
                    model_attributions = [
                        attr for attr in decision.model_attributions 
                        if attr.model_type.value == model_type
                    ]
                    summary['average_confidence'] = np.mean([attr.confidence for attr in model_attributions])
            
            breakdown['attribution_summary'] = attribution_summary
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Error getting decision attribution breakdown: {e}")
            return {}
    
    def create_attribution_visualization(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """Create visualization data for model attribution"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            
            # Filter recent decisions
            recent_decisions = [
                d for d in self.decision_history 
                if d.timestamp >= cutoff_time
            ]
            
            if not recent_decisions:
                return {}
            
            # Prepare data for visualization
            model_data = {}
            decision_data = []
            
            for decision in recent_decisions:
                decision_record = {
                    'timestamp': decision.timestamp,
                    'symbol': decision.symbol,
                    'action': decision.action,
                    'total_confidence': decision.total_confidence
                }
                
                # Add model contributions
                for attribution in decision.model_attributions:
                    model_type = attribution.model_type.value
                    decision_record[f'{model_type}_weight'] = attribution.weight
                    decision_record[f'{model_type}_confidence'] = attribution.confidence
                    
                    if model_type not in model_data:
                        model_data[model_type] = {
                            'weights': [],
                            'confidences': [],
                            'timestamps': []
                        }
                    
                    model_data[model_type]['weights'].append(attribution.weight)
                    model_data[model_type]['confidences'].append(attribution.confidence)
                    model_data[model_type]['timestamps'].append(decision.timestamp)
                
                decision_data.append(decision_record)
            
            # Create visualization data
            visualization_data = {
                'model_attribution_chart': self._create_model_attribution_chart(model_data),
                'decision_confidence_chart': self._create_decision_confidence_chart(decision_data),
                'model_performance_chart': self._create_model_performance_chart(),
                'attribution_heatmap': self._create_attribution_heatmap(decision_data)
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error creating attribution visualization: {e}")
            return {}
    
    def _create_model_attribution_chart(self, model_data: Dict) -> Dict:
        """Create model attribution chart data"""
        try:
            chart_data = {
                'type': 'bar',
                'data': {
                    'models': list(model_data.keys()),
                    'average_weights': [np.mean(model_data[model]['weights']) for model in model_data.keys()],
                    'average_confidences': [np.mean(model_data[model]['confidences']) for model in model_data.keys()]
                },
                'layout': {
                    'title': 'Model Attribution Weights and Confidence',
                    'xaxis': {'title': 'Model Type'},
                    'yaxis': {'title': 'Value'},
                    'barmode': 'group'
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating model attribution chart: {e}")
            return {}
    
    def _create_decision_confidence_chart(self, decision_data: List[Dict]) -> Dict:
        """Create decision confidence chart data"""
        try:
            if not decision_data:
                return {}
            
            timestamps = [d['timestamp'] for d in decision_data]
            confidences = [d['total_confidence'] for d in decision_data]
            symbols = [d['symbol'] for d in decision_data]
            
            chart_data = {
                'type': 'scatter',
                'data': {
                    'timestamps': timestamps,
                    'confidences': confidences,
                    'symbols': symbols
                },
                'layout': {
                    'title': 'Decision Confidence Over Time',
                    'xaxis': {'title': 'Time'},
                    'yaxis': {'title': 'Confidence Score'},
                    'mode': 'markers+lines'
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating decision confidence chart: {e}")
            return {}
    
    def _create_model_performance_chart(self) -> Dict:
        """Create model performance chart data"""
        try:
            models = list(self.model_performance.keys())
            performance_scores = [self.model_performance[model]['performance_score'] for model in models]
            accuracies = [self.model_performance[model]['accuracy'] for model in models]
            
            chart_data = {
                'type': 'bar',
                'data': {
                    'models': models,
                    'performance_scores': performance_scores,
                    'accuracies': accuracies
                },
                'layout': {
                    'title': 'Model Performance Comparison',
                    'xaxis': {'title': 'Model Type'},
                    'yaxis': {'title': 'Performance Score'},
                    'barmode': 'group'
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating model performance chart: {e}")
            return {}
    
    def _create_attribution_heatmap(self, decision_data: List[Dict]) -> Dict:
        """Create attribution heatmap data"""
        try:
            if not decision_data:
                return {}
            
            # Extract model types from decision data
            model_types = set()
            for decision in decision_data:
                for key in decision.keys():
                    if key.endswith('_weight'):
                        model_type = key.replace('_weight', '')
                        model_types.add(model_type)
            
            model_types = list(model_types)
            
            # Create heatmap data
            heatmap_data = []
            for i, decision in enumerate(decision_data):
                row = [i]  # Decision index
                for model_type in model_types:
                    weight_key = f'{model_type}_weight'
                    weight = decision.get(weight_key, 0.0)
                    row.append(weight)
                heatmap_data.append(row)
            
            chart_data = {
                'type': 'heatmap',
                'data': {
                    'z': heatmap_data,
                    'x': ['Decision'] + model_types,
                    'y': [f'Decision {i}' for i in range(len(decision_data))]
                },
                'layout': {
                    'title': 'Model Attribution Heatmap',
                    'xaxis': {'title': 'Model Type'},
                    'yaxis': {'title': 'Decision'}
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating attribution heatmap: {e}")
            return {}
    
    def get_model_ranking(self, metric: str = 'performance_score') -> List[Dict]:
        """Get model ranking by specified metric"""
        try:
            ranking = []
            
            for model_type, performance in self.model_performance.items():
                if performance['total_decisions'] > 0:
                    ranking.append({
                        'model_type': model_type,
                        'metric_value': performance.get(metric, 0.0),
                        'total_decisions': performance['total_decisions'],
                        'accuracy': performance['accuracy'],
                        'average_confidence': performance['average_confidence'],
                        'performance_score': performance['performance_score']
                    })
            
            # Sort by metric value (descending)
            ranking.sort(key=lambda x: x['metric_value'], reverse=True)
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error getting model ranking: {e}")
            return []
    
    def get_attribution_statistics(self) -> Dict:
        """Get comprehensive attribution statistics"""
        try:
            if not self.attribution_history:
                return {}
            
            # Calculate statistics
            total_attributions = len(self.attribution_history)
            model_type_counts = {}
            attribution_type_counts = {}
            
            for attribution in self.attribution_history:
                model_type = attribution.model_type.value
                attr_type = attribution.attribution_type.value
                
                model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
                attribution_type_counts[attr_type] = attribution_type_counts.get(attr_type, 0) + 1
            
            # Calculate averages
            average_confidence = np.mean([attr.confidence for attr in self.attribution_history])
            average_weight = np.mean([attr.weight for attr in self.attribution_history])
            
            return {
                'total_attributions': total_attributions,
                'total_decisions': len(self.decision_history),
                'model_type_counts': model_type_counts,
                'attribution_type_counts': attribution_type_counts,
                'average_confidence': average_confidence,
                'average_weight': average_weight,
                'model_performance': self.model_performance,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting attribution statistics: {e}")
            return {}
    
    def export_attribution_data(self, filepath: str, time_period_hours: int = 24):
        """Export attribution data to file"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            
            # Filter recent data
            recent_decisions = [
                d for d in self.decision_history 
                if d.timestamp >= cutoff_time
            ]
            
            recent_attributions = [
                attr for attr in self.attribution_history 
                if attr.timestamp >= cutoff_time
            ]
            
            # Prepare export data
            export_data = {
                'export_info': {
                    'time_period_hours': time_period_hours,
                    'cutoff_time': cutoff_time.isoformat(),
                    'export_timestamp': datetime.now().isoformat(),
                    'total_decisions': len(recent_decisions),
                    'total_attributions': len(recent_attributions)
                },
                'decisions': [
                    {
                        'decision_id': d.decision_id,
                        'timestamp': d.timestamp.isoformat(),
                        'symbol': d.symbol,
                        'action': d.action,
                        'final_decision': d.final_decision,
                        'total_confidence': d.total_confidence,
                        'decision_factors': d.decision_factors
                    }
                    for d in recent_decisions
                ],
                'attributions': [
                    {
                        'model_id': attr.model_id,
                        'model_type': attr.model_type.value,
                        'attribution_type': attr.attribution_type.value,
                        'value': attr.value,
                        'weight': attr.weight,
                        'confidence': attr.confidence,
                        'timestamp': attr.timestamp.isoformat(),
                        'context': attr.context
                    }
                    for attr in recent_attributions
                ],
                'model_performance': self.model_performance
            }
            
            # Export to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported attribution data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting attribution data: {e}")
    
    def update_model_outcome(self, decision_id: str, outcome: str, actual_pnl: float):
        """Update model performance with actual outcome"""
        try:
            # Find decision
            decision = None
            for d in self.decision_history:
                if d.decision_id == decision_id:
                    decision = d
                    break
            
            if not decision:
                logger.warning(f"Decision {decision_id} not found for outcome update")
                return
            
            # Update model performance based on outcome
            for attribution in decision.model_attributions:
                model_type = attribution.model_type.value
                
                if model_type in self.model_performance:
                    # Determine if decision was correct (simplified logic)
                    is_correct = (outcome == 'profit' and actual_pnl > 0) or (outcome == 'loss' and actual_pnl < 0)
                    
                    if is_correct:
                        self.model_performance[model_type]['correct_decisions'] += 1
                    
                    # Recalculate accuracy
                    total_decisions = self.model_performance[model_type]['total_decisions']
                    if total_decisions > 0:
                        self.model_performance[model_type]['accuracy'] = (
                            self.model_performance[model_type]['correct_decisions'] / total_decisions
                        )
            
            # Recalculate performance scores
            self._calculate_performance_scores()
            
            logger.info(f"Updated model outcomes for decision {decision_id}")
            
        except Exception as e:
            logger.error(f"Error updating model outcome: {e}")
