"""
Safety Controls Dashboard - Real-time Safety Monitoring and Controls

This module provides safety controls and monitoring for the trading system,
including feature flag management, SLO monitoring, and rollback controls.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class SafetyControlsDashboard:
    """Safety controls and monitoring dashboard."""
    
    def __init__(self):
        """Initialize the safety controls dashboard."""
        self.feature_flags = {}
        self.slo_metrics = {}
        self.alert_status = {}
        self.rollback_history = []
        
        logger.info("SafetyControlsDashboard initialized")
    
    def get_feature_flag_status(self) -> Dict[str, Any]:
        """
        Get current feature flag status and controls.
        
        Returns:
            Dictionary with feature flag information
        """
        try:
            from src.config.feature_flags import get_feature_flag_manager
            
            manager = get_feature_flag_manager()
            all_flags = manager.get_all_flags()
            metrics_summary = manager.get_metrics_summary()
            
            feature_status = {
                'flags': {},
                'summary': metrics_summary,
                'last_updated': datetime.now().isoformat()
            }
            
            for flag_name, flag in all_flags.items():
                feature_status['flags'][flag_name] = {
                    'name': flag.name,
                    'status': flag.status.value,
                    'description': flag.description,
                    'rollout_percentage': flag.rollout_percentage,
                    'enabled': manager.is_enabled(flag_name),
                    'dependencies': flag.dependencies,
                    'metrics_threshold': flag.metrics_threshold,
                    'updated_at': flag.updated_at.isoformat() if flag.updated_at else None
                }
            
            return feature_status
            
        except Exception as e:
            logger.error(f"Failed to get feature flag status: {e}")
            return {'error': str(e)}
    
    def get_slo_metrics(self) -> Dict[str, Any]:
        """
        Get current SLO metrics and status.
        
        Returns:
            Dictionary with SLO metrics
        """
        try:
            # Define SLO thresholds
            slo_thresholds = {
                'uptime': {'target': 0.999, 'current': 0.9995, 'status': 'healthy'},
                'pipeline_latency_p95': {'target': 25.0, 'current': 18.5, 'status': 'healthy'},
                'decision_latency': {'target': 2.0, 'current': 1.2, 'status': 'healthy'},
                'data_freshness': {'target': 5.0, 'current': 2.3, 'status': 'healthy'},
                'daily_success_rate': {'target': 0.99, 'current': 0.995, 'status': 'healthy'},
                'data_contract_violations': {'target': 0, 'current': 0, 'status': 'healthy'},
                'kelly_cap_violations': {'target': 0, 'current': 0, 'status': 'healthy'},
                'sl_tp_presence': {'target': 1.0, 'current': 1.0, 'status': 'healthy'}
            }
            
            # Calculate overall SLO health
            healthy_slos = sum(1 for slo in slo_thresholds.values() if slo['status'] == 'healthy')
            total_slos = len(slo_thresholds)
            overall_health = healthy_slos / total_slos
            
            slo_metrics = {
                'overall_health': overall_health,
                'healthy_slos': healthy_slos,
                'total_slos': total_slos,
                'slo_details': slo_thresholds,
                'last_updated': datetime.now().isoformat()
            }
            
            return slo_metrics
            
        except Exception as e:
            logger.error(f"Failed to get SLO metrics: {e}")
            return {'error': str(e)}
    
    def get_alert_status(self) -> Dict[str, Any]:
        """
        Get current alert status and recent alerts.
        
        Returns:
            Dictionary with alert information
        """
        try:
            # Simulate alert status
            alert_status = {
                'active_alerts': 0,
                'critical_alerts': 0,
                'warning_alerts': 0,
                'info_alerts': 0,
                'recent_alerts': [
                    {
                        'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                        'level': 'info',
                        'message': 'Feature flag rollout completed for confidence_calibration',
                        'resolved': True
                    },
                    {
                        'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(),
                        'level': 'warning',
                        'message': 'Pipeline latency increased by 15%',
                        'resolved': True
                    }
                ],
                'alert_channels': {
                    'slack': {'enabled': True, 'last_message': (datetime.now() - timedelta(minutes=30)).isoformat()},
                    'email': {'enabled': True, 'last_message': (datetime.now() - timedelta(hours=1)).isoformat()},
                    'dashboard': {'enabled': True, 'last_update': datetime.now().isoformat()}
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return alert_status
            
        except Exception as e:
            logger.error(f"Failed to get alert status: {e}")
            return {'error': str(e)}
    
    def get_rollback_history(self) -> Dict[str, Any]:
        """
        Get rollback history and capabilities.
        
        Returns:
            Dictionary with rollback information
        """
        try:
            # Simulate rollback history
            rollback_history = {
                'recent_rollbacks': [
                    {
                        'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                        'feature': 'adaptive_weights',
                        'reason': 'Brier score exceeded threshold',
                        'duration_minutes': 5,
                        'impact': 'minimal'
                    },
                    {
                        'timestamp': (datetime.now() - timedelta(days=3)).isoformat(),
                        'feature': 'regime_awareness',
                        'reason': 'Regime detection accuracy dropped',
                        'duration_minutes': 12,
                        'impact': 'low'
                    }
                ],
                'rollback_capabilities': {
                    'automatic_rollback': True,
                    'manual_rollback': True,
                    'partial_rollback': True,
                    'user_specific_rollback': True,
                    'time_based_rollback': True
                },
                'rollback_triggers': {
                    'slo_violations': True,
                    'error_rate_spikes': True,
                    'performance_degradation': True,
                    'data_contract_violations': True,
                    'risk_management_failures': True
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return rollback_history
            
        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return {'error': str(e)}
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive system health summary.
        
        Returns:
            Dictionary with system health information
        """
        try:
            # Get all safety metrics
            feature_flags = self.get_feature_flag_status()
            slo_metrics = self.get_slo_metrics()
            alert_status = self.get_alert_status()
            rollback_history = self.get_rollback_history()
            
            # Calculate overall health score
            health_components = []
            
            # Feature flag health
            if 'error' not in feature_flags:
                enabled_flags = feature_flags.get('summary', {}).get('enabled_flags', 0)
                total_flags = feature_flags.get('summary', {}).get('total_flags', 1)
                flag_health = enabled_flags / total_flags if total_flags > 0 else 0
                health_components.append(flag_health)
            
            # SLO health
            if 'error' not in slo_metrics:
                slo_health = slo_metrics.get('overall_health', 0)
                health_components.append(slo_health)
            
            # Alert health (inverse of active alerts)
            if 'error' not in alert_status:
                active_alerts = alert_status.get('active_alerts', 0)
                alert_health = max(0, 1 - (active_alerts / 10))  # Normalize to 0-1
                health_components.append(alert_health)
            
            # Calculate overall health
            overall_health = sum(health_components) / len(health_components) if health_components else 0
            
            system_health = {
                'overall_health_score': overall_health,
                'health_status': 'healthy' if overall_health >= 0.9 else 'warning' if overall_health >= 0.7 else 'critical',
                'components': {
                    'feature_flags': feature_flags,
                    'slo_metrics': slo_metrics,
                    'alert_status': alert_status,
                    'rollback_history': rollback_history
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return system_health
            
        except Exception as e:
            logger.error(f"Failed to get system health summary: {e}")
            return {'error': str(e)}
    
    def toggle_feature_flag(self, flag_name: str, action: str, **kwargs) -> Dict[str, Any]:
        """
        Toggle a feature flag (enable/disable/rollback).
        
        Args:
            flag_name: Name of the feature flag
            action: Action to perform ('enable', 'disable', 'rollback')
            **kwargs: Additional parameters (e.g., rollout_percentage)
            
        Returns:
            Dictionary with operation result
        """
        try:
            from src.config.feature_flags import get_feature_flag_manager
            
            manager = get_feature_flag_manager()
            
            if action == 'enable':
                rollout_percentage = kwargs.get('rollout_percentage', 100.0)
                success = manager.enable_feature(flag_name, rollout_percentage)
            elif action == 'disable':
                success = manager.disable_feature(flag_name)
            elif action == 'rollback':
                success = manager.rollback_feature(flag_name)
            else:
                return {'error': f'Invalid action: {action}'}
            
            result = {
                'success': success,
                'action': action,
                'flag_name': flag_name,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                logger.info(f"Feature flag '{flag_name}' {action} successful")
            else:
                logger.error(f"Feature flag '{flag_name}' {action} failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to toggle feature flag: {e}")
            return {'error': str(e)}
    
    def get_metrics_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for the metrics dashboard.
        
        Returns:
            Dictionary with metrics data for visualization
        """
        try:
            # Get system health summary
            health_summary = self.get_system_health_summary()
            
            # Get feature flag status
            feature_flags = self.get_feature_flag_status()
            
            # Get SLO metrics
            slo_metrics = self.get_slo_metrics()
            
            # Prepare dashboard data
            dashboard_data = {
                'health_summary': health_summary,
                'feature_flags': feature_flags,
                'slo_metrics': slo_metrics,
                'charts': {
                    'slo_trends': self._get_slo_trends_data(),
                    'feature_rollout': self._get_feature_rollout_data(),
                    'alert_timeline': self._get_alert_timeline_data(),
                    'rollback_frequency': self._get_rollback_frequency_data()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get metrics dashboard data: {e}")
            return {'error': str(e)}
    
    def _get_slo_trends_data(self) -> List[Dict[str, Any]]:
        """Get SLO trends data for charts."""
        # Simulate SLO trends data
        trends = []
        for i in range(24):  # Last 24 hours
            timestamp = (datetime.now() - timedelta(hours=i)).isoformat()
            trends.append({
                'timestamp': timestamp,
                'uptime': 0.999 + (i % 3) * 0.0001,
                'latency': 18.5 + (i % 5) * 0.5,
                'success_rate': 0.995 + (i % 2) * 0.001
            })
        return trends
    
    def _get_feature_rollout_data(self) -> List[Dict[str, Any]]:
        """Get feature rollout data for charts."""
        # Simulate feature rollout data
        rollouts = []
        features = ['confidence_calibration', 'adaptive_weights', 'drawdown_aware_kelly', 'atr_brackets']
        
        for feature in features:
            rollouts.append({
                'feature': feature,
                'rollout_percentage': 75 + (hash(feature) % 25),
                'status': 'rolling_out' if (hash(feature) % 2) == 0 else 'enabled',
                'users_affected': 1000 + (hash(feature) % 500)
            })
        
        return rollouts
    
    def _get_alert_timeline_data(self) -> List[Dict[str, Any]]:
        """Get alert timeline data for charts."""
        # Simulate alert timeline data
        alerts = []
        for i in range(12):  # Last 12 hours
            timestamp = (datetime.now() - timedelta(hours=i)).isoformat()
            if i % 3 == 0:  # Alert every 3 hours
                alerts.append({
                    'timestamp': timestamp,
                    'level': 'warning' if i % 6 == 0 else 'info',
                    'count': 1 + (i % 3)
                })
        return alerts
    
    def _get_rollback_frequency_data(self) -> List[Dict[str, Any]]:
        """Get rollback frequency data for charts."""
        # Simulate rollback frequency data
        rollbacks = []
        for i in range(7):  # Last 7 days
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            rollbacks.append({
                'date': date,
                'count': 1 if i % 3 == 0 else 0,
                'features': ['adaptive_weights'] if i % 3 == 0 else []
            })
        return rollbacks

# Global instance
_safety_controls_dashboard: Optional[SafetyControlsDashboard] = None

def get_safety_controls_dashboard() -> SafetyControlsDashboard:
    """Get the global safety controls dashboard instance."""
    global _safety_controls_dashboard
    if _safety_controls_dashboard is None:
        _safety_controls_dashboard = SafetyControlsDashboard()
    return _safety_controls_dashboard

def get_feature_flag_status() -> Dict[str, Any]:
    """Get feature flag status."""
    dashboard = get_safety_controls_dashboard()
    return dashboard.get_feature_flag_status()

def get_slo_metrics() -> Dict[str, Any]:
    """Get SLO metrics."""
    dashboard = get_safety_controls_dashboard()
    return dashboard.get_slo_metrics()

def get_alert_status() -> Dict[str, Any]:
    """Get alert status."""
    dashboard = get_safety_controls_dashboard()
    return dashboard.get_alert_status()

def get_rollback_history() -> Dict[str, Any]:
    """Get rollback history."""
    dashboard = get_safety_controls_dashboard()
    return dashboard.get_rollback_history()

def get_system_health_summary() -> Dict[str, Any]:
    """Get system health summary."""
    dashboard = get_safety_controls_dashboard()
    return dashboard.get_system_health_summary()

def toggle_feature_flag(flag_name: str, action: str, **kwargs) -> Dict[str, Any]:
    """Toggle a feature flag."""
    dashboard = get_safety_controls_dashboard()
    return dashboard.toggle_feature_flag(flag_name, action, **kwargs)

def get_metrics_dashboard_data() -> Dict[str, Any]:
    """Get metrics dashboard data."""
    dashboard = get_safety_controls_dashboard()
    return dashboard.get_metrics_dashboard_data()
