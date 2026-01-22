"""
Policy Versioning & Comparison
Enables objective tracking of learning improvement
"""

import os
import json
import pickle
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PolicyStatus(Enum):
    """Policy status"""
    TRAINING = "training"
    VALIDATION = "validation"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"

@dataclass
class PolicyMetrics:
    """Policy performance metrics"""
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    volatility: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    trades_count: int
    avg_trade_duration: float
    timestamp: datetime

@dataclass
class PolicyVersion:
    """Policy version information"""
    policy_id: str
    version: str
    timestamp: datetime
    status: PolicyStatus
    metrics: PolicyMetrics
    model_path: str
    config: Dict[str, Any]
    training_data: Dict[str, Any]
    validation_period: Tuple[datetime, datetime]
    performance_summary: Dict[str, float]

class PolicyVersionManager:
    """Manages policy versions and comparisons"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.policies_dir = config.get('policies_dir', 'models/policies')
        self.max_versions = config.get('max_versions', 10)
        self.validation_period_days = config.get('validation_period_days', 30)
        self.promotion_threshold = config.get('promotion_threshold', 0.1)  # 10% improvement
        
        # Create directories
        os.makedirs(self.policies_dir, exist_ok=True)
        os.makedirs(os.path.join(self.policies_dir, 'training'), exist_ok=True)
        os.makedirs(os.path.join(self.policies_dir, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(self.policies_dir, 'production'), exist_ok=True)
        os.makedirs(os.path.join(self.policies_dir, 'archived'), exist_ok=True)
        
        # Load existing policies
        self.policies = self._load_policies()
        
        logger.info("Policy Version Manager initialized")
    
    def create_policy_version(self, policy_id: str, model, config: Dict, 
                            training_data: Dict) -> str:
        """Create a new policy version"""
        try:
            # Generate version string
            version = self._generate_version_string()
            
            # Create policy version
            policy_version = PolicyVersion(
                policy_id=policy_id,
                version=version,
                timestamp=datetime.now(),
                status=PolicyStatus.TRAINING,
                metrics=PolicyMetrics(
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    total_return=0.0,
                    volatility=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    calmar_ratio=0.0,
                    sortino_ratio=0.0,
                    trades_count=0,
                    avg_trade_duration=0.0,
                    timestamp=datetime.now()
                ),
                model_path="",
                config=config,
                training_data=training_data,
                validation_period=(datetime.now(), datetime.now() + timedelta(days=self.validation_period_days)),
                performance_summary={}
            )
            
            # Save model
            model_path = self._save_model(policy_id, version, model)
            policy_version.model_path = model_path
            
            # Save policy metadata
            self._save_policy_metadata(policy_version)
            
            # Add to policies
            if policy_id not in self.policies:
                self.policies[policy_id] = []
            self.policies[policy_id].append(policy_version)
            
            logger.info(f"Created policy version {policy_id}:{version}")
            return version
            
        except Exception as e:
            logger.error(f"Error creating policy version: {e}")
            return ""
    
    def _generate_version_string(self) -> str:
        """Generate version string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _save_model(self, policy_id: str, version: str, model) -> str:
        """Save model to disk"""
        try:
            model_dir = os.path.join(self.policies_dir, 'training', policy_id, version)
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, 'model.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    def _save_policy_metadata(self, policy_version: PolicyVersion):
        """Save policy metadata"""
        try:
            metadata_dir = os.path.join(self.policies_dir, 'training', 
                                      policy_version.policy_id, policy_version.version)
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata_path = os.path.join(metadata_dir, 'metadata.json')
            
            # Convert to dict for JSON serialization
            metadata = asdict(policy_version)
            metadata['timestamp'] = policy_version.timestamp.isoformat()
            metadata['metrics']['timestamp'] = policy_version.metrics.timestamp.isoformat()
            metadata['validation_period'] = [
                policy_version.validation_period[0].isoformat(),
                policy_version.validation_period[1].isoformat()
            ]
            metadata['status'] = policy_version.status.value
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving policy metadata: {e}")
    
    def _load_policies(self) -> Dict[str, List[PolicyVersion]]:
        """Load existing policies from disk"""
        try:
            policies = {}
            
            for status_dir in ['training', 'validation', 'production', 'archived']:
                status_path = os.path.join(self.policies_dir, status_dir)
                if not os.path.exists(status_path):
                    continue
                
                for policy_id in os.listdir(status_path):
                    policy_path = os.path.join(status_path, policy_id)
                    if not os.path.isdir(policy_path):
                        continue
                    
                    if policy_id not in policies:
                        policies[policy_id] = []
                    
                    for version in os.listdir(policy_path):
                        version_path = os.path.join(policy_path, version)
                        if not os.path.isdir(version_path):
                            continue
                        
                        metadata_path = os.path.join(version_path, 'metadata.json')
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                
                                # Convert back to PolicyVersion
                                policy_version = self._dict_to_policy_version(metadata)
                                policies[policy_id].append(policy_version)
                                
                            except Exception as e:
                                logger.error(f"Error loading policy {policy_id}:{version}: {e}")
            
            # Sort policies by timestamp
            for policy_id in policies:
                policies[policy_id].sort(key=lambda x: x.timestamp, reverse=True)
            
            return policies
            
        except Exception as e:
            logger.error(f"Error loading policies: {e}")
            return {}
    
    def _dict_to_policy_version(self, metadata: Dict) -> PolicyVersion:
        """Convert dict to PolicyVersion"""
        try:
            # Convert metrics
            metrics_data = metadata['metrics']
            metrics = PolicyMetrics(
                sharpe_ratio=metrics_data['sharpe_ratio'],
                max_drawdown=metrics_data['max_drawdown'],
                total_return=metrics_data['total_return'],
                volatility=metrics_data['volatility'],
                win_rate=metrics_data['win_rate'],
                profit_factor=metrics_data['profit_factor'],
                calmar_ratio=metrics_data['calmar_ratio'],
                sortino_ratio=metrics_data['sortino_ratio'],
                trades_count=metrics_data['trades_count'],
                avg_trade_duration=metrics_data['avg_trade_duration'],
                timestamp=datetime.fromisoformat(metrics_data['timestamp'])
            )
            
            # Convert validation period
            validation_period = (
                datetime.fromisoformat(metadata['validation_period'][0]),
                datetime.fromisoformat(metadata['validation_period'][1])
            )
            
            return PolicyVersion(
                policy_id=metadata['policy_id'],
                version=metadata['version'],
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                status=PolicyStatus(metadata['status']),
                metrics=metrics,
                model_path=metadata['model_path'],
                config=metadata['config'],
                training_data=metadata['training_data'],
                validation_period=validation_period,
                performance_summary=metadata['performance_summary']
            )
            
        except Exception as e:
            logger.error(f"Error converting dict to PolicyVersion: {e}")
            return None
    
    def update_policy_metrics(self, policy_id: str, version: str, metrics: PolicyMetrics):
        """Update policy metrics"""
        try:
            if policy_id not in self.policies:
                logger.error(f"Policy {policy_id} not found")
                return
            
            # Find the policy version
            policy_version = None
            for pv in self.policies[policy_id]:
                if pv.version == version:
                    policy_version = pv
                    break
            
            if not policy_version:
                logger.error(f"Policy version {policy_id}:{version} not found")
                return
            
            # Update metrics
            policy_version.metrics = metrics
            policy_version.performance_summary = self._calculate_performance_summary(metrics)
            
            # Save updated metadata
            self._save_policy_metadata(policy_version)
            
            logger.info(f"Updated metrics for policy {policy_id}:{version}")
            
        except Exception as e:
            logger.error(f"Error updating policy metrics: {e}")
    
    def _calculate_performance_summary(self, metrics: PolicyMetrics) -> Dict[str, float]:
        """Calculate performance summary"""
        try:
            return {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'total_return': metrics.total_return,
                'volatility': metrics.volatility,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'calmar_ratio': metrics.calmar_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'trades_count': metrics.trades_count,
                'avg_trade_duration': metrics.avg_trade_duration,
                'risk_adjusted_return': metrics.sharpe_ratio * metrics.total_return,
                'stability_score': 1.0 / (1.0 + metrics.max_drawdown) if metrics.max_drawdown > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {}
    
    def promote_policy(self, policy_id: str, version: str) -> bool:
        """Promote policy to production if it meets criteria"""
        try:
            if policy_id not in self.policies:
                logger.error(f"Policy {policy_id} not found")
                return False
            
            # Find the policy version
            policy_version = None
            for pv in self.policies[policy_id]:
                if pv.version == version:
                    policy_version = pv
                    break
            
            if not policy_version:
                logger.error(f"Policy version {policy_id}:{version} not found")
                return False
            
            # Check if policy meets promotion criteria
            if not self._meets_promotion_criteria(policy_id, policy_version):
                logger.info(f"Policy {policy_id}:{version} does not meet promotion criteria")
                return False
            
            # Move to production
            self._move_policy(policy_version, PolicyStatus.PRODUCTION)
            
            # Archive old production policies
            self._archive_old_production_policies(policy_id)
            
            logger.info(f"Promoted policy {policy_id}:{version} to production")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting policy: {e}")
            return False
    
    def _meets_promotion_criteria(self, policy_id: str, policy_version: PolicyVersion) -> bool:
        """Check if policy meets promotion criteria"""
        try:
            # Get current production policy
            current_production = self.get_current_production_policy(policy_id)
            
            if not current_production:
                # No current production policy, promote if metrics are reasonable
                return (policy_version.metrics.sharpe_ratio > 0.5 and 
                       policy_version.metrics.max_drawdown < 0.2)
            
            # Compare with current production
            improvement = self._calculate_improvement(current_production.metrics, policy_version.metrics)
            
            return improvement >= self.promotion_threshold
            
        except Exception as e:
            logger.error(f"Error checking promotion criteria: {e}")
            return False
    
    def _calculate_improvement(self, current_metrics: PolicyMetrics, 
                             new_metrics: PolicyMetrics) -> float:
        """Calculate improvement over current policy"""
        try:
            # Weighted improvement calculation
            sharpe_improvement = (new_metrics.sharpe_ratio - current_metrics.sharpe_ratio) / max(current_metrics.sharpe_ratio, 0.1)
            drawdown_improvement = (current_metrics.max_drawdown - new_metrics.max_drawdown) / max(current_metrics.max_drawdown, 0.01)
            return_improvement = (new_metrics.total_return - current_metrics.total_return) / max(abs(current_metrics.total_return), 0.01)
            
            # Weighted average
            improvement = (sharpe_improvement * 0.4 + 
                          drawdown_improvement * 0.3 + 
                          return_improvement * 0.3)
            
            return improvement
            
        except Exception as e:
            logger.error(f"Error calculating improvement: {e}")
            return 0.0
    
    def _move_policy(self, policy_version: PolicyVersion, new_status: PolicyStatus):
        """Move policy to new status directory"""
        try:
            # Get current path
            current_path = os.path.dirname(policy_version.model_path)
            
            # Get new path
            new_path = os.path.join(self.policies_dir, new_status.value, 
                                  policy_version.policy_id, policy_version.version)
            
            # Move directory
            if os.path.exists(current_path) and current_path != new_path:
                shutil.move(current_path, new_path)
                
                # Update model path
                policy_version.model_path = os.path.join(new_path, 'model.pkl')
            
            # Update status
            policy_version.status = new_status
            
            # Save updated metadata
            self._save_policy_metadata(policy_version)
            
        except Exception as e:
            logger.error(f"Error moving policy: {e}")
    
    def _archive_old_production_policies(self, policy_id: str):
        """Archive old production policies"""
        try:
            production_policies = [
                pv for pv in self.policies[policy_id] 
                if pv.status == PolicyStatus.PRODUCTION
            ]
            
            # Keep only the latest production policy
            if len(production_policies) > 1:
                production_policies.sort(key=lambda x: x.timestamp, reverse=True)
                
                for old_policy in production_policies[1:]:
                    self._move_policy(old_policy, PolicyStatus.ARCHIVED)
            
        except Exception as e:
            logger.error(f"Error archiving old production policies: {e}")
    
    def get_current_production_policy(self, policy_id: str) -> Optional[PolicyVersion]:
        """Get current production policy"""
        try:
            if policy_id not in self.policies:
                return None
            
            production_policies = [
                pv for pv in self.policies[policy_id] 
                if pv.status == PolicyStatus.PRODUCTION
            ]
            
            if not production_policies:
                return None
            
            # Return the latest production policy
            return max(production_policies, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Error getting current production policy: {e}")
            return None
    
    def get_policy_comparison(self, policy_id: str) -> pd.DataFrame:
        """Get policy comparison table"""
        try:
            if policy_id not in self.policies:
                return pd.DataFrame()
            
            comparison_data = []
            for policy_version in self.policies[policy_id]:
                comparison_data.append({
                    'version': policy_version.version,
                    'status': policy_version.status.value,
                    'timestamp': policy_version.timestamp,
                    'sharpe_ratio': policy_version.metrics.sharpe_ratio,
                    'max_drawdown': policy_version.metrics.max_drawdown,
                    'total_return': policy_version.metrics.total_return,
                    'volatility': policy_version.metrics.volatility,
                    'win_rate': policy_version.metrics.win_rate,
                    'profit_factor': policy_version.metrics.profit_factor,
                    'calmar_ratio': policy_version.metrics.calmar_ratio,
                    'sortino_ratio': policy_version.metrics.sortino_ratio,
                    'trades_count': policy_version.metrics.trades_count
                })
            
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('timestamp', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting policy comparison: {e}")
            return pd.DataFrame()
    
    def get_best_policy(self, policy_id: str, metric: str = 'sharpe_ratio') -> Optional[PolicyVersion]:
        """Get best policy based on metric"""
        try:
            if policy_id not in self.policies:
                return None
            
            if not self.policies[policy_id]:
                return None
            
            # Get all policies with valid metrics
            valid_policies = [
                pv for pv in self.policies[policy_id] 
                if hasattr(pv.metrics, metric)
            ]
            
            if not valid_policies:
                return None
            
            # Return best policy
            return max(valid_policies, key=lambda x: getattr(x.metrics, metric))
            
        except Exception as e:
            logger.error(f"Error getting best policy: {e}")
            return None
    
    def cleanup_old_versions(self, policy_id: str = None):
        """Cleanup old policy versions"""
        try:
            policies_to_clean = [policy_id] if policy_id else list(self.policies.keys())
            
            for pid in policies_to_clean:
                if pid not in self.policies:
                    continue
                
                # Keep only the latest max_versions
                if len(self.policies[pid]) > self.max_versions:
                    # Sort by timestamp and keep latest
                    self.policies[pid].sort(key=lambda x: x.timestamp, reverse=True)
                    
                    # Archive old versions
                    for old_policy in self.policies[pid][self.max_versions:]:
                        if old_policy.status != PolicyStatus.PRODUCTION:
                            self._move_policy(old_policy, PolicyStatus.ARCHIVED)
                    
                    # Update policies list
                    self.policies[pid] = self.policies[pid][:self.max_versions]
            
            logger.info("Cleaned up old policy versions")
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
    
    def get_policy_summary(self) -> Dict:
        """Get summary of all policies"""
        try:
            summary = {
                'total_policies': len(self.policies),
                'policies': {}
            }
            
            for policy_id, versions in self.policies.items():
                summary['policies'][policy_id] = {
                    'total_versions': len(versions),
                    'production_versions': len([v for v in versions if v.status == PolicyStatus.PRODUCTION]),
                    'validation_versions': len([v for v in versions if v.status == PolicyStatus.VALIDATION]),
                    'training_versions': len([v for v in versions if v.status == PolicyStatus.TRAINING]),
                    'archived_versions': len([v for v in versions if v.status == PolicyStatus.ARCHIVED]),
                    'latest_version': versions[0].version if versions else None,
                    'best_sharpe': max([v.metrics.sharpe_ratio for v in versions]) if versions else 0.0
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting policy summary: {e}")
            return {}
