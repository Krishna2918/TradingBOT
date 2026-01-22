"""
Feature Conflict Checker
Prevents models from relying on redundant or correlated features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of feature conflicts"""
    HIGH_CORRELATION = "high_correlation"
    REDUNDANT = "redundant"
    MULTICOLLINEARITY = "multicollinearity"
    LEAKAGE = "data_leakage"
    DUPLICATE = "duplicate"

@dataclass
class FeatureConflict:
    """Feature conflict information"""
    feature1: str
    feature2: str
    conflict_type: ConflictType
    correlation: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendation: str
    timestamp: datetime

@dataclass
class FeatureAnalysis:
    """Feature analysis results"""
    feature_name: str
    correlation_with_target: float
    variance: float
    missing_ratio: float
    uniqueness_ratio: float
    importance_score: float
    conflicts: List[FeatureConflict]
    recommendation: str

class FeatureConflictChecker:
    """Checks for feature conflicts and provides recommendations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.correlation_threshold = config.get('correlation_threshold', 0.9)
        self.vif_threshold = config.get('vif_threshold', 10.0)
        self.importance_threshold = config.get('importance_threshold', 0.01)
        self.conflict_history = []
        
        logger.info("Feature Conflict Checker initialized")
    
    def analyze_features(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, FeatureAnalysis]:
        """Analyze all features for conflicts and issues"""
        try:
            feature_analyses = {}
            
            for feature in X.columns:
                analysis = self._analyze_single_feature(X, y, feature)
                feature_analyses[feature] = analysis
            
            return feature_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing features: {e}")
            return {}
    
    def _analyze_single_feature(self, X: pd.DataFrame, y: pd.Series, feature: str) -> FeatureAnalysis:
        """Analyze a single feature"""
        try:
            feature_data = X[feature]
            
            # Basic statistics
            correlation_with_target = 0.0
            if y is not None:
                correlation_with_target = abs(feature_data.corr(y))
            
            variance = feature_data.var()
            missing_ratio = feature_data.isnull().sum() / len(feature_data)
            uniqueness_ratio = feature_data.nunique() / len(feature_data)
            
            # Calculate importance score (simplified)
            importance_score = self._calculate_importance_score(feature_data, y)
            
            # Find conflicts
            conflicts = self._find_feature_conflicts(X, feature)
            
            # Generate recommendation
            recommendation = self._generate_feature_recommendation(
                feature, correlation_with_target, variance, missing_ratio, 
                uniqueness_ratio, importance_score, conflicts
            )
            
            return FeatureAnalysis(
                feature_name=feature,
                correlation_with_target=correlation_with_target,
                variance=variance,
                missing_ratio=missing_ratio,
                uniqueness_ratio=uniqueness_ratio,
                importance_score=importance_score,
                conflicts=conflicts,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing feature {feature}: {e}")
            return FeatureAnalysis(
                feature_name=feature,
                correlation_with_target=0.0,
                variance=0.0,
                missing_ratio=0.0,
                uniqueness_ratio=0.0,
                importance_score=0.0,
                conflicts=[],
                recommendation="error"
            )
    
    def _calculate_importance_score(self, feature_data: pd.Series, y: pd.Series) -> float:
        """Calculate feature importance score"""
        try:
            if y is None:
                return 0.0
            
            # Simple correlation-based importance
            correlation = abs(feature_data.corr(y))
            
            # Adjust for variance (higher variance = higher importance)
            variance_score = min(1.0, feature_data.var() / feature_data.mean() if feature_data.mean() != 0 else 0.0)
            
            # Combine scores
            importance = (correlation * 0.7) + (variance_score * 0.3)
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating importance score: {e}")
            return 0.0
    
    def _find_feature_conflicts(self, X: pd.DataFrame, feature: str) -> List[FeatureConflict]:
        """Find conflicts for a specific feature"""
        try:
            conflicts = []
            feature_data = X[feature]
            
            for other_feature in X.columns:
                if other_feature == feature:
                    continue
                
                other_data = X[other_feature]
                
                # Check correlation
                correlation = abs(feature_data.corr(other_data))
                
                if correlation > self.correlation_threshold:
                    conflict_type = ConflictType.HIGH_CORRELATION
                    severity = self._determine_severity(correlation)
                    recommendation = self._get_correlation_recommendation(feature, other_feature, correlation)
                    
                    conflict = FeatureConflict(
                        feature1=feature,
                        feature2=other_feature,
                        conflict_type=conflict_type,
                        correlation=correlation,
                        severity=severity,
                        recommendation=recommendation,
                        timestamp=datetime.now()
                    )
                    conflicts.append(conflict)
                
                # Check for duplicates
                if self._is_duplicate_feature(feature_data, other_data):
                    conflict = FeatureConflict(
                        feature1=feature,
                        feature2=other_feature,
                        conflict_type=ConflictType.DUPLICATE,
                        correlation=1.0,
                        severity='critical',
                        recommendation=f"Remove duplicate feature: {other_feature}",
                        timestamp=datetime.now()
                    )
                    conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error finding conflicts for {feature}: {e}")
            return []
    
    def _determine_severity(self, correlation: float) -> str:
        """Determine conflict severity based on correlation"""
        if correlation >= 0.99:
            return 'critical'
        elif correlation >= 0.95:
            return 'high'
        elif correlation >= 0.9:
            return 'medium'
        else:
            return 'low'
    
    def _get_correlation_recommendation(self, feature1: str, feature2: str, correlation: float) -> str:
        """Get recommendation for correlated features"""
        if correlation >= 0.99:
            return f"Remove {feature2} - nearly identical to {feature1}"
        elif correlation >= 0.95:
            return f"Consider removing {feature2} - highly correlated with {feature1}"
        else:
            return f"Monitor correlation between {feature1} and {feature2}"
    
    def _is_duplicate_feature(self, feature1: pd.Series, feature2: pd.Series) -> bool:
        """Check if two features are duplicates"""
        try:
            # Check if they're identical
            if feature1.equals(feature2):
                return True
            
            # Check if one is a scaled version of the other
            if len(feature1) != len(feature2):
                return False
            
            # Check for linear relationship
            correlation = abs(feature1.corr(feature2))
            if correlation > 0.999:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate features: {e}")
            return False
    
    def _generate_feature_recommendation(self, feature: str, correlation_with_target: float,
                                       variance: float, missing_ratio: float,
                                       uniqueness_ratio: float, importance_score: float,
                                       conflicts: List[FeatureConflict]) -> str:
        """Generate recommendation for a feature"""
        try:
            recommendations = []
            
            # Check importance
            if importance_score < self.importance_threshold:
                recommendations.append("Consider removing - low importance")
            
            # Check missing values
            if missing_ratio > 0.5:
                recommendations.append("High missing values - consider imputation or removal")
            
            # Check variance
            if variance < 1e-10:
                recommendations.append("Low variance - consider removing")
            
            # Check uniqueness
            if uniqueness_ratio < 0.01:
                recommendations.append("Low uniqueness - may be categorical")
            
            # Check conflicts
            critical_conflicts = [c for c in conflicts if c.severity == 'critical']
            if critical_conflicts:
                recommendations.append("Critical conflicts detected - review immediately")
            
            if not recommendations:
                return "Feature looks good"
            
            return "; ".join(recommendations)
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Error in analysis"
    
    def check_multicollinearity(self, X: pd.DataFrame) -> Dict[str, float]:
        """Check for multicollinearity using VIF"""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                              for i in range(len(X.columns))]
            
            # Filter high VIF features
            high_vif = vif_data[vif_data["VIF"] > self.vif_threshold]
            
            return dict(zip(high_vif["Feature"], high_vif["VIF"]))
            
        except ImportError:
            logger.warning("statsmodels not available for VIF calculation")
            return {}
        except Exception as e:
            logger.error(f"Error checking multicollinearity: {e}")
            return {}
    
    def get_feature_ranking(self, feature_analyses: Dict[str, FeatureAnalysis]) -> pd.DataFrame:
        """Rank features by importance and conflicts"""
        try:
            ranking_data = []
            
            for feature, analysis in feature_analyses.items():
                # Calculate overall score
                importance_score = analysis.importance_score
                conflict_penalty = len(analysis.conflicts) * 0.1
                missing_penalty = analysis.missing_ratio * 0.2
                
                overall_score = importance_score - conflict_penalty - missing_penalty
                
                ranking_data.append({
                    'feature': feature,
                    'importance_score': importance_score,
                    'correlation_with_target': analysis.correlation_with_target,
                    'variance': analysis.variance,
                    'missing_ratio': analysis.missing_ratio,
                    'conflict_count': len(analysis.conflicts),
                    'overall_score': overall_score,
                    'recommendation': analysis.recommendation
                })
            
            df = pd.DataFrame(ranking_data)
            df = df.sort_values('overall_score', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating feature ranking: {e}")
            return pd.DataFrame()
    
    def get_conflict_summary(self, feature_analyses: Dict[str, FeatureAnalysis]) -> Dict:
        """Get summary of all conflicts"""
        try:
            all_conflicts = []
            for analysis in feature_analyses.values():
                all_conflicts.extend(analysis.conflicts)
            
            # Group by severity
            severity_counts = {}
            conflict_types = {}
            
            for conflict in all_conflicts:
                severity_counts[conflict.severity] = severity_counts.get(conflict.severity, 0) + 1
                conflict_types[conflict.conflict_type.value] = conflict_types.get(conflict.conflict_type.value, 0) + 1
            
            return {
                'total_conflicts': len(all_conflicts),
                'severity_counts': severity_counts,
                'conflict_types': conflict_types,
                'critical_conflicts': [c for c in all_conflicts if c.severity == 'critical'],
                'high_conflicts': [c for c in all_conflicts if c.severity == 'high']
            }
            
        except Exception as e:
            logger.error(f"Error creating conflict summary: {e}")
            return {}
    
    def generate_feature_report(self, X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """Generate comprehensive feature analysis report"""
        try:
            # Analyze features
            feature_analyses = self.analyze_features(X, y)
            
            # Check multicollinearity
            multicollinearity = self.check_multicollinearity(X)
            
            # Get feature ranking
            feature_ranking = self.get_feature_ranking(feature_analyses)
            
            # Get conflict summary
            conflict_summary = self.get_conflict_summary(feature_analyses)
            
            # Generate recommendations
            recommendations = self._generate_global_recommendations(
                feature_analyses, multicollinearity, conflict_summary
            )
            
            return {
                'feature_analyses': feature_analyses,
                'feature_ranking': feature_ranking,
                'multicollinearity': multicollinearity,
                'conflict_summary': conflict_summary,
                'recommendations': recommendations,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating feature report: {e}")
            return {}
    
    def _generate_global_recommendations(self, feature_analyses: Dict[str, FeatureAnalysis],
                                       multicollinearity: Dict[str, float],
                                       conflict_summary: Dict) -> List[str]:
        """Generate global recommendations"""
        try:
            recommendations = []
            
            # Check for critical conflicts
            if conflict_summary.get('critical_conflicts'):
                recommendations.append("Remove features with critical conflicts immediately")
            
            # Check for high VIF
            if multicollinearity:
                high_vif_features = list(multicollinearity.keys())
                recommendations.append(f"Address multicollinearity in: {', '.join(high_vif_features)}")
            
            # Check for low importance features
            low_importance_features = [
                f for f, analysis in feature_analyses.items()
                if analysis.importance_score < self.importance_threshold
            ]
            if low_importance_features:
                recommendations.append(f"Consider removing low importance features: {', '.join(low_importance_features)}")
            
            # Check for high missing values
            high_missing_features = [
                f for f, analysis in feature_analyses.items()
                if analysis.missing_ratio > 0.3
            ]
            if high_missing_features:
                recommendations.append(f"Address missing values in: {', '.join(high_missing_features)}")
            
            if not recommendations:
                recommendations.append("Feature set looks good - no major issues detected")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating global recommendations: {e}")
            return ["Error in recommendation generation"]

class FeatureManager:
    """Manages feature selection and conflict resolution"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.checker = FeatureConflictChecker(config)
        self.feature_history = []
        
        logger.info("Feature Manager initialized")
    
    def select_features(self, X: pd.DataFrame, y: pd.Series = None, 
                       max_features: int = None) -> List[str]:
        """Select best features based on analysis"""
        try:
            # Generate feature report
            report = self.checker.generate_feature_report(X, y)
            
            # Get feature ranking
            feature_ranking = report['feature_ranking']
            
            # Filter out problematic features
            good_features = []
            for _, row in feature_ranking.iterrows():
                feature = row['feature']
                
                # Skip if recommendation suggests removal
                if 'remove' in row['recommendation'].lower():
                    continue
                
                # Skip if too many conflicts
                if row['conflict_count'] > 3:
                    continue
                
                # Skip if low importance
                if row['importance_score'] < self.checker.importance_threshold:
                    continue
                
                good_features.append(feature)
            
            # Limit number of features if specified
            if max_features and len(good_features) > max_features:
                good_features = good_features[:max_features]
            
            # Store selection history
            self.feature_history.append({
                'selected_features': good_features,
                'total_features': len(X.columns),
                'timestamp': datetime.now()
            })
            
            return good_features
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return list(X.columns)
    
    def resolve_conflicts(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Resolve feature conflicts by removing problematic features"""
        try:
            # Generate feature report
            report = self.checker.generate_feature_report(X, y)
            
            # Get features to remove
            features_to_remove = set()
            
            # Remove features with critical conflicts
            for conflict in report['conflict_summary'].get('critical_conflicts', []):
                features_to_remove.add(conflict.feature2)
            
            # Remove features with high VIF
            features_to_remove.update(report['multicollinearity'].keys())
            
            # Remove low importance features
            for feature, analysis in report['feature_analyses'].items():
                if analysis.importance_score < self.checker.importance_threshold:
                    features_to_remove.add(feature)
            
            # Remove features with high missing values
            for feature, analysis in report['feature_analyses'].items():
                if analysis.missing_ratio > 0.5:
                    features_to_remove.add(feature)
            
            # Create cleaned dataset
            cleaned_features = [f for f in X.columns if f not in features_to_remove]
            X_cleaned = X[cleaned_features]
            
            logger.info(f"Removed {len(features_to_remove)} problematic features")
            logger.info(f"Remaining features: {len(cleaned_features)}")
            
            return X_cleaned
            
        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            return X
    
    def get_feature_importance_report(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Get feature importance report"""
        try:
            report = self.checker.generate_feature_report(X, y)
            return report['feature_ranking']
            
        except Exception as e:
            logger.error(f"Error getting feature importance report: {e}")
            return pd.DataFrame()
    
    def monitor_feature_drift(self, X_old: pd.DataFrame, X_new: pd.DataFrame) -> Dict:
        """Monitor feature drift between datasets"""
        try:
            drift_report = {}
            
            for feature in X_old.columns:
                if feature not in X_new.columns:
                    continue
                
                old_data = X_old[feature]
                new_data = X_new[feature]
                
                # Calculate drift metrics
                mean_drift = abs(new_data.mean() - old_data.mean())
                std_drift = abs(new_data.std() - old_data.std())
                
                # KS test for distribution drift
                try:
                    from scipy import stats
                    ks_stat, p_value = stats.ks_2samp(old_data.dropna(), new_data.dropna())
                except:
                    ks_stat, p_value = 0.0, 1.0
                
                drift_report[feature] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant_drift': p_value < 0.05
                }
            
            return drift_report
            
        except Exception as e:
            logger.error(f"Error monitoring feature drift: {e}")
            return {}
