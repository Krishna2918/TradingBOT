"""
Feature Conflict Detection Module
Detects conflicts and redundancies between trading features using local LLM
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from enum import Enum

# Import local LLM integration
try:
    from src.ai.local_llm_integration import LocalLLMClient, LLMRequest, LLMResponse
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    logger.warning("Local LLM integration not available")

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of feature conflicts"""
    REDUNDANT = "redundant"
    CORRELATED = "correlated"
    CONTRADICTORY = "contradictory"
    OVERFITTING = "overfitting"
    LOGICAL_CONFLICT = "logical_conflict"

class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FeatureDefinition:
    """Definition of a trading feature"""
    name: str
    description: str
    category: str  # technical, fundamental, sentiment, etc.
    calculation_method: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    expected_range: Tuple[float, float]
    interpretation: str

@dataclass
class FeatureConflict:
    """Represents a conflict between features"""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    features_involved: List[str]
    description: str
    evidence: Dict[str, Any]
    recommendation: str
    confidence: float
    timestamp: datetime

class FeatureConflictDetector:
    """
    Detects conflicts and redundancies between trading features
    Uses local LLM for intelligent analysis and correlation detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features = {}
        self.conflicts = []
        self.correlation_matrix = None
        self.feature_importance = {}
        
        # Initialize local LLM client if available
        self.local_llm_client = None
        if LOCAL_LLM_AVAILABLE and config.get('use_local_llm', True):
            try:
                llm_config = config.get('local_llm_config', {})
                self.local_llm_client = LocalLLMClient(llm_config)
                logger.info("Local LLM client initialized for feature conflict detection")
            except Exception as e:
                logger.warning(f"Failed to initialize local LLM client: {e}")
                self.local_llm_client = None
        
        # Statistical thresholds
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.redundancy_threshold = config.get('redundancy_threshold', 0.9)
        self.contradiction_threshold = config.get('contradiction_threshold', -0.7)
        
        logger.info("Feature Conflict Detector initialized")
    
    def register_feature(self, feature: FeatureDefinition):
        """Register a new feature for conflict detection"""
        try:
            self.features[feature.name] = feature
            logger.debug(f"Registered feature: {feature.name}")
            
        except Exception as e:
            logger.error(f"Error registering feature {feature.name}: {e}")
    
    def register_features_batch(self, features: List[FeatureDefinition]):
        """Register multiple features at once"""
        try:
            for feature in features:
                self.register_feature(feature)
            
            logger.info(f"Registered {len(features)} features")
            
        except Exception as e:
            logger.error(f"Error registering features batch: {e}")
    
    def detect_conflicts(self, feature_data: Optional[Dict[str, np.ndarray]] = None) -> List[FeatureConflict]:
        """Detect conflicts between registered features"""
        try:
            conflicts = []
            
            # Statistical conflict detection
            if feature_data:
                statistical_conflicts = self._detect_statistical_conflicts(feature_data)
                conflicts.extend(statistical_conflicts)
            
            # LLM-based conflict detection
            if self.local_llm_client:
                llm_conflicts = self._detect_llm_conflicts()
                conflicts.extend(llm_conflicts)
            
            # Logical conflict detection
            logical_conflicts = self._detect_logical_conflicts()
            conflicts.extend(logical_conflicts)
            
            # Store conflicts
            self.conflicts.extend(conflicts)
            
            # Remove duplicates and sort by severity
            conflicts = self._deduplicate_conflicts(conflicts)
            conflicts.sort(key=lambda x: self._get_severity_score(x.severity), reverse=True)
            
            logger.info(f"Detected {len(conflicts)} feature conflicts")
            return conflicts
            
        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            return []
    
    def _detect_statistical_conflicts(self, feature_data: Dict[str, np.ndarray]) -> List[FeatureConflict]:
        """Detect conflicts using statistical analysis"""
        try:
            conflicts = []
            feature_names = list(feature_data.keys())
            
            if len(feature_names) < 2:
                return conflicts
            
            # Calculate correlation matrix
            data_matrix = np.column_stack([feature_data[name] for name in feature_names])
            correlation_matrix = np.corrcoef(data_matrix.T)
            
            # Store correlation matrix
            self.correlation_matrix = pd.DataFrame(
                correlation_matrix, 
                index=feature_names, 
                columns=feature_names
            )
            
            # Detect high correlations (redundancy)
            for i, feature1 in enumerate(feature_names):
                for j, feature2 in enumerate(feature_names[i+1:], i+1):
                    correlation = abs(correlation_matrix[i, j])
                    
                    if correlation > self.redundancy_threshold:
                        conflict = FeatureConflict(
                            conflict_id=f"redundant_{feature1}_{feature2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            conflict_type=ConflictType.REDUNDANT,
                            severity=ConflictSeverity.HIGH if correlation > 0.95 else ConflictSeverity.MEDIUM,
                            features_involved=[feature1, feature2],
                            description=f"Features {feature1} and {feature2} are highly correlated (r={correlation:.3f})",
                            evidence={
                                'correlation': correlation,
                                'threshold': self.redundancy_threshold,
                                'analysis_type': 'statistical'
                            },
                            recommendation=f"Consider removing one of {feature1} or {feature2} to reduce redundancy",
                            confidence=correlation,
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict)
                    
                    elif correlation > self.correlation_threshold:
                        conflict = FeatureConflict(
                            conflict_id=f"correlated_{feature1}_{feature2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            conflict_type=ConflictType.CORRELATED,
                            severity=ConflictSeverity.MEDIUM,
                            features_involved=[feature1, feature2],
                            description=f"Features {feature1} and {feature2} show significant correlation (r={correlation:.3f})",
                            evidence={
                                'correlation': correlation,
                                'threshold': self.correlation_threshold,
                                'analysis_type': 'statistical'
                            },
                            recommendation=f"Monitor correlation between {feature1} and {feature2} for potential redundancy",
                            confidence=correlation,
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict)
                    
                    elif correlation < self.contradiction_threshold:
                        conflict = FeatureConflict(
                            conflict_id=f"contradictory_{feature1}_{feature2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            conflict_type=ConflictType.CONTRADICTORY,
                            severity=ConflictSeverity.HIGH,
                            features_involved=[feature1, feature2],
                            description=f"Features {feature1} and {feature2} show strong negative correlation (r={correlation:.3f})",
                            evidence={
                                'correlation': correlation,
                                'threshold': self.contradiction_threshold,
                                'analysis_type': 'statistical'
                            },
                            recommendation=f"Investigate contradictory signals between {feature1} and {feature2}",
                            confidence=abs(correlation),
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error in statistical conflict detection: {e}")
            return []
    
    def _detect_llm_conflicts(self) -> List[FeatureConflict]:
        """Detect conflicts using local LLM analysis"""
        try:
            if not self.local_llm_client or not self.features:
                return []
            
            conflicts = []
            
            # Prepare feature descriptions for LLM
            feature_descriptions = []
            for name, feature in self.features.items():
                feature_descriptions.append({
                    'name': feature.name,
                    'description': feature.description,
                    'category': feature.category,
                    'calculation_method': feature.calculation_method,
                    'interpretation': feature.interpretation
                })
            
            # Create LLM request
            request = LLMRequest(
                prompt=self._build_conflict_detection_prompt(feature_descriptions),
                task_type="feature_conflict_detection",
                temperature=0.3,  # Lower temperature for more deterministic analysis
                max_tokens=1024,
                context={'features': feature_descriptions}
            )
            
            # Get LLM response
            response = self.local_llm_client._make_request(request)
            
            if response and response.content:
                # Parse LLM response for conflicts
                llm_conflicts = self._parse_llm_conflict_response(response.content)
                conflicts.extend(llm_conflicts)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error in LLM conflict detection: {e}")
            return []
    
    def _detect_logical_conflicts(self) -> List[FeatureConflict]:
        """Detect logical conflicts between features"""
        try:
            conflicts = []
            
            # Check for features with contradictory interpretations
            for name1, feature1 in self.features.items():
                for name2, feature2 in self.features.items():
                    if name1 >= name2:  # Avoid duplicates
                        continue
                    
                    # Check for contradictory categories
                    if self._are_contradictory_categories(feature1.category, feature2.category):
                        conflict = FeatureConflict(
                            conflict_id=f"logical_{name1}_{name2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            conflict_type=ConflictType.LOGICAL_CONFLICT,
                            severity=ConflictSeverity.MEDIUM,
                            features_involved=[name1, name2],
                            description=f"Features {name1} ({feature1.category}) and {name2} ({feature2.category}) may have contradictory interpretations",
                            evidence={
                                'category1': feature1.category,
                                'category2': feature2.category,
                                'analysis_type': 'logical'
                            },
                            recommendation=f"Review interpretation logic for {name1} and {name2}",
                            confidence=0.7,
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict)
                    
                    # Check for overlapping calculation methods
                    if self._have_overlapping_calculations(feature1, feature2):
                        conflict = FeatureConflict(
                            conflict_id=f"overlap_{name1}_{name2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            conflict_type=ConflictType.OVERFITTING,
                            severity=ConflictSeverity.LOW,
                            features_involved=[name1, name2],
                            description=f"Features {name1} and {name2} may have overlapping calculation methods",
                            evidence={
                                'method1': feature1.calculation_method,
                                'method2': feature2.calculation_method,
                                'analysis_type': 'logical'
                            },
                            recommendation=f"Review calculation methods for {name1} and {name2} to avoid overfitting",
                            confidence=0.6,
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error in logical conflict detection: {e}")
            return []
    
    def _build_conflict_detection_prompt(self, feature_descriptions: List[Dict[str, Any]]) -> str:
        """Build prompt for LLM conflict detection"""
        feature_list = []
        for i, feature in enumerate(feature_descriptions, 1):
            feature_list.append(f"""
{i}. {feature['name']}
   Category: {feature['category']}
   Description: {feature['description']}
   Calculation: {feature['calculation_method']}
   Interpretation: {feature['interpretation']}
""")
        
        return f"""
You are a quantitative analyst reviewing trading features for conflicts and redundancies. Analyze the following features:

Features:
{''.join(feature_list)}

Identify potential conflicts including:
1. Features that measure the same underlying market phenomenon
2. Features with contradictory logic or interpretation
3. Features that could lead to overfitting
4. Features with overlapping calculation methods
5. Features that might provide redundant signals

For each conflict identified, provide:
- Conflict type (redundant, correlated, contradictory, overfitting, logical_conflict)
- Severity (low, medium, high, critical)
- Features involved
- Brief description
- Evidence supporting the conflict
- Recommendation for resolution

Format your response as a structured analysis with specific examples and actionable recommendations.
"""
    
    def _parse_llm_conflict_response(self, response: str) -> List[FeatureConflict]:
        """Parse LLM response to extract conflicts"""
        try:
            conflicts = []
            
            # Simple parsing - in production, you might want more sophisticated parsing
            lines = response.split('\n')
            current_conflict = None
            
            for line in lines:
                line = line.strip()
                
                if 'conflict type:' in line.lower() or 'conflict:' in line.lower():
                    if current_conflict:
                        conflicts.append(current_conflict)
                    
                    # Extract conflict type and severity
                    conflict_type = ConflictType.CORRELATED  # Default
                    severity = ConflictSeverity.MEDIUM  # Default
                    
                    if 'redundant' in line.lower():
                        conflict_type = ConflictType.REDUNDANT
                    elif 'contradictory' in line.lower():
                        conflict_type = ConflictType.CONTRADICTORY
                    elif 'overfitting' in line.lower():
                        conflict_type = ConflictType.OVERFITTING
                    elif 'logical' in line.lower():
                        conflict_type = ConflictType.LOGICAL_CONFLICT
                    
                    if 'critical' in line.lower():
                        severity = ConflictSeverity.CRITICAL
                    elif 'high' in line.lower():
                        severity = ConflictSeverity.HIGH
                    elif 'low' in line.lower():
                        severity = ConflictSeverity.LOW
                    
                    current_conflict = FeatureConflict(
                        conflict_id=f"llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        conflict_type=conflict_type,
                        severity=severity,
                        features_involved=[],
                        description="",
                        evidence={'source': 'llm_analysis'},
                        recommendation="",
                        confidence=0.7,
                        timestamp=datetime.now()
                    )
                
                elif current_conflict and 'features:' in line.lower():
                    # Extract feature names
                    features_text = line.split(':')[1].strip()
                    current_conflict.features_involved = [f.strip() for f in features_text.split(',')]
                
                elif current_conflict and 'description:' in line.lower():
                    current_conflict.description = line.split(':')[1].strip()
                
                elif current_conflict and 'recommendation:' in line.lower():
                    current_conflict.recommendation = line.split(':')[1].strip()
            
            # Add the last conflict if exists
            if current_conflict:
                conflicts.append(current_conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error parsing LLM conflict response: {e}")
            return []
    
    def _are_contradictory_categories(self, category1: str, category2: str) -> bool:
        """Check if two feature categories are contradictory"""
        contradictory_pairs = [
            ('bullish', 'bearish'),
            ('momentum', 'mean_reversion'),
            ('trend_following', 'contrarian'),
            ('volatility_breakout', 'volatility_mean_reversion')
        ]
        
        for pair in contradictory_pairs:
            if (category1.lower() in pair[0] and category2.lower() in pair[1]) or \
               (category1.lower() in pair[1] and category2.lower() in pair[0]):
                return True
        
        return False
    
    def _have_overlapping_calculations(self, feature1: FeatureDefinition, feature2: FeatureDefinition) -> bool:
        """Check if two features have overlapping calculation methods"""
        method1 = feature1.calculation_method.lower()
        method2 = feature2.calculation_method.lower()
        
        # Check for common calculation patterns
        overlapping_patterns = [
            ('sma', 'moving_average'),
            ('ema', 'exponential'),
            ('rsi', 'relative_strength'),
            ('macd', 'moving_average_convergence'),
            ('bollinger', 'bands')
        ]
        
        for pattern1, pattern2 in overlapping_patterns:
            if (pattern1 in method1 and pattern2 in method2) or \
               (pattern1 in method2 and pattern2 in method1):
                return True
        
        return False
    
    def _deduplicate_conflicts(self, conflicts: List[FeatureConflict]) -> List[FeatureConflict]:
        """Remove duplicate conflicts"""
        try:
            unique_conflicts = []
            seen_pairs = set()
            
            for conflict in conflicts:
                # Create a key based on features involved
                features_key = tuple(sorted(conflict.features_involved))
                
                if features_key not in seen_pairs:
                    unique_conflicts.append(conflict)
                    seen_pairs.add(features_key)
            
            return unique_conflicts
            
        except Exception as e:
            logger.error(f"Error deduplicating conflicts: {e}")
            return conflicts
    
    def _get_severity_score(self, severity: ConflictSeverity) -> int:
        """Get numeric score for severity ranking"""
        severity_scores = {
            ConflictSeverity.LOW: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.HIGH: 3,
            ConflictSeverity.CRITICAL: 4
        }
        return severity_scores.get(severity, 0)
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get summary of detected conflicts"""
        try:
            if not self.conflicts:
                return {'total_conflicts': 0}
            
            # Count by type and severity
            type_counts = {}
            severity_counts = {}
            
            for conflict in self.conflicts:
                conflict_type = conflict.conflict_type.value
                severity = conflict.severity.value
                
                type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Get most problematic features
            feature_conflict_counts = {}
            for conflict in self.conflicts:
                for feature in conflict.features_involved:
                    feature_conflict_counts[feature] = feature_conflict_counts.get(feature, 0) + 1
            
            most_problematic = sorted(
                feature_conflict_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            return {
                'total_conflicts': len(self.conflicts),
                'type_counts': type_counts,
                'severity_counts': severity_counts,
                'most_problematic_features': most_problematic,
                'correlation_matrix_available': self.correlation_matrix is not None,
                'llm_available': self.local_llm_client is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting conflict summary: {e}")
            return {}
    
    def export_conflicts(self, filepath: str):
        """Export conflicts to file"""
        try:
            conflicts_data = []
            for conflict in self.conflicts:
                conflicts_data.append({
                    'conflict_id': conflict.conflict_id,
                    'conflict_type': conflict.conflict_type.value,
                    'severity': conflict.severity.value,
                    'features_involved': conflict.features_involved,
                    'description': conflict.description,
                    'evidence': conflict.evidence,
                    'recommendation': conflict.recommendation,
                    'confidence': conflict.confidence,
                    'timestamp': conflict.timestamp.isoformat()
                })
            
            with open(filepath, 'w') as f:
                json.dump(conflicts_data, f, indent=2)
            
            logger.info(f"Exported {len(conflicts_data)} conflicts to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting conflicts: {e}")
    
    def clear_conflicts(self):
        """Clear all detected conflicts"""
        self.conflicts.clear()
        logger.info("Cleared all conflicts")
    
    def get_feature_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations for feature optimization"""
        try:
            recommendations = {}
            
            # Analyze conflicts to generate recommendations
            for conflict in self.conflicts:
                if conflict.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]:
                    for feature in conflict.features_involved:
                        if feature not in recommendations:
                            recommendations[feature] = []
                        
                        recommendations[feature].append(conflict.recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting feature recommendations: {e}")
            return {}
