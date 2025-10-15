"""
SLA Analytics System

This module implements a comprehensive SLA analytics system with
trend analysis, forecasting, insights generation, and predictive
analytics for enterprise-grade SLA monitoring and optimization.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import sqlite3
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction types."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    VOLATILE = "VOLATILE"

class ForecastAccuracy(Enum):
    """Forecast accuracy levels."""
    HIGH = "HIGH"      # > 90%
    MEDIUM = "MEDIUM"  # 70% - 90%
    LOW = "LOW"        # < 70%

class InsightType(Enum):
    """Insight types."""
    PERFORMANCE_TREND = "PERFORMANCE_TREND"
    CAPACITY_PLANNING = "CAPACITY_PLANNING"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    SEASONALITY = "SEASONALITY"
    CORRELATION = "CORRELATION"
    PREDICTION = "PREDICTION"

@dataclass
class SLATrend:
    """SLA trend definition."""
    trend_id: str
    sla_id: str
    metric_type: str
    trend_direction: TrendDirection
    trend_strength: float  # -1 to 1
    confidence: float      # 0 to 1
    start_date: datetime
    end_date: datetime
    data_points: int
    slope: float
    r_squared: float
    description: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SLAForecast:
    """SLA forecast definition."""
    forecast_id: str
    sla_id: str
    metric_type: str
    forecast_period: int  # days
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    accuracy: ForecastAccuracy
    model_type: str
    r_squared: float
    mse: float
    forecast_date: datetime
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SLAInsight:
    """SLA insight definition."""
    insight_id: str
    sla_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence: float
    impact: str
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

class SLAAnalytics:
    """
    Comprehensive SLA analytics system.
    
    Features:
    - Trend analysis and pattern recognition
    - Predictive forecasting
    - Anomaly detection
    - Seasonality analysis
    - Correlation analysis
    - Capacity planning insights
    - Performance optimization recommendations
    """
    
    def __init__(self, db_path: str = "data/sla_analytics.db"):
        """
        Initialize SLA analytics.
        
        Args:
            db_path: Path to SLA analytics database
        """
        self.db_path = db_path
        self.trends: List[SLATrend] = []
        self.forecasts: List[SLAForecast] = []
        self.insights: List[SLAInsight] = []
        
        # Analytics configuration
        self.analytics_config = {
            'trend_analysis_window_days': 30,
            'forecast_period_days': 7,
            'min_data_points_trend': 10,
            'min_data_points_forecast': 20,
            'confidence_threshold': 0.7,
            'anomaly_threshold': 3.0,  # z-score
            'seasonality_detection_window': 7  # days
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("SLA Analytics initialized")
    
    def _init_database(self) -> None:
        """Initialize SLA analytics database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create SLA trends table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_trends (
                trend_id TEXT PRIMARY KEY,
                sla_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                trend_direction TEXT NOT NULL,
                trend_strength REAL NOT NULL,
                confidence REAL NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                data_points INTEGER NOT NULL,
                slope REAL NOT NULL,
                r_squared REAL NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create SLA forecasts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_forecasts (
                forecast_id TEXT PRIMARY KEY,
                sla_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                forecast_period INTEGER NOT NULL,
                forecast_values TEXT NOT NULL,
                confidence_intervals TEXT NOT NULL,
                accuracy TEXT NOT NULL,
                model_type TEXT NOT NULL,
                r_squared REAL NOT NULL,
                mse REAL NOT NULL,
                forecast_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create SLA insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_insights (
                insight_id TEXT PRIMARY KEY,
                sla_id TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                confidence REAL NOT NULL,
                impact TEXT,
                recommendations TEXT,
                supporting_data TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def analyze_trends(self, sla_id: str, metric_type: str, 
                      start_date: datetime, end_date: datetime) -> Optional[SLATrend]:
        """
        Analyze SLA trends.
        
        Args:
            sla_id: SLA ID
            metric_type: Metric type
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            SLA trend analysis
        """
        try:
            # Get historical data
            historical_data = self._get_historical_data(sla_id, metric_type, start_date, end_date)
            
            if len(historical_data) < self.analytics_config['min_data_points_trend']:
                logger.warning(f"Insufficient data for trend analysis: {len(historical_data)} points")
                return None
            
            # Prepare data for analysis
            timestamps = [d['timestamp'] for d in historical_data]
            values = [d['value'] for d in historical_data]
            
            # Convert timestamps to numeric values
            time_numeric = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]  # hours
            
            # Perform linear regression
            X = np.array(time_numeric).reshape(-1, 1)
            y = np.array(values)
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate trend metrics
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            trend_strength = abs(slope) / np.std(y) if np.std(y) > 0 else 0
            
            # Determine trend direction
            if abs(slope) < 0.01:  # Very small slope
                trend_direction = TrendDirection.STABLE
            elif slope > 0:
                trend_direction = TrendDirection.IMPROVING
            else:
                trend_direction = TrendDirection.DEGRADING
            
            # Calculate confidence
            confidence = min(r_squared, 1.0)
            
            # Create trend object
            trend = SLATrend(
                trend_id=f"TREND_{sla_id}_{metric_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                sla_id=sla_id,
                metric_type=metric_type,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                confidence=confidence,
                start_date=start_date,
                end_date=end_date,
                data_points=len(historical_data),
                slope=slope,
                r_squared=r_squared,
                description=self._generate_trend_description(trend_direction, trend_strength, confidence)
            )
            
            self.trends.append(trend)
            self._store_sla_trend(trend)
            
            logger.info(f"Trend analysis completed: {trend.trend_id}")
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing trends for {sla_id}/{metric_type}: {e}")
            return None
    
    def generate_forecast(self, sla_id: str, metric_type: str, 
                         forecast_days: int = None) -> Optional[SLAForecast]:
        """
        Generate SLA forecast.
        
        Args:
            sla_id: SLA ID
            metric_type: Metric type
            forecast_days: Forecast period in days
            
        Returns:
            SLA forecast
        """
        try:
            if forecast_days is None:
                forecast_days = self.analytics_config['forecast_period_days']
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.analytics_config['trend_analysis_window_days'])
            historical_data = self._get_historical_data(sla_id, metric_type, start_date, end_date)
            
            if len(historical_data) < self.analytics_config['min_data_points_forecast']:
                logger.warning(f"Insufficient data for forecasting: {len(historical_data)} points")
                return None
            
            # Prepare data for forecasting
            timestamps = [d['timestamp'] for d in historical_data]
            values = [d['value'] for d in historical_data]
            
            # Convert timestamps to numeric values
            time_numeric = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]  # hours
            
            # Try different forecasting models
            forecast_results = []
            
            # Linear regression forecast
            linear_forecast = self._linear_forecast(time_numeric, values, forecast_days)
            if linear_forecast:
                forecast_results.append(('linear', linear_forecast))
            
            # Polynomial regression forecast
            poly_forecast = self._polynomial_forecast(time_numeric, values, forecast_days)
            if poly_forecast:
                forecast_results.append(('polynomial', poly_forecast))
            
            # Moving average forecast
            ma_forecast = self._moving_average_forecast(values, forecast_days)
            if ma_forecast:
                forecast_results.append(('moving_average', ma_forecast))
            
            if not forecast_results:
                logger.warning("No valid forecasts generated")
                return None
            
            # Select best forecast based on R-squared
            best_forecast = max(forecast_results, key=lambda x: x[1]['r_squared'])
            model_type, forecast_data = best_forecast
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                forecast_data['forecast_values'], forecast_data['mse']
            )
            
            # Determine forecast accuracy
            accuracy = self._determine_forecast_accuracy(forecast_data['r_squared'])
            
            # Create forecast object
            forecast = SLAForecast(
                forecast_id=f"FORECAST_{sla_id}_{metric_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                sla_id=sla_id,
                metric_type=metric_type,
                forecast_period=forecast_days,
                forecast_values=forecast_data['forecast_values'],
                confidence_intervals=confidence_intervals,
                accuracy=accuracy,
                model_type=model_type,
                r_squared=forecast_data['r_squared'],
                mse=forecast_data['mse'],
                forecast_date=datetime.now()
            )
            
            self.forecasts.append(forecast)
            self._store_sla_forecast(forecast)
            
            logger.info(f"Forecast generated: {forecast.forecast_id}")
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast for {sla_id}/{metric_type}: {e}")
            return None
    
    def _linear_forecast(self, time_numeric: List[float], values: List[float], 
                        forecast_days: int) -> Optional[Dict[str, Any]]:
        """Generate linear regression forecast."""
        try:
            X = np.array(time_numeric).reshape(-1, 1)
            y = np.array(values)
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate forecast
            last_time = time_numeric[-1]
            forecast_times = [last_time + (i + 1) * 24 for i in range(forecast_days)]  # 24 hours per day
            forecast_X = np.array(forecast_times).reshape(-1, 1)
            forecast_values = model.predict(forecast_X).tolist()
            
            # Calculate metrics
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r_squared = r2_score(y, y_pred)
            
            return {
                'forecast_values': forecast_values,
                'mse': mse,
                'r_squared': r_squared
            }
            
        except Exception as e:
            logger.error(f"Error in linear forecast: {e}")
            return None
    
    def _polynomial_forecast(self, time_numeric: List[float], values: List[float], 
                           forecast_days: int) -> Optional[Dict[str, Any]]:
        """Generate polynomial regression forecast."""
        try:
            X = np.array(time_numeric).reshape(-1, 1)
            y = np.array(values)
            
            # Use polynomial features
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Generate forecast
            last_time = time_numeric[-1]
            forecast_times = [last_time + (i + 1) * 24 for i in range(forecast_days)]
            forecast_X = np.array(forecast_times).reshape(-1, 1)
            forecast_X_poly = poly_features.transform(forecast_X)
            forecast_values = model.predict(forecast_X_poly).tolist()
            
            # Calculate metrics
            y_pred = model.predict(X_poly)
            mse = mean_squared_error(y, y_pred)
            r_squared = r2_score(y, y_pred)
            
            return {
                'forecast_values': forecast_values,
                'mse': mse,
                'r_squared': r_squared
            }
            
        except Exception as e:
            logger.error(f"Error in polynomial forecast: {e}")
            return None
    
    def _moving_average_forecast(self, values: List[float], forecast_days: int) -> Optional[Dict[str, Any]]:
        """Generate moving average forecast."""
        try:
            # Calculate moving average
            window_size = min(7, len(values) // 3)  # 7-day window or 1/3 of data
            if window_size < 2:
                return None
            
            ma_values = []
            for i in range(window_size, len(values)):
                ma_values.append(np.mean(values[i-window_size:i]))
            
            if not ma_values:
                return None
            
            # Simple trend calculation
            if len(ma_values) > 1:
                trend = (ma_values[-1] - ma_values[0]) / len(ma_values)
            else:
                trend = 0
            
            # Generate forecast
            forecast_values = []
            last_ma = ma_values[-1]
            for i in range(forecast_days):
                forecast_values.append(last_ma + trend * (i + 1))
            
            # Calculate metrics (simplified)
            mse = np.var(values)  # Use variance as proxy for MSE
            r_squared = 0.5  # Conservative estimate for moving average
            
            return {
                'forecast_values': forecast_values,
                'mse': mse,
                'r_squared': r_squared
            }
            
        except Exception as e:
            logger.error(f"Error in moving average forecast: {e}")
            return None
    
    def _calculate_confidence_intervals(self, forecast_values: List[float], mse: float) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for forecast."""
        confidence_intervals = []
        std_error = np.sqrt(mse)
        
        for value in forecast_values:
            # 95% confidence interval (1.96 * std_error)
            margin = 1.96 * std_error
            lower = value - margin
            upper = value + margin
            confidence_intervals.append((lower, upper))
        
        return confidence_intervals
    
    def _determine_forecast_accuracy(self, r_squared: float) -> ForecastAccuracy:
        """Determine forecast accuracy based on R-squared."""
        if r_squared >= 0.9:
            return ForecastAccuracy.HIGH
        elif r_squared >= 0.7:
            return ForecastAccuracy.MEDIUM
        else:
            return ForecastAccuracy.LOW
    
    def detect_anomalies(self, sla_id: str, metric_type: str, 
                        start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Detect anomalies in SLA data.
        
        Args:
            sla_id: SLA ID
            metric_type: Metric type
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            List of detected anomalies
        """
        try:
            # Get historical data
            historical_data = self._get_historical_data(sla_id, metric_type, start_date, end_date)
            
            if len(historical_data) < 10:
                logger.warning(f"Insufficient data for anomaly detection: {len(historical_data)} points")
                return []
            
            values = [d['value'] for d in historical_data]
            timestamps = [d['timestamp'] for d in historical_data]
            
            # Calculate z-scores
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            if std_value == 0:
                return []
            
            z_scores = [(v - mean_value) / std_value for v in values]
            
            # Detect anomalies
            anomalies = []
            threshold = self.analytics_config['anomaly_threshold']
            
            for i, z_score in enumerate(z_scores):
                if abs(z_score) > threshold:
                    anomaly = {
                        'timestamp': timestamps[i],
                        'value': values[i],
                        'z_score': z_score,
                        'severity': 'HIGH' if abs(z_score) > threshold * 2 else 'MEDIUM',
                        'description': f"Anomaly detected: {metric_type} = {values[i]:.2f} (z-score: {z_score:.2f})"
                    }
                    anomalies.append(anomaly)
            
            logger.info(f"Detected {len(anomalies)} anomalies for {sla_id}/{metric_type}")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {sla_id}/{metric_type}: {e}")
            return []
    
    def analyze_seasonality(self, sla_id: str, metric_type: str, 
                          start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Analyze seasonality patterns in SLA data.
        
        Args:
            sla_id: SLA ID
            metric_type: Metric type
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Seasonality analysis results
        """
        try:
            # Get historical data
            historical_data = self._get_historical_data(sla_id, metric_type, start_date, end_date)
            
            if len(historical_data) < 14:  # Need at least 2 weeks
                logger.warning(f"Insufficient data for seasonality analysis: {len(historical_data)} points")
                return {}
            
            # Group data by day of week and hour
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            
            # Calculate averages by day of week
            daily_patterns = df.groupby('day_of_week')['value'].agg(['mean', 'std']).to_dict()
            
            # Calculate averages by hour
            hourly_patterns = df.groupby('hour')['value'].agg(['mean', 'std']).to_dict()
            
            # Detect significant patterns
            daily_variation = np.std(list(daily_patterns['mean'].values()))
            hourly_variation = np.std(list(hourly_patterns['mean'].values()))
            
            # Determine if seasonality exists
            has_daily_seasonality = daily_variation > np.std(df['value']) * 0.1
            has_hourly_seasonality = hourly_variation > np.std(df['value']) * 0.1
            
            return {
                'has_daily_seasonality': has_daily_seasonality,
                'has_hourly_seasonality': has_hourly_seasonality,
                'daily_patterns': daily_patterns,
                'hourly_patterns': hourly_patterns,
                'daily_variation': daily_variation,
                'hourly_variation': hourly_variation,
                'overall_variation': np.std(df['value'])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality for {sla_id}/{metric_type}: {e}")
            return {}
    
    def generate_insights(self, sla_id: str, start_date: datetime, end_date: datetime) -> List[SLAInsight]:
        """
        Generate SLA insights.
        
        Args:
            sla_id: SLA ID
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            List of generated insights
        """
        insights = []
        
        try:
            # Get all metric types for this SLA
            metric_types = self._get_metric_types(sla_id, start_date, end_date)
            
            for metric_type in metric_types:
                # Analyze trends
                trend = self.analyze_trends(sla_id, metric_type, start_date, end_date)
                if trend and trend.confidence > self.analytics_config['confidence_threshold']:
                    insight = self._create_trend_insight(sla_id, metric_type, trend)
                    if insight:
                        insights.append(insight)
                
                # Detect anomalies
                anomalies = self.detect_anomalies(sla_id, metric_type, start_date, end_date)
                if anomalies:
                    insight = self._create_anomaly_insight(sla_id, metric_type, anomalies)
                    if insight:
                        insights.append(insight)
                
                # Analyze seasonality
                seasonality = self.analyze_seasonality(sla_id, metric_type, start_date, end_date)
                if seasonality.get('has_daily_seasonality') or seasonality.get('has_hourly_seasonality'):
                    insight = self._create_seasonality_insight(sla_id, metric_type, seasonality)
                    if insight:
                        insights.append(insight)
            
            # Generate capacity planning insights
            capacity_insight = self._create_capacity_planning_insight(sla_id, start_date, end_date)
            if capacity_insight:
                insights.append(capacity_insight)
            
            # Store insights
            for insight in insights:
                self.insights.append(insight)
                self._store_sla_insight(insight)
            
            logger.info(f"Generated {len(insights)} insights for SLA {sla_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights for SLA {sla_id}: {e}")
            return []
    
    def _create_trend_insight(self, sla_id: str, metric_type: str, trend: SLATrend) -> Optional[SLAInsight]:
        """Create trend insight."""
        if trend.confidence < self.analytics_config['confidence_threshold']:
            return None
        
        if trend.trend_direction == TrendDirection.DEGRADING:
            title = f"Performance Degradation Detected in {metric_type}"
            description = f"Analysis shows a degrading trend in {metric_type} with {trend.confidence:.1%} confidence."
            impact = "HIGH - Performance degradation may lead to SLA violations"
            recommendations = [
                "Investigate root cause of performance degradation",
                "Implement performance optimization measures",
                "Consider capacity scaling",
                "Monitor closely for SLA violations"
            ]
        elif trend.trend_direction == TrendDirection.IMPROVING:
            title = f"Performance Improvement Detected in {metric_type}"
            description = f"Analysis shows an improving trend in {metric_type} with {trend.confidence:.1%} confidence."
            impact = "POSITIVE - Performance improvements are beneficial"
            recommendations = [
                "Continue current optimization efforts",
                "Document successful changes",
                "Consider applying similar optimizations to other components"
            ]
        else:
            return None
        
        return SLAInsight(
            insight_id=f"INSIGHT_{sla_id}_{metric_type}_TREND_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sla_id=sla_id,
            insight_type=InsightType.PERFORMANCE_TREND,
            title=title,
            description=description,
            confidence=trend.confidence,
            impact=impact,
            recommendations=recommendations,
            supporting_data={
                'trend_direction': trend.trend_direction.value,
                'trend_strength': trend.trend_strength,
                'slope': trend.slope,
                'r_squared': trend.r_squared,
                'data_points': trend.data_points
            }
        )
    
    def _create_anomaly_insight(self, sla_id: str, metric_type: str, anomalies: List[Dict[str, Any]]) -> Optional[SLAInsight]:
        """Create anomaly insight."""
        if not anomalies:
            return None
        
        high_severity_anomalies = [a for a in anomalies if a['severity'] == 'HIGH']
        
        if high_severity_anomalies:
            title = f"Critical Anomalies Detected in {metric_type}"
            description = f"Detected {len(high_severity_anomalies)} critical anomalies in {metric_type}."
            impact = "CRITICAL - Anomalies may indicate system issues"
            recommendations = [
                "Investigate anomaly causes immediately",
                "Check system logs for errors",
                "Verify system health and performance",
                "Consider implementing additional monitoring"
            ]
        else:
            title = f"Anomalies Detected in {metric_type}"
            description = f"Detected {len(anomalies)} anomalies in {metric_type}."
            impact = "MEDIUM - Anomalies should be investigated"
            recommendations = [
                "Review anomaly patterns",
                "Check for correlation with other metrics",
                "Monitor for recurrence",
                "Consider threshold adjustments"
            ]
        
        return SLAInsight(
            insight_id=f"INSIGHT_{sla_id}_{metric_type}_ANOMALY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sla_id=sla_id,
            insight_type=InsightType.ANOMALY_DETECTION,
            title=title,
            description=description,
            confidence=0.8,  # High confidence for anomaly detection
            impact=impact,
            recommendations=recommendations,
            supporting_data={
                'anomaly_count': len(anomalies),
                'high_severity_count': len(high_severity_anomalies),
                'anomalies': anomalies[:5]  # Include first 5 anomalies
            }
        )
    
    def _create_seasonality_insight(self, sla_id: str, metric_type: str, seasonality: Dict[str, Any]) -> Optional[SLAInsight]:
        """Create seasonality insight."""
        title = f"Seasonality Patterns Detected in {metric_type}"
        description = f"Analysis reveals seasonal patterns in {metric_type}."
        
        if seasonality.get('has_daily_seasonality') and seasonality.get('has_hourly_seasonality'):
            description += " Both daily and hourly patterns are present."
            impact = "MEDIUM - Seasonal patterns affect capacity planning"
            recommendations = [
                "Implement time-based capacity scaling",
                "Adjust monitoring thresholds for different time periods",
                "Plan maintenance during low-activity periods",
                "Consider predictive scaling based on patterns"
            ]
        elif seasonality.get('has_daily_seasonality'):
            description += " Daily patterns are present."
            impact = "LOW - Daily patterns are normal"
            recommendations = [
                "Monitor daily patterns for changes",
                "Adjust capacity planning for daily variations",
                "Consider daily maintenance windows"
            ]
        else:
            description += " Hourly patterns are present."
            impact = "LOW - Hourly patterns are normal"
            recommendations = [
                "Monitor hourly patterns for changes",
                "Adjust capacity planning for hourly variations"
            ]
        
        return SLAInsight(
            insight_id=f"INSIGHT_{sla_id}_{metric_type}_SEASONALITY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sla_id=sla_id,
            insight_type=InsightType.SEASONALITY,
            title=title,
            description=description,
            confidence=0.7,  # Medium confidence for seasonality
            impact=impact,
            recommendations=recommendations,
            supporting_data={
                'has_daily_seasonality': seasonality.get('has_daily_seasonality', False),
                'has_hourly_seasonality': seasonality.get('has_hourly_seasonality', False),
                'daily_variation': seasonality.get('daily_variation', 0),
                'hourly_variation': seasonality.get('hourly_variation', 0)
            }
        )
    
    def _create_capacity_planning_insight(self, sla_id: str, start_date: datetime, end_date: datetime) -> Optional[SLAInsight]:
        """Create capacity planning insight."""
        try:
            # Get all metrics for capacity analysis
            all_metrics = self._get_all_metrics(sla_id, start_date, end_date)
            
            if not all_metrics:
                return None
            
            # Analyze capacity trends
            capacity_issues = []
            
            for metric_type, data in all_metrics.items():
                if metric_type in ['cpu_usage', 'memory_usage', 'disk_usage']:
                    recent_values = data[-7:]  # Last 7 data points
                    if recent_values:
                        avg_usage = np.mean(recent_values)
                        if avg_usage > 80:  # High usage threshold
                            capacity_issues.append({
                                'metric': metric_type,
                                'usage': avg_usage,
                                'trend': 'increasing' if len(data) > 1 and data[-1] > data[-2] else 'stable'
                            })
            
            if not capacity_issues:
                return None
            
            title = "Capacity Planning Recommendations"
            description = f"Analysis indicates potential capacity issues in {len(capacity_issues)} areas."
            impact = "HIGH - Capacity issues may lead to SLA violations"
            recommendations = [
                "Monitor capacity metrics closely",
                "Plan for capacity scaling",
                "Implement auto-scaling policies",
                "Review resource allocation"
            ]
            
            return SLAInsight(
                insight_id=f"INSIGHT_{sla_id}_CAPACITY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                sla_id=sla_id,
                insight_type=InsightType.CAPACITY_PLANNING,
                title=title,
                description=description,
                confidence=0.8,
                impact=impact,
                recommendations=recommendations,
                supporting_data={
                    'capacity_issues': capacity_issues,
                    'analysis_period': f"{start_date.isoformat()} to {end_date.isoformat()}"
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating capacity planning insight: {e}")
            return None
    
    def _get_historical_data(self, sla_id: str, metric_type: str, 
                           start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical data for analysis."""
        # This would typically query the SLA monitoring database
        # For now, return simulated data
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Simulate data with some trend and noise
            base_value = 50 + (current_date - start_date).days * 0.1  # Slight upward trend
            noise = np.random.normal(0, 5)  # Random noise
            value = max(0, base_value + noise)
            
            data.append({
                'timestamp': current_date,
                'value': value
            })
            
            current_date += timedelta(hours=1)  # Hourly data points
        
        return data
    
    def _get_metric_types(self, sla_id: str, start_date: datetime, end_date: datetime) -> List[str]:
        """Get available metric types for SLA."""
        # This would typically query the database
        return ['uptime', 'response_time', 'throughput', 'error_rate', 'availability']
    
    def _get_all_metrics(self, sla_id: str, start_date: datetime, end_date: datetime) -> Dict[str, List[float]]:
        """Get all metrics for SLA."""
        metrics = {}
        metric_types = self._get_metric_types(sla_id, start_date, end_date)
        
        for metric_type in metric_types:
            data = self._get_historical_data(sla_id, metric_type, start_date, end_date)
            metrics[metric_type] = [d['value'] for d in data]
        
        return metrics
    
    def _generate_trend_description(self, trend_direction: TrendDirection, 
                                  trend_strength: float, confidence: float) -> str:
        """Generate trend description."""
        direction_desc = {
            TrendDirection.IMPROVING: "improving",
            TrendDirection.STABLE: "stable",
            TrendDirection.DEGRADING: "degrading",
            TrendDirection.VOLATILE: "volatile"
        }
        
        strength_desc = "strong" if trend_strength > 0.5 else "moderate" if trend_strength > 0.2 else "weak"
        
        return f"{strength_desc.capitalize()} {direction_desc[trend_direction]} trend with {confidence:.1%} confidence"
    
    def _store_sla_trend(self, trend: SLATrend) -> None:
        """Store SLA trend in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sla_trends 
            (trend_id, sla_id, metric_type, trend_direction, trend_strength, confidence,
             start_date, end_date, data_points, slope, r_squared, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trend.trend_id, trend.sla_id, trend.metric_type, trend.trend_direction.value,
            trend.trend_strength, trend.confidence, trend.start_date.isoformat(),
            trend.end_date.isoformat(), trend.data_points, trend.slope, trend.r_squared,
            trend.description, trend.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_sla_forecast(self, forecast: SLAForecast) -> None:
        """Store SLA forecast in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sla_forecasts 
            (forecast_id, sla_id, metric_type, forecast_period, forecast_values,
             confidence_intervals, accuracy, model_type, r_squared, mse, forecast_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            forecast.forecast_id, forecast.sla_id, forecast.metric_type, forecast.forecast_period,
            json.dumps(forecast.forecast_values), json.dumps(forecast.confidence_intervals),
            forecast.accuracy.value, forecast.model_type, forecast.r_squared, forecast.mse,
            forecast.forecast_date.isoformat(), forecast.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_sla_insight(self, insight: SLAInsight) -> None:
        """Store SLA insight in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sla_insights 
            (insight_id, sla_id, insight_type, title, description, confidence,
             impact, recommendations, supporting_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight.insight_id, insight.sla_id, insight.insight_type.value, insight.title,
            insight.description, insight.confidence, insight.impact,
            json.dumps(insight.recommendations), json.dumps(insight.supporting_data),
            insight.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_analytics_summary(self, sla_id: str = None, days: int = 30) -> Dict[str, Any]:
        """
        Get analytics summary.
        
        Args:
            sla_id: Filter by SLA ID
            days: Time window in days
            
        Returns:
            Analytics summary dictionary
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter data
        trends = [t for t in self.trends if start_date <= t.created_at <= end_date]
        forecasts = [f for f in self.forecasts if start_date <= f.created_at <= end_date]
        insights = [i for i in self.insights if start_date <= i.created_at <= end_date]
        
        if sla_id:
            trends = [t for t in trends if t.sla_id == sla_id]
            forecasts = [f for f in forecasts if f.sla_id == sla_id]
            insights = [i for i in insights if i.sla_id == sla_id]
        
        # Calculate statistics
        trend_directions = {}
        for trend in trends:
            direction = trend.trend_direction.value
            trend_directions[direction] = trend_directions.get(direction, 0) + 1
        
        insight_types = {}
        for insight in insights:
            insight_type = insight.insight_type.value
            insight_types[insight_type] = insight_types.get(insight_type, 0) + 1
        
        forecast_accuracies = {}
        for forecast in forecasts:
            accuracy = forecast.accuracy.value
            forecast_accuracies[accuracy] = forecast_accuracies.get(accuracy, 0) + 1
        
        return {
            'summary': {
                'total_trends': len(trends),
                'total_forecasts': len(forecasts),
                'total_insights': len(insights),
                'analysis_period_days': days
            },
            'trend_breakdown': trend_directions,
            'insight_breakdown': insight_types,
            'forecast_accuracy_breakdown': forecast_accuracies,
            'recent_insights': [
                {
                    'insight_id': i.insight_id,
                    'title': i.title,
                    'type': i.insight_type.value,
                    'confidence': i.confidence,
                    'impact': i.impact,
                    'created_at': i.created_at.isoformat()
                }
                for i in insights[-10:]  # Last 10 insights
            ]
        }
