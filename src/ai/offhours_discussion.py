"""
Off-Hours Discussion System

This module implements the off-hours AI discussion system where models
analyze completed trades, discuss patterns, and generate improvements
during non-trading hours.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

from src.ai.multi_model import get_multi_model_manager, ModelOpinion, ModelRole
from src.ai.collaborative_discussion import get_collaborative_discussion
from src.trading.positions import get_position_manager, get_open_positions
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

class DiscussionTopic(Enum):
    """Off-hours discussion topic enumeration."""
    TRADE_ANALYSIS = "TRADE_ANALYSIS"
    SUCCESS_PATTERNS = "SUCCESS_PATTERNS"
    FAILURE_ANALYSIS = "FAILURE_ANALYSIS"
    STRATEGY_IMPROVEMENT = "STRATEGY_IMPROVEMENT"
    MARKET_STUDY = "MARKET_STUDY"
    RISK_REVIEW = "RISK_REVIEW"

@dataclass
class TradeInsight:
    """Insight from trade analysis."""
    trade_id: str
    symbol: str
    insight_type: str  # "SUCCESS", "FAILURE", "LEARNING"
    key_factors: List[str]
    lessons_learned: List[str]
    improvement_suggestions: List[str]
    confidence: float
    timestamp: datetime

@dataclass
class PatternAnalysis:
    """Analysis of trading patterns."""
    pattern_type: str  # "SUCCESS", "FAILURE", "MARKET_CONDITION"
    pattern_description: str
    frequency: int
    success_rate: float
    key_indicators: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: datetime

@dataclass
class ImprovementSuggestion:
    """Suggestion for system improvement."""
    category: str  # "STRATEGY", "RISK", "EXECUTION", "ANALYSIS"
    suggestion: str
    rationale: str
    expected_impact: str
    implementation_difficulty: str  # "LOW", "MEDIUM", "HIGH"
    priority: int  # 1-10
    timestamp: datetime

@dataclass
class OffHoursSession:
    """Off-hours discussion session."""
    session_id: str
    topic: DiscussionTopic
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    participants: List[str]
    insights: List[TradeInsight]
    patterns: List[PatternAnalysis]
    improvements: List[ImprovementSuggestion]
    discussion_summary: str
    action_items: List[str]

class OffHoursAI:
    """Manages off-hours AI discussions and analysis."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.multi_model_manager = get_multi_model_manager(mode)
        self.collaborative_discussion = get_collaborative_discussion(mode)
        self.position_manager = get_position_manager()
        
        # Off-hours sessions
        self.offhours_sessions = []
        
        # Analysis data
        self.trade_insights = []
        self.pattern_analyses = []
        self.improvement_suggestions = []
        
        logger.info(f"Off-Hours AI initialized for {mode} mode")
    
    async def conduct_offhours_analysis(self, topic: DiscussionTopic) -> OffHoursSession:
        """Conduct off-hours analysis session."""
        try:
            session_id = f"offhours_{topic.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_time = datetime.now()
            
            logger.info(f"Starting off-hours analysis: {topic.value}")
            
            # Initialize session
            session = OffHoursSession(
                session_id=session_id,
                topic=topic,
                start_time=start_time,
                end_time=None,
                duration_seconds=None,
                participants=[],
                insights=[],
                patterns=[],
                improvements=[],
                discussion_summary="",
                action_items=[]
            )
            
            # Conduct topic-specific analysis
            if topic == DiscussionTopic.TRADE_ANALYSIS:
                await self._analyze_completed_trades(session)
            elif topic == DiscussionTopic.SUCCESS_PATTERNS:
                await self._analyze_success_patterns(session)
            elif topic == DiscussionTopic.FAILURE_ANALYSIS:
                await self._analyze_failure_patterns(session)
            elif topic == DiscussionTopic.STRATEGY_IMPROVEMENT:
                await self._discuss_strategy_improvements(session)
            elif topic == DiscussionTopic.MARKET_STUDY:
                await self._conduct_market_study(session)
            elif topic == DiscussionTopic.RISK_REVIEW:
                await self._review_risk_management(session)
            
            # Complete session
            session.end_time = datetime.now()
            session.duration_seconds = (session.end_time - session.start_time).total_seconds()
            
            # Store session
            self.offhours_sessions.append(session)
            
            logger.info(f"Off-hours analysis completed in {session.duration_seconds:.2f} seconds")
            return session
            
        except Exception as e:
            logger.error(f"Error in off-hours analysis: {e}")
            return session
    
    async def _analyze_completed_trades(self, session: OffHoursSession):
        """Analyze completed trades for insights."""
        try:
            # Get recent completed trades
            # This would integrate with actual trade history in production
            completed_trades = self._get_recent_completed_trades()
            
            if not completed_trades:
                logger.info("No completed trades to analyze")
                return
            
            # Analyze each trade
            for trade in completed_trades:
                trade_analysis_prompt = f"""
Analyze this completed trade for insights:

Trade Details:
- Symbol: {trade.get('symbol', 'N/A')}
- Entry Price: {trade.get('entry_price', 'N/A')}
- Exit Price: {trade.get('exit_price', 'N/A')}
- P&L: {trade.get('pnl', 'N/A')}
- P&L %: {trade.get('pnl_percent', 'N/A')}
- Hold Time: {trade.get('hold_days', 'N/A')} days
- Exit Reason: {trade.get('exit_reason', 'N/A')}

Please provide:
1. Key factors that led to success/failure
2. Lessons learned
3. Improvement suggestions
4. Confidence in analysis (0.0-1.0)
"""
                
                # Get analysis from all models
                market_data = {"trade_data": trade}
                opinions = await self.multi_model_manager.get_all_model_opinions(
                    trade_analysis_prompt, market_data
                )
                
                # Create trade insight
                insight = self._create_trade_insight(trade, opinions)
                session.insights.append(insight)
                self.trade_insights.append(insight)
            
            logger.info(f"Analyzed {len(completed_trades)} completed trades")
            
        except Exception as e:
            logger.error(f"Error analyzing completed trades: {e}")
    
    async def _analyze_success_patterns(self, session: OffHoursSession):
        """Analyze patterns in successful trades."""
        try:
            # Get successful trades
            successful_trades = [insight for insight in self.trade_insights 
                               if insight.insight_type == "SUCCESS"]
            
            if not successful_trades:
                logger.info("No successful trades to analyze for patterns")
                return
            
            pattern_analysis_prompt = f"""
Analyze the following successful trades for common patterns:

Successful Trades: {len(successful_trades)}

Key factors from successful trades:
{self._extract_key_factors(successful_trades)}

Please identify:
1. Common patterns in successful trades
2. Key indicators that predict success
3. Recommendations for future trades
4. Confidence in pattern analysis (0.0-1.0)
"""
            
            # Get pattern analysis from all models
            market_data = {"successful_trades": len(successful_trades)}
            opinions = await self.multi_model_manager.get_all_model_opinions(
                pattern_analysis_prompt, market_data
            )
            
            # Create pattern analysis
            pattern = self._create_pattern_analysis("SUCCESS", opinions, successful_trades)
            session.patterns.append(pattern)
            self.pattern_analyses.append(pattern)
            
            logger.info("Analyzed success patterns")
            
        except Exception as e:
            logger.error(f"Error analyzing success patterns: {e}")
    
    async def _analyze_failure_patterns(self, session: OffHoursSession):
        """Analyze patterns in failed trades."""
        try:
            # Get failed trades
            failed_trades = [insight for insight in self.trade_insights 
                           if insight.insight_type == "FAILURE"]
            
            if not failed_trades:
                logger.info("No failed trades to analyze for patterns")
                return
            
            pattern_analysis_prompt = f"""
Analyze the following failed trades for common patterns:

Failed Trades: {len(failed_trades)}

Key factors from failed trades:
{self._extract_key_factors(failed_trades)}

Please identify:
1. Common patterns in failed trades
2. Warning signs that predict failure
3. Recommendations to avoid similar failures
4. Confidence in pattern analysis (0.0-1.0)
"""
            
            # Get pattern analysis from all models
            market_data = {"failed_trades": len(failed_trades)}
            opinions = await self.multi_model_manager.get_all_model_opinions(
                pattern_analysis_prompt, market_data
            )
            
            # Create pattern analysis
            pattern = self._create_pattern_analysis("FAILURE", opinions, failed_trades)
            session.patterns.append(pattern)
            self.pattern_analyses.append(pattern)
            
            logger.info("Analyzed failure patterns")
            
        except Exception as e:
            logger.error(f"Error analyzing failure patterns: {e}")
    
    async def _discuss_strategy_improvements(self, session: OffHoursSession):
        """Discuss strategy improvements."""
        try:
            # Collect all insights and patterns
            all_insights = self.trade_insights
            all_patterns = self.pattern_analyses
            
            improvement_prompt = f"""
Based on our trading analysis, suggest improvements to our trading strategy:

Trade Insights: {len(all_insights)}
Pattern Analyses: {len(all_patterns)}

Recent insights:
{self._summarize_insights(all_insights[-10:])}

Recent patterns:
{self._summarize_patterns(all_patterns[-5:])}

Please suggest improvements in these categories:
1. STRATEGY: Changes to trading approach
2. RISK: Risk management improvements
3. EXECUTION: Trade execution improvements
4. ANALYSIS: Analysis method improvements

For each suggestion, provide:
- Rationale
- Expected impact
- Implementation difficulty (LOW/MEDIUM/HIGH)
- Priority (1-10)
"""
            
            # Get improvement suggestions from all models
            market_data = {"insights_count": len(all_insights), "patterns_count": len(all_patterns)}
            opinions = await self.multi_model_manager.get_all_model_opinions(
                improvement_prompt, market_data
            )
            
            # Create improvement suggestions
            improvements = self._create_improvement_suggestions(opinions)
            session.improvements.extend(improvements)
            self.improvement_suggestions.extend(improvements)
            
            logger.info(f"Generated {len(improvements)} improvement suggestions")
            
        except Exception as e:
            logger.error(f"Error discussing strategy improvements: {e}")
    
    async def _conduct_market_study(self, session: OffHoursSession):
        """Conduct market study and analysis."""
        try:
            market_study_prompt = f"""
Conduct a comprehensive market study based on our trading history:

Trading History:
- Total trades analyzed: {len(self.trade_insights)}
- Success patterns: {len([p for p in self.pattern_analyses if p.pattern_type == 'SUCCESS'])}
- Failure patterns: {len([p for p in self.pattern_analyses if p.pattern_type == 'FAILURE'])}

Please analyze:
1. Market conditions that favor our strategy
2. Market conditions that hurt our strategy
3. Seasonal or cyclical patterns
4. Volatility impact on performance
5. Recommendations for market adaptation
"""
            
            # Get market study from all models
            market_data = {"trading_history": len(self.trade_insights)}
            opinions = await self.multi_model_manager.get_all_model_opinions(
                market_study_prompt, market_data
            )
            
            # Create market study insights
            market_insights = self._create_market_insights(opinions)
            session.insights.extend(market_insights)
            
            logger.info("Conducted market study")
            
        except Exception as e:
            logger.error(f"Error conducting market study: {e}")
    
    async def _review_risk_management(self, session: OffHoursSession):
        """Review risk management practices."""
        try:
            risk_review_prompt = f"""
Review our risk management practices based on trading history:

Risk Analysis Data:
- Total trades: {len(self.trade_insights)}
- High-risk trades: {len([i for i in self.trade_insights if 'risk' in i.key_factors])}
- Risk-related failures: {len([i for i in self.trade_insights if i.insight_type == 'FAILURE' and 'risk' in i.key_factors])}

Please review:
1. Risk management effectiveness
2. Areas where risk was not properly managed
3. Suggestions for improving risk management
4. Risk tolerance adjustments needed
5. Position sizing improvements
"""
            
            # Get risk review from all models
            market_data = {"risk_data": len(self.trade_insights)}
            opinions = await self.multi_model_manager.get_all_model_opinions(
                risk_review_prompt, market_data
            )
            
            # Create risk management improvements
            risk_improvements = self._create_risk_improvements(opinions)
            session.improvements.extend(risk_improvements)
            
            logger.info("Reviewed risk management")
            
        except Exception as e:
            logger.error(f"Error reviewing risk management: {e}")
    
    def _get_recent_completed_trades(self) -> List[Dict[str, Any]]:
        """Get recent completed trades for analysis."""
        try:
            # This would integrate with actual trade history in production
            # For now, return simulated data
            return [
                {
                    "symbol": "AAPL",
                    "entry_price": 150.0,
                    "exit_price": 155.0,
                    "pnl": 500.0,
                    "pnl_percent": 3.33,
                    "hold_days": 5,
                    "exit_reason": "TAKE_PROFIT"
                },
                {
                    "symbol": "MSFT",
                    "entry_price": 300.0,
                    "exit_price": 285.0,
                    "pnl": -150.0,
                    "pnl_percent": -5.0,
                    "hold_days": 3,
                    "exit_reason": "STOP_LOSS"
                }
            ]
        except Exception as e:
            logger.error(f"Error getting completed trades: {e}")
            return []
    
    def _create_trade_insight(self, trade: Dict[str, Any], opinions: List[ModelOpinion]) -> TradeInsight:
        """Create trade insight from model opinions."""
        try:
            # Determine insight type based on P&L
            pnl_percent = trade.get('pnl_percent', 0)
            if pnl_percent > 2:
                insight_type = "SUCCESS"
            elif pnl_percent < -2:
                insight_type = "FAILURE"
            else:
                insight_type = "LEARNING"
            
            # Combine insights from all models
            all_factors = []
            all_lessons = []
            all_suggestions = []
            confidences = []
            
            for opinion in opinions:
                all_factors.extend(opinion.supporting_evidence)
                all_lessons.extend(opinion.concerns)  # Concerns become lessons
                all_suggestions.append(opinion.recommendation)
                confidences.append(opinion.confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            return TradeInsight(
                trade_id=f"trade_{trade.get('symbol', 'unknown')}_{datetime.now().strftime('%Y%m%d')}",
                symbol=trade.get('symbol', 'unknown'),
                insight_type=insight_type,
                key_factors=list(set(all_factors))[:5],  # Top 5 unique factors
                lessons_learned=list(set(all_lessons))[:5],  # Top 5 unique lessons
                improvement_suggestions=list(set(all_suggestions))[:3],  # Top 3 suggestions
                confidence=avg_confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating trade insight: {e}")
            return TradeInsight(
                trade_id="error",
                symbol="unknown",
                insight_type="LEARNING",
                key_factors=[],
                lessons_learned=[],
                improvement_suggestions=[],
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def _create_pattern_analysis(self, pattern_type: str, opinions: List[ModelOpinion],
                               trades: List[TradeInsight]) -> PatternAnalysis:
        """Create pattern analysis from model opinions."""
        try:
            # Combine insights from all models
            all_indicators = []
            all_recommendations = []
            confidences = []
            
            for opinion in opinions:
                all_indicators.extend(opinion.supporting_evidence)
                all_recommendations.append(opinion.recommendation)
                confidences.append(opinion.confidence)
            
            # Calculate metrics
            frequency = len(trades)
            success_rate = len([t for t in trades if t.insight_type == "SUCCESS"]) / frequency if frequency > 0 else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            return PatternAnalysis(
                pattern_type=pattern_type,
                pattern_description=f"{pattern_type} pattern in {frequency} trades",
                frequency=frequency,
                success_rate=success_rate,
                key_indicators=list(set(all_indicators))[:5],
                recommendations=list(set(all_recommendations))[:3],
                confidence=avg_confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating pattern analysis: {e}")
            return PatternAnalysis(
                pattern_type=pattern_type,
                pattern_description="Error in analysis",
                frequency=0,
                success_rate=0.0,
                key_indicators=[],
                recommendations=[],
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def _create_improvement_suggestions(self, opinions: List[ModelOpinion]) -> List[ImprovementSuggestion]:
        """Create improvement suggestions from model opinions."""
        suggestions = []
        
        try:
            for opinion in opinions:
                # Parse improvement suggestions from opinion
                suggestion = ImprovementSuggestion(
                    category="STRATEGY",  # Default category
                    suggestion=opinion.recommendation,
                    rationale=opinion.opinion,
                    expected_impact="MEDIUM",  # Default impact
                    implementation_difficulty="MEDIUM",  # Default difficulty
                    priority=5,  # Default priority
                    timestamp=datetime.now()
                )
                suggestions.append(suggestion)
            
        except Exception as e:
            logger.error(f"Error creating improvement suggestions: {e}")
        
        return suggestions
    
    def _create_market_insights(self, opinions: List[ModelOpinion]) -> List[TradeInsight]:
        """Create market insights from model opinions."""
        insights = []
        
        try:
            for opinion in opinions:
                insight = TradeInsight(
                    trade_id=f"market_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol="MARKET",
                    insight_type="LEARNING",
                    key_factors=opinion.supporting_evidence,
                    lessons_learned=opinion.concerns,
                    improvement_suggestions=[opinion.recommendation],
                    confidence=opinion.confidence,
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error creating market insights: {e}")
        
        return insights
    
    def _create_risk_improvements(self, opinions: List[ModelOpinion]) -> List[ImprovementSuggestion]:
        """Create risk management improvements from model opinions."""
        improvements = []
        
        try:
            for opinion in opinions:
                improvement = ImprovementSuggestion(
                    category="RISK",
                    suggestion=opinion.recommendation,
                    rationale=opinion.opinion,
                    expected_impact="HIGH",
                    implementation_difficulty="MEDIUM",
                    priority=8,  # High priority for risk improvements
                    timestamp=datetime.now()
                )
                improvements.append(improvement)
            
        except Exception as e:
            logger.error(f"Error creating risk improvements: {e}")
        
        return improvements
    
    def _extract_key_factors(self, trades: List[TradeInsight]) -> str:
        """Extract key factors from trades."""
        all_factors = []
        for trade in trades:
            all_factors.extend(trade.key_factors)
        
        # Count frequency of factors
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Get top factors
        top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return ", ".join([f"{factor} ({count})" for factor, count in top_factors])
    
    def _summarize_insights(self, insights: List[TradeInsight]) -> str:
        """Summarize recent insights."""
        if not insights:
            return "No recent insights"
        
        summary_parts = []
        for insight in insights[-3:]:  # Last 3 insights
            summary_parts.append(f"{insight.symbol}: {insight.insight_type} - {', '.join(insight.key_factors[:2])}")
        
        return "; ".join(summary_parts)
    
    def _summarize_patterns(self, patterns: List[PatternAnalysis]) -> str:
        """Summarize recent patterns."""
        if not patterns:
            return "No recent patterns"
        
        summary_parts = []
        for pattern in patterns[-2:]:  # Last 2 patterns
            summary_parts.append(f"{pattern.pattern_type}: {pattern.pattern_description}")
        
        return "; ".join(summary_parts)
    
    def get_offhours_history(self, limit: int = 10) -> List[OffHoursSession]:
        """Get recent off-hours sessions."""
        return self.offhours_sessions[-limit:] if self.offhours_sessions else []
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of all insights."""
        return {
            "total_insights": len(self.trade_insights),
            "success_insights": len([i for i in self.trade_insights if i.insight_type == "SUCCESS"]),
            "failure_insights": len([i for i in self.trade_insights if i.insight_type == "FAILURE"]),
            "learning_insights": len([i for i in self.trade_insights if i.insight_type == "LEARNING"]),
            "total_patterns": len(self.pattern_analyses),
            "total_improvements": len(self.improvement_suggestions),
            "recent_insights": self.trade_insights[-5:] if self.trade_insights else []
        }
    
    def validate_offhours_system(self) -> bool:
        """Validate off-hours system."""
        try:
            # Check if multi-model manager is available
            if not self.multi_model_manager:
                logger.error("Multi-model manager not available")
                return False
            
            # Check if collaborative discussion is available
            if not self.collaborative_discussion:
                logger.error("Collaborative discussion not available")
                return False
            
            logger.info("Off-hours system validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Off-hours system validation error: {e}")
            return False

# Global off-hours AI instance
_offhours_ai: Optional[OffHoursAI] = None

def get_offhours_ai(mode: str = None) -> OffHoursAI:
    """Get the global off-hours AI instance."""
    global _offhours_ai
    if _offhours_ai is None:
        if mode is None:
            mode = get_current_mode()
        _offhours_ai = OffHoursAI(mode)
    return _offhours_ai

async def conduct_offhours_analysis(topic: DiscussionTopic, mode: str = None) -> OffHoursSession:
    """Conduct off-hours analysis."""
    return await get_offhours_ai(mode).conduct_offhours_analysis(topic)

def get_offhours_history(limit: int = 10, mode: str = None) -> List[OffHoursSession]:
    """Get off-hours history."""
    return get_offhours_ai(mode).get_offhours_history(limit)

def get_insights_summary(mode: str = None) -> Dict[str, Any]:
    """Get insights summary."""
    return get_offhours_ai(mode).get_insights_summary()

def validate_offhours_system(mode: str = None) -> bool:
    """Validate off-hours system."""
    return get_offhours_ai(mode).validate_offhours_system()
