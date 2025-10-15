"""
Comprehensive AI Decision Logger
Captures detailed information from the AI trading pipeline
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class AIDecisionLogger:
    """Comprehensive logger for AI trading decisions and pipeline data"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.decisions_log = self.log_dir / "ai_decisions.jsonl"
        self.pipeline_log = self.log_dir / "ai_pipeline.jsonl"
        self.performance_log = self.log_dir / "ai_performance.jsonl"
        self.market_analysis_log = self.log_dir / "market_analysis.jsonl"
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging for AI decisions"""
        handler = logging.FileHandler(self.log_dir / "ai_activity.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def log_decision(self, 
                    symbol: str,
                    action: str,
                    confidence: float,
                    reasoning: List[str],
                    model_consensus: Dict[str, Any],
                    risk_assessment: Dict[str, Any],
                    market_context: Dict[str, Any],
                    execution_details: Dict[str, Any],
                    pipeline_metrics: Dict[str, Any]):
        """Log a complete AI trading decision with full pipeline data"""
        
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "model_consensus": model_consensus,
            "risk_assessment": risk_assessment,
            "market_context": market_context,
            "execution_details": execution_details,
            "pipeline_metrics": pipeline_metrics
        }
        
        # Write to decisions log
        with open(self.decisions_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(decision_entry) + '\n')
        
        # Log to console
        logger.info(f"AI Decision: {action} {symbol} (Confidence: {confidence:.1%})")
        logger.info(f"Reasoning: {'; '.join(reasoning)}")
        
        return decision_entry
    
    def log_pipeline_step(self,
                         step_name: str,
                         input_data: Dict[str, Any],
                         output_data: Dict[str, Any],
                         processing_time: float,
                         success: bool,
                         error_message: Optional[str] = None):
        """Log individual pipeline step execution"""
        
        pipeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "input_data": input_data,
            "output_data": output_data,
            "processing_time_ms": processing_time * 1000,
            "success": success,
            "error_message": error_message
        }
        
        # Write to pipeline log
        with open(self.pipeline_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(pipeline_entry) + '\n')
        
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Pipeline Step: {step_name} - {status} ({processing_time:.3f}s)")
        
        if error_message:
            logger.error(f"Pipeline Error in {step_name}: {error_message}")
        
        return pipeline_entry
    
    def log_market_analysis(self,
                           symbol: str,
                           market_data: pd.DataFrame,
                           technical_indicators: Dict[str, float],
                           sentiment_analysis: Dict[str, Any],
                           regime_detection: Dict[str, Any],
                           volatility_metrics: Dict[str, float]):
        """Log comprehensive market analysis"""
        
        analysis_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "market_data_summary": {
                "price": float(market_data['Close'].iloc[-1]) if 'Close' in market_data.columns else None,
                "volume": int(market_data['Volume'].iloc[-1]) if 'Volume' in market_data.columns else None,
                "high": float(market_data['High'].iloc[-1]) if 'High' in market_data.columns else None,
                "low": float(market_data['Low'].iloc[-1]) if 'Low' in market_data.columns else None,
                "data_points": len(market_data)
            },
            "technical_indicators": technical_indicators,
            "sentiment_analysis": sentiment_analysis,
            "regime_detection": regime_detection,
            "volatility_metrics": volatility_metrics
        }
        
        # Write to market analysis log
        with open(self.market_analysis_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(analysis_entry) + '\n')
        
        logger.info(f"Market Analysis: {symbol} - Price: ${analysis_entry['market_data_summary']['price']:.2f}")
        
        return analysis_entry
    
    def log_performance_metrics(self,
                               symbol: str,
                               trade_result: Dict[str, Any],
                               model_performance: Dict[str, float],
                               risk_metrics: Dict[str, float],
                               learning_insights: Dict[str, Any]):
        """Log performance metrics and learning insights"""
        
        performance_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "trade_result": trade_result,
            "model_performance": model_performance,
            "risk_metrics": risk_metrics,
            "learning_insights": learning_insights
        }
        
        # Write to performance log
        with open(self.performance_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(performance_entry) + '\n')
        
        pnl = trade_result.get('pnl', 0)
        logger.info(f"Performance: {symbol} - P&L: ${pnl:.2f}")
        
        return performance_entry
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent AI decisions"""
        decisions = []
        try:
            with open(self.decisions_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    decisions.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass
        return decisions
    
    def clear_logs(self):
        """Clear all log files for new session"""
        try:
            # Clear all log files
            for log_file in [self.decisions_log, self.pipeline_log, self.performance_log, self.market_analysis_log]:
                if log_file.exists():
                    log_file.unlink()
            logger.info("AI logger files cleared for new session")
        except Exception as e:
            logger.error(f"Error clearing log files: {e}")
    
    def get_pipeline_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Get pipeline performance metrics for the last N hours"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        pipeline_stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "average_processing_time": 0.0,
            "step_breakdown": {}
        }
        
        try:
            with open(self.pipeline_log, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                    
                    if entry_time >= cutoff_time:
                        pipeline_stats["total_steps"] += 1
                        
                        if entry['success']:
                            pipeline_stats["successful_steps"] += 1
                        else:
                            pipeline_stats["failed_steps"] += 1
                        
                        step_name = entry['step_name']
                        if step_name not in pipeline_stats["step_breakdown"]:
                            pipeline_stats["step_breakdown"][step_name] = {
                                "count": 0,
                                "success_rate": 0.0,
                                "avg_time": 0.0
                            }
                        
                        step_stats = pipeline_stats["step_breakdown"][step_name]
                        step_stats["count"] += 1
                        step_stats["avg_time"] = (
                            (step_stats["avg_time"] * (step_stats["count"] - 1) + entry['processing_time_ms']) 
                            / step_stats["count"]
                        )
            
            # Calculate success rates
            for step_stats in pipeline_stats["step_breakdown"].values():
                step_stats["success_rate"] = (
                    pipeline_stats["successful_steps"] / pipeline_stats["total_steps"] * 100
                    if pipeline_stats["total_steps"] > 0 else 0
                )
            
            pipeline_stats["average_processing_time"] = (
                sum(step["avg_time"] for step in pipeline_stats["step_breakdown"].values()) 
                / len(pipeline_stats["step_breakdown"])
                if pipeline_stats["step_breakdown"] else 0
            )
            
        except FileNotFoundError:
            pass
        
        return pipeline_stats
    
    def get_ai_insights(self) -> Dict[str, Any]:
        """Get comprehensive AI insights and statistics"""
        insights = {
            "total_decisions": 0,
            "decision_breakdown": {"BUY": 0, "SELL": 0, "HOLD": 0},
            "average_confidence": 0.0,
            "top_performing_symbols": [],
            "recent_accuracy": 0.0,
            "pipeline_health": "Unknown"
        }
        
        try:
            # Analyze decisions
            with open(self.decisions_log, 'r', encoding='utf-8') as f:
                decisions = []
                for line in f:
                    decisions.append(json.loads(line.strip()))
                
                insights["total_decisions"] = len(decisions)
                
                if decisions:
                    # Decision breakdown
                    for decision in decisions:
                        action = decision.get('action', 'HOLD')
                        if action in insights["decision_breakdown"]:
                            insights["decision_breakdown"][action] += 1
                    
                    # Average confidence
                    confidences = [d.get('confidence', 0) for d in decisions]
                    insights["average_confidence"] = sum(confidences) / len(confidences)
            
            # Get pipeline performance
            pipeline_perf = self.get_pipeline_performance(24)
            if pipeline_perf["total_steps"] > 0:
                success_rate = pipeline_perf["successful_steps"] / pipeline_perf["total_steps"]
                if success_rate >= 0.95:
                    insights["pipeline_health"] = "Excellent"
                elif success_rate >= 0.85:
                    insights["pipeline_health"] = "Good"
                elif success_rate >= 0.70:
                    insights["pipeline_health"] = "Fair"
                else:
                    insights["pipeline_health"] = "Poor"
            
        except FileNotFoundError:
            pass
        
        return insights

# Global logger instance
ai_logger = AIDecisionLogger()

def log_ai_decision(symbol: str, action: str, confidence: float, reasoning: List[str], 
                   model_consensus: Dict[str, Any] = None, risk_assessment: Dict[str, Any] = None,
                   market_context: Dict[str, Any] = None, execution_details: Dict[str, Any] = None,
                   pipeline_metrics: Dict[str, Any] = None):
    """Convenience function to log AI decisions"""
    return ai_logger.log_decision(
        symbol=symbol,
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        model_consensus=model_consensus or {},
        risk_assessment=risk_assessment or {},
        market_context=market_context or {},
        execution_details=execution_details or {},
        pipeline_metrics=pipeline_metrics or {}
    )

def log_pipeline_step(step_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any],
                     processing_time: float, success: bool, error_message: str = None):
    """Convenience function to log pipeline steps"""
    return ai_logger.log_pipeline_step(
        step_name=step_name,
        input_data=input_data,
        output_data=output_data,
        processing_time=processing_time,
        success=success,
        error_message=error_message
    )
