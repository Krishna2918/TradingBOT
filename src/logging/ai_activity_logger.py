"""
AI Activity Logger
Comprehensive logging system to track all AI trading activities
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
from queue import Queue, Empty

class AIActivityLogger:
    """
    Dedicated logger for AI trading activities
    Tracks all AI decisions, signals, trades, and reasoning
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate log files for different activities
        self.ai_log_file = self.log_dir / "ai_activity.log"
        self.ai_json_file = self.log_dir / "ai_activity.json"
        self.trades_log_file = self.log_dir / "ai_trades.log"
        self.signals_log_file = self.log_dir / "ai_signals.log"
        self.decisions_log_file = self.log_dir / "ai_decisions.log"
        
        # Setup loggers
        self._setup_loggers()
        
        # Activity queue for async logging
        self.activity_queue = Queue()
        self.logging_thread = None
        self.running = False
        
        # Start background logging thread
        self.start_background_logging()
        
        # Activity counters
        self.activity_counters = {
            'signals_generated': 0,
            'trades_executed': 0,
            'decisions_made': 0,
            'errors_encountered': 0,
            'market_analysis': 0,
            'risk_assessments': 0
        }
        
        self.log_ai_startup()
    
    def _setup_loggers(self):
        """Setup dedicated loggers for different AI activities"""
        
        # Main AI activity logger
        self.ai_logger = logging.getLogger('ai_activity')
        self.ai_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.ai_logger.handlers.clear()
        
        # File handler for AI activity
        ai_handler = logging.FileHandler(self.ai_log_file, encoding='utf-8')
        ai_handler.setLevel(logging.INFO)
        
        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ai_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.ai_logger.addHandler(ai_handler)
        self.ai_logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.ai_logger.propagate = False
    
    def start_background_logging(self):
        """Start background thread for async logging"""
        if self.logging_thread is None or not self.logging_thread.is_alive():
            self.running = True
            self.logging_thread = threading.Thread(target=self._background_logger, daemon=True)
            self.logging_thread.start()
            self.ai_logger.info("AI Activity Logger: Background logging started")
    
    def stop_background_logging(self):
        """Stop background logging thread"""
        self.running = False
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=5)
        self.ai_logger.info("AI Activity Logger: Background logging stopped")
    
    def _background_logger(self):
        """Background thread for processing log queue"""
        while self.running:
            try:
                # Get activity from queue with timeout
                activity = self.activity_queue.get(timeout=1.0)
                self._process_activity(activity)
                self.activity_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"Error in background logger: {e}")
    
    def _process_activity(self, activity: Dict[str, Any]):
        """Process and log an activity"""
        try:
            activity_type = activity.get('type', 'unknown')
            timestamp = activity.get('timestamp', datetime.now().isoformat())
            
            # Log to appropriate file
            if activity_type == 'trade':
                self._log_trade(activity)
            elif activity_type == 'signal':
                self._log_signal(activity)
            elif activity_type == 'decision':
                self._log_decision(activity)
            else:
                self._log_general_activity(activity)
            
            # Log to JSON file for structured data
            self._log_to_json(activity)
            
            # Update counters
            if activity_type in self.activity_counters:
                self.activity_counters[activity_type] += 1
            
        except Exception as e:
            print(f"Error processing activity: {e}")
    
    def _log_trade(self, activity: Dict[str, Any]):
        """Log trade execution"""
        symbol = activity.get('symbol', 'UNKNOWN')
        action = activity.get('action', 'UNKNOWN')
        quantity = activity.get('quantity', 0)
        price = activity.get('price', 0.0)
        pnl = activity.get('pnl', 0.0)
        confidence = activity.get('confidence', 0.0)
        reasoning = activity.get('reasoning', [])
        
        trade_msg = f"TRADE EXECUTED | {symbol} | {action} {quantity} @ ${price:.2f} | P&L: ${pnl:.2f} | Confidence: {confidence:.2f}"
        self.ai_logger.info(trade_msg)
        
        # Detailed trade log
        with open(self.trades_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {trade_msg}\n")
            if reasoning:
                f.write(f"  Reasoning: {'; '.join(reasoning)}\n")
            f.write("\n")
    
    def _log_signal(self, activity: Dict[str, Any]):
        """Log signal generation"""
        symbol = activity.get('symbol', 'UNKNOWN')
        signal_type = activity.get('signal_type', 'UNKNOWN')
        confidence = activity.get('confidence', 0.0)
        score = activity.get('score', 0.0)
        sources = activity.get('sources', {})
        
        signal_msg = f"SIGNAL GENERATED | {symbol} | {signal_type} | Score: {score:.3f} | Confidence: {confidence:.2f}"
        self.ai_logger.info(signal_msg)
        
        # Detailed signal log
        with open(self.signals_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {signal_msg}\n")
            f.write(f"  Sources: {sources}\n")
            f.write("\n")
    
    def _log_decision(self, activity: Dict[str, Any]):
        """Log AI decision making"""
        decision_type = activity.get('decision_type', 'UNKNOWN')
        symbol = activity.get('symbol', 'UNKNOWN')
        decision = activity.get('decision', 'UNKNOWN')
        reasoning = activity.get('reasoning', [])
        risk_factors = activity.get('risk_factors', {})
        
        decision_msg = f"DECISION MADE | {decision_type} | {symbol} | {decision}"
        self.ai_logger.info(decision_msg)
        
        # Detailed decision log
        with open(self.decisions_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {decision_msg}\n")
            if reasoning:
                f.write(f"  Reasoning: {'; '.join(reasoning)}\n")
            if risk_factors:
                f.write(f"  Risk Factors: {risk_factors}\n")
            f.write("\n")
    
    def _log_general_activity(self, activity: Dict[str, Any]):
        """Log general AI activity"""
        activity_type = activity.get('type', 'unknown')
        message = activity.get('message', 'No message')
        details = activity.get('details', {})
        
        activity_msg = f"AI ACTIVITY | {activity_type.upper()} | {message}"
        self.ai_logger.info(activity_msg)
        
        if details:
            for key, value in details.items():
                self.ai_logger.info(f"  {key}: {value}")
    
    def _log_to_json(self, activity: Dict[str, Any]):
        """Log activity to JSON file for structured data"""
        try:
            with open(self.ai_json_file, 'a', encoding='utf-8') as f:
                json.dump(activity, f, ensure_ascii=False, indent=None)
                f.write('\n')
        except Exception as e:
            print(f"Error writing to JSON log: {e}")
    
    def log_ai_startup(self):
        """Log AI system startup"""
        startup_activity = {
            'type': 'startup',
            'timestamp': datetime.now().isoformat(),
            'message': 'AI Trading System Started',
            'details': {
                'log_files': {
                    'ai_activity': str(self.ai_log_file),
                    'ai_trades': str(self.trades_log_file),
                    'ai_signals': str(self.signals_log_file),
                    'ai_decisions': str(self.decisions_log_file),
                    'ai_json': str(self.ai_json_file)
                },
                'background_logging': self.running
            }
        }
        self.log_activity(startup_activity)
    
    def log_activity(self, activity: Dict[str, Any]):
        """Add activity to logging queue"""
        activity['timestamp'] = datetime.now().isoformat()
        self.activity_queue.put(activity)
    
    def log_signal_generation(self, symbol: str, signal_type: str, confidence: float, 
                            score: float, sources: Dict[str, float], reasoning: List[str] = None):
        """Log AI signal generation"""
        activity = {
            'type': 'signal',
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'score': score,
            'sources': sources,
            'reasoning': reasoning or []
        }
        self.log_activity(activity)
    
    def log_trade_execution(self, symbol: str, action: str, quantity: float, 
                          price: float, pnl: float, confidence: float, reasoning: List[str] = None):
        """Log trade execution"""
        activity = {
            'type': 'trade',
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'confidence': confidence,
            'reasoning': reasoning or []
        }
        self.log_activity(activity)
    
    def log_decision_making(self, decision_type: str, symbol: str, decision: str, 
                          reasoning: List[str], risk_factors: Dict[str, Any] = None):
        """Log AI decision making process"""
        activity = {
            'type': 'decision',
            'decision_type': decision_type,
            'symbol': symbol,
            'decision': decision,
            'reasoning': reasoning,
            'risk_factors': risk_factors or {}
        }
        self.log_activity(activity)
    
    def log_market_analysis(self, analysis_type: str, symbols: List[str], 
                          results: Dict[str, Any], confidence: float):
        """Log market analysis activities"""
        activity = {
            'type': 'market_analysis',
            'analysis_type': analysis_type,
            'symbols': symbols,
            'results': results,
            'confidence': confidence,
            'message': f"Market analysis: {analysis_type} for {len(symbols)} symbols"
        }
        self.log_activity(activity)
    
    def log_risk_assessment(self, symbol: str, risk_factors: Dict[str, float], 
                          risk_score: float, recommendation: str):
        """Log risk assessment activities"""
        activity = {
            'type': 'risk_assessment',
            'symbol': symbol,
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'recommendation': recommendation,
            'message': f"Risk assessment for {symbol}: {recommendation} (score: {risk_score:.2f})"
        }
        self.log_activity(activity)
    
    def log_error(self, error_type: str, error_message: str, symbol: str = None, 
                 details: Dict[str, Any] = None):
        """Log AI errors and exceptions"""
        activity = {
            'type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'symbol': symbol,
            'details': details or {},
            'message': f"AI Error: {error_type} - {error_message}"
        }
        self.log_activity(activity)
        self.activity_counters['errors_encountered'] += 1
    
    def log_performance_update(self, portfolio_value: float, pnl: float, 
                             win_rate: float, total_trades: int):
        """Log performance updates"""
        activity = {
            'type': 'performance_update',
            'portfolio_value': portfolio_value,
            'pnl': pnl,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'message': f"Performance: ${portfolio_value:.2f} | P&L: ${pnl:.2f} | Win Rate: {win_rate:.1f}% | Trades: {total_trades}"
        }
        self.log_activity(activity)
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """Get summary of AI activities"""
        return {
            'counters': self.activity_counters.copy(),
            'log_files': {
                'ai_activity': str(self.ai_log_file),
                'ai_trades': str(self.trades_log_file),
                'ai_signals': str(self.signals_log_file),
                'ai_decisions': str(self.decisions_log_file),
                'ai_json': str(self.ai_json_file)
            },
            'background_logging_active': self.running,
            'queue_size': self.activity_queue.qsize()
        }
    
    def get_recent_activities(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent activities from JSON log"""
        activities = []
        try:
            if self.ai_json_file.exists():
                with open(self.ai_json_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Get last N lines
                    recent_lines = lines[-count:] if len(lines) > count else lines
                    for line in recent_lines:
                        try:
                            activity = json.loads(line.strip())
                            activities.append(activity)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error reading recent activities: {e}")
        
        return activities

# Global instance
ai_logger = AIActivityLogger()

# Convenience functions
def log_ai_signal(symbol: str, signal_type: str, confidence: float, score: float, 
                 sources: Dict[str, float], reasoning: List[str] = None):
    """Log AI signal generation"""
    ai_logger.log_signal_generation(symbol, signal_type, confidence, score, sources, reasoning)

def log_ai_trade(symbol: str, action: str, quantity: float, price: float, 
                pnl: float, confidence: float, reasoning: List[str] = None):
    """Log AI trade execution"""
    ai_logger.log_trade_execution(symbol, action, quantity, price, pnl, confidence, reasoning)

def log_ai_decision(decision_type: str, symbol: str, decision: str, 
                   reasoning: List[str], risk_factors: Dict[str, Any] = None):
    """Log AI decision making"""
    ai_logger.log_decision_making(decision_type, symbol, decision, reasoning, risk_factors)

def log_ai_activity(activity_type: str, message: str, details: Dict[str, Any] = None):
    """Log general AI activity"""
    activity = {
        'type': activity_type,
        'message': message,
        'details': details or {}
    }
    ai_logger.log_activity(activity)

def get_ai_activity_summary() -> Dict[str, Any]:
    """Get AI activity summary"""
    return ai_logger.get_activity_summary()

def get_recent_ai_activities(count: int = 50) -> List[Dict[str, Any]]:
    """Get recent AI activities"""
    return ai_logger.get_recent_activities(count)
