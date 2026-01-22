"""
Helper methods for Hybrid Control Plane integration
"""

from typing import Dict, List
from datetime import datetime, timedelta
from src.ai.meta_ensemble_blender import PredictionSignal

def convert_analysis_to_predictions(analysis: Dict) -> List[PredictionSignal]:
    """Convert analysis to prediction signals for meta-ensemble blender"""
    predictions = []
    
    # Convert AI predictions
    if analysis.get('ai_predictions'):
        for symbol, pred in analysis['ai_predictions'].items():
            if pred.get('direction') == 'UP':
                predictions.append(PredictionSignal(
                    symbol=symbol,
                    direction=pred.get('confidence', 0.5),
                    confidence=pred.get('confidence', 0.5),
                    source='lstm',
                    timestamp=datetime.now().isoformat()
                ))
            elif pred.get('direction') == 'DOWN':
                predictions.append(PredictionSignal(
                    symbol=symbol,
                    direction=-pred.get('confidence', 0.5),
                    confidence=pred.get('confidence', 0.5),
                    source='lstm',
                    timestamp=datetime.now().isoformat()
                ))
    
    # Convert sentiment signals
    if analysis.get('sentiment'):
        for symbol, sent in analysis['sentiment'].items():
            if sent.get('compound', 0) != 0:
                predictions.append(PredictionSignal(
                    symbol=symbol,
                    direction=sent['compound'],
                    confidence=abs(sent['compound']),
                    source='sentiment',
                    timestamp=datetime.now().isoformat()
                ))
    
    return predictions

def calculate_daily_pnl(trades: List[Dict], initial_capital: float) -> float:
    """Calculate daily P&L percentage"""
    if not trades:
        return 0.0
    
    today = datetime.now().date()
    today_trades = [t for t in trades if t['timestamp'].date() == today]
    
    if not today_trades:
        return 0.0
    
    total_pnl = sum(t.get('pnl', 0) for t in today_trades)
    return total_pnl / initial_capital if initial_capital > 0 else 0.0

def calculate_max_drawdown_5d(trades: List[Dict], initial_capital: float) -> float:
    """Calculate 5-day maximum drawdown"""
    if len(trades) < 2:
        return 0.0
    
    # Get last 5 days of trades
    five_days_ago = datetime.now() - timedelta(days=5)
    recent_trades = [t for t in trades if t['timestamp'] >= five_days_ago]
    
    if not recent_trades:
        return 0.0
    
    # Calculate running P&L
    running_pnl = []
    cumulative_pnl = 0.0
    
    for trade in recent_trades:
        cumulative_pnl += trade.get('pnl', 0)
        running_pnl.append(cumulative_pnl)
    
    if not running_pnl:
        return 0.0
    
    # Calculate max drawdown
    peak = running_pnl[0]
    max_dd = 0.0
    
    for pnl in running_pnl:
        if pnl > peak:
            peak = pnl
        dd = (peak - pnl) / initial_capital if initial_capital > 0 else 0.0
        max_dd = max(max_dd, dd)
    
    return max_dd

def calculate_volatility_zscore(trades: List[Dict]) -> float:
    """Calculate volatility z-score"""
    if len(trades) < 10:
        return 0.0
    
    # Get recent trade P&L
    recent_trades = trades[-10:]
    pnl_values = [t.get('pnl', 0) for t in recent_trades]
    
    if not pnl_values:
        return 0.0
    
    # Calculate z-score
    mean_pnl = sum(pnl_values) / len(pnl_values)
    variance = sum((x - mean_pnl) ** 2 for x in pnl_values) / len(pnl_values)
    std_dev = variance ** 0.5
    
    if std_dev == 0:
        return 0.0
    
    current_pnl = pnl_values[-1]
    z_score = (current_pnl - mean_pnl) / std_dev
    
    return z_score

def calculate_ensemble_accuracy(trades: List[Dict]) -> float:
    """Calculate ensemble accuracy"""
    if not trades:
        return 0.5
    
    # Calculate accuracy based on recent trades
    recent_trades = trades[-20:] if len(trades) >= 20 else trades
    correct_predictions = 0
    
    for trade in recent_trades:
        if trade.get('pnl', 0) > 0:
            correct_predictions += 1
    
    return correct_predictions / len(recent_trades) if recent_trades else 0.5

def calculate_sharpe_ratio(trades: List[Dict], initial_capital: float) -> float:
    """Calculate Sharpe ratio"""
    if len(trades) < 10:
        return 0.0
    
    # Get recent trade returns
    recent_trades = trades[-20:] if len(trades) >= 20 else trades
    returns = [t.get('pnl', 0) / initial_capital for t in recent_trades if initial_capital > 0]
    
    if not returns:
        return 0.0
    
    # Calculate Sharpe ratio
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5
    
    if std_dev == 0:
        return 0.0
    
    sharpe = mean_return / std_dev
    return sharpe
