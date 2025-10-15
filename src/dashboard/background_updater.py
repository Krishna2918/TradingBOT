"""
Background Update System for Real-time Data
Handles price updates, portfolio tracking, and system monitoring
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

from .state_manager import trading_state, save_trading_state
from .services import update_holdings_prices, is_market_open, get_live_price

logger = logging.getLogger(__name__)

class BackgroundUpdater:
    """Background system for real-time updates"""
    
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval  # seconds
        self.running = False
        self.thread = None
        self.last_portfolio_update = None
        self.last_price_update = None
        
    def start(self):
        """Start the background update thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            logger.info("Background updater started")
    
    def stop(self):
        """Stop the background update thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Background updater stopped")
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                self._update_portfolio_history()
                self._update_holdings_prices()
                self._update_system_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Background update error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_portfolio_history(self):
        """Update portfolio value history"""
        try:
            current_capital = trading_state.get('current_capital', 0)
            holdings = trading_state.get('holdings', [])
            
            # Calculate total portfolio value
            holdings_value = 0
            if isinstance(holdings, list):
                for holding in holdings:
                    if holding.get('qty', 0) > 0:
                        holdings_value += holding.get('qty', 0) * holding.get('current_price', 0)
            
            total_portfolio_value = current_capital + holdings_value
            
            # Update portfolio history
            now = datetime.now()
            portfolio_history = trading_state.get('portfolio_history', [])
            
            # Add current value if it's been more than 5 minutes since last update
            if (not self.last_portfolio_update or 
                (now - self.last_portfolio_update).total_seconds() > 300):
                
                portfolio_history.append({
                    'timestamp': now.isoformat(),
                    'value': total_portfolio_value,
                    'cash': current_capital,
                    'holdings_value': holdings_value
                })
                
                # Keep only last 7 days
                cutoff = now - timedelta(days=7)
                portfolio_history = [
                    h for h in portfolio_history 
                    if datetime.fromisoformat(h['timestamp']) > cutoff
                ]
                
                trading_state['portfolio_history'] = portfolio_history
                self.last_portfolio_update = now
                
                logger.info(f"Portfolio updated: ${total_portfolio_value:,.2f}")
                
        except Exception as e:
            logger.error(f"Error updating portfolio history: {e}")
    
    def _update_holdings_prices(self):
        """Update holdings prices with real market data"""
        try:
            if is_market_open():
                # Only update prices when market is open
                update_holdings_prices()
                self.last_price_update = datetime.now()
                logger.info("Holdings prices updated")
            else:
                # When market is closed, update less frequently
                if (not self.last_price_update or 
                    (datetime.now() - self.last_price_update).total_seconds() > 3600):  # 1 hour
                    update_holdings_prices()
                    self.last_price_update = datetime.now()
                    logger.info("Holdings prices updated (market closed)")
                    
        except Exception as e:
            logger.error(f"Error updating holdings prices: {e}")
    
    def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # Update AI performance metrics
            ai_decisions_today = trading_state.get('ai_decisions_today', 0)
            trades = trading_state.get('trades', [])
            
            # Calculate performance metrics
            winning_trades = sum(1 for trade in trades if trade.get('pnl') is not None and trade.get('pnl', 0) > 0)
            total_trades = len([trade for trade in trades if trade.get('pnl') is not None])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Store metrics
            trading_state['performance_metrics'] = {
                'ai_decisions_today': ai_decisions_today,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'last_updated': datetime.now().isoformat()
            }
            
            # Update market status
            trading_state['market_status'] = {
                'is_open': is_market_open(),
                'last_checked': datetime.now().isoformat()
            }
            
            save_trading_state()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

# Global background updater instance
background_updater = BackgroundUpdater()

def start_background_updates():
    """Start background updates"""
    background_updater.start()

def stop_background_updates():
    """Stop background updates"""
    background_updater.stop()
