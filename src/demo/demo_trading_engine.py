"""
Demo Trading Engine - AI-Controlled Practice Trading
Uses real Canadian market data with simulated order execution
"""

import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import random
import yfinance as yf
import sys
import os
from pathlib import Path

# Add src to path for AI ensemble
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ai.ai_ensemble import AIEnsemble

logger = logging.getLogger(__name__)


class DemoAccount:
    """Simulated trading account with real market prices"""
    
    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.pnl_history: List[Dict] = []
        
        logger.info(f" Demo account initialized with ${starting_capital:,.2f} CAD")
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total account value"""
        positions_value = sum(
            pos['quantity'] * current_prices.get(symbol, pos['avg_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of a position"""
        if symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        return pos['quantity'] * current_price
    
    def buy(self, symbol: str, quantity: int, price: float, strategy: str) -> bool:
        """Execute buy order"""
        cost = quantity * price
        
        if cost > self.cash:
            logger.warning(f" Insufficient cash for {symbol}: need ${cost:.2f}, have ${self.cash:.2f}")
            return False
        
        self.cash -= cost
        
        if symbol in self.positions:
            # Add to existing position
            pos = self.positions[symbol]
            total_qty = pos['quantity'] + quantity
            total_cost = (pos['avg_price'] * pos['quantity']) + (price * quantity)
            pos['quantity'] = total_qty
            pos['avg_price'] = total_cost / total_qty
        else:
            # New position
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'entry_time': datetime.now(),
                'strategy': strategy
            }
        
        # Log trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': 'BUY',
            'quantity': quantity,
            'price': price,
            'value': cost,
            'strategy': strategy
        }
        self.trade_history.append(trade)
        
        logger.info(f" BUY {quantity} {symbol} @ ${price:.2f} | Strategy: {strategy}")
        return True
    
    def sell(self, symbol: str, quantity: int, price: float, strategy: str) -> bool:
        """Execute sell order"""
        if symbol not in self.positions:
            logger.warning(f" No position in {symbol} to sell")
            return False
        
        pos = self.positions[symbol]
        if quantity > pos['quantity']:
            logger.warning(f" Cannot sell {quantity} {symbol}, only have {pos['quantity']}")
            return False
        
        proceeds = quantity * price
        self.cash += proceeds
        
        # Calculate P&L for this sale
        cost_basis = pos['avg_price'] * quantity
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100
        
        # Update position
        pos['quantity'] -= quantity
        if pos['quantity'] == 0:
            del self.positions[symbol]
        
        # Log trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': 'SELL',
            'quantity': quantity,
            'price': price,
            'value': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'strategy': strategy
        }
        self.trade_history.append(trade)
        
        pnl_symbol = "+" if pnl >= 0 else ""
        logger.info(f" SELL {quantity} {symbol} @ ${price:.2f} | P&L: {pnl_symbol}${pnl:.2f} ({pnl_symbol}{pnl_pct:.2f}%) | Strategy: {strategy}")
        return True
    
    def get_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get account summary"""
        total_value = self.get_total_value(current_prices)
        total_pnl = total_value - self.starting_capital
        total_return = (total_pnl / self.starting_capital) * 100
        
        # Calculate positions P&L
        positions_pnl = 0.0
        for symbol, pos in self.positions.items():
            current_price = current_prices.get(symbol, pos['avg_price'])
            current_value = pos['quantity'] * current_price
            cost_basis = pos['quantity'] * pos['avg_price']
            positions_pnl += (current_value - cost_basis)
        
        # Calculate realized P&L from closed trades
        realized_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl'))
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'unrealized_pnl': positions_pnl,
            'realized_pnl': realized_pnl,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }


class DemoTradingEngine:
    """Main demo trading engine with AI control"""
    
    def __init__(self, config_path: str = "config/demo_config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['demo_mode']
        
        # Initialize demo account
        starting_capital = self.config['starting_capital']['total']
        self.account = DemoAccount(starting_capital)
        
        # Market data
        self.symbols = (
            self.config['market_data']['symbols']['large_caps'] +
            self.config['market_data']['symbols']['mid_caps']
        )
        self.current_prices: Dict[str, float] = {}
        
        # Demo tracking
        self.demo_start = datetime.now()
        self.demo_end = self.demo_start + timedelta(days=self.config['duration_days'])
        self.is_running = False
        
        # Performance tracking
        self.daily_snapshots: List[Dict] = []
        
        # AI Ensemble
        try:
            self.ai_ensemble = AIEnsemble()
            self.ai_enabled = True
            logger.info(" AI Ensemble initialized (Grok + Kimi + Claude)")
        except Exception as e:
            logger.warning(f" AI Ensemble not available: {e}")
            self.ai_ensemble = None
            self.ai_enabled = False
        
        logger.info(" Demo Trading Engine initialized")
        logger.info(f" Demo period: {self.demo_start.strftime('%Y-%m-%d')} to {self.demo_end.strftime('%Y-%m-%d')}")
        logger.info(f" Starting capital: ${starting_capital:,.2f} CAD")
        logger.info(f" Tracking {len(self.symbols)} Canadian stocks")
    
    def fetch_real_prices(self) -> Dict[str, float]:
        """Fetch real-time prices from Yahoo Finance"""
        prices = {}
        
        try:
            # Fetch all symbols at once
            tickers = yf.Tickers(' '.join(self.symbols))
            
            for symbol in self.symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    
                    # Try to get current price
                    price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                    
                    if price and price > 0:
                        prices[symbol] = float(price)
                    else:
                        # Fallback: get last close
                        hist = ticker.history(period='1d')
                        if not hist.empty:
                            prices[symbol] = float(hist['Close'].iloc[-1])
                
                except Exception as e:
                    logger.warning(f" Could not fetch price for {symbol}: {e}")
            
            logger.info(f" Fetched prices for {len(prices)}/{len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f" Error fetching market data: {e}")
        
        return prices
    
    def generate_ai_signals(self) -> List[Dict]:
        """Generate AI trading signals using ensemble of Grok, Kimi, and Claude"""
        signals = []
        
        if not self.current_prices:
            return signals
        
        # Use AI ensemble if available
        if self.ai_enabled and self.ai_ensemble:
            try:
                # Get AI ensemble analysis
                ai_analysis = self.ai_ensemble.analyze_market(
                    self.current_prices, 
                    self.account.positions
                )
                
                ensemble_decision = ai_analysis['ensemble_decision']
                
                # Convert ensemble decision to trading signals
                if ensemble_decision['signal'] in ['buy', 'sell'] and ensemble_decision['confidence'] > 0.6:
                    # Find best symbol for the signal
                    best_symbol = self._find_best_symbol_for_signal(ensemble_decision['signal'])
                    
                    if best_symbol:
                        signals.append({
                            'symbol': best_symbol,
                            'action': ensemble_decision['signal'].upper(),
                            'strategy': 'AI Ensemble',
                            'confidence': ensemble_decision['confidence'],
                            'price': self.current_prices[best_symbol],
                            'reason': ensemble_decision['reason'],
                            'ai_consensus': ensemble_decision.get('consensus', 'unknown'),
                            'individual_ais': ensemble_decision.get('individual_signals', {})
                        })
                
                logger.info(f" AI Ensemble: {ensemble_decision['signal']} signal with {ensemble_decision['confidence']:.2f} confidence")
                
            except Exception as e:
                logger.error(f" AI ensemble analysis failed: {e}")
                # Fallback to simplified signals
                signals = self._generate_fallback_signals()
        else:
            # Fallback to simplified signals if AI not available
            signals = self._generate_fallback_signals()
        
        return signals
    
    def _find_best_symbol_for_signal(self, signal: str) -> Optional[str]:
        """Find the best symbol for a buy/sell signal"""
        if signal == 'buy':
            # Find symbol with best potential (simplified)
            available_symbols = [s for s in self.symbols if s in self.current_prices and self.account.cash > 1000]
            if available_symbols:
                return random.choice(available_symbols[:5])  # Top 5 for demo
        elif signal == 'sell':
            # Find symbol we have position in
            available_positions = [s for s in self.account.positions.keys() if s in self.current_prices]
            if available_positions:
                return random.choice(available_positions)
        return None
    
    def _generate_fallback_signals(self) -> List[Dict]:
        """Generate fallback signals when AI ensemble is not available"""
        signals = []
        
        # Simulate each strategy generating signals
        strategies = {
            'Momentum Scalping': 0.20,
            'News-Volatility': 0.20,
            'Gamma/OI Squeeze': 0.15,
            'Arbitrage': 0.15,
            'AI/ML Patterns': 0.30
        }
        
        for symbol in self.symbols[:10]:  # Focus on top 10 for demo
            if symbol not in self.current_prices:
                continue
            
            # Randomly generate signals (simplified fallback)
            if random.random() < 0.10:  # 10% chance per symbol (lower than AI)
                strategy = random.choice(list(strategies.keys()))
                action = random.choice(['BUY', 'SELL'])
                confidence = random.uniform(0.50, 0.80)  # Lower confidence for fallback
                
                # Only BUY if we have cash, only SELL if we have position
                if action == 'BUY' and self.account.cash > 1000:
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'strategy': strategy,
                        'confidence': confidence,
                        'price': self.current_prices[symbol],
                        'reason': f"{strategy} detected bullish opportunity (fallback mode)"
                    })
                elif action == 'SELL' and symbol in self.account.positions:
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'strategy': strategy,
                        'confidence': confidence,
                        'price': self.current_prices[symbol],
                        'reason': f"{strategy} detected sell signal (fallback mode)"
                    })
        
        return signals
    
    def execute_signal(self, signal: Dict) -> bool:
        """Execute a trading signal"""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        strategy = signal['strategy']
        
        # Apply slippage
        slippage = self.config['simulation']['slippage']
        if action == 'BUY':
            price *= (1 + slippage)
        else:
            price *= (1 - slippage)
        
        # Calculate position size
        if action == 'BUY':
            max_position_size = self.account.cash * self.config['risk_management']['max_position_size']
            quantity = int(max_position_size / price)
            
            if quantity > 0:
                return self.account.buy(symbol, quantity, price, strategy)
        
        elif action == 'SELL':
            if symbol in self.account.positions:
                quantity = self.account.positions[symbol]['quantity']
                return self.account.sell(symbol, quantity, price, strategy)
        
        return False
    
    def check_stop_loss_take_profit(self):
        """Check all positions for stop loss / take profit"""
        for symbol in list(self.account.positions.keys()):
            pos = self.account.positions[symbol]
            current_price = self.current_prices.get(symbol)
            
            if not current_price:
                continue
            
            # Calculate P&L %
            pnl_pct = ((current_price - pos['avg_price']) / pos['avg_price']) * 100
            
            # Stop loss
            stop_loss = -self.config['risk_management']['stop_loss_percent'] * 100
            if pnl_pct <= stop_loss:
                logger.warning(f" Stop loss triggered for {symbol}: {pnl_pct:.2f}%")
                self.account.sell(symbol, pos['quantity'], current_price, "Stop Loss")
                continue
            
            # Take profit
            take_profit = self.config['risk_management']['take_profit_percent'] * 100
            if pnl_pct >= take_profit:
                logger.info(f" Take profit triggered for {symbol}: {pnl_pct:.2f}%")
                self.account.sell(symbol, pos['quantity'], current_price, "Take Profit")
    
    def run_demo_cycle(self):
        """Run one demo trading cycle"""
        # 1. Fetch real market prices
        self.current_prices = self.fetch_real_prices()
        
        if not self.current_prices:
            logger.warning(" No market data available, skipping cycle")
            return
        
        # 2. Check stop loss / take profit
        self.check_stop_loss_take_profit()
        
        # 3. Generate AI signals
        signals = self.generate_ai_signals()
        
        if signals:
            logger.info(f" Generated {len(signals)} AI trading signals")
        
        # 4. Execute signals
        for signal in signals:
            self.execute_signal(signal)
        
        # 5. Log current status
        summary = self.account.get_summary(self.current_prices)
        
        pnl_symbol = "+" if summary['total_pnl'] >= 0 else ""
        logger.info(f" Portfolio: ${summary['total_value']:,.2f} | P&L: {pnl_symbol}${summary['total_pnl']:,.2f} ({pnl_symbol}{summary['total_return_pct']:.2f}%) | Positions: {summary['num_positions']}")
    
    def run_demo(self, cycles: int = None):
        """Run demo trading for specified cycles or until demo ends"""
        self.is_running = True
        cycle_count = 0
        
        logger.info("=" * 70)
        logger.info(" STARTING DEMO TRADING MODE")
        logger.info("=" * 70)
        
        try:
            while self.is_running:
                if datetime.now() >= self.demo_end:
                    logger.info("⏰ Demo period ended")
                    break
                
                if cycles and cycle_count >= cycles:
                    break
                
                self.run_demo_cycle()
                cycle_count += 1
                
                # Wait before next cycle
                time.sleep(self.config['market_data']['refresh_rate_seconds'])
        
        except KeyboardInterrupt:
            logger.info("⏸  Demo stopped by user")
        
        finally:
            self.is_running = False
            self.generate_final_report()
    
    def generate_final_report(self):
        """Generate end-of-demo report"""
        summary = self.account.get_summary(self.current_prices)
        
        logger.info("=" * 70)
        logger.info(" DEMO TRADING FINAL REPORT")
        logger.info("=" * 70)
        logger.info(f"Starting Capital: ${self.account.starting_capital:,.2f} CAD")
        logger.info(f"Final Value: ${summary['total_value']:,.2f} CAD")
        logger.info(f"Total P&L: ${summary['total_pnl']:+,.2f} CAD ({summary['total_return_pct']:+.2f}%)")
        logger.info(f"Realized P&L: ${summary['realized_pnl']:+,.2f} CAD")
        logger.info(f"Unrealized P&L: ${summary['unrealized_pnl']:+,.2f} CAD")
        logger.info(f"Total Trades: {summary['num_trades']}")
        logger.info(f"Open Positions: {summary['num_positions']}")
        logger.info("=" * 70)
        
        # Save report
        report_path = f"logs/demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write("DEMO TRADING FINAL REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Starting Capital: ${self.account.starting_capital:,.2f} CAD\n")
            f.write(f"Final Value: ${summary['total_value']:,.2f} CAD\n")
            f.write(f"Total P&L: ${summary['total_pnl']:+,.2f} CAD ({summary['total_return_pct']:+.2f}%)\n")
            f.write(f"Total Trades: {summary['num_trades']}\n")
            f.write("\nTRADE HISTORY:\n")
            for trade in self.account.trade_history:
                f.write(f"{trade['timestamp']} | {trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}\n")
        
        logger.info(f" Report saved to: {report_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Run demo
    engine = DemoTradingEngine()
    engine.run_demo(cycles=20)  # Run 20 cycles for testing

