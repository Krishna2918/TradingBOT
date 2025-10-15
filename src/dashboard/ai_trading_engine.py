"""
Clean AI trading engine that updates state_manager directly
"""
import logging
from datetime import datetime
from typing import Optional
import asyncio

from .clean_state_manager import state_manager, Trade
from src.integration.master_orchestrator import MasterOrchestrator
from src.dashboard.services import get_random_tsx_stock, is_market_open

logger = logging.getLogger(__name__)

class AITradingEngine:
    def __init__(self):
        self.orchestrator: Optional[MasterOrchestrator] = None
        self.running = False
        
    def initialize(self):
        """Initialize AI components"""
        try:
            self.orchestrator = MasterOrchestrator()
            logger.info("✅ AI Trading Engine initialized")
            return True
        except Exception as e:
            logger.error(f"❌ AI Trading Engine initialization failed: {e}")
            return False
    
    async def execute_trading_cycle(self):
        """Execute one complete AI trading decision cycle"""
        if not self.orchestrator:
            logger.error("Orchestrator not initialized")
            return
        
        # Check market hours
        if not is_market_open():
            logger.info("⏸️ Market closed - analysis mode only")
            await self._run_market_analysis()
            return
        
        try:
            # Get current positions
            state = state_manager.get_current_state()
            current_positions = state.get('positions', [])
            
            # Decision 1: Check if we should SELL any existing positions
            for position in current_positions:
                symbol = position['symbol']
                market_data = await self._fetch_market_data(symbol)
                
                if market_data is None:
                    continue
                
                # Run AI decision pipeline for existing position
                decision = await self.orchestrator.run_decision_pipeline(market_data)
                
                if decision and decision.action == 'sell':
                    logger.info(f"AI Decision: SELL {symbol} (existing position)")
                    await self._execute_trade(decision, symbol)
                    return  # Execute one trade per cycle
            
            # Decision 2: Check if we should BUY a new position
            # Only if we have available capital and less than 5 positions
            if len(current_positions) < 5 and state['current_capital'] > 100:
                symbol = get_random_tsx_stock()
                
                # Make sure we don't already own this stock
                if any(p['symbol'] == symbol for p in current_positions):
                    logger.info(f"AI Decision: SKIP {symbol} (already owned)")
                    return
                
                market_data = await self._fetch_market_data(symbol)
                
                if market_data is None:
                    logger.warning(f"No market data for {symbol}")
                    return
                
                # Run AI decision pipeline for new position
                decision = await self.orchestrator.run_decision_pipeline(market_data)
                
                if decision and decision.action == 'buy':
                    logger.info(f"AI Decision: BUY {symbol} (new position)")
                    await self._execute_trade(decision, symbol)
                elif decision and decision.action == 'hold':
                    # For new positions, 'hold' means 'pass' (don't buy)
                    logger.info(f"AI Decision: PASS {symbol} (hold decision for new position)")
                else:
                    logger.info(f"AI Decision: PASS {symbol} (no decision or other action)")
            else:
                logger.info(f"AI Decision: WAIT (positions: {len(current_positions)}, capital: ${state['current_capital']:.0f})")
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
    
    async def _fetch_market_data(self, symbol: str):
        """Fetch real market data (use demo prices to avoid rate limiting)"""
        from src.dashboard.services import get_demo_price
        import yfinance as yf
        import pandas as pd
        
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d', interval='1d')
            
            if hist.empty:
                return None
            
            # Add current demo price
            current_price = get_demo_price(symbol)
            if current_price:
                # Append current price as latest data point
                latest_row = hist.iloc[-1].copy()
                latest_row['Close'] = current_price
                hist = pd.concat([hist, pd.DataFrame([latest_row], index=[datetime.now()])])
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def _execute_trade(self, decision, symbol: str):
        """Execute trade and update state"""
        from src.dashboard.services import get_demo_price
        
        # Get current price
        price = get_demo_price(symbol)
        if not price:
            logger.error(f"Cannot get price for {symbol}")
            return
        
        # Get current state
        state = state_manager.get_current_state()
        current_capital = state['current_capital']
        
        # Calculate position size (use decision.position_size if available)
        position_size = getattr(decision, 'position_size', 0.02)  # 2% default
        trade_value = current_capital * position_size
        quantity = int(trade_value / price)
        
        if quantity < 1:
            logger.warning(f"Position size too small: {quantity} shares")
            return
        
        # Adjust quantity if cost exceeds capital
        cost = quantity * price
        if cost > current_capital:
            quantity = int(current_capital / price)
            if quantity < 1:
                return
            cost = quantity * price
        
        # Create trade
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action=decision.action.upper(),
            quantity=quantity,
            price=price,
            reasoning=getattr(decision, 'reasoning', 'AI decision'),
            confidence=getattr(decision, 'confidence', 0.5),
            pnl=None
        )
        
        # Update state manager (SINGLE SOURCE OF TRUTH)
        state_manager.add_trade(trade)
        
        logger.info(f"✅ Executed: {trade.action} {quantity} {symbol} @ ${price:.2f}")
    
    async def _run_market_analysis(self):
        """Run analysis when market is closed (no trading)"""
        # Analyze a few random stocks for market research
        symbols = [get_random_tsx_stock() for _ in range(3)]
        
        for symbol in symbols:
            market_data = await self._fetch_market_data(symbol)
            
            if market_data is not None and self.orchestrator:
                try:
                    # Run pipeline for analysis only
                    decision = await self.orchestrator.run_decision_pipeline(market_data)
                    logger.info(f"📊 Market Analysis: {symbol} - {decision.action if decision else 'N/A'}")
                except Exception as e:
                    logger.error(f"Market analysis error for {symbol}: {e}")

# Global instance
ai_engine = AITradingEngine()
