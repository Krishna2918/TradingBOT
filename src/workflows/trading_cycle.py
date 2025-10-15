"""
Complete Trading Workflow - BUY→HOLD→SELL Cycle

This module implements the complete trading cycle that orchestrates
the entire BUY→HOLD→SELL workflow with real-time monitoring and execution.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.trading.positions import Position, get_position_manager, track_position, get_open_positions, close_position
from src.trading.execution import execute_buy_order, execute_sell_order, OrderType, ExecutionResult
from src.trading.risk import calculate_position_size, check_portfolio_limits
from src.trading.exit_strategy import generate_sell_signals, SellSignal
from src.ai.enhanced_ensemble import analyze_for_entry, analyze_for_exit, get_enhanced_ensemble
from src.config.mode_manager import get_current_mode
from src.monitoring.performance_analytics import start_phase_timer, end_phase_timer, track_phase_performance
from src.monitoring.system_monitor import get_system_monitor

# Phase 3: Import data quality validation
from src.validation.data_quality import get_data_quality_validator, should_skip_sizing
from src.config.database import get_data_quality_violations

logger = logging.getLogger(__name__)

class CyclePhase(Enum):
    """Trading cycle phase enumeration."""
    ENTRY = "ENTRY"
    MONITORING = "MONITORING"
    EXIT = "EXIT"
    ANALYSIS = "ANALYSIS"

@dataclass
class CycleResults:
    """Results of a complete trading cycle."""
    phase: CyclePhase
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    positions_analyzed: int
    positions_opened: int
    positions_closed: int
    signals_generated: int
    orders_executed: int
    successful_orders: int
    failed_orders: int
    total_pnl: float
    mode: str
    errors: List[str]
    warnings: List[str]

@dataclass
class PositionUpdate:
    """Position update during monitoring phase."""
    position_id: int
    symbol: str
    current_price: float
    pnl: float
    pnl_percent: float
    holding_days: int
    status: str
    timestamp: datetime

@dataclass
class ClosedPosition:
    """Closed position result."""
    position_id: int
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    holding_days: int
    exit_reason: str
    timestamp: datetime

class TradingCycle:
    """Manages the complete trading cycle workflow."""
    
    def __init__(self):
        """Initialize Trading Cycle."""
        self.position_manager = get_position_manager()
        self.enhanced_ensemble = get_enhanced_ensemble()
        
        # Cycle configuration
        self.max_positions = 10
        self.analysis_interval = 60  # seconds
        self.monitoring_interval = 30  # seconds
        self.max_holding_days = 30
        
        # Cycle state
        self.current_phase = CyclePhase.ANALYSIS
        self.cycle_start_time = None
        self.is_running = False
        
        logger.info("Trading Cycle initialized")
    
    def run_buy_phase(self, mode: Optional[str] = None) -> List[Position]:
        """Run the BUY phase - analyze and open new positions."""
        if mode is None:
            mode = get_current_mode()
        
        # Phase 1: Start phase timer and structured logging
        start_phase_timer("buy_phase")
        system_monitor = get_system_monitor()
        system_monitor.log_phase_event("buy_phase", "started", mode, {"step": "entry_analysis"})
        
        logger.info(f"Starting BUY phase for {mode} mode")
        
        try:
            # Check portfolio limits
            portfolio_limits = check_portfolio_limits(mode)
            if not portfolio_limits['can_open_new_position']:
                logger.info("Cannot open new positions - portfolio limits reached")
                return []
            
            # Get current open positions
            open_positions = get_open_positions(mode)
            current_position_count = len(open_positions)
            
            if current_position_count >= self.max_positions:
                logger.info(f"Maximum positions reached ({current_position_count}/{self.max_positions})")
                return []
            
            # Analyze market for entry opportunities
            new_positions = []
            symbols_to_analyze = self._get_symbols_to_analyze(mode)
            
            for symbol in symbols_to_analyze:
                try:
                    # Check if position already exists
                    existing_position = self.position_manager.get_position_by_symbol(symbol, mode)
                    if existing_position:
                        continue
                    
                    # Get market features for analysis
                    market_features = self._get_market_features(symbol)
                    
                    # Phase 3: Check data quality before trading decision
                    quality_validator = get_data_quality_validator()
                    quality_report = quality_validator.validate_dataframe(market_features, symbol)
                    
                    # Check if we should skip sizing due to poor data quality
                    if should_skip_sizing(quality_report, threshold=0.7):
                        logger.warning(f"Skipping {symbol} due to poor data quality: {quality_report.overall_score:.2%} "
                                     f"({quality_report.quality_level.value})")
                        system_monitor.log_phase_event("buy_phase", "quality_gate_failed", mode, {
                            "symbol": symbol,
                            "quality_score": quality_report.overall_score,
                            "quality_level": quality_report.quality_level.value,
                            "violations": len(quality_report.violations)
                        })
                        continue
                    
                    # Log quality gate passed
                    system_monitor.log_phase_event("buy_phase", "quality_gate_passed", mode, {
                        "symbol": symbol,
                        "quality_score": quality_report.overall_score,
                        "quality_level": quality_report.quality_level.value
                    })
                    
                    # Analyze for entry
                    buy_signal = analyze_for_entry(symbol, market_features, mode)
                    
                    if buy_signal and buy_signal.confidence > 0.7:
                        # Execute buy order
                        execution_result = execute_buy_order(
                            symbol=buy_signal.symbol,
                            quantity=buy_signal.position_size,
                            price=buy_signal.entry_price,
                            order_type=OrderType.MARKET
                        )
                        
                        if execution_result.success:
                            # Track position
                            position = track_position(
                                symbol=buy_signal.symbol,
                                entry_price=buy_signal.entry_price,
                                quantity=buy_signal.position_size,
                                stop_loss=buy_signal.stop_loss,
                                take_profit=buy_signal.take_profit
                            )
                            
                            new_positions.append(position)
                            logger.info(f"Opened new position: {symbol} {buy_signal.position_size} shares @ ${buy_signal.entry_price:.2f}")
                            
                            # Check if we've reached max positions
                            if len(new_positions) >= (self.max_positions - current_position_count):
                                break
                        else:
                            logger.warning(f"Failed to execute buy order for {symbol}: {execution_result.error_message}")
                
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} for entry: {e}")
                    continue
            
            # Phase 1: End phase timer and log completion
            duration = end_phase_timer("buy_phase")
            system_monitor.log_phase_event("buy_phase", "completed", mode, {
                "step": "entry_analysis",
                "positions_opened": len(new_positions),
                "duration_ms": duration * 1000
            })
            
            logger.info(f"BUY phase completed: {len(new_positions)} new positions opened")
            return new_positions
            
        except Exception as e:
            # Phase 1: Log error and end timer
            duration = end_phase_timer("buy_phase")
            system_monitor.log_phase_event("buy_phase", "error", mode, {
                "step": "entry_analysis",
                "error": str(e),
                "duration_ms": duration * 1000
            })
            logger.error(f"Error in BUY phase: {e}")
            return []
    
    def run_hold_phase(self, mode: Optional[str] = None) -> List[PositionUpdate]:
        """Run the HOLD phase - monitor existing positions."""
        if mode is None:
            mode = get_current_mode()
        
        # Phase 1: Start phase timer and structured logging
        start_phase_timer("hold_phase")
        system_monitor = get_system_monitor()
        system_monitor.log_phase_event("hold_phase", "started", mode, {"step": "position_monitoring"})
        
        logger.info(f"Starting HOLD phase for {mode} mode")
        
        try:
            # Get all open positions
            open_positions = get_open_positions(mode)
            position_updates = []
            
            for position in open_positions:
                try:
                    # Get current market data
                    current_price = self._get_current_price(position.symbol)
                    
                    # Update position P&L
                    self.position_manager.update_position_pnl(position.symbol, current_price, mode)
                    
                    # Calculate holding period
                    holding_days = (datetime.now() - position.entry_time).days
                    
                    # Calculate current P&L
                    pnl = (current_price - position.entry_price) * position.quantity
                    pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                    
                    # Create position update
                    update = PositionUpdate(
                        position_id=position.id,
                        symbol=position.symbol,
                        current_price=current_price,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        holding_days=holding_days,
                        status="MONITORING",
                        timestamp=datetime.now()
                    )
                    
                    position_updates.append(update)
                    
                    # Log significant P&L changes
                    if abs(pnl_percent) > 5:  # 5% change
                        logger.info(f"Position {position.symbol}: P&L {pnl_percent:.2f}% (${pnl:.2f})")
                
                except Exception as e:
                    logger.error(f"Error monitoring position {position.symbol}: {e}")
                    continue
            
            # Phase 1: End phase timer and log completion
            duration = end_phase_timer("hold_phase")
            system_monitor.log_phase_event("hold_phase", "completed", mode, {
                "step": "position_monitoring",
                "positions_monitored": len(position_updates),
                "duration_ms": duration * 1000
            })
            
            logger.info(f"HOLD phase completed: {len(position_updates)} positions monitored")
            return position_updates
            
        except Exception as e:
            # Phase 1: Log error and end timer
            duration = end_phase_timer("hold_phase")
            system_monitor.log_phase_event("hold_phase", "error", mode, {
                "step": "position_monitoring",
                "error": str(e),
                "duration_ms": duration * 1000
            })
            logger.error(f"Error in HOLD phase: {e}")
            return []
    
    def run_sell_phase(self, mode: Optional[str] = None) -> List[ClosedPosition]:
        """Run the SELL phase - analyze and close positions."""
        if mode is None:
            mode = get_current_mode()
        
        # Phase 1: Start phase timer and structured logging
        start_phase_timer("sell_phase")
        system_monitor = get_system_monitor()
        system_monitor.log_phase_event("sell_phase", "started", mode, {"step": "exit_analysis"})
        
        logger.info(f"Starting SELL phase for {mode} mode")
        
        try:
            # Get all open positions
            open_positions = get_open_positions(mode)
            closed_positions = []
            
            for position in open_positions:
                try:
                    # Check for exit signals
                    sell_signals = generate_sell_signals([position], mode)
                    
                    should_sell = False
                    exit_reason = "MANUAL"
                    
                    if sell_signals:
                        # Use the first sell signal
                        sell_signal = sell_signals[0]
                        should_sell = True
                        exit_reason = sell_signal.exit_reason.value
                    else:
                        # Check for time-based exit
                        holding_days = (datetime.now() - position.entry_time).days
                        if holding_days >= self.max_holding_days:
                            should_sell = True
                            exit_reason = "TIME_BASED"
                    
                    if should_sell:
                        # Get current price
                        current_price = self._get_current_price(position.symbol)
                        
                        # Execute sell order
                        execution_result = execute_sell_order(position, current_price, OrderType.MARKET)
                        
                        if execution_result.success:
                            # Close position
                            closed_position = close_position(position.symbol, current_price, mode)
                            
                            if closed_position:
                                # Calculate final metrics
                                holding_days = (closed_position.exit_time - closed_position.entry_time).days
                                
                                closed_pos = ClosedPosition(
                                    position_id=closed_position.id,
                                    symbol=closed_position.symbol,
                                    entry_price=closed_position.entry_price,
                                    exit_price=closed_position.exit_price,
                                    quantity=closed_position.quantity,
                                    pnl=closed_position.pnl,
                                    pnl_percent=closed_position.pnl_percent,
                                    holding_days=holding_days,
                                    exit_reason=exit_reason,
                                    timestamp=closed_position.exit_time
                                )
                                
                                closed_positions.append(closed_pos)
                                logger.info(f"Closed position: {position.symbol} @ ${current_price:.2f}, P&L: ${closed_position.pnl:.2f} ({closed_position.pnl_percent:.2f}%)")
                        else:
                            logger.warning(f"Failed to execute sell order for {position.symbol}: {execution_result.error_message}")
                
                except Exception as e:
                    logger.error(f"Error processing position {position.symbol} for exit: {e}")
                    continue
            
            # Phase 1: End phase timer and log completion
            duration = end_phase_timer("sell_phase")
            system_monitor.log_phase_event("sell_phase", "completed", mode, {
                "step": "exit_analysis",
                "positions_closed": len(closed_positions),
                "duration_ms": duration * 1000
            })
            
            logger.info(f"SELL phase completed: {len(closed_positions)} positions closed")
            return closed_positions
            
        except Exception as e:
            # Phase 1: Log error and end timer
            duration = end_phase_timer("sell_phase")
            system_monitor.log_phase_event("sell_phase", "error", mode, {
                "step": "exit_analysis",
                "error": str(e),
                "duration_ms": duration * 1000
            })
            logger.error(f"Error in SELL phase: {e}")
            return []
    
    def execute_complete_cycle(self, mode: Optional[str] = None) -> CycleResults:
        """Execute a complete trading cycle."""
        if mode is None:
            mode = get_current_mode()
        
        # Phase 1: Start complete cycle timer and structured logging
        start_phase_timer("complete_cycle")
        system_monitor = get_system_monitor()
        system_monitor.log_phase_event("complete_cycle", "started", mode, {"step": "cycle_orchestration"})
        
        self.cycle_start_time = datetime.now()
        self.is_running = True
        
        logger.info(f"Starting complete trading cycle for {mode} mode")
        
        errors = []
        warnings = []
        positions_opened = 0
        positions_closed = 0
        orders_executed = 0
        successful_orders = 0
        failed_orders = 0
        total_pnl = 0.0
        
        try:
            # Phase 1: BUY - Open new positions
            self.current_phase = CyclePhase.ENTRY
            new_positions = self.run_buy_phase(mode)
            positions_opened = len(new_positions)
            
            # Phase 2: HOLD - Monitor existing positions
            self.current_phase = CyclePhase.MONITORING
            position_updates = self.run_hold_phase(mode)
            
            # Phase 3: SELL - Close positions
            self.current_phase = CyclePhase.EXIT
            closed_positions = self.run_sell_phase(mode)
            positions_closed = len(closed_positions)
            
            # Calculate total P&L from closed positions
            total_pnl = sum(pos.pnl for pos in closed_positions)
            
            # Phase 4: ANALYSIS - Analyze results
            self.current_phase = CyclePhase.ANALYSIS
            self._analyze_cycle_results(new_positions, position_updates, closed_positions, mode)
            
        except Exception as e:
            error_msg = f"Error in trading cycle: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        finally:
            self.is_running = False
            cycle_end_time = datetime.now()
            duration = (cycle_end_time - self.cycle_start_time).total_seconds()
            
            # Phase 1: End complete cycle timer and log completion
            cycle_duration = end_phase_timer("complete_cycle")
            system_monitor.log_phase_event("complete_cycle", "completed", mode, {
                "step": "cycle_orchestration",
                "positions_opened": positions_opened,
                "positions_closed": positions_closed,
                "total_pnl": total_pnl,
                "duration_ms": cycle_duration * 1000
            })
            
            # Create cycle results
            results = CycleResults(
                phase=CyclePhase.ANALYSIS,
                start_time=self.cycle_start_time,
                end_time=cycle_end_time,
                duration_seconds=duration,
                positions_analyzed=len(get_open_positions(mode)),
                positions_opened=positions_opened,
                positions_closed=positions_closed,
                signals_generated=positions_opened + positions_closed,
                orders_executed=orders_executed,
                successful_orders=successful_orders,
                failed_orders=failed_orders,
                total_pnl=total_pnl,
                mode=mode,
                errors=errors,
                warnings=warnings
            )
            
            logger.info(f"Trading cycle completed: {positions_opened} opened, {positions_closed} closed, P&L: ${total_pnl:.2f}")
            return results
    
    def run_mode_specific_cycle(self, mode: str) -> CycleResults:
        """Run trading cycle for specific mode."""
        return self.execute_complete_cycle(mode)
    
    def _get_symbols_to_analyze(self, mode: str) -> List[str]:
        """Get list of symbols to analyze for entry."""
        # This would integrate with the stock universe in production
        # For now, return a predefined list
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "META", "NVDA", "NFLX", "AMD", "INTC"
        ]
        
        # Filter out symbols that already have positions
        open_positions = get_open_positions(mode)
        existing_symbols = {pos.symbol for pos in open_positions}
        
        return [symbol for symbol in symbols if symbol not in existing_symbols]
    
    def _get_market_features(self, symbol: str) -> Dict[str, Any]:
        """Get market features for a symbol."""
        # This would integrate with real market data in production
        # For now, return simulated data
        import random
        
        return {
            "current_price": 150.0 + random.uniform(-10, 10),
            "rsi": random.uniform(20, 80),
            "macd": random.uniform(-2, 2),
            "sma_20": 150.0 + random.uniform(-5, 5),
            "sma_50": 150.0 + random.uniform(-8, 8),
            "bollinger_position": random.uniform(0, 1),
            "volume_ratio": random.uniform(0.5, 2.0),
            "atr": random.uniform(0.01, 0.05),
            "sentiment_score": random.uniform(-1, 1),
            "fundamental_score": random.uniform(0.3, 0.8),
            "market_regime": random.choice(["BULL", "BEAR", "SIDEWAYS", "NEUTRAL"]),
            "sector_performance": random.uniform(-0.1, 0.1),
            "news_impact": random.uniform(-0.5, 0.5),
            "volatility": random.uniform(0.01, 0.05),
            "volume_trend": random.choice(["HIGH", "NORMAL", "LOW"])
        }
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # This would integrate with real market data in production
        # For now, return simulated price
        import random
        return 150.0 + random.uniform(-5, 5)
    
    def _analyze_cycle_results(self, new_positions: List[Position], 
                             position_updates: List[PositionUpdate],
                             closed_positions: List[ClosedPosition], 
                             mode: str) -> None:
        """Analyze the results of a trading cycle."""
        logger.info(f"Analyzing cycle results for {mode} mode:")
        logger.info(f"  - New positions opened: {len(new_positions)}")
        logger.info(f"  - Positions monitored: {len(position_updates)}")
        logger.info(f"  - Positions closed: {len(closed_positions)}")
        
        if closed_positions:
            total_pnl = sum(pos.pnl for pos in closed_positions)
            avg_pnl = total_pnl / len(closed_positions)
            winning_trades = len([pos for pos in closed_positions if pos.pnl > 0])
            win_rate = (winning_trades / len(closed_positions)) * 100
            
            logger.info(f"  - Total P&L: ${total_pnl:.2f}")
            logger.info(f"  - Average P&L: ${avg_pnl:.2f}")
            logger.info(f"  - Win rate: {win_rate:.1f}%")
    
    def validate_cycle(self) -> bool:
        """Validate the trading cycle configuration."""
        try:
            # Validate configuration
            if self.max_positions <= 0:
                logger.error("Invalid max_positions")
                return False
            
            if self.analysis_interval <= 0:
                logger.error("Invalid analysis_interval")
                return False
            
            if self.monitoring_interval <= 0:
                logger.error("Invalid monitoring_interval")
                return False
            
            if self.max_holding_days <= 0:
                logger.error("Invalid max_holding_days")
                return False
            
            # Validate components
            if not self.position_manager:
                logger.error("Position manager not initialized")
                return False
            
            if not self.enhanced_ensemble:
                logger.error("Enhanced ensemble not initialized")
                return False
            
            logger.info("Trading cycle validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Trading cycle validation error: {e}")
            return False

# Global trading cycle instance
_trading_cycle: Optional[TradingCycle] = None

def get_trading_cycle() -> TradingCycle:
    """Get the global trading cycle instance."""
    global _trading_cycle
    if _trading_cycle is None:
        _trading_cycle = TradingCycle()
    return _trading_cycle

def run_buy_phase(mode: Optional[str] = None) -> List[Position]:
    """Run the BUY phase."""
    return get_trading_cycle().run_buy_phase(mode)

def run_hold_phase(mode: Optional[str] = None) -> List[PositionUpdate]:
    """Run the HOLD phase."""
    return get_trading_cycle().run_hold_phase(mode)

def run_sell_phase(mode: Optional[str] = None) -> List[ClosedPosition]:
    """Run the SELL phase."""
    return get_trading_cycle().run_sell_phase(mode)

def execute_complete_cycle(mode: Optional[str] = None) -> CycleResults:
    """Execute a complete trading cycle."""
    return get_trading_cycle().execute_complete_cycle(mode)
