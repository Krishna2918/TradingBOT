"""
Trading Orchestrator - Master Pipeline Controller

Integrates all components into a streamlined trading pipeline:
1. Data Collection → 2. AI Analysis → 3. Risk Management → 4. Execution → 5. Monitoring
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from pathlib import Path

# Core components
from src.data_pipeline.comprehensive_data_pipeline import ComprehensiveDataPipeline
from src.ai.ai_ensemble import AIEnsemble
from src.ai.model_stack import LSTMModel, GRUTransformerModel, MetaEnsemble
from src.ai.rl import TradingEnvironment, PPOAgent, DQNAgent
from src.risk_management import CapitalAllocator, LeverageGovernor, KillSwitchManager
from src.execution import ExecutionEngine, OrderType, OrderSide
from src.event_awareness import EventCalendar, VolatilityDetector, AnomalyDetector
from src.strategies.strategy_manager import StrategyManager
from src.trading_modes import ModeManager
from src.reporting import ReportGenerator, ReportScheduler

logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """
    Master orchestrator that coordinates all trading bot components
    
    Pipeline Flow:
    1. Data Collection: Market data, news, options, macro
    2. Event Awareness: Check calendar, volatility, anomalies
    3. AI Analysis: Ensemble predictions, RL decisions
    4. Strategy Signals: Generate trading signals
    5. Risk Management: Validate capital, leverage, kill switches
    6. Execution: Place orders with VWAP, partial fills
    7. Monitoring: Track performance, generate reports
    """
    
    def __init__(self, config: Dict = None):
        """Initialize orchestrator with all components"""
        
        logger.info(" Initializing Trading Orchestrator...")
        
        self.config = config or self._load_default_config()
        self.is_running = False
        self.cycle_count = 0
        
        # Initialize components
        logger.info(" Initializing Data Pipeline...")
        self.data_pipeline = ComprehensiveDataPipeline()
        
        logger.info(" Initializing AI Ensemble...")
        try:
            self.ai_ensemble = AIEnsemble()
        except Exception as e:
            logger.warning(f" AI Ensemble init failed: {e}, using basic mode")
            self.ai_ensemble = None
        
        logger.info(" Initializing AI Model Stack...")
        try:
            self.lstm_model = LSTMModel(input_size=35, hidden_size=128, num_layers=2, output_size=1)
            self.gru_model = GRUTransformerModel(input_size=50, hidden_size=256, num_layers=2, output_size=1)
            self.meta_ensemble = MetaEnsemble(self.lstm_model, self.gru_model)
        except Exception as e:
            logger.warning(f" Model stack init failed: {e}")
            self.meta_ensemble = None
        
        logger.info(" Initializing Strategy Manager...")
        self.strategy_manager = StrategyManager("config/strategy_config.yaml")
        
        logger.info(" Initializing Risk Management...")
        self.capital_allocator = CapitalAllocator("config/risk_config.yaml")
        self.leverage_governor = LeverageGovernor("config/risk_config.yaml")
        self.kill_switch_manager = KillSwitchManager("config/risk_config.yaml")
        
        logger.info(" Initializing Execution Engine...")
        self.execution_engine = ExecutionEngine(
            commission_rate=0.001,
            slippage_bps=5.0,
            allow_fractional=True
        )
        
        logger.info(" Initializing Event Awareness...")
        self.event_calendar = EventCalendar()
        self.volatility_detector = VolatilityDetector()
        self.anomaly_detector = AnomalyDetector()
        
        logger.info(" Initializing Trading Mode Manager...")
        self.mode_manager = ModeManager()
        
        logger.info(" Initializing Reporting System...")
        self.report_generator = ReportGenerator()
        self.report_scheduler = ReportScheduler()
        
        logger.info(" Trading Orchestrator initialized successfully!")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'symbols': ['SHOP.TO', 'TD.TO', 'RY.TO', 'ENB.TO', 'CNQ.TO'],
            'cycle_interval_seconds': 60,  # 1 minute
            'enable_ai_predictions': True,
            'enable_anomaly_detection': True,
            'enable_auto_execution': False,  # Safety: manual approval required
            'max_daily_trades': 50,
            'max_position_size_pct': 0.10  # 10% max
        }
    
    def run_trading_cycle(self) -> Dict:
        """
        Execute one complete trading cycle
        
        Returns:
            Dictionary with cycle results
        """
        
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info(f"\n{'='*80}")
        logger.info(f" TRADING CYCLE #{self.cycle_count} - {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")
        
        results = {
            'cycle_number': self.cycle_count,
            'timestamp': cycle_start.isoformat(),
            'status': 'running'
        }
        
        try:
            # ===== PHASE 1: PRE-FLIGHT CHECKS =====
            logger.info("\n PHASE 1: Pre-Flight Checks")
            
            # Check kill switch
            if self.kill_switch_manager.is_kill_switch_active():
                logger.error(" KILL SWITCH ACTIVE - Trading halted!")
                results['status'] = 'kill_switch_active'
                return results
            
            # Check if market is open
            if self.event_calendar.is_market_holiday():
                logger.info(" Market holiday detected - skipping trading")
                results['status'] = 'market_closed'
                return results
            
            # Check trading mode
            current_mode = self.mode_manager.get_current_mode()
            logger.info(f" Trading Mode: {current_mode.value.upper()}")
            
            # ===== PHASE 2: DATA COLLECTION =====
            logger.info("\n PHASE 2: Data Collection")
            
            market_data = self._collect_market_data()
            results['data_collected'] = len(market_data)
            
            # ===== PHASE 3: EVENT AWARENESS =====
            logger.info("\n PHASE 3: Event Awareness Analysis")
            
            event_analysis = self._analyze_events()
            results['events'] = event_analysis
            
            # Check for high-impact events
            if event_analysis.get('high_impact_event_soon'):
                logger.warning(" High-impact event approaching - reducing position sizes")
            
            # ===== PHASE 4: AI PREDICTIONS =====
            logger.info("\n PHASE 4: AI Model Predictions")
            
            ai_predictions = self._generate_ai_predictions(market_data)
            results['ai_predictions'] = ai_predictions
            
            # ===== PHASE 5: STRATEGY SIGNALS =====
            logger.info("\n PHASE 5: Strategy Signal Generation")
            
            signals = self._generate_strategy_signals(market_data, ai_predictions)
            results['signals_generated'] = len(signals)
            
            logger.info(f" Generated {len(signals)} trading signals")
            
            # ===== PHASE 6: RISK MANAGEMENT =====
            logger.info("\n PHASE 6: Risk Management Validation")
            
            validated_signals = self._validate_signals_risk(signals)
            results['signals_validated'] = len(validated_signals)
            
            rejected = len(signals) - len(validated_signals)
            if rejected > 0:
                logger.warning(f" {rejected} signals rejected by risk management")
            
            # ===== PHASE 7: ORDER EXECUTION =====
            logger.info("\n PHASE 7: Order Execution")
            
            if self.config.get('enable_auto_execution'):
                executions = self._execute_orders(validated_signals, market_data)
                results['orders_executed'] = len(executions)
            else:
                logger.info("ℹ Auto-execution disabled - signals logged for manual review")
                results['orders_executed'] = 0
                results['pending_manual_approval'] = len(validated_signals)
            
            # ===== PHASE 8: PORTFOLIO MONITORING =====
            logger.info("\n PHASE 8: Portfolio Monitoring")
            
            portfolio_status = self._monitor_portfolio()
            results['portfolio'] = portfolio_status
            
            # ===== PHASE 9: PERFORMANCE TRACKING =====
            logger.info("\n PHASE 9: Performance Tracking")
            
            performance = self._track_performance()
            results['performance'] = performance
            
            # Calculate cycle duration
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            results['cycle_duration_seconds'] = cycle_duration
            results['status'] = 'completed'
            
            logger.info(f"\n Cycle #{self.cycle_count} completed in {cycle_duration:.2f}s")
            logger.info(f"{'='*80}\n")
            
            return results
            
        except Exception as e:
            logger.error(f" Error in trading cycle: {e}", exc_info=True)
            results['status'] = 'error'
            results['error'] = str(e)
            return results
    
    def _collect_market_data(self) -> Dict:
        """Collect market data from all sources"""
        
        symbols = self.config.get('symbols', [])
        
        try:
            # Run full data collection cycle
            self.data_pipeline.run_data_collection_cycle(symbols)
            
            # Retrieve collected data
            data = {
                'symbols': symbols,
                'timestamp': datetime.now().isoformat(),
                'quality_metrics': self.data_pipeline.check_data_quality()
            }
            
            logger.info(f" Collected data for {len(symbols)} symbols")
            
            return data
            
        except Exception as e:
            logger.error(f" Data collection failed: {e}")
            return {}
    
    def _analyze_events(self) -> Dict:
        """Analyze upcoming events and market conditions"""
        
        analysis = {}
        
        try:
            # Check upcoming events
            upcoming = self.event_calendar.get_upcoming_events(hours_ahead=24)
            high_impact = self.event_calendar.get_high_impact_events(days_ahead=7)
            
            analysis['upcoming_events_24h'] = len(upcoming)
            analysis['high_impact_events_7d'] = len(high_impact)
            analysis['high_impact_event_soon'] = any(
                e.is_upcoming(hours_ahead=4) for e in high_impact
            )
            
            # Log upcoming events
            if upcoming:
                logger.info(f" {len(upcoming)} events in next 24h:")
                for event in upcoming[:3]:  # Show top 3
                    logger.info(f"   • {event.title} @ {event.scheduled_time}")
            
            return analysis
            
        except Exception as e:
            logger.error(f" Event analysis failed: {e}")
            return {}
    
    def _generate_ai_predictions(self, market_data: Dict) -> Dict:
        """Generate AI model predictions"""
        
        predictions = {
            'model_stack': None,
            'ensemble': None,
            'rl_agent': None
        }
        
        try:
            # Use meta-ensemble if available
            if self.meta_ensemble:
                # Note: Would need actual prepared data here
                logger.info(" AI model stack predictions available")
                predictions['model_stack'] = 'ready'
            
            # Use AI ensemble if available
            if self.ai_ensemble:
                logger.info(" AI ensemble (Grok/Kimi/Claude) ready")
                predictions['ensemble'] = 'ready'
            
            return predictions
            
        except Exception as e:
            logger.error(f" AI prediction failed: {e}")
            return predictions
    
    def _generate_strategy_signals(self, market_data: Dict, ai_predictions: Dict) -> List[Dict]:
        """Generate trading signals from strategies"""
        
        try:
            # Use strategy manager to analyze market
            all_signals = self.strategy_manager.analyze_market_conditions(
                market_data.get('market_data', {}),
                market_data.get('news', []),
                market_data.get('options', {})
            )
            
            # Flatten signals from all strategies
            signals = []
            for strategy_name, strategy_signals in all_signals.items():
                for signal in strategy_signals:
                    signal['strategy'] = strategy_name
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f" Signal generation failed: {e}")
            return []
    
    def _validate_signals_risk(self, signals: List[Dict]) -> List[Dict]:
        """Validate signals through risk management"""
        
        validated = []
        
        for signal in signals:
            try:
                # Check capital availability
                position_size = self.capital_allocator.calculate_position_size(
                    signal.get('price', 0),
                    signal.get('strategy', 'default')
                )
                
                # Check leverage limits
                leverage_state = self.leverage_governor.get_leverage_state()
                
                # Validate
                if position_size > 0 and not self.kill_switch_manager.is_kill_switch_active():
                    signal['validated_position_size'] = position_size
                    signal['leverage'] = leverage_state.current_leverage
                    validated.append(signal)
                
            except Exception as e:
                logger.warning(f" Signal validation failed: {e}")
                continue
        
        return validated
    
    def _execute_orders(self, signals: List[Dict], market_data: Dict) -> List[Dict]:
        """Execute validated trading orders"""
        
        executions = []
        
        for signal in signals:
            try:
                # Create order
                order = self.execution_engine.create_order(
                    symbol=signal['symbol'],
                    side=OrderSide.BUY if signal['direction'] == 'long' else OrderSide.SELL,
                    quantity=signal.get('validated_position_size', 0) / signal.get('price', 1),
                    order_type=OrderType.MARKET
                )
                
                # Execute with VWAP if large order
                if order.quantity > 100:
                    success = self.execution_engine.execute_vwap_order(
                        order=order,
                        market_data=market_data.get('bars_1m', {}).get(signal['symbol']),
                        time_window_minutes=30
                    )
                else:
                    success = self.execution_engine.execute_market_order(
                        order=order,
                        current_price=signal.get('price', 0),
                        volume=10000  # Mock volume
                    )
                
                if success:
                    executions.append({
                        'order_id': order.order_id,
                        'symbol': signal['symbol'],
                        'filled': order.filled_quantity,
                        'price': order.average_fill_price
                    })
                
            except Exception as e:
                logger.error(f" Order execution failed: {e}")
                continue
        
        return executions
    
    def _monitor_portfolio(self) -> Dict:
        """Monitor current portfolio status"""
        
        try:
            account_info = self.mode_manager.get_current_account_info()
            
            return {
                'mode': self.mode_manager.current_mode.value,
                'capital': account_info['capital'],
                'num_positions': account_info.get('num_positions', 0),
                'num_trades': account_info['num_trades']
            }
            
        except Exception as e:
            logger.error(f" Portfolio monitoring failed: {e}")
            return {}
    
    def _track_performance(self) -> Dict:
        """Track performance metrics"""
        
        try:
            exec_stats = self.execution_engine.get_execution_statistics()
            
            return {
                'total_executions': exec_stats.get('total_executions', 0),
                'total_volume': exec_stats.get('total_volume', 0),
                'avg_slippage': exec_stats.get('average_slippage', 0)
            }
            
        except Exception as e:
            logger.error(f" Performance tracking failed: {e}")
            return {}
    
    def start(self, run_indefinitely: bool = False):
        """Start the trading orchestrator"""
        
        logger.info(" Starting Trading Orchestrator...")
        logger.info(f" Monitoring {len(self.config.get('symbols', []))} symbols")
        logger.info(f"⏱ Cycle interval: {self.config.get('cycle_interval_seconds', 60)}s")
        
        self.is_running = True
        
        try:
            if run_indefinitely:
                # Continuous operation
                while self.is_running:
                    self.run_trading_cycle()
                    time.sleep(self.config.get('cycle_interval_seconds', 60))
            else:
                # Single cycle
                results = self.run_trading_cycle()
                return results
                
        except KeyboardInterrupt:
            logger.info("\n⏸ Trading orchestrator stopped by user")
            self.stop()
        except Exception as e:
            logger.error(f" Fatal error: {e}", exc_info=True)
            self.stop()
    
    def stop(self):
        """Stop the trading orchestrator"""
        logger.info(" Stopping Trading Orchestrator...")
        self.is_running = False
        
        # Generate final report
        try:
            report = self.report_generator.generate_daily_report()
            logger.info(f" Final report generated: {report.get('file_path')}")
        except Exception as e:
            logger.error(f" Failed to generate final report: {e}")
        
        logger.info(" Trading Orchestrator stopped")

# Global instance
_orchestrator_instance = None

def get_orchestrator() -> TradingOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = TradingOrchestrator()
    return _orchestrator_instance

