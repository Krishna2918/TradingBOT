"""
Slippage & Latency Simulator
Simulates realistic market impact, slippage, and latency for backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time
import random

logger = logging.getLogger(__name__)

@dataclass
class SlippageModel:
    """Slippage model configuration"""
    linear_impact: float = 0.0001  # Linear impact per share
    square_root_impact: float = 0.00005  # Square root impact
    temporary_impact: float = 0.0002  # Temporary impact
    permanent_impact: float = 0.0001  # Permanent impact
    volatility_multiplier: float = 1.5  # Volatility multiplier

@dataclass
class LatencyModel:
    """Latency model configuration"""
    base_latency_ms: float = 10.0  # Base latency in milliseconds
    network_latency_ms: float = 5.0  # Network latency
    processing_latency_ms: float = 3.0  # Processing latency
    jitter_ms: float = 2.0  # Random jitter
    venue_latency_ms: Dict[str, float] = None  # Venue-specific latency
    
    def __post_init__(self):
        if self.venue_latency_ms is None:
            self.venue_latency_ms = {
                'TSX': 8.0,
                'NASDAQ': 12.0,
                'NYSE': 15.0,
                'OTC': 25.0
            }

class SlippageSimulator:
    """Simulates market impact and slippage"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.slippage_model = SlippageModel()
        self.impact_history = []
        
        logger.info("Slippage Simulator initialized")
    
    def calculate_market_impact(self, symbol: str, quantity: int, side: str, 
                              market_data: Dict) -> Dict:
        """Calculate market impact for an order"""
        try:
            # Get market data
            current_price = market_data.get('price', 100.0)
            volume = market_data.get('volume', 10000)
            volatility = market_data.get('volatility', 0.02)
            spread = market_data.get('spread', 0.01)
            
            # Calculate participation rate
            participation_rate = quantity / volume if volume > 0 else 0
            
            # Linear impact component
            linear_impact = self.slippage_model.linear_impact * quantity
            
            # Square root impact component (Almgren-Chriss model)
            square_root_impact = self.slippage_model.square_root_impact * np.sqrt(quantity)
            
            # Volatility adjustment
            volatility_impact = volatility * self.slippage_model.volatility_multiplier
            
            # Temporary impact (reverts quickly)
            temporary_impact = self.slippage_model.temporary_impact * participation_rate
            
            # Permanent impact (persists)
            permanent_impact = self.slippage_model.permanent_impact * participation_rate
            
            # Total impact
            total_impact = (linear_impact + square_root_impact + 
                          volatility_impact + temporary_impact + permanent_impact)
            
            # Calculate execution price
            if side == 'BUY':
                execution_price = current_price * (1 + total_impact)
            else:  # SELL
                execution_price = current_price * (1 - total_impact)
            
            # Add spread cost
            spread_cost = current_price * spread / 2
            
            if side == 'BUY':
                execution_price += spread_cost
            else:
                execution_price -= spread_cost
            
            # Calculate slippage
            slippage = abs(execution_price - current_price) / current_price
            
            impact_data = {
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'current_price': current_price,
                'execution_price': execution_price,
                'slippage': slippage,
                'linear_impact': linear_impact,
                'square_root_impact': square_root_impact,
                'volatility_impact': volatility_impact,
                'temporary_impact': temporary_impact,
                'permanent_impact': permanent_impact,
                'total_impact': total_impact,
                'spread_cost': spread_cost,
                'participation_rate': participation_rate,
                'timestamp': datetime.now()
            }
            
            # Store impact history
            self.impact_history.append(impact_data)
            
            return impact_data
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return {}
    
    def calculate_implementation_shortfall(self, symbol: str, quantity: int, 
                                         side: str, market_data: Dict,
                                         execution_prices: List[float]) -> Dict:
        """Calculate implementation shortfall (IS)"""
        try:
            current_price = market_data.get('price', 100.0)
            
            if not execution_prices:
                return {}
            
            # Calculate volume-weighted average price
            vwap = np.mean(execution_prices)
            
            # Calculate implementation shortfall
            if side == 'BUY':
                is_per_share = vwap - current_price
            else:  # SELL
                is_per_share = current_price - vwap
            
            is_percentage = is_per_share / current_price
            
            # Calculate total IS
            total_is = is_per_share * quantity
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'current_price': current_price,
                'vwap': vwap,
                'is_per_share': is_per_share,
                'is_percentage': is_percentage,
                'total_is': total_is,
                'execution_prices': execution_prices,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating implementation shortfall: {e}")
            return {}
    
    def get_impact_statistics(self, symbol: str = None) -> Dict:
        """Get impact statistics"""
        try:
            if not self.impact_history:
                return {}
            
            # Filter by symbol if specified
            data = self.impact_history
            if symbol:
                data = [d for d in data if d['symbol'] == symbol]
            
            if not data:
                return {}
            
            # Calculate statistics
            slippages = [d['slippage'] for d in data]
            impacts = [d['total_impact'] for d in data]
            participation_rates = [d['participation_rate'] for d in data]
            
            return {
                'total_orders': len(data),
                'average_slippage': np.mean(slippages),
                'median_slippage': np.median(slippages),
                'max_slippage': np.max(slippages),
                'min_slippage': np.min(slippages),
                'slippage_std': np.std(slippages),
                'average_impact': np.mean(impacts),
                'average_participation_rate': np.mean(participation_rates),
                'symbol': symbol or 'all'
            }
            
        except Exception as e:
            logger.error(f"Error getting impact statistics: {e}")
            return {}

class LatencySimulator:
    """Simulates execution latency and delays"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.latency_model = LatencyModel()
        self.latency_history = []
        
        logger.info("Latency Simulator initialized")
    
    def simulate_latency(self, venue: str = 'TSX', order_type: str = 'market') -> float:
        """Simulate execution latency"""
        try:
            # Base latency
            base_latency = self.latency_model.base_latency_ms
            
            # Network latency
            network_latency = self.latency_model.network_latency_ms
            
            # Processing latency
            processing_latency = self.latency_model.processing_latency_ms
            
            # Venue-specific latency
            venue_latency = self.latency_model.venue_latency_ms.get(venue, 10.0)
            
            # Random jitter
            jitter = random.uniform(-self.latency_model.jitter_ms, 
                                  self.latency_model.jitter_ms)
            
            # Order type adjustment
            order_type_multiplier = 1.0
            if order_type == 'limit':
                order_type_multiplier = 1.2  # Limit orders take longer
            elif order_type == 'iceberg':
                order_type_multiplier = 1.5  # Iceberg orders take longer
            
            # Calculate total latency
            total_latency = (base_latency + network_latency + processing_latency + 
                           venue_latency + jitter) * order_type_multiplier
            
            # Ensure minimum latency
            total_latency = max(total_latency, 1.0)
            
            # Store latency record
            latency_record = {
                'venue': venue,
                'order_type': order_type,
                'total_latency_ms': total_latency,
                'base_latency_ms': base_latency,
                'network_latency_ms': network_latency,
                'processing_latency_ms': processing_latency,
                'venue_latency_ms': venue_latency,
                'jitter_ms': jitter,
                'timestamp': datetime.now()
            }
            
            self.latency_history.append(latency_record)
            
            return total_latency
            
        except Exception as e:
            logger.error(f"Error simulating latency: {e}")
            return 10.0  # Default latency
    
    def simulate_network_delay(self, data_size_kb: float = 1.0) -> float:
        """Simulate network delay based on data size"""
        try:
            # Base network delay
            base_delay = 5.0  # ms
            
            # Data size factor
            size_factor = data_size_kb * 0.1  # 0.1ms per KB
            
            # Random network variation
            network_variation = random.uniform(0.5, 2.0)
            
            total_delay = (base_delay + size_factor) * network_variation
            
            return total_delay
            
        except Exception as e:
            logger.error(f"Error simulating network delay: {e}")
            return 5.0
    
    def simulate_venue_processing_time(self, venue: str, order_complexity: str = 'simple') -> float:
        """Simulate venue processing time"""
        try:
            # Base processing times by venue
            venue_processing = {
                'TSX': 8.0,
                'NASDAQ': 12.0,
                'NYSE': 15.0,
                'OTC': 25.0,
                'DARK_POOL': 20.0
            }
            
            base_time = venue_processing.get(venue, 10.0)
            
            # Complexity adjustment
            complexity_multiplier = {
                'simple': 1.0,
                'medium': 1.3,
                'complex': 1.8,
                'algorithmic': 2.2
            }
            
            multiplier = complexity_multiplier.get(order_complexity, 1.0)
            
            # Random variation
            variation = random.uniform(0.8, 1.2)
            
            total_time = base_time * multiplier * variation
            
            return total_time
            
        except Exception as e:
            logger.error(f"Error simulating venue processing time: {e}")
            return 10.0
    
    def get_latency_statistics(self, venue: str = None) -> Dict:
        """Get latency statistics"""
        try:
            if not self.latency_history:
                return {}
            
            # Filter by venue if specified
            data = self.latency_history
            if venue:
                data = [d for d in data if d['venue'] == venue]
            
            if not data:
                return {}
            
            # Calculate statistics
            latencies = [d['total_latency_ms'] for d in data]
            
            return {
                'total_orders': len(data),
                'average_latency_ms': np.mean(latencies),
                'median_latency_ms': np.median(latencies),
                'max_latency_ms': np.max(latencies),
                'min_latency_ms': np.min(latencies),
                'latency_std_ms': np.std(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'venue': venue or 'all'
            }
            
        except Exception as e:
            logger.error(f"Error getting latency statistics: {e}")
            return {}

class SlippageLatencySimulator:
    """Combined slippage and latency simulator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.slippage_simulator = SlippageSimulator(config)
        self.latency_simulator = LatencySimulator(config)
        
        logger.info("Slippage & Latency Simulator initialized")
    
    def simulate_order_execution(self, symbol: str, quantity: int, side: str,
                               order_type: str, venue: str, market_data: Dict) -> Dict:
        """Simulate complete order execution with slippage and latency"""
        try:
            # Simulate latency
            latency_ms = self.latency_simulator.simulate_latency(venue, order_type)
            
            # Simulate slippage
            impact_data = self.slippage_simulator.calculate_market_impact(
                symbol, quantity, side, market_data
            )
            
            # Combine results
            execution_data = {
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'order_type': order_type,
                'venue': venue,
                'latency_ms': latency_ms,
                'execution_price': impact_data.get('execution_price', market_data.get('price', 100.0)),
                'slippage': impact_data.get('slippage', 0.0),
                'market_impact': impact_data.get('total_impact', 0.0),
                'spread_cost': impact_data.get('spread_cost', 0.0),
                'participation_rate': impact_data.get('participation_rate', 0.0),
                'timestamp': datetime.now()
            }
            
            return execution_data
            
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return {}
    
    def simulate_batch_execution(self, orders: List[Dict], market_data: Dict) -> List[Dict]:
        """Simulate batch execution of multiple orders"""
        try:
            executions = []
            
            for order in orders:
                execution = self.simulate_order_execution(
                    order['symbol'],
                    order['quantity'],
                    order['side'],
                    order.get('order_type', 'market'),
                    order.get('venue', 'TSX'),
                    market_data
                )
                
                executions.append(execution)
            
            return executions
            
        except Exception as e:
            logger.error(f"Error simulating batch execution: {e}")
            return []
    
    def get_simulation_statistics(self) -> Dict:
        """Get comprehensive simulation statistics"""
        try:
            slippage_stats = self.slippage_simulator.get_impact_statistics()
            latency_stats = self.latency_simulator.get_latency_statistics()
            
            return {
                'slippage_statistics': slippage_stats,
                'latency_statistics': latency_stats,
                'total_simulations': len(self.slippage_simulator.impact_history),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting simulation statistics: {e}")
            return {}
    
    def calibrate_models(self, historical_data: pd.DataFrame) -> Dict:
        """Calibrate slippage and latency models using historical data"""
        try:
            if historical_data.empty:
                return {}
            
            # Calibrate slippage model
            slippage_calibration = self._calibrate_slippage_model(historical_data)
            
            # Calibrate latency model
            latency_calibration = self._calibrate_latency_model(historical_data)
            
            return {
                'slippage_calibration': slippage_calibration,
                'latency_calibration': latency_calibration,
                'calibration_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calibrating models: {e}")
            return {}
    
    def _calibrate_slippage_model(self, data: pd.DataFrame) -> Dict:
        """Calibrate slippage model parameters"""
        try:
            # This would use actual historical data to calibrate parameters
            # For now, return default calibration
            return {
                'linear_impact': 0.0001,
                'square_root_impact': 0.00005,
                'temporary_impact': 0.0002,
                'permanent_impact': 0.0001,
                'volatility_multiplier': 1.5,
                'calibration_method': 'historical_regression'
            }
            
        except Exception as e:
            logger.error(f"Error calibrating slippage model: {e}")
            return {}
    
    def _calibrate_latency_model(self, data: pd.DataFrame) -> Dict:
        """Calibrate latency model parameters"""
        try:
            # This would use actual historical data to calibrate parameters
            # For now, return default calibration
            return {
                'base_latency_ms': 10.0,
                'network_latency_ms': 5.0,
                'processing_latency_ms': 3.0,
                'jitter_ms': 2.0,
                'calibration_method': 'historical_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error calibrating latency model: {e}")
            return {}
