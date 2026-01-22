"""
Tick and Level II Data Processor
Handles high-frequency market data for micro-structure awareness
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
import aiohttp
import json
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq
import os

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Individual tick data point"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    trade_type: str  # 'B', 'S', 'T' (Buy, Sell, Trade)

@dataclass
class LevelIIData:
    """Level II order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # [(price, size), ...]
    asks: List[Tuple[float, int]]  # [(price, size), ...]
    spread: float
    mid_price: float

class TickDataProcessor:
    """Processes and stores tick-level market data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config.get('data_dir', 'data/tick_data')
        self.compression_interval = config.get('compression_interval', 60)  # seconds
        self.max_memory_ticks = config.get('max_memory_ticks', 10000)
        
        # In-memory storage
        self.tick_buffer: Dict[str, List[TickData]] = {}
        self.level2_buffer: Dict[str, List[LevelIIData]] = {}
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("Tick Data Processor initialized")
    
    async def fetch_tick_data(self, symbol: str, duration_minutes: int = 1) -> List[TickData]:
        """Fetch tick data for a symbol"""
        try:
            # Simulate tick data fetching (replace with actual Questrade API)
            ticks = []
            base_price = 100.0  # This would come from real market data
            
            for i in range(duration_minutes * 60):  # 1 tick per second
                timestamp = datetime.now() - timedelta(seconds=i)
                
                # Simulate price movement
                price_change = np.random.normal(0, 0.01)
                price = base_price + price_change
                
                # Simulate bid/ask spread
                spread = 0.01
                bid = price - spread/2
                ask = price + spread/2
                
                tick = TickData(
                    timestamp=timestamp,
                    symbol=symbol,
                    price=price,
                    volume=np.random.randint(100, 1000),
                    bid=bid,
                    ask=ask,
                    bid_size=np.random.randint(100, 500),
                    ask_size=np.random.randint(100, 500),
                    trade_type=np.random.choice(['B', 'S', 'T'])
                )
                ticks.append(tick)
            
            return ticks
            
        except Exception as e:
            logger.error(f"Error fetching tick data for {symbol}: {e}")
            return []
    
    async def fetch_level2_data(self, symbol: str) -> LevelIIData:
        """Fetch Level II order book data"""
        try:
            # Simulate Level II data (replace with actual Questrade API)
            base_price = 100.0
            
            # Generate bid levels
            bids = []
            for i in range(5):
                price = base_price - (i + 1) * 0.01
                size = np.random.randint(100, 1000)
                bids.append((price, size))
            
            # Generate ask levels
            asks = []
            for i in range(5):
                price = base_price + (i + 1) * 0.01
                size = np.random.randint(100, 1000)
                asks.append((price, size))
            
            spread = asks[0][0] - bids[0][0]
            mid_price = (bids[0][0] + asks[0][0]) / 2
            
            return LevelIIData(
                timestamp=datetime.now(),
                symbol=symbol,
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mid_price
            )
            
        except Exception as e:
            logger.error(f"Error fetching Level II data for {symbol}: {e}")
            return None
    
    def store_tick_data(self, symbol: str, ticks: List[TickData]):
        """Store tick data to buffer and compress to disk"""
        if symbol not in self.tick_buffer:
            self.tick_buffer[symbol] = []
        
        self.tick_buffer[symbol].extend(ticks)
        
        # Compress to disk when buffer is full
        if len(self.tick_buffer[symbol]) >= self.max_memory_ticks:
            self._compress_to_disk(symbol)
    
    def _compress_to_disk(self, symbol: str):
        """Compress tick data to Parquet format"""
        try:
            ticks = self.tick_buffer[symbol]
            
            # Convert to DataFrame
            data = {
                'timestamp': [t.timestamp for t in ticks],
                'symbol': [t.symbol for t in ticks],
                'price': [t.price for t in ticks],
                'volume': [t.volume for t in ticks],
                'bid': [t.bid for t in ticks],
                'ask': [t.ask for t in ticks],
                'bid_size': [t.bid_size for t in ticks],
                'ask_size': [t.ask_size for t in ticks],
                'trade_type': [t.trade_type for t in ticks]
            }
            
            df = pd.DataFrame(data)
            
            # Save to Parquet
            filename = f"{symbol}_ticks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = os.path.join(self.data_dir, filename)
            
            df.to_parquet(filepath, compression='snappy')
            
            # Clear buffer
            self.tick_buffer[symbol] = []
            
            logger.info(f"Compressed {len(ticks)} ticks for {symbol} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error compressing tick data for {symbol}: {e}")
    
    def aggregate_to_bars(self, symbol: str, interval_minutes: int = 1) -> pd.DataFrame:
        """Aggregate tick data to OHLCV bars"""
        try:
            if symbol not in self.tick_buffer or not self.tick_buffer[symbol]:
                return pd.DataFrame()
            
            ticks = self.tick_buffer[symbol]
            df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'price': t.price,
                'volume': t.volume
            } for t in ticks])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Resample to bars
            bars = df['price'].resample(f'{interval_minutes}T').ohlc()
            volume = df['volume'].resample(f'{interval_minutes}T').sum()
            
            bars['volume'] = volume
            bars.dropna(inplace=True)
            
            return bars
            
        except Exception as e:
            logger.error(f"Error aggregating bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_vwap(self, symbol: str, period_minutes: int = 5) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            if symbol not in self.tick_buffer or not self.tick_buffer[symbol]:
                return 0.0
            
            ticks = self.tick_buffer[symbol]
            cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
            
            recent_ticks = [t for t in ticks if t.timestamp >= cutoff_time]
            
            if not recent_ticks:
                return 0.0
            
            total_volume = sum(t.volume for t in recent_ticks)
            if total_volume == 0:
                return 0.0
            
            vwap = sum(t.price * t.volume for t in recent_ticks) / total_volume
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP for {symbol}: {e}")
            return 0.0
    
    def calculate_slippage(self, symbol: str, order_size: int, order_type: str = 'MARKET') -> float:
        """Calculate expected slippage for an order"""
        try:
            if symbol not in self.tick_buffer or not self.tick_buffer[symbol]:
                return 0.0
            
            ticks = self.tick_buffer[symbol]
            if not ticks:
                return 0.0
            
            latest_tick = ticks[-1]
            
            if order_type == 'MARKET':
                # Market order slippage based on spread
                spread = latest_tick.ask - latest_tick.bid
                slippage = spread / 2  # Half spread for market orders
            else:
                # Limit order slippage (minimal)
                slippage = 0.0
            
            # Adjust for order size vs available liquidity
            if order_type == 'MARKET':
                available_liquidity = latest_tick.bid_size if order_type == 'SELL' else latest_tick.ask_size
                if order_size > available_liquidity:
                    slippage *= (order_size / available_liquidity)
            
            return slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage for {symbol}: {e}")
            return 0.0
    
    def get_liquidity_score(self, symbol: str) -> float:
        """Calculate liquidity score (0-1) based on recent tick data"""
        try:
            if symbol not in self.tick_buffer or not self.tick_buffer[symbol]:
                return 0.0
            
            ticks = self.tick_buffer[symbol]
            if not ticks:
                return 0.0
            
            # Use last 10 ticks
            recent_ticks = ticks[-10:]
            
            # Calculate average volume and spread
            avg_volume = np.mean([t.volume for t in recent_ticks])
            avg_spread = np.mean([t.ask - t.bid for t in recent_ticks])
            
            # Normalize to 0-1 scale
            volume_score = min(avg_volume / 1000, 1.0)  # Max at 1000 volume
            spread_score = max(0, 1.0 - (avg_spread / 0.05))  # Max spread 0.05
            
            liquidity_score = (volume_score + spread_score) / 2
            return liquidity_score
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score for {symbol}: {e}")
            return 0.0

class TickDataManager:
    """Manages tick data collection and processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processor = TickDataProcessor(config)
        self.symbols = config.get('symbols', ['TD.TO', 'RY.TO', 'SHOP.TO'])
        self.collection_interval = config.get('collection_interval', 1)  # seconds
        self.is_running = False
        
        logger.info("Tick Data Manager initialized")
    
    async def start_collection(self):
        """Start continuous tick data collection"""
        self.is_running = True
        logger.info("Starting tick data collection...")
        
        while self.is_running:
            try:
                # Collect tick data for all symbols
                for symbol in self.symbols:
                    ticks = await self.processor.fetch_tick_data(symbol, 1)
                    self.processor.store_tick_data(symbol, ticks)
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in tick data collection: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop_collection(self):
        """Stop tick data collection"""
        self.is_running = False
        logger.info("Stopped tick data collection")
    
    def get_recent_bars(self, symbol: str, interval_minutes: int = 1, count: int = 100) -> pd.DataFrame:
        """Get recent OHLCV bars for a symbol"""
        return self.processor.aggregate_to_bars(symbol, interval_minutes).tail(count)
    
    def get_vwap(self, symbol: str, period_minutes: int = 5) -> float:
        """Get VWAP for a symbol"""
        return self.processor.calculate_vwap(symbol, period_minutes)
    
    def get_slippage_estimate(self, symbol: str, order_size: int, order_type: str = 'MARKET') -> float:
        """Get slippage estimate for an order"""
        return self.processor.calculate_slippage(symbol, order_size, order_type)
    
    def get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for a symbol"""
        return self.processor.get_liquidity_score(symbol)
