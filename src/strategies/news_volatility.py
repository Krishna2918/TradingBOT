"""
News-Volatility Strategy
Trades based on news sentiment and volatility spikes
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

@dataclass
class NewsSignal:
    """News-based trading signal"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    news_source: str
    news_headline: str
    sentiment_score: float
    timestamp: datetime

class NewsVolatilityStrategy:
    """News-Volatility Strategy for Canadian Markets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.allocation = config.get('allocation', 0.20)
        self.max_leverage = config.get('max_leverage', 1.5)
        
        # Strategy parameters
        self.sentiment_threshold = config.get('signals', {}).get('news_sentiment_threshold', 0.7)
        self.volatility_spike = config.get('signals', {}).get('volatility_spike', 1.5)
        self.volume_surge = config.get('signals', {}).get('volume_surge', 3.0)
        self.stop_loss = config.get('risk', {}).get('stop_loss', 0.5)
        self.take_profit = config.get('risk', {}).get('take_profit', 1.0)
        self.max_holding_time = config.get('risk', {}).get('max_holding_time', 45)
        
        # Canadian market instruments
        self.instruments = config.get('instruments', [
            "SU.TO", "ENB.TO", "TRP.TO", "ABX.TO", "WCN.TO"
        ])
        
        # News sentiment keywords
        self.bullish_keywords = [
            'earnings beat', 'strong growth', 'positive outlook', 'upgrade',
            'acquisition', 'partnership', 'expansion', 'dividend increase',
            'buyback', 'guidance raise', 'strong demand', 'market share gain'
        ]
        
        self.bearish_keywords = [
            'earnings miss', 'weak guidance', 'downgrade', 'layoffs',
            'revenue decline', 'profit warning', 'regulatory issues',
            'competition', 'market share loss', 'debt concerns', 'liquidity issues'
        ]
        
        self.active_positions = {}
        self.signal_history = []
        self.news_cache = {}
    
    def analyze_news_and_volatility(self, news_data: List[Dict], market_data: Dict) -> List[NewsSignal]:
        """Analyze news sentiment and volatility to generate signals"""
        try:
            signals = []
            
            # Process news data
            for news_item in news_data:
                signal = self._analyze_news_item(news_item, market_data)
                if signal:
                    signals.append(signal)
            
            logger.info(f"Generated {len(signals)} news-volatility signals")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to analyze news and volatility: {e}")
            return []
    
    def _analyze_news_item(self, news_item: Dict, market_data: Dict) -> Optional[NewsSignal]:
        """Analyze individual news item for trading signals"""
        try:
            headline = news_item.get('title', '').lower()
            source = news_item.get('source', '')
            timestamp = news_item.get('timestamp', datetime.now())
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(headline)
            
            # Only process if sentiment is strong enough
            if abs(sentiment_score) < self.sentiment_threshold:
                return None
            
            # Find relevant symbols
            relevant_symbols = self._find_relevant_symbols(headline, source)
            
            if not relevant_symbols:
                return None
            
            # Generate signals for each relevant symbol
            signals = []
            for symbol in relevant_symbols:
                if symbol in market_data:
                    signal = self._create_news_signal(
                        symbol, news_item, sentiment_score, market_data[symbol]
                    )
                    if signal:
                        signals.append(signal)
            
            return signals[0] if signals else None
            
        except Exception as e:
            logger.error(f"Failed to analyze news item: {e}")
            return None
    
    def _calculate_sentiment_score(self, headline: str) -> float:
        """Calculate sentiment score from headline"""
        try:
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in headline)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in headline)
            
            total_keywords = bullish_count + bearish_count
            if total_keywords == 0:
                return 0.0
            
            # Calculate sentiment score (-1.0 to 1.0)
            sentiment = (bullish_count - bearish_count) / total_keywords
            
            # Apply additional scoring based on intensity words
            intensity_words = ['strong', 'significant', 'major', 'substantial', 'dramatic']
            intensity_multiplier = 1.0
            
            for word in intensity_words:
                if word in headline:
                    intensity_multiplier += 0.2
            
            return sentiment * intensity_multiplier
            
        except Exception as e:
            logger.error(f"Failed to calculate sentiment score: {e}")
            return 0.0
    
    def _find_relevant_symbols(self, headline: str, source: str) -> List[str]:
        """Find symbols relevant to the news"""
        try:
            relevant_symbols = []
            
            # Map company names to symbols
            company_mapping = {
                'royal bank': 'RY.TO',
                'td bank': 'TD.TO',
                'shopify': 'SHOP.TO',
                'suncor': 'SU.TO',
                'enbridge': 'ENB.TO',
                'tc energy': 'TRP.TO',
                'barrick': 'ABX.TO',
                'waste connections': 'WCN.TO',
                'canadian national': 'CNR.TO',
                'canadian pacific': 'CP.TO'
            }
            
            headline_lower = headline.lower()
            
            for company, symbol in company_mapping.items():
                if company in headline_lower:
                    relevant_symbols.append(symbol)
            
            # If no specific company mentioned, check for sector keywords
            if not relevant_symbols:
                sector_keywords = {
                    'banking': ['RY.TO', 'TD.TO'],
                    'energy': ['SU.TO', 'ENB.TO', 'TRP.TO'],
                    'mining': ['ABX.TO'],
                    'railway': ['CNR.TO', 'CP.TO'],
                    'technology': ['SHOP.TO']
                }
                
                for sector, symbols in sector_keywords.items():
                    if sector in headline_lower:
                        relevant_symbols.extend(symbols)
            
            return relevant_symbols[:3]  # Limit to top 3 relevant symbols
            
        except Exception as e:
            logger.error(f"Failed to find relevant symbols: {e}")
            return []
    
    def _create_news_signal(self, symbol: str, news_item: Dict, sentiment_score: float, market_data: Dict) -> Optional[NewsSignal]:
        """Create trading signal from news and market data"""
        try:
            current_price = market_data.get('close', 0)
            volume = market_data.get('volume', 0)
            
            if current_price <= 0:
                return None
            
            # Check volume surge
            volume_ratio = self._calculate_volume_ratio(symbol, volume)
            if volume_ratio < self.volume_surge:
                return None
            
            # Determine trade direction
            side = 'BUY' if sentiment_score > 0 else 'SELL'
            
            # Calculate signal strength
            strength = min(abs(sentiment_score), 1.0)
            
            # Calculate confidence based on volume and sentiment
            confidence = (strength + min(volume_ratio / self.volume_surge, 1.0)) / 2
            
            return NewsSignal(
                symbol=symbol,
                side=side,
                strength=strength,
                entry_price=current_price,
                stop_loss=self._calculate_stop_loss(current_price, side),
                take_profit=self._calculate_take_profit(current_price, side),
                confidence=confidence,
                news_source=news_item.get('source', ''),
                news_headline=news_item.get('title', ''),
                sentiment_score=sentiment_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to create news signal: {e}")
            return None
    
    def _calculate_volume_ratio(self, symbol: str, current_volume: int) -> float:
        """Calculate volume ratio vs average"""
        try:
            # Mock average volume calculation
            avg_volume = 1500000  # Mock average volume for news-sensitive stocks
            
            if avg_volume > 0:
                return current_volume / avg_volume
            return 1.0
            
        except Exception as e:
            logger.error(f"Failed to calculate volume ratio: {e}")
            return 1.0
    
    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side == 'BUY':
            return entry_price * (1 - self.stop_loss / 100)
        else:
            return entry_price * (1 + self.stop_loss / 100)
    
    def _calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side == 'BUY':
            return entry_price * (1 + self.take_profit / 100)
        else:
            return entry_price * (1 - self.take_profit / 100)
    
    def should_exit_position(self, symbol: str, current_price: float, entry_time: datetime) -> bool:
        """Check if position should be exited"""
        try:
            # Check holding time
            holding_time = datetime.now() - entry_time
            if holding_time.total_seconds() > (self.max_holding_time * 60):
                logger.info(f"Exiting {symbol} due to max holding time")
                return True
            
            # Check stop loss and take profit
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                side = position['side']
                
                if side == 'BUY':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        return True
                else:
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check exit conditions for {symbol}: {e}")
            return False
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status"""
        return {
            'enabled': self.enabled,
            'allocation': self.allocation,
            'max_leverage': self.max_leverage,
            'active_positions': len(self.active_positions),
            'signals_generated': len(self.signal_history),
            'instruments': self.instruments,
            'parameters': {
                'sentiment_threshold': self.sentiment_threshold,
                'volatility_spike': self.volatility_spike,
                'volume_surge': self.volume_surge,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'max_holding_time': self.max_holding_time
            }
        }

# Test the strategy
if __name__ == "__main__":
    config = {
        'enabled': True,
        'allocation': 0.20,
        'max_leverage': 1.5,
        'instruments': ['SU.TO', 'ENB.TO', 'ABX.TO'],
        'signals': {
            'news_sentiment_threshold': 0.7,
            'volatility_spike': 1.5,
            'volume_surge': 3.0
        },
        'risk': {
            'stop_loss': 0.5,
            'take_profit': 1.0,
            'max_holding_time': 45
        }
    }
    
    strategy = NewsVolatilityStrategy(config)
    
    # Test with mock news data
    mock_news = [
        {
            'title': 'Suncor Energy reports strong earnings beat and positive outlook',
            'source': 'financialpost',
            'timestamp': datetime.now()
        }
    ]
    
    mock_market_data = {
        'SU.TO': {
            'close': 45.50,
            'volume': 5000000
        }
    }
    
    signals = strategy.analyze_news_and_volatility(mock_news, mock_market_data)
    print(f"Generated {len(signals)} news signals")
    
    status = strategy.get_strategy_status()
    print(f"Strategy status: {status}")
