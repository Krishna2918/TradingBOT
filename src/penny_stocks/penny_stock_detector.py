"""
Penny Stock Detection & Analysis Module

Detects and analyzes penny stocks with:
- Abnormal volume detection
- Sentiment analysis
- Liquidity filtering
- Dynamic position sizing
- RL feedback integration
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PennyStockProfile:
    """Penny stock profile data"""
    symbol: str
    price: float
    market_cap: float
    avg_volume: float
    current_volume: float
    volume_ratio: float
    volatility: float
    liquidity_score: float
    sentiment_score: float
    risk_level: str
    is_tradeable: bool
    max_position_size: float
    timestamp: datetime

class PennyStockDetector:
    """
    Penny Stock Detector
    
    Identifies and analyzes Canadian penny stocks (< $5 CAD)
    Focuses on TSXV listings with momentum potential
    
    Features:
    - Volume spike detection
    - Sentiment analysis
    - Liquidity filtering
    - Risk assessment
    - Dynamic position sizing
    """
    
    def __init__(
        self,
        price_threshold: float = 5.0,  # $5 CAD max
        min_volume_threshold: int = 50000,  # Minimum daily volume
        volume_spike_multiplier: float = 3.0,  # 3x average = spike
        min_liquidity_score: float = 0.3,  # 0-1 scale
        max_position_pct: float = 0.02  # 2% max position size
    ):
        self.price_threshold = price_threshold
        self.min_volume_threshold = min_volume_threshold
        self.volume_spike_multiplier = volume_spike_multiplier
        self.min_liquidity_score = min_liquidity_score
        self.max_position_pct = max_position_pct
        
        self.tracked_stocks: Dict[str, PennyStockProfile] = {}
        self.watchlist: List[str] = []
        self.blacklist: List[str] = []  # Illiquid or manipulated stocks
        
        logger.info(" Penny Stock Detector initialized")
    
    def is_penny_stock(self, symbol: str, price: float, exchange: str = None) -> bool:
        """
        Determine if a stock qualifies as a penny stock
        
        Args:
            symbol: Stock symbol
            price: Current price
            exchange: Exchange (TSX, TSXV, CSE)
        
        Returns:
            True if penny stock
        """
        
        # Price criteria
        if price >= self.price_threshold:
            return False
        
        # Exchange criteria (TSXV is prime penny stock exchange)
        if exchange and exchange.upper() in ['TSX', 'NYSE', 'NASDAQ']:
            return False  # Larger exchanges, less risky
        
        # Blacklist check
        if symbol in self.blacklist:
            logger.debug(f" {symbol} is blacklisted")
            return False
        
        return True
    
    def detect_volume_spike(
        self,
        symbol: str,
        current_volume: float,
        historical_volume: pd.Series
    ) -> Tuple[bool, float]:
        """
        Detect abnormal volume spikes
        
        Args:
            symbol: Stock symbol
            current_volume: Current trading volume
            historical_volume: Historical volume series
        
        Returns:
            (is_spike, volume_ratio)
        """
        
        if len(historical_volume) < 20:
            return False, 0.0
        
        # Calculate average volume
        avg_volume = historical_volume.mean()
        
        if avg_volume == 0:
            return False, 0.0
        
        # Calculate volume ratio
        volume_ratio = current_volume / avg_volume
        
        # Detect spike
        is_spike = volume_ratio >= self.volume_spike_multiplier
        
        if is_spike:
            logger.info(
                f" Volume spike detected for {symbol}: "
                f"{current_volume:,.0f} vs avg {avg_volume:,.0f} ({volume_ratio:.1f}x)"
            )
        
        return is_spike, volume_ratio
    
    def calculate_liquidity_score(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate liquidity score (0-1)
        
        Factors:
        - Average daily volume
        - Bid-ask spread
        - Price volatility
        - Trading frequency
        
        Returns:
            Liquidity score (0 = illiquid, 1 = highly liquid)
        """
        
        if len(data) < 20:
            return 0.0
        
        scores = []
        
        # 1. Volume score
        avg_volume = data['volume'].mean()
        if avg_volume >= 1000000:  # 1M+ shares
            volume_score = 1.0
        elif avg_volume >= 500000:  # 500K+
            volume_score = 0.8
        elif avg_volume >= 100000:  # 100K+
            volume_score = 0.5
        elif avg_volume >= 50000:  # 50K+ (minimum)
            volume_score = 0.3
        else:
            volume_score = 0.1
        
        scores.append(volume_score)
        
        # 2. Spread score (if available)
        if 'bid' in data.columns and 'ask' in data.columns:
            avg_spread = ((data['ask'] - data['bid']) / data['close']).mean()
            spread_score = max(0, 1 - (avg_spread * 10))  # Lower spread = better
            scores.append(spread_score)
        
        # 3. Trading frequency score
        non_zero_volume_days = (data['volume'] > 0).sum()
        frequency_score = non_zero_volume_days / len(data)
        scores.append(frequency_score)
        
        # 4. Volatility score (moderate volatility is good)
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        if 0.02 <= volatility <= 0.05:  # Sweet spot
            volatility_score = 1.0
        elif volatility < 0.02:  # Too low
            volatility_score = 0.5
        elif volatility > 0.10:  # Too high (risky)
            volatility_score = 0.3
        else:
            volatility_score = 0.7
        
        scores.append(volatility_score)
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2][:len(scores)]  # Match weights to scores length
        liquidity_score = np.average(scores, weights=weights)
        
        return liquidity_score
    
    def analyze_sentiment(
        self,
        symbol: str,
        news_data: List[Dict]
    ) -> Tuple[float, str]:
        """
        Analyze sentiment from news and social media
        
        Args:
            symbol: Stock symbol
            news_data: List of news articles/posts
        
        Returns:
            (sentiment_score, sentiment_label)
            Score: -1 (very negative) to +1 (very positive)
        """
        
        if not news_data:
            return 0.0, "neutral"
        
        sentiments = []
        
        for article in news_data:
            # Simple keyword-based sentiment (would use NLP in production)
            content = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            positive_keywords = ['surge', 'spike', 'rally', 'gain', 'profit', 'growth', 'bullish', 'breakthrough']
            negative_keywords = ['drop', 'fall', 'loss', 'decline', 'crash', 'bearish', 'concern', 'risk']
            
            pos_count = sum(1 for kw in positive_keywords if kw in content)
            neg_count = sum(1 for kw in negative_keywords if kw in content)
            
            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                sentiments.append(sentiment)
        
        if not sentiments:
            return 0.0, "neutral"
        
        avg_sentiment = np.mean(sentiments)
        
        # Classify
        if avg_sentiment >= 0.5:
            label = "very_positive"
        elif avg_sentiment >= 0.2:
            label = "positive"
        elif avg_sentiment >= -0.2:
            label = "neutral"
        elif avg_sentiment >= -0.5:
            label = "negative"
        else:
            label = "very_negative"
        
        return avg_sentiment, label
    
    def assess_risk_level(
        self,
        profile: PennyStockProfile
    ) -> str:
        """
        Assess overall risk level
        
        Returns:
            Risk level: low, medium, high, extreme
        """
        
        risk_score = 0
        
        # Price risk (lower price = higher risk)
        if profile.price < 0.50:
            risk_score += 3
        elif profile.price < 1.00:
            risk_score += 2
        elif profile.price < 2.00:
            risk_score += 1
        
        # Liquidity risk
        if profile.liquidity_score < 0.3:
            risk_score += 3
        elif profile.liquidity_score < 0.5:
            risk_score += 2
        elif profile.liquidity_score < 0.7:
            risk_score += 1
        
        # Volatility risk
        if profile.volatility > 0.10:
            risk_score += 2
        elif profile.volatility > 0.07:
            risk_score += 1
        
        # Volume risk
        if profile.avg_volume < 50000:
            risk_score += 2
        elif profile.avg_volume < 100000:
            risk_score += 1
        
        # Classify
        if risk_score >= 7:
            return "extreme"
        elif risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
    
    def calculate_position_size(
        self,
        profile: PennyStockProfile,
        total_capital: float
    ) -> float:
        """
        Calculate dynamic position size based on risk
        
        Args:
            profile: Penny stock profile
            total_capital: Total available capital
        
        Returns:
            Maximum position size in CAD
        """
        
        # Base position size (2% max for penny stocks)
        base_size = total_capital * self.max_position_pct
        
        # Adjust by liquidity
        liquidity_multiplier = profile.liquidity_score
        
        # Adjust by risk
        risk_multipliers = {
            'low': 1.0,
            'medium': 0.7,
            'high': 0.4,
            'extreme': 0.2
        }
        risk_multiplier = risk_multipliers.get(profile.risk_level, 0.5)
        
        # Adjust by sentiment
        sentiment_multiplier = max(0.5, min(1.2, 1.0 + (profile.sentiment_score * 0.2)))
        
        # Calculate final position size
        position_size = base_size * liquidity_multiplier * risk_multiplier * sentiment_multiplier
        
        # Cap at liquidity (don't exceed 5% of average daily volume value)
        max_liquidity_size = (profile.avg_volume * profile.price * 0.05)
        position_size = min(position_size, max_liquidity_size)
        
        return position_size
    
    def analyze_penny_stock(
        self,
        symbol: str,
        data: pd.DataFrame,
        news_data: List[Dict] = None
    ) -> Optional[PennyStockProfile]:
        """
        Comprehensive penny stock analysis
        
        Args:
            symbol: Stock symbol
            data: OHLCV data
            news_data: News articles
        
        Returns:
            PennyStockProfile if tradeable, None otherwise
        """
        
        if len(data) < 20:
            logger.warning(f" Insufficient data for {symbol}")
            return None
        
        try:
            # Get current metrics
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1]
            historical_volume = data['volume'].iloc[:-1]
            avg_volume = historical_volume.mean()
            
            # Check if penny stock
            if not self.is_penny_stock(symbol, current_price):
                return None
            
            # Volume spike detection
            is_spike, volume_ratio = self.detect_volume_spike(
                symbol, current_volume, historical_volume
            )
            
            # Liquidity score
            liquidity_score = self.calculate_liquidity_score(symbol, data)
            
            # Sentiment analysis
            sentiment_score, sentiment_label = self.analyze_sentiment(
                symbol, news_data or []
            )
            
            # Volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Market cap estimation (if available)
            market_cap = 0  # Would calculate from shares outstanding
            
            # Create profile
            profile = PennyStockProfile(
                symbol=symbol,
                price=current_price,
                market_cap=market_cap,
                avg_volume=avg_volume,
                current_volume=current_volume,
                volume_ratio=volume_ratio,
                volatility=volatility,
                liquidity_score=liquidity_score,
                sentiment_score=sentiment_score,
                risk_level="",  # Will be set below
                is_tradeable=False,  # Will be determined
                max_position_size=0.0,  # Will be calculated
                timestamp=datetime.now()
            )
            
            # Assess risk
            profile.risk_level = self.assess_risk_level(profile)
            
            # Determine if tradeable
            profile.is_tradeable = (
                liquidity_score >= self.min_liquidity_score and
                avg_volume >= self.min_volume_threshold and
                profile.risk_level != "extreme" and
                symbol not in self.blacklist
            )
            
            # Log analysis
            logger.info(
                f" {symbol}: ${current_price:.2f} | "
                f"Volume: {volume_ratio:.1f}x | "
                f"Liquidity: {liquidity_score:.2f} | "
                f"Sentiment: {sentiment_label} | "
                f"Risk: {profile.risk_level} | "
                f"Tradeable: {profile.is_tradeable}"
            )
            
            # Store in tracking
            self.tracked_stocks[symbol] = profile
            
            # Add to watchlist if promising
            if profile.is_tradeable and is_spike and sentiment_score > 0:
                if symbol not in self.watchlist:
                    self.watchlist.append(symbol)
                    logger.info(f" Added {symbol} to penny stock watchlist")
            
            return profile
            
        except Exception as e:
            logger.error(f" Failed to analyze {symbol}: {e}")
            return None
    
    def get_watchlist(self) -> List[PennyStockProfile]:
        """Get current penny stock watchlist"""
        return [
            self.tracked_stocks[symbol]
            for symbol in self.watchlist
            if symbol in self.tracked_stocks
        ]
    
    def blacklist_stock(self, symbol: str, reason: str = ""):
        """Add stock to blacklist"""
        if symbol not in self.blacklist:
            self.blacklist.append(symbol)
            logger.warning(f" Blacklisted {symbol}: {reason}")
    
    def get_summary(self) -> Dict:
        """Get penny stock detector summary"""
        tradeable_count = sum(
            1 for p in self.tracked_stocks.values() if p.is_tradeable
        )
        
        return {
            'tracked_stocks': len(self.tracked_stocks),
            'tradeable_stocks': tradeable_count,
            'watchlist_size': len(self.watchlist),
            'blacklist_size': len(self.blacklist),
            'timestamp': datetime.now().isoformat()
        }

# Global detector instance
_detector_instance = None

def get_penny_stock_detector() -> PennyStockDetector:
    """Get global penny stock detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PennyStockDetector()
    return _detector_instance

