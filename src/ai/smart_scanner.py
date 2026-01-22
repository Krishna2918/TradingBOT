"""
Multi-Tier Smart Scanner for Intelligent Stock Monitoring

Implements a 4-tier priority system:
- Tier 1 (Hot List): 30 stocks scanned every 5 seconds (high-priority movers)
- Tier 2 (Core): 60 stocks scanned every 30 seconds (blue chips, high liquidity)
- Tier 3 (Extended): 200 stocks scanned every 2 minutes (mid/small caps)
- Tier 4 (Discovery): Full universe scanned every 30 minutes (find new opportunities)

Auto-promotes/demotes stocks based on real-time signals:
- Volume spikes, price movements, news sentiment, options activity
"""

import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class SmartScanner:
    """
    Intelligent stock scanner with dynamic tier-based prioritization
    """
    
    def __init__(self, full_universe: List[str]):
        """
        Initialize scanner with full stock universe
        
        Args:
            full_universe: Complete list of all stocks to monitor
        """
        self.full_universe = full_universe
        
        # Tier definitions
        self.tiers: Dict[int, Dict[str, Any]] = {
            1: {'name': 'Hot List', 'max_size': 30, 'interval_seconds': 5, 'stocks': set()},
            2: {'name': 'Core Universe', 'max_size': 60, 'interval_seconds': 30, 'stocks': set()},
            3: {'name': 'Extended Watch', 'max_size': 200, 'interval_seconds': 120, 'stocks': set()},
            4: {'name': 'Discovery', 'max_size': 99999, 'interval_seconds': 1800, 'stocks': set()},
        }
        
        # Initialize with default assignments
        self._initialize_default_tiers()
        
        # Tracking metrics for promotion/demotion logic
        self.stock_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'volume_avg_20d': 0.0,
            'volume_current': 0.0,
            'price_change_1h': 0.0,
            'price_change_1d': 0.0,
            'news_sentiment': 0.0,
            'last_update': None,
            'promotion_count': 0,
            'tier_entry_time': None,
        })
        
        # Last scan times per tier
        self.last_scan_times: Dict[int, datetime] = {}
        
        logger.info(f"Smart Scanner initialized with {len(full_universe)} stocks")
        logger.info(f"  Tier 1 (Hot List): {len(self.tiers[1]['stocks'])} stocks @ {self.tiers[1]['interval_seconds']}s")
        logger.info(f"  Tier 2 (Core): {len(self.tiers[2]['stocks'])} stocks @ {self.tiers[2]['interval_seconds']}s")
        logger.info(f"  Tier 3 (Extended): {len(self.tiers[3]['stocks'])} stocks @ {self.tiers[3]['interval_seconds']}s")
        logger.info(f"  Tier 4 (Discovery): {len(self.tiers[4]['stocks'])} stocks @ {self.tiers[4]['interval_seconds']}s")
    
    def _initialize_default_tiers(self):
        """Initialize tiers with sensible defaults based on stock characteristics"""
        
        # Tier 4 (Discovery) gets everything by default
        self.tiers[4]['stocks'] = set(self.full_universe)
        
        # Tier 2 (Core) - Blue chips and high liquidity stocks
        core_patterns = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',  # Big 5 banks
                        'CNQ.TO', 'SU.TO', 'IMO.TO', 'CVE.TO',  # Energy giants
                        'ENB.TO', 'TRP.TO', 'FTS.TO',  # Pipelines
                        'SHOP.TO', 'BB.TO', 'LSPD.TO',  # Tech
                        'CP.TO', 'CNR.TO',  # Railroads
                        'BCE.TO', 'T.TO', 'RCI-B.TO',  # Telecoms
                        'MFC.TO', 'SLF.TO',  # Insurance
                        'ABX.TO', 'WPM.TO', 'FNV.TO',  # Miners
                        'XIU.TO', 'XIC.TO', 'VFV.TO', 'ZCN.TO']  # ETFs
        
        for stock in core_patterns:
            if stock in self.full_universe:
                self.tiers[2]['stocks'].add(stock)
                self.tiers[4]['stocks'].discard(stock)
        
        # Tier 3 (Extended) - Mid-caps, cannabis, crypto
        extended_patterns = []
        for stock in self.full_universe:
            # Cannabis stocks
            if any(x in stock for x in ['WEED', 'ACB', 'TLRY', 'HEXO', 'OGI', 'CRON', 'VFF']):
                extended_patterns.append(stock)
            # Crypto stocks
            elif any(x in stock for x in ['HUT', 'BITF', 'HIVE', 'DM']):
                extended_patterns.append(stock)
            # Healthcare/biotech
            elif any(x in stock for x in ['WELL', 'DOC', 'QIPT', 'PHM']):
                extended_patterns.append(stock)
        
        for stock in extended_patterns[:200]:  # Cap at 200
            if stock in self.full_universe and stock not in self.tiers[2]['stocks']:
                self.tiers[3]['stocks'].add(stock)
                self.tiers[4]['stocks'].discard(stock)
        
        # Tier 1 (Hot List) starts empty - populated dynamically by promotions
        logger.info("Default tier assignments complete")
    
    def get_stocks_to_scan(self) -> Dict[int, List[str]]:
        """
        Returns which stocks to scan for each tier based on time since last scan
        
        Returns:
            Dict mapping tier number to list of stocks to scan
        """
        now = datetime.now()
        to_scan = {}
        
        for tier_num, tier_config in self.tiers.items():
            last_scan = self.last_scan_times.get(tier_num)
            interval = timedelta(seconds=tier_config['interval_seconds'])
            
            # Check if it's time to scan this tier
            if last_scan is None or (now - last_scan) >= interval:
                to_scan[tier_num] = list(tier_config['stocks'])
                self.last_scan_times[tier_num] = now
        
        return to_scan
    
    def update_stock_metrics(self, symbol: str, metrics: Dict[str, Any]):
        """
        Update metrics for a stock (called after fetching live data)
        
        Args:
            symbol: Stock symbol
            metrics: Dictionary with keys:
                - volume_current: Current volume
                - volume_avg_20d: 20-day average volume
                - price_change_1h: 1-hour price change %
                - price_change_1d: 1-day price change %
                - news_sentiment: Sentiment score (-1 to 1)
        """
        self.stock_metrics[symbol].update(metrics)
        self.stock_metrics[symbol]['last_update'] = datetime.now()
    
    def evaluate_promotions_demotions(self) -> Dict[str, Any]:
        """
        Evaluate all stocks and determine which should be promoted/demoted
        
        Returns:
            Dict with 'promoted' and 'demoted' lists
        """
        now = datetime.now()
        promoted = []
        demoted = []
        
        # PROMOTION RULES (Tier 3/4 → Tier 1)
        for tier_num in [3, 4]:
            for symbol in list(self.tiers[tier_num]['stocks']):
                metrics = self.stock_metrics.get(symbol)
                if not metrics or not metrics.get('last_update'):
                    continue
                
                # Calculate promotion score
                score = 0.0
                reasons = []
                
                # Volume spike (>3x average)
                vol_ratio = metrics.get('volume_current', 0) / max(metrics.get('volume_avg_20d', 1), 1)
                if vol_ratio >= 3.0:
                    score += 40
                    reasons.append(f"volume_spike_{vol_ratio:.1f}x")
                
                # Price move (>5% intraday)
                price_change = abs(metrics.get('price_change_1h', 0))
                if price_change >= 5.0:
                    score += 30
                    reasons.append(f"price_move_{price_change:.1f}%")
                
                # News sentiment spike (>0.7)
                sentiment = metrics.get('news_sentiment', 0)
                if abs(sentiment) >= 0.7:
                    score += 20
                    reasons.append(f"sentiment_{sentiment:.2f}")
                
                # Strong daily move (>10%)
                daily_change = abs(metrics.get('price_change_1d', 0))
                if daily_change >= 10.0:
                    score += 10
                    reasons.append(f"daily_move_{daily_change:.1f}%")
                
                # Promote if score >= 50
                if score >= 50 and len(self.tiers[1]['stocks']) < self.tiers[1]['max_size']:
                    self._promote_stock(symbol, tier_num, 1, reasons)
                    promoted.append({'symbol': symbol, 'from_tier': tier_num, 'to_tier': 1, 'score': score, 'reasons': reasons})
        
        # DEMOTION RULES (Tier 1 → Tier 3)
        for symbol in list(self.tiers[1]['stocks']):
            metrics = self.stock_metrics.get(symbol)
            if not metrics:
                continue
            
            entry_time = metrics.get('tier_entry_time')
            if not entry_time:
                continue
            
            # Time in Tier 1
            time_in_tier = (now - entry_time).total_seconds() / 3600  # hours
            
            # Demote if:
            # 1. No significant moves for 2+ hours
            # 2. Volume returned to normal (<1.5x average)
            # 3. Sentiment cooled off (<0.3)
            
            if time_in_tier >= 2.0:
                vol_ratio = metrics.get('volume_current', 0) / max(metrics.get('volume_avg_20d', 1), 1)
                price_change = abs(metrics.get('price_change_1h', 0))
                sentiment = abs(metrics.get('news_sentiment', 0))
                
                if vol_ratio < 1.5 and price_change < 2.0 and sentiment < 0.3:
                    self._demote_stock(symbol, 1, 3, ["cooled_off", f"time_in_tier={time_in_tier:.1f}h"])
                    demoted.append({'symbol': symbol, 'from_tier': 1, 'to_tier': 3, 'reason': 'cooled_off'})
        
        if promoted:
            logger.info(f"🔥 PROMOTED {len(promoted)} stocks to Tier 1 (Hot List)")
            for p in promoted:
                logger.info(f"   {p['symbol']}: {' + '.join(p['reasons'])}")
        
        if demoted:
            logger.info(f"❄️  DEMOTED {len(demoted)} stocks from Tier 1")
            for d in demoted:
                logger.info(f"   {d['symbol']}: {d['reason']}")
        
        return {'promoted': promoted, 'demoted': demoted}
    
    def _promote_stock(self, symbol: str, from_tier: int, to_tier: int, reasons: List[str]):
        """Move stock to higher priority tier"""
        if symbol in self.tiers[from_tier]['stocks']:
            self.tiers[from_tier]['stocks'].discard(symbol)
        
        self.tiers[to_tier]['stocks'].add(symbol)
        self.stock_metrics[symbol]['tier_entry_time'] = datetime.now()
        self.stock_metrics[symbol]['promotion_count'] += 1
        
        logger.debug(f"📈 {symbol} promoted: Tier {from_tier} → Tier {to_tier} ({', '.join(reasons)})")
    
    def _demote_stock(self, symbol: str, from_tier: int, to_tier: int, reasons: List[str]):
        """Move stock to lower priority tier"""
        if symbol in self.tiers[from_tier]['stocks']:
            self.tiers[from_tier]['stocks'].discard(symbol)
        
        self.tiers[to_tier]['stocks'].add(symbol)
        self.stock_metrics[symbol]['tier_entry_time'] = datetime.now()
        
        logger.debug(f"📉 {symbol} demoted: Tier {from_tier} → Tier {to_tier} ({', '.join(reasons)})")
    
    def get_tier_for_stock(self, symbol: str) -> int:
        """Get the current tier number for a stock"""
        for tier_num, tier_config in self.tiers.items():
            if symbol in tier_config['stocks']:
                return tier_num
        return 4  # Default to discovery tier
    
    def get_hot_list(self) -> List[str]:
        """Get current Tier 1 (Hot List) stocks"""
        return list(self.tiers[1]['stocks'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        return {
            'total_universe': len(self.full_universe),
            'tier_1_count': len(self.tiers[1]['stocks']),
            'tier_2_count': len(self.tiers[2]['stocks']),
            'tier_3_count': len(self.tiers[3]['stocks']),
            'tier_4_count': len(self.tiers[4]['stocks']),
            'hot_list': list(self.tiers[1]['stocks'])[:10],  # Top 10
            'last_scan_times': {k: v.isoformat() if v else None for k, v in self.last_scan_times.items()},
        }


def create_smart_scanner(universe: Optional[List[str]] = None) -> SmartScanner:
    """
    Factory function to create a SmartScanner instance
    
    Args:
        universe: List of stocks (defaults to full TSX/TSXV universe)
    
    Returns:
        Configured SmartScanner instance
    """
    if universe is None:
        from src.data_pipeline.tsx_stock_universe import get_full_tsx_universe
        universe = get_full_tsx_universe()
    
    scanner = SmartScanner(universe)
    logger.info(f"Smart Scanner created with {len(universe)} stocks")
    return scanner

