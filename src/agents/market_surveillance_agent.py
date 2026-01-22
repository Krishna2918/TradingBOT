"""
Market Surveillance Agent - Dynamic Top 101 Stock Discovery

Monitors the entire Canadian market (TSX + TSXV) after market close to identify
the top 101 performing stocks based on multiple criteria including penny stocks,
volume spikes, price momentum, and emerging opportunities.
"""

import logging
import time
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from ..data_collection.market_calendar import MarketCalendar
from ..data_collection.enhanced_collectors import MultiSourceDataCollector
from ..data_collection.storage_manager import StorageManager

logger = logging.getLogger(__name__)

@dataclass
class StockMetrics:
    """Comprehensive stock performance metrics"""
    symbol: str
    name: str
    price: float
    volume: int
    market_cap: float
    
    # Performance metrics
    daily_return: float
    weekly_return: float
    monthly_return: float
    volatility: float
    
    # Volume metrics
    volume_ratio: float  # Current volume vs average
    volume_spike: bool
    
    # Technical metrics
    rsi: float
    momentum_score: float
    breakout_score: float
    
    # Classification
    is_penny_stock: bool
    sector: str
    exchange: str
    
    # Overall ranking score
    composite_score: float

class MarketSurveillanceAgent:
    """Intelligent agent that monitors entire Canadian market"""
    
    def __init__(self):
        self.market_calendar = MarketCalendar()
        self.data_collector = MultiSourceDataCollector()
        self.storage_manager = StorageManager()
        
        # Surveillance configuration
        self.config = {
            "scan_frequency_hours": 4,  # Scan every 4 hours when markets closed
            "top_stocks_count": 101,
            "penny_stock_threshold": 5.0,  # Under $5 = penny stock
            "min_volume": 10000,  # Minimum daily volume
            "min_market_cap": 1000000,  # $1M minimum market cap
            "volume_spike_threshold": 3.0,  # 3x average volume = spike
            "momentum_lookback_days": 20,
            "max_concurrent_scans": 1,  # Sequential processing for rate limiting
            "alpha_vantage_calls_per_minute": 74,  # Stay under 75/minute limit
            "batch_delay_seconds": 1.0  # Delay between calls
        }
        
        # Canadian stock universe (will be dynamically updated)
        self.canadian_universe = self._get_initial_canadian_universe()
        
        # Current top 101 stocks
        self.current_top_101 = []
        self.last_scan_time = None
        self.scan_results_history = []
        
        logger.info("🕵️ Market Surveillance Agent initialized")
        logger.info(f"📊 Monitoring {len(self.canadian_universe)} Canadian stocks")
    
    def _get_initial_canadian_universe(self) -> List[str]:
        """Get comprehensive list of Canadian stocks to monitor"""
        
        # Start with known major stocks
        major_stocks = [
            # TSX 60 Core
            "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
            "CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO", "CVE.TO",
            "SHOP.TO", "CSU.TO", "LSPD.TO", "WCN.TO", "WSP.TO",
            "AEM.TO", "K.TO", "NTR.TO", "FNV.TO", "ATD.TO",
            
            # Growth & Tech
            "HUT.TO", "BITF.TO", "HIVE.TO", "DMGI.TO", "MMED.TO",
            "NUMI.TO", "CYBN.TO", "FTRP.TO", "DCBO.TO", "PYR.TO",
            "HPQ.TO", "NOU.TO", "TOI.TO", "WELL.TO", "CTS.TO",
            
            # Penny Stock Candidates
            "PMET.TO", "LAC.TO", "CRE.TO", "FL.TO", "WEED.TO",
            "ACB.TO", "HEXO.TO", "OGI.TO", "FOOD.TO",
            
            # Utilities & REITs
            "H.TO", "CU.TO", "AQN.TO", "BEP-UN.TO", "BIP-UN.TO",
            "CAR-UN.TO", "REI-UN.TO", "HR-UN.TO", "CHP-UN.TO",
            
            # Mining & Resources
            "FM.TO", "TKO.TO", "CS.TO", "HBM.TO", "LUN.TO",
            "ABX.TO", "WPM.TO", "KL.TO", "ELD.TO", "YRI.TO",
            
            # Additional high-volume candidates
            "AC.TO", "CNR.TO", "CP.TO", "BCE.TO", "T.TO", "RCI-B.TO"
        ]
        
        # Add more symbols dynamically (this would be expanded with market screeners)
        additional_symbols = self._discover_additional_symbols()
        
        all_symbols = list(set(major_stocks + additional_symbols))
        
        logger.info(f"📈 Initial universe: {len(all_symbols)} Canadian stocks")
        return all_symbols
    
    def _discover_additional_symbols(self) -> List[str]:
        """Discover additional Canadian symbols through various methods"""
        
        # This would integrate with:
        # 1. TSX/TSXV official listings
        # 2. Financial data providers
        # 3. Screener APIs
        # 4. News mentions and social media
        
        # For now, return additional known symbols
        additional = [
            # More TSXV growth stocks
            "GDNP.V", "NTAR.TO", "QUIS.V", "RECO.V", "SEDI.V",
            "TGOD.TO", "VLNS.TO", "WLLW.TO", "XTRA.V", "ZENA.TO",
            
            # More mining and resources
            "ARTG.TO", "BTO.TO", "CG.TO", "DNG.TO", "EQX.TO",
            "FSZ.TO", "GDX.TO", "HEO.TO", "IMG.TO", "JAG.TO",
            
            # More tech and innovation
            "BB.TO", "DOCN.TO", "GOOS.TO", "NVEI.TO", "REAL.TO",
            "TFII.TO", "TOY.TO", "XPEL.TO", "ZYME.TO"
        ]
        
        return additional
    
    def scan_entire_market(self) -> Dict[str, StockMetrics]:
        """Scan entire Canadian market and calculate performance metrics"""
        
        logger.info(f"🔍 Starting market scan of {len(self.canadian_universe)} stocks")
        
        # Check if we should scan now
        if not self._should_scan_now():
            logger.info("⏸️ Market scan postponed - not optimal timing")
            return {}
        
        scan_results = {}
        failed_symbols = []
        
        # Sequential scanning with Alpha Vantage rate limiting (74 calls/minute)
        calls_made = 0
        start_time = time.time()
        
        for symbol in self.canadian_universe:
            try:
                # Rate limiting: ensure we don't exceed 74 calls per minute
                if calls_made >= self.config["alpha_vantage_calls_per_minute"]:
                    elapsed = time.time() - start_time
                    if elapsed < 60:
                        sleep_time = 60 - elapsed
                        logger.info(f"⏳ Rate limiting: sleeping {sleep_time:.1f}s to respect Alpha Vantage limits")
                        time.sleep(sleep_time)
                    calls_made = 0
                    start_time = time.time()
                
                # Scan the stock
                metrics = self._scan_single_stock(symbol)
                calls_made += 1
                
                if metrics:
                    scan_results[symbol] = metrics
                    logger.debug(f"✅ Scanned {symbol}: Score {metrics.composite_score:.3f}")
                else:
                    failed_symbols.append(symbol)
                    logger.debug(f"⚠️ No data for {symbol}")
                
                # Small delay between calls
                time.sleep(self.config["batch_delay_seconds"])
                
                # Progress update every 10 stocks
                if len(scan_results) % 10 == 0:
                    logger.info(f"📊 Progress: {len(scan_results)} stocks analyzed, {len(failed_symbols)} failed")
                        
            except Exception as e:
                failed_symbols.append(symbol)
                logger.warning(f"❌ Error scanning {symbol}: {e}")
                calls_made += 1
        
        logger.info(f"✅ Market scan complete: {len(scan_results)} stocks analyzed, {len(failed_symbols)} failed")
        
        # Save scan results
        self._save_scan_results(scan_results)
        
        return scan_results
    
    def _scan_single_stock(self, symbol: str) -> Optional[StockMetrics]:
        """Scan a single stock and calculate all metrics"""
        
        try:
            # Use Alpha Vantage primarily for better Canadian stock coverage
            data, source = self.data_collector.alpha_vantage_collector.fetch_daily_data(symbol)
            
            # If Alpha Vantage fails, fallback to Yahoo Finance
            if data is None or data.empty:
                logger.debug(f"Alpha Vantage failed for {symbol}, trying Yahoo Finance")
                data, source = self.data_collector.yahoo_collector.fetch_data(symbol, period="1mo", interval="1d")
            
            if data is None or data.empty or len(data) < 5:
                return None
            
            # Calculate basic metrics
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            # Performance metrics
            daily_return = data['Close'].pct_change().iloc[-1] if len(data) > 1 else 0.0
            weekly_return = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) if len(data) >= 5 else 0.0
            monthly_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) if len(data) > 0 else 0.0
            
            # Volatility (20-day)
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
            
            # Volume analysis
            avg_volume = data['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0
            volume_spike = volume_ratio > self.config["volume_spike_threshold"]
            
            # Technical indicators
            rsi = self._calculate_rsi(data['Close'])
            momentum_score = self._calculate_momentum_score(data)
            breakout_score = self._calculate_breakout_score(data)
            
            # Classification
            is_penny_stock = current_price < self.config["penny_stock_threshold"]
            market_cap = self._estimate_market_cap(symbol, current_price, current_volume)
            sector = self._classify_sector(symbol)
            exchange = "TSXV" if ".V" in symbol else "TSX"
            
            # Composite scoring
            composite_score = self._calculate_composite_score(
                daily_return, weekly_return, monthly_return, volatility,
                volume_ratio, rsi, momentum_score, breakout_score,
                is_penny_stock, market_cap
            )
            
            return StockMetrics(
                symbol=symbol,
                name=self._get_company_name(symbol),
                price=current_price,
                volume=current_volume,
                market_cap=market_cap,
                daily_return=daily_return,
                weekly_return=weekly_return,
                monthly_return=monthly_return,
                volatility=volatility,
                volume_ratio=volume_ratio,
                volume_spike=volume_spike,
                rsi=rsi,
                momentum_score=momentum_score,
                breakout_score=breakout_score,
                is_penny_stock=is_penny_stock,
                sector=sector,
                exchange=exchange,
                composite_score=composite_score
            )
            
        except Exception as e:
            logger.debug(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score based on price and volume trends"""
        if len(data) < 10:
            return 0.0
        
        # Price momentum (recent vs older)
        recent_avg = data['Close'].tail(5).mean()
        older_avg = data['Close'].head(5).mean()
        price_momentum = (recent_avg / older_avg - 1) if older_avg > 0 else 0.0
        
        # Volume momentum
        recent_vol = data['Volume'].tail(5).mean()
        older_vol = data['Volume'].head(5).mean()
        volume_momentum = (recent_vol / older_vol - 1) if older_vol > 0 else 0.0
        
        # Combined momentum score
        momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3)
        
        return max(-1.0, min(1.0, momentum_score))  # Clamp between -1 and 1
    
    def _calculate_breakout_score(self, data: pd.DataFrame) -> float:
        """Calculate breakout potential score"""
        if len(data) < 20:
            return 0.0
        
        # Check if price is near recent highs
        recent_high = data['High'].tail(20).max()
        current_price = data['Close'].iloc[-1]
        high_proximity = current_price / recent_high if recent_high > 0 else 0.0
        
        # Check volume confirmation
        avg_volume = data['Volume'].tail(20).mean()
        recent_volume = data['Volume'].tail(3).mean()
        volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 0.0
        
        # Breakout score
        breakout_score = (high_proximity * 0.6 + min(volume_confirmation, 2.0) * 0.4)
        
        return max(0.0, min(2.0, breakout_score))
    
    def _calculate_composite_score(self, daily_ret: float, weekly_ret: float, monthly_ret: float,
                                 volatility: float, volume_ratio: float, rsi: float,
                                 momentum: float, breakout: float, is_penny: bool, market_cap: float) -> float:
        """Calculate composite ranking score"""
        
        score = 0.0
        
        # Performance component (40% weight)
        performance_score = (
            daily_ret * 0.3 +      # Recent performance
            weekly_ret * 0.4 +     # Short-term trend
            monthly_ret * 0.3      # Medium-term trend
        )
        score += performance_score * 0.4
        
        # Volume component (25% weight)
        volume_score = min(volume_ratio / 5.0, 1.0)  # Normalize volume ratio
        score += volume_score * 0.25
        
        # Technical component (20% weight)
        technical_score = (momentum * 0.5 + breakout * 0.3 + (rsi - 50) / 50 * 0.2)
        score += technical_score * 0.2
        
        # Opportunity component (15% weight)
        opportunity_score = 0.0
        if is_penny:
            opportunity_score += 0.3  # Penny stocks get opportunity bonus
        if market_cap < 100000000:  # Small cap bonus
            opportunity_score += 0.2
        if volatility > 0.3:  # High volatility bonus (more trading opportunities)
            opportunity_score += 0.1
        
        score += opportunity_score * 0.15
        
        return score
    
    def _estimate_market_cap(self, symbol: str, price: float, volume: int) -> float:
        """Estimate market cap (simplified)"""
        # This is a rough estimate - in production would use actual shares outstanding
        # For now, use volume as proxy for company size
        if volume > 1000000:
            return price * volume * 0.1  # Large volume stocks
        elif volume > 100000:
            return price * volume * 0.5  # Medium volume stocks
        else:
            return price * volume * 1.0  # Small volume stocks
    
    def _classify_sector(self, symbol: str) -> str:
        """Classify stock sector based on symbol patterns"""
        
        # Banking
        if symbol in ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO"]:
            return "Banking"
        
        # Energy
        elif symbol in ["CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO", "CVE.TO"]:
            return "Energy"
        
        # Technology
        elif symbol in ["SHOP.TO", "CSU.TO", "LSPD.TO", "BB.TO", "NVEI.TO"]:
            return "Technology"
        
        # Mining
        elif symbol in ["AEM.TO", "K.TO", "ABX.TO", "WPM.TO", "KL.TO"]:
            return "Mining"
        
        # Cannabis
        elif symbol in ["WEED.TO", "ACB.TO", "HEXO.TO", "OGI.TO"]:
            return "Cannabis"
        
        # Crypto/Blockchain
        elif symbol in ["HUT.TO", "BITF.TO", "HIVE.TO", "DMGI.TO"]:
            return "Crypto"
        
        else:
            return "Other"
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name (simplified)"""
        # In production, this would query company info
        # For now, return symbol without .TO/.V
        return symbol.replace(".TO", "").replace(".V", "")
    
    def _should_scan_now(self) -> bool:
        """Determine if we should scan the market now"""
        
        # Only scan when markets are closed
        if self.market_calendar.is_market_open_now():
            return False
        
        # Check if enough time has passed since last scan
        if self.last_scan_time:
            hours_since_scan = (datetime.now() - self.last_scan_time).total_seconds() / 3600
            if hours_since_scan < self.config["scan_frequency_hours"]:
                return False
        
        return True
    
    def update_top_101_stocks(self) -> Dict[str, any]:
        """Update the top 101 stocks based on current market scan"""
        
        logger.info("🔄 Updating top 101 stocks based on market surveillance")
        
        # Scan the market
        scan_results = self.scan_entire_market()
        
        if not scan_results:
            logger.warning("⚠️ No scan results available")
            return {"status": "failed", "reason": "no_scan_results"}
        
        # Rank all stocks by composite score
        ranked_stocks = sorted(
            scan_results.values(),
            key=lambda x: x.composite_score,
            reverse=True
        )
        
        # Select top 101
        new_top_101 = ranked_stocks[:self.config["top_stocks_count"]]
        
        # Analyze changes from previous top 101
        changes = self._analyze_changes(new_top_101)
        
        # Update current top 101
        self.current_top_101 = new_top_101
        self.last_scan_time = datetime.now()
        
        # Save updated list
        self._save_top_101_list(new_top_101, changes)
        
        # Generate report
        report = {
            "status": "success",
            "scan_time": self.last_scan_time.isoformat(),
            "total_scanned": len(scan_results),
            "top_101_updated": True,
            "changes": changes,
            "top_performers": [stock.symbol for stock in new_top_101[:10]],
            "penny_stock_count": sum(1 for stock in new_top_101 if stock.is_penny_stock),
            "sector_distribution": self._get_sector_distribution(new_top_101),
            "average_score": np.mean([stock.composite_score for stock in new_top_101])
        }
        
        logger.info(f"✅ Top 101 updated: {changes['new_entries']} new, {changes['dropped']} dropped")
        
        return report
    
    def _analyze_changes(self, new_top_101: List[StockMetrics]) -> Dict[str, any]:
        """Analyze changes in top 101 list"""
        
        new_symbols = {stock.symbol for stock in new_top_101}
        old_symbols = {stock.symbol for stock in self.current_top_101} if self.current_top_101 else set()
        
        new_entries = new_symbols - old_symbols
        dropped = old_symbols - new_symbols
        maintained = new_symbols & old_symbols
        
        return {
            "new_entries": len(new_entries),
            "dropped": len(dropped),
            "maintained": len(maintained),
            "new_symbols": list(new_entries),
            "dropped_symbols": list(dropped),
            "turnover_rate": len(new_entries) / 101 if new_entries else 0.0
        }
    
    def _get_sector_distribution(self, stocks: List[StockMetrics]) -> Dict[str, int]:
        """Get sector distribution of top stocks"""
        sector_counts = {}
        for stock in stocks:
            sector_counts[stock.sector] = sector_counts.get(stock.sector, 0) + 1
        return sector_counts
    
    def _save_top_101_list(self, top_stocks: List[StockMetrics], changes: Dict):
        """Save the updated top 101 list"""
        
        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "market_status": self.market_calendar.get_market_status(),
            "total_stocks": len(top_stocks),
            "changes": changes,
            "top_101_stocks": [asdict(stock) for stock in top_stocks],
            "summary_stats": {
                "avg_price": np.mean([stock.price for stock in top_stocks]),
                "avg_volume": np.mean([stock.volume for stock in top_stocks]),
                "avg_market_cap": np.mean([stock.market_cap for stock in top_stocks]),
                "penny_stock_count": sum(1 for stock in top_stocks if stock.is_penny_stock),
                "sector_distribution": self._get_sector_distribution(top_stocks)
            }
        }
        
        # Save to file
        with open("data/top_101_stocks.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save simple symbol list for easy use
        symbol_list = [stock.symbol for stock in top_stocks]
        with open("data/top_101_symbols.txt", "w") as f:
            f.write("\n".join(symbol_list))
        
        logger.info("💾 Top 101 list saved to data/top_101_stocks.json")
    
    def _save_scan_results(self, scan_results: Dict[str, StockMetrics]):
        """Save complete scan results for analysis"""
        
        # Add to history
        scan_record = {
            "timestamp": datetime.now().isoformat(),
            "total_scanned": len(scan_results),
            "results": {symbol: asdict(metrics) for symbol, metrics in scan_results.items()}
        }
        
        self.scan_results_history.append(scan_record)
        
        # Keep only last 30 scans
        if len(self.scan_results_history) > 30:
            self.scan_results_history = self.scan_results_history[-30:]
        
        # Save to file
        with open("data/market_scan_history.json", "w") as f:
            json.dump(self.scan_results_history, f, indent=2, default=str)
    
    def get_current_top_101(self) -> List[str]:
        """Get current top 101 stock symbols"""
        if self.current_top_101:
            return [stock.symbol for stock in self.current_top_101]
        else:
            # Return default list if no scan has been performed
            from ..data_collection.symbol_manager import SymbolManager
            sm = SymbolManager()
            return sm.get_all_symbols()
    
    async def start_surveillance(self):
        """Start the market surveillance agent (async version)"""
        
        logger.info("🕵️ Starting Market Surveillance Agent")
        
        # Perform initial scan
        logger.info("🔍 Performing initial market scan...")
        report = self.update_top_101_stocks()
        
        if report["status"] == "success":
            logger.info(f"✅ Initial scan complete!")
            logger.info(f"📊 Analyzed: {report['total_scanned']} stocks")
            logger.info(f"🏆 Top 10: {', '.join(report['top_performers'][:10])}")
            logger.info(f"💰 Penny stocks: {report['penny_stock_count']}/101")
            logger.info(f"📈 Average score: {report['average_score']:.3f}")
            
            # Show sector distribution
            sectors = report.get('sector_distribution', {})
            logger.info(f"🏢 Sectors: {', '.join([f'{k}({v})' for k, v in sectors.items()])}")
            
        else:
            logger.warning(f"⚠️ Initial scan failed: {report.get('reason', 'unknown')}")
            return
        
        # Continue with periodic surveillance
        logger.info("🔄 Starting continuous surveillance mode...")
        
        while True:
            try:
                if self._should_scan_now():
                    logger.info("🔍 Performing scheduled market scan")
                    report = self.update_top_101_stocks()
                    
                    if report["status"] == "success":
                        logger.info(f"✅ Scan complete: {report['changes']['new_entries']} new stocks discovered")
                        
                        # Log significant changes
                        if report['changes']['new_entries'] > 5:
                            logger.info(f"🚨 Significant market shift: {report['changes']['new_entries']} new entries")
                            logger.info(f"📈 New stocks: {', '.join(report['changes']['new_symbols'][:5])}")
                        
                    else:
                        logger.warning(f"⚠️ Scan failed: {report.get('reason', 'unknown')}")
                
                # Wait before next check (check every hour, scan every 4 hours)
                await asyncio.sleep(3600)  # 1 hour
                
            except KeyboardInterrupt:
                logger.info("🛑 Surveillance stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Surveillance error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def run_continuous_surveillance(self):
        """Run continuous market surveillance (for production)"""
        
        logger.info("🕵️ Starting continuous market surveillance")
        
        while True:
            try:
                if self._should_scan_now():
                    logger.info("🔍 Performing scheduled market scan")
                    report = self.update_top_101_stocks()
                    
                    if report["status"] == "success":
                        logger.info(f"✅ Scan complete: {report['changes']['new_entries']} new stocks discovered")
                    else:
                        logger.warning(f"⚠️ Scan failed: {report.get('reason', 'unknown')}")
                
                # Wait before next check (check every hour, scan every 4 hours)
                time.sleep(3600)  # 1 hour
                
            except KeyboardInterrupt:
                logger.info("🛑 Surveillance stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Surveillance error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = MarketSurveillanceAgent()
    
    print("🕵️ MARKET SURVEILLANCE AGENT TEST")
    print("=" * 50)
    
    # Test with a few symbols
    test_symbols = ["RY.TO", "SHOP.TO", "HUT.TO", "PMET.TO", "WEED.TO"]
    
    print(f"🧪 Testing surveillance on {len(test_symbols)} symbols...")
    
    # Simulate market scan
    agent.canadian_universe = test_symbols  # Limit for testing
    
    report = agent.update_top_101_stocks()
    
    if report["status"] == "success":
        print(f"✅ Surveillance test successful!")
        print(f"📊 Scanned: {report['total_scanned']} stocks")
        print(f"🏆 Top performers: {', '.join(report['top_performers'])}")
        print(f"💰 Penny stocks found: {report['penny_stock_count']}")
    else:
        print(f"❌ Surveillance test failed: {report.get('reason', 'unknown')}")