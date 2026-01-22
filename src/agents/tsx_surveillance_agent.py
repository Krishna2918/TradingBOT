"""
TSX Market Surveillance Agent - Alpha Vantage Primary

Focuses on TSX-listed companies using Alpha Vantage as primary data source
with proper 74 calls/minute rate limiting. Yahoo Finance as fallback only.
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

from ..data_collection.market_calendar import MarketCalendar
from ..data_collection.alpha_vantage_collector import AlphaVantageCollector
from ..data_collection.enhanced_collectors import EnhancedYahooFinanceCollector

logger = logging.getLogger(__name__)

@dataclass
class TSXStockMetrics:
    """TSX stock performance metrics"""
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
    volume_ratio: float
    volume_spike: bool
    
    # Technical metrics
    rsi: float
    momentum_score: float
    breakout_score: float
    
    # Classification
    is_penny_stock: bool
    sector: str
    
    # Overall ranking score
    composite_score: float
    data_source: str  # Track which API provided the data

class TSXSurveillanceAgent:
    """TSX-focused surveillance agent with Alpha Vantage primary"""
    
    def __init__(self):
        self.market_calendar = MarketCalendar()
        self.alpha_vantage = AlphaVantageCollector()  # PRIMARY
        self.yahoo_fallback = EnhancedYahooFinanceCollector()  # FALLBACK ONLY
        
        # Alpha Vantage focused configuration
        self.config = {
            "alpha_vantage_calls_per_minute": 74,  # Stay under 75/minute limit
            "call_delay": 0.82,  # 60/74 = 0.81 seconds between calls
            "top_stocks_count": 101,
            "penny_stock_threshold": 5.0,
            "min_volume": 10000,
            "min_market_cap": 1000000,
            "volume_spike_threshold": 3.0,
            "progress_batch_size": 15,  # Progress updates every 15 stocks
        }
        
        # TSX-focused stock universe (Alpha Vantage compatible)
        self.tsx_universe = self._get_tsx_universe()
        self.current_top_101 = []
        self.last_scan_time = None
        
        # Rate limiting tracking
        self.alpha_vantage_calls = 0
        self.minute_start = time.time()
        
        logger.info(f"🏢 TSX Surveillance Agent initialized")
        logger.info(f"📊 Monitoring {len(self.tsx_universe)} TSX companies")
        logger.info(f"🔑 Primary: Alpha Vantage ({self.config['alpha_vantage_calls_per_minute']} calls/min)")
        logger.info(f"🔄 Fallback: Yahoo Finance (when needed)")
    
    def _get_tsx_universe(self) -> List[str]:
        """Get TSX companies that work well with Alpha Vantage"""
        
        # Focus on major TSX companies (remove .TO for Alpha Vantage)
        tsx_companies = [
            # Big 6 Canadian Banks
            "RY", "TD", "BNS", "BMO", "CM", "NA",
            
            # Energy Sector
            "CNQ", "SU", "ENB", "TRP", "CVE", "IMO", "PPL", "KEY", "ARX", "WCP",
            
            # Technology
            "SHOP", "CSU", "LSPD", "NVEI", "BB", "DOCN",
            
            # Mining & Resources
            "ABX", "K", "AEM", "FNV", "WPM", "KL", "NTR", "FM", "TKO", "CS",
            
            # Telecommunications
            "BCE", "T", "RCI.B", "QBR.B",
            
            # Utilities
            "FTS", "EMA", "CU", "AQN", "H",
            
            # Transportation
            "CNR", "CP", "AC", "TFII",
            
            # Consumer & Retail
            "ATD", "L", "MG", "GOOS", "DOL",
            
            # Healthcare
            "CSH", "CRH", "WELL", "PHM",
            
            # Cannabis (Penny Stock Opportunities)
            "WEED", "ACB", "HEXO", "OGI", "TLRY",
            
            # Crypto/Blockchain
            "HUT", "BITF", "HIVE", "DMGI",
            
            # REITs (some trade on US exchanges too)
            "CAR.UN", "REI.UN", "HR.UN", "CHP.UN",
            
            # Growth & Small Caps
            "NUMI", "PYR", "HPQ", "NOU", "PMET", "LAC", "CRE", "FL",
        ]
        
        logger.info(f"🏢 TSX universe: {len(tsx_companies)} companies")
        return tsx_companies
    
    async def start_surveillance(self):
        """Start TSX surveillance with Alpha Vantage primary"""
        
        logger.info("🚀 Starting TSX Market Surveillance (Alpha Vantage Primary)")
        
        # Perform rate-limited scan
        scan_results = await self.perform_alpha_vantage_scan()
        
        if scan_results:
            # Update top 101 stocks
            await self.update_top_101_stocks(scan_results)
            
            # Generate surveillance report
            await self.generate_surveillance_report(scan_results)
            
            logger.info("✅ TSX surveillance completed successfully")
        else:
            logger.warning("⚠️ No valid scan results obtained")
    
    async def perform_alpha_vantage_scan(self) -> Dict[str, TSXStockMetrics]:
        """Perform TSX scan with Alpha Vantage primary, Yahoo fallback"""
        
        logger.info(f"🔍 Starting Alpha Vantage scan of {len(self.tsx_universe)} TSX companies")
        
        estimated_time = len(self.tsx_universe) * self.config["call_delay"] / 60
        logger.info(f"⏱️ Estimated time: {estimated_time:.1f} minutes")
        
        scan_results = {}
        alpha_vantage_success = 0
        yahoo_fallback_used = 0
        failed_symbols = []
        
        # Reset rate limiting
        self.alpha_vantage_calls = 0
        self.minute_start = time.time()
        
        for i, symbol in enumerate(self.tsx_universe):
            try:
                # Rate limiting for Alpha Vantage
                await self._enforce_alpha_vantage_rate_limit()
                
                # Try Alpha Vantage first (PRIMARY)
                metrics = await self._scan_with_alpha_vantage(symbol)
                self.alpha_vantage_calls += 1
                
                if metrics:
                    metrics.data_source = "Alpha Vantage"
                    scan_results[symbol] = metrics
                    alpha_vantage_success += 1
                    logger.debug(f"✅ Alpha Vantage: {symbol} - Score: {metrics.composite_score:.3f}")
                else:
                    # Fallback to Yahoo Finance
                    logger.debug(f"🔄 Alpha Vantage failed for {symbol}, trying Yahoo fallback")
                    metrics = await self._scan_with_yahoo_fallback(symbol)
                    
                    if metrics:
                        metrics.data_source = "Yahoo Finance (Fallback)"
                        scan_results[symbol] = metrics
                        yahoo_fallback_used += 1
                        logger.debug(f"✅ Yahoo Fallback: {symbol} - Score: {metrics.composite_score:.3f}")
                    else:
                        failed_symbols.append(symbol)
                        logger.debug(f"❌ Both APIs failed for {symbol}")
                
                # Progress updates
                if (i + 1) % self.config["progress_batch_size"] == 0:
                    progress = (i + 1) / len(self.tsx_universe) * 100
                    logger.info(f"📊 Progress: {progress:.1f}% ({i+1}/{len(self.tsx_universe)}) - "
                              f"Alpha Vantage: {alpha_vantage_success}, Yahoo: {yahoo_fallback_used}, Failed: {len(failed_symbols)}")
                
                # Delay between calls
                await asyncio.sleep(self.config["call_delay"])
                
            except Exception as e:
                failed_symbols.append(symbol)
                logger.warning(f"❌ Error scanning {symbol}: {e}")
        
        # Final summary
        total_success = len(scan_results)
        logger.info(f"✅ TSX scan complete:")
        logger.info(f"   📊 Total analyzed: {total_success}/{len(self.tsx_universe)}")
        logger.info(f"   🔑 Alpha Vantage success: {alpha_vantage_success}")
        logger.info(f"   🔄 Yahoo fallback used: {yahoo_fallback_used}")
        logger.info(f"   ❌ Failed: {len(failed_symbols)}")
        
        return scan_results
    
    async def _enforce_alpha_vantage_rate_limit(self):
        """Enforce Alpha Vantage 74 calls/minute rate limit"""
        
        current_time = time.time()
        elapsed = current_time - self.minute_start
        
        # If we've made 74 calls and less than 60 seconds have passed, wait
        if self.alpha_vantage_calls >= self.config["alpha_vantage_calls_per_minute"] and elapsed < 60:
            sleep_time = 60 - elapsed
            logger.info(f"⏳ Alpha Vantage rate limit: sleeping {sleep_time:.1f}s (made {self.alpha_vantage_calls} calls)")
            await asyncio.sleep(sleep_time)
            
            # Reset counters
            self.alpha_vantage_calls = 0
            self.minute_start = time.time()
        
        # If more than 60 seconds have passed, reset counters
        elif elapsed >= 60:
            self.alpha_vantage_calls = 0
            self.minute_start = current_time
    
    async def _scan_with_alpha_vantage(self, symbol: str) -> Optional[TSXStockMetrics]:
        """Scan using Alpha Vantage (PRIMARY)"""
        
        try:
            # Try both symbol formats for Alpha Vantage
            data = None
            
            # Try without .TO first (Alpha Vantage prefers this)
            clean_symbol = symbol.replace(".TO", "").replace(".V", "")
            data, source = self.alpha_vantage.fetch_daily_data(clean_symbol)
            
            # If that fails, try with .TO
            if data is None or data.empty:
                data, source = self.alpha_vantage.fetch_daily_data(f"{clean_symbol}.TO")
            
            if data is None or data.empty or len(data) < 5:
                return None
            
            return self._calculate_tsx_metrics(symbol, data)
            
        except Exception as e:
            logger.debug(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    async def _scan_with_yahoo_fallback(self, symbol: str) -> Optional[TSXStockMetrics]:
        """Scan using Yahoo Finance (FALLBACK ONLY)"""
        
        try:
            # Yahoo Finance expects .TO format
            yahoo_symbol = f"{symbol}.TO" if not symbol.endswith(".TO") else symbol
            data, source = self.yahoo_fallback.fetch_data(yahoo_symbol, period="1mo", interval="1d")
            
            if data is None or data.empty or len(data) < 5:
                return None
            
            return self._calculate_tsx_metrics(symbol, data)
            
        except Exception as e:
            logger.debug(f"Yahoo fallback error for {symbol}: {e}")
            return None
    
    def _calculate_tsx_metrics(self, symbol: str, data: pd.DataFrame) -> TSXStockMetrics:
        """Calculate comprehensive TSX stock metrics"""
        
        # Basic metrics
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        
        # Performance metrics
        daily_return = data['Close'].pct_change().iloc[-1] if len(data) > 1 else 0.0
        weekly_return = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) if len(data) >= 5 else 0.0
        monthly_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) if len(data) > 0 else 0.0
        
        # Volatility
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
        sector = self._classify_tsx_sector(symbol)
        
        # Composite scoring
        composite_score = self._calculate_composite_score(
            daily_return, weekly_return, monthly_return, volatility,
            volume_ratio, rsi, momentum_score, breakout_score,
            is_penny_stock, market_cap
        )
        
        return TSXStockMetrics(
            symbol=symbol,
            name=self._get_company_name(symbol),
            price=current_price,
            volume=int(current_volume),
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
            composite_score=composite_score,
            data_source=""  # Will be set by caller
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        if len(data) < 10:
            return 0.0
        
        recent_avg = data['Close'].tail(5).mean()
        older_avg = data['Close'].head(5).mean()
        price_momentum = (recent_avg / older_avg - 1) if older_avg > 0 else 0.0
        
        recent_vol = data['Volume'].tail(5).mean()
        older_vol = data['Volume'].head(5).mean()
        volume_momentum = (recent_vol / older_vol - 1) if older_vol > 0 else 0.0
        
        momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3)
        return max(-1.0, min(1.0, momentum_score))
    
    def _calculate_breakout_score(self, data: pd.DataFrame) -> float:
        """Calculate breakout potential score"""
        if len(data) < 20:
            return 0.0
        
        recent_high = data['High'].tail(20).max()
        current_price = data['Close'].iloc[-1]
        high_proximity = current_price / recent_high if recent_high > 0 else 0.0
        
        avg_volume = data['Volume'].tail(20).mean()
        recent_volume = data['Volume'].tail(3).mean()
        volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 0.0
        
        breakout_score = (high_proximity * 0.6 + min(volume_confirmation, 2.0) * 0.4)
        return max(0.0, min(2.0, breakout_score))
    
    def _calculate_composite_score(self, daily_ret: float, weekly_ret: float, monthly_ret: float,
                                 volatility: float, volume_ratio: float, rsi: float,
                                 momentum: float, breakout: float, is_penny: bool, market_cap: float) -> float:
        """Calculate composite ranking score"""
        
        score = 0.0
        
        # Performance component (40%)
        performance_score = (daily_ret * 0.3 + weekly_ret * 0.4 + monthly_ret * 0.3)
        score += performance_score * 0.4
        
        # Volume component (25%)
        volume_score = min(volume_ratio / 5.0, 1.0)
        score += volume_score * 0.25
        
        # Technical component (20%)
        technical_score = (momentum * 0.5 + breakout * 0.3 + (rsi - 50) / 50 * 0.2)
        score += technical_score * 0.2
        
        # Opportunity component (15%)
        opportunity_score = 0.0
        if is_penny:
            opportunity_score += 0.3
        if market_cap < 100000000:
            opportunity_score += 0.2
        if volatility > 0.3:
            opportunity_score += 0.1
        
        score += opportunity_score * 0.15
        
        return score
    
    def _estimate_market_cap(self, symbol: str, price: float, volume: int) -> float:
        """Estimate market cap"""
        if volume > 1000000:
            return price * volume * 0.1
        elif volume > 100000:
            return price * volume * 0.5
        else:
            return price * volume * 1.0
    
    def _classify_tsx_sector(self, symbol: str) -> str:
        """Classify TSX sector"""
        if symbol in ["RY", "TD", "BNS", "BMO", "CM", "NA"]:
            return "Banking"
        elif symbol in ["CNQ", "SU", "ENB", "TRP", "CVE", "IMO", "PPL", "KEY", "ARX", "WCP"]:
            return "Energy"
        elif symbol in ["SHOP", "CSU", "LSPD", "BB", "NVEI", "DOCN"]:
            return "Technology"
        elif symbol in ["ABX", "K", "AEM", "WPM", "KL", "NTR", "FM", "TKO", "CS"]:
            return "Mining"
        elif symbol in ["WEED", "ACB", "HEXO", "OGI", "TLRY"]:
            return "Cannabis"
        elif symbol in ["HUT", "BITF", "HIVE", "DMGI"]:
            return "Crypto"
        elif symbol in ["BCE", "T", "RCI.B", "QBR.B"]:
            return "Telecom"
        elif symbol in ["FTS", "EMA", "CU", "AQN", "H"]:
            return "Utilities"
        else:
            return "Other"
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name"""
        return symbol.replace(".TO", "").replace(".V", "")
    
    async def update_top_101_stocks(self, scan_results: Dict[str, TSXStockMetrics]):
        """Update top 101 TSX stocks"""
        
        logger.info("🏆 Ranking TSX stocks and selecting top 101...")
        
        # Sort by composite score
        ranked_stocks = sorted(scan_results.values(), key=lambda x: x.composite_score, reverse=True)
        
        # Select top 101
        top_101 = ranked_stocks[:self.config["top_stocks_count"]]
        
        # Update current list
        old_symbols = set(self.current_top_101)
        self.current_top_101 = [stock.symbol for stock in top_101]
        new_symbols = set(self.current_top_101)
        
        # Calculate changes
        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols
        
        logger.info(f"📊 Top 101 TSX updated: {len(added)} added, {len(removed)} removed")
        
        # Save results
        await self._save_tsx_results(top_101, scan_results)
    
    async def _save_tsx_results(self, top_101: List[TSXStockMetrics], all_results: Dict[str, TSXStockMetrics]):
        """Save TSX surveillance results"""
        
        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "scan_summary": {
                "total_scanned": len(all_results),
                "top_101_count": len(top_101),
                "primary_source": "Alpha Vantage",
                "fallback_source": "Yahoo Finance",
                "rate_limit": f"{self.config['alpha_vantage_calls_per_minute']} calls/minute"
            },
            "data_source_breakdown": {
                "alpha_vantage": len([s for s in all_results.values() if s.data_source == "Alpha Vantage"]),
                "yahoo_fallback": len([s for s in all_results.values() if "Yahoo" in s.data_source])
            },
            "top_101_stocks": [asdict(stock) for stock in top_101],
            "market_composition": {
                "penny_stocks": len([s for s in top_101 if s.is_penny_stock]),
                "sectors": {},
                "data_sources": {}
            }
        }
        
        # Calculate composition
        for stock in top_101:
            # Sector distribution
            sector = stock.sector
            report["market_composition"]["sectors"][sector] = \
                report["market_composition"]["sectors"].get(sector, 0) + 1
            
            # Data source distribution
            source = stock.data_source
            report["market_composition"]["data_sources"][source] = \
                report["market_composition"]["data_sources"].get(source, 0) + 1
        
        # Save to file
        output_file = "tsx_surveillance_results.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save simple symbol list
        with open("top_101_tsx_symbols.txt", 'w') as f:
            f.write("\n".join(self.current_top_101))
        
        logger.info(f"💾 TSX results saved to {output_file}")
    
    async def generate_surveillance_report(self, scan_results: Dict[str, TSXStockMetrics]):
        """Generate final TSX surveillance report"""
        
        if not self.current_top_101:
            return
        
        logger.info("📊 Generating TSX surveillance report...")
        
        # Get top stocks
        top_10 = self.current_top_101[:10]
        
        # Data source analysis
        alpha_vantage_count = len([s for s in scan_results.values() if s.data_source == "Alpha Vantage"])
        yahoo_count = len([s for s in scan_results.values() if "Yahoo" in s.data_source])
        
        logger.info("🏆 TOP 10 TSX OPPORTUNITIES:")
        for i, symbol in enumerate(top_10, 1):
            logger.info(f"   {i:2d}. {symbol}")
        
        logger.info(f"📊 Data Source Performance:")
        logger.info(f"   🔑 Alpha Vantage: {alpha_vantage_count} stocks")
        logger.info(f"   🔄 Yahoo Fallback: {yahoo_count} stocks")
        
        logger.info(f"💰 Penny stocks in top 101: {len([s for s in scan_results.values() if s.is_penny_stock])}")
        logger.info(f"🚀 Total TSX opportunities: {len(self.current_top_101)}")
        logger.info("✅ TSX surveillance complete!")

# Usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = TSXSurveillanceAgent()
        await agent.start_surveillance()
    
    asyncio.run(main())