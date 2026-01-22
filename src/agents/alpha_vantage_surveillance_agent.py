"""
Alpha Vantage Market Surveillance Agent - Optimized for Canadian Markets

Uses Alpha Vantage as primary data source with proper rate limiting (74 calls/minute)
for comprehensive Canadian market surveillance and top 101 stock selection.
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

class AlphaVantageMarketSurveillanceAgent:
    """Optimized surveillance agent using Alpha Vantage with proper rate limiting"""
    
    def __init__(self):
        self.market_calendar = MarketCalendar()
        self.alpha_vantage = AlphaVantageCollector()
        self.yahoo_fallback = EnhancedYahooFinanceCollector()
        
        # Rate limiting configuration
        self.config = {
            "calls_per_minute": 74,  # Stay under 75/minute Alpha Vantage limit
            "call_delay": 0.82,      # 60/74 = 0.81 seconds between calls
            "top_stocks_count": 101,
            "penny_stock_threshold": 5.0,
            "min_volume": 10000,
            "min_market_cap": 1000000,
            "volume_spike_threshold": 3.0,
            "batch_size": 20,        # Process in batches for progress updates
        }
        
        # Canadian stock universe
        self.canadian_universe = self._get_canadian_universe()
        self.current_top_101 = []
        self.last_scan_time = None
        
        # Rate limiting tracking
        self.calls_made = 0
        self.minute_start = time.time()
        
        logger.info(f"🕵️ Alpha Vantage Surveillance Agent initialized")
        logger.info(f"📊 Monitoring {len(self.canadian_universe)} Canadian stocks")
        logger.info(f"⏱️ Rate limit: {self.config['calls_per_minute']} calls/minute")
    
    def _get_canadian_universe(self) -> List[str]:
        """Get comprehensive list of Canadian stocks optimized for Alpha Vantage"""
        
        # Focus on major Canadian stocks that Alpha Vantage covers well
        major_stocks = [
            # Big 6 Banks
            "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
            
            # Energy Giants
            "CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO", "CVE.TO", "IMO.TO",
            "PPL.TO", "KEY.TO", "ARX.TO", "WCP.TO", "BTE.TO",
            
            # Tech Leaders
            "SHOP.TO", "CSU.TO", "LSPD.TO", "NVEI.TO", "DOCN.TO", "BB.TO",
            
            # Mining & Resources
            "ABX.TO", "K.TO", "AEM.TO", "FNV.TO", "WPM.TO", "KL.TO",
            "NTR.TO", "FM.TO", "TKO.TO", "CS.TO", "HBM.TO", "LUN.TO",
            
            # Telecom
            "BCE.TO", "T.TO", "RCI-B.TO", "QBR-B.TO",
            
            # Utilities
            "FTS.TO", "EMA.TO", "CU.TO", "AQN.TO", "H.TO",
            
            # REITs
            "CAR-UN.TO", "REI-UN.TO", "HR-UN.TO", "CHP-UN.TO",
            
            # Transportation
            "CNR.TO", "CP.TO", "AC.TO", "TFII.TO",
            
            # Consumer
            "ATD.TO", "L.TO", "MG.TO", "GOOS.TO", "DOL.TO",
            
            # Healthcare & Pharma
            "CSH.TO", "CRH.TO", "WELL.TO", "PHM.TO",
            
            # Cannabis (Penny Stock Opportunities)
            "WEED.TO", "ACB.TO", "HEXO.TO", "OGI.TO", "TLRY.TO",
            
            # Crypto/Blockchain
            "HUT.TO", "BITF.TO", "HIVE.TO", "DMGI.TO",
            
            # Growth & Small Caps
            "NUMI.TO", "CYBN.TO", "PYR.TO", "HPQ.TO", "NOU.TO",
            "PMET.TO", "LAC.TO", "CRE.TO", "FL.TO", "TOI.TO",
        ]
        
        logger.info(f"📈 Canadian universe: {len(major_stocks)} high-quality stocks")
        return major_stocks
    
    async def start_surveillance(self):
        """Start the Alpha Vantage market surveillance"""
        
        logger.info("🚀 Starting Alpha Vantage Market Surveillance")
        
        # Perform market scan with proper rate limiting
        scan_results = await self.perform_rate_limited_scan()
        
        if scan_results:
            # Update top 101 stocks
            await self.update_top_101_stocks(scan_results)
            
            # Generate surveillance report
            await self.generate_surveillance_report()
            
            logger.info("✅ Market surveillance completed successfully")
        else:
            logger.warning("⚠️ No valid scan results obtained")
    
    async def perform_rate_limited_scan(self) -> Dict[str, StockMetrics]:
        """Perform market scan with Alpha Vantage rate limiting"""
        
        logger.info(f"🔍 Starting rate-limited scan of {len(self.canadian_universe)} stocks")
        logger.info(f"⏱️ Estimated time: {len(self.canadian_universe) * self.config['call_delay'] / 60:.1f} minutes")
        
        scan_results = {}
        failed_symbols = []
        
        # Reset rate limiting counters
        self.calls_made = 0
        self.minute_start = time.time()
        
        for i, symbol in enumerate(self.canadian_universe):
            try:
                # Rate limiting check
                await self._enforce_rate_limit()
                
                # Scan the stock
                metrics = await self._scan_single_stock_alpha_vantage(symbol)
                self.calls_made += 1
                
                if metrics:
                    scan_results[symbol] = metrics
                    logger.debug(f"✅ {symbol}: Score {metrics.composite_score:.3f}")
                else:
                    failed_symbols.append(symbol)
                    logger.debug(f"⚠️ No data: {symbol}")
                
                # Progress updates
                if (i + 1) % self.config["batch_size"] == 0:
                    progress = (i + 1) / len(self.canadian_universe) * 100
                    logger.info(f"📊 Progress: {progress:.1f}% ({i+1}/{len(self.canadian_universe)}) - {len(scan_results)} valid stocks")
                
                # Small delay between calls
                await asyncio.sleep(self.config["call_delay"])
                
            except Exception as e:
                failed_symbols.append(symbol)
                logger.warning(f"❌ Error scanning {symbol}: {e}")
                self.calls_made += 1
        
        logger.info(f"✅ Scan complete: {len(scan_results)} stocks analyzed, {len(failed_symbols)} failed")
        return scan_results
    
    async def _enforce_rate_limit(self):
        """Enforce Alpha Vantage rate limiting (74 calls/minute)"""
        
        current_time = time.time()
        elapsed = current_time - self.minute_start
        
        # If we've made 74 calls and less than 60 seconds have passed, wait
        if self.calls_made >= self.config["calls_per_minute"] and elapsed < 60:
            sleep_time = 60 - elapsed
            logger.info(f"⏳ Rate limit: sleeping {sleep_time:.1f}s (made {self.calls_made} calls)")
            await asyncio.sleep(sleep_time)
            
            # Reset counters
            self.calls_made = 0
            self.minute_start = time.time()
        
        # If more than 60 seconds have passed, reset counters
        elif elapsed >= 60:
            self.calls_made = 0
            self.minute_start = current_time
    
    async def _scan_single_stock_alpha_vantage(self, symbol: str) -> Optional[StockMetrics]:
        """Scan a single stock using Alpha Vantage with fallback to Yahoo"""
        
        try:
            # Try Alpha Vantage first
            data, source = self.alpha_vantage.fetch_daily_data(symbol)
            
            # If Alpha Vantage fails, try Yahoo Finance as fallback
            if data is None or data.empty:
                logger.debug(f"Alpha Vantage failed for {symbol}, trying Yahoo fallback")
                data, source = self.yahoo_fallback.fetch_data(symbol, period="1mo", interval="1d")
            
            if data is None or data.empty or len(data) < 5:
                return None
            
            # Calculate metrics
            return self._calculate_stock_metrics(symbol, data)
            
        except Exception as e:
            logger.debug(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_stock_metrics(self, symbol: str, data: pd.DataFrame) -> StockMetrics:
        """Calculate comprehensive stock metrics"""
        
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
            exchange=exchange,
            composite_score=composite_score
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
    
    def _classify_sector(self, symbol: str) -> str:
        """Classify stock sector"""
        if symbol in ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO"]:
            return "Banking"
        elif symbol in ["CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO", "CVE.TO"]:
            return "Energy"
        elif symbol in ["SHOP.TO", "CSU.TO", "LSPD.TO", "BB.TO", "NVEI.TO"]:
            return "Technology"
        elif symbol in ["ABX.TO", "K.TO", "AEM.TO", "WPM.TO", "KL.TO"]:
            return "Mining"
        elif symbol in ["WEED.TO", "ACB.TO", "HEXO.TO", "OGI.TO"]:
            return "Cannabis"
        elif symbol in ["HUT.TO", "BITF.TO", "HIVE.TO", "DMGI.TO"]:
            return "Crypto"
        else:
            return "Other"
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name"""
        return symbol.replace(".TO", "").replace(".V", "")
    
    async def update_top_101_stocks(self, scan_results: Dict[str, StockMetrics]):
        """Update top 101 stocks based on scan results"""
        
        logger.info("🏆 Ranking stocks and selecting top 101...")
        
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
        
        logger.info(f"📊 Top 101 updated: {len(added)} added, {len(removed)} removed")
        
        # Save results
        await self._save_surveillance_results(top_101, scan_results)
    
    async def _save_surveillance_results(self, top_101: List[StockMetrics], all_results: Dict[str, StockMetrics]):
        """Save surveillance results to files"""
        
        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "scan_summary": {
                "total_scanned": len(all_results),
                "top_101_count": len(top_101),
                "data_source": "Alpha Vantage (primary) + Yahoo Finance (fallback)",
                "rate_limit": f"{self.config['calls_per_minute']} calls/minute"
            },
            "top_101_stocks": [asdict(stock) for stock in top_101],
            "market_composition": {
                "penny_stocks": len([s for s in top_101 if s.is_penny_stock]),
                "sectors": {},
                "exchanges": {}
            }
        }
        
        # Calculate composition
        for stock in top_101:
            # Sector distribution
            sector = stock.sector
            report["market_composition"]["sectors"][sector] = \
                report["market_composition"]["sectors"].get(sector, 0) + 1
            
            # Exchange distribution
            exchange = stock.exchange
            report["market_composition"]["exchanges"][exchange] = \
                report["market_composition"]["exchanges"].get(exchange, 0) + 1
        
        # Save to file
        output_file = "alpha_vantage_surveillance_results.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save simple symbol list
        with open("top_101_symbols_alpha_vantage.txt", 'w') as f:
            f.write("\n".join(self.current_top_101))
        
        logger.info(f"💾 Results saved to {output_file}")
    
    async def generate_surveillance_report(self):
        """Generate final surveillance report"""
        
        if not self.current_top_101:
            return
        
        logger.info("📊 Generating surveillance intelligence report...")
        
        # Get top stocks data
        top_stocks = [stock for stock in self.current_top_101[:10]]
        
        logger.info("🏆 TOP 10 OPPORTUNITIES:")
        for i, symbol in enumerate(top_stocks, 1):
            logger.info(f"   {i:2d}. {symbol}")
        
        logger.info(f"💰 Penny stocks in top 101: {len([s for s in self.current_top_101 if '.TO' in s])}")
        logger.info(f"🚀 Total opportunities identified: {len(self.current_top_101)}")
        logger.info("✅ Market surveillance complete!")

# Usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = AlphaVantageMarketSurveillanceAgent()
        await agent.start_surveillance()
    
    asyncio.run(main())