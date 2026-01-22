"""
US Market Surveillance Agent - Alpha Vantage Optimized

Focuses on US markets using Alpha Vantage paid subscription with proper rate limiting.
Monitors NYSE, NASDAQ for top 101 opportunities including growth stocks, penny stocks, and blue chips.
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
class USStockMetrics:
    """US stock performance metrics"""
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
    exchange: str
    
    # Overall ranking score
    composite_score: float

class USMarketSurveillanceAgent:
    """US market surveillance agent optimized for Alpha Vantage paid subscription"""
    
    def __init__(self):
        self.market_calendar = MarketCalendar()
        self.alpha_vantage = AlphaVantageCollector()  # PRIMARY - paid subscription
        self.yahoo_fallback = EnhancedYahooFinanceCollector()  # FALLBACK ONLY
        
        # Alpha Vantage paid subscription configuration
        self.config = {
            "alpha_vantage_calls_per_minute": 74,  # Conservative rate limiting
            "call_delay": 0.82,  # 60/74 seconds between calls
            "top_stocks_count": 101,
            "penny_stock_threshold": 5.0,  # Under $5 = penny stock
            "min_volume": 50000,  # Higher volume for US markets
            "min_market_cap": 10000000,  # $10M minimum market cap
            "volume_spike_threshold": 3.0,
            "progress_batch_size": 20,  # Progress updates every 20 stocks
        }
        
        # US market universe (Alpha Vantage optimized)
        self.us_universe = self._get_us_market_universe()
        self.current_top_101 = []
        self.last_scan_time = None
        
        # Rate limiting tracking
        self.alpha_vantage_calls = 0
        self.minute_start = time.time()
        
        logger.info(f"🇺🇸 US Market Surveillance Agent initialized")
        logger.info(f"📊 Monitoring {len(self.us_universe)} US stocks")
        logger.info(f"🔑 Alpha Vantage paid subscription ({self.config['alpha_vantage_calls_per_minute']} calls/min)")
        logger.info(f"🔄 Yahoo Finance fallback available")
    
    def _get_us_market_universe(self) -> List[str]:
        """Get comprehensive US stock universe optimized for Alpha Vantage"""
        
        us_stocks = [
            # FAANG + Mega Tech
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA",
            "NFLX", "CRM", "ORCL", "ADBE", "INTC", "AMD", "QCOM", "AVGO",
            
            # Financial Giants
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF",
            "AXP", "BLK", "SCHW", "CB", "AIG", "MET", "PRU", "ALL", "TRV",
            
            # Healthcare & Pharma
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
            "AMGN", "GILD", "BIIB", "REGN", "VRTX", "ILMN", "MRNA", "BNTX",
            
            # Consumer & Retail
            "WMT", "HD", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT",
            "COST", "LOW", "DIS", "CMCSA", "VZ", "T", "NFLX", "ROKU",
            
            # Industrial & Manufacturing
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX",
            "NOC", "GD", "DE", "EMR", "ETN", "PH", "ITW", "ROK", "DOV",
            
            # Energy & Utilities
            "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "KMI",
            "OKE", "WMB", "EPD", "ET", "NEE", "DUK", "SO", "AEP", "EXC",
            
            # Growth & Emerging Tech
            "PLTR", "SNOW", "CRWD", "ZS", "OKTA", "DDOG", "NET", "FSLY",
            "TWLO", "ZM", "DOCU", "WORK", "UBER", "LYFT", "DASH", "ABNB",
            
            # Biotech & Healthcare Innovation
            "MRNA", "BNTX", "NVAX", "TDOC", "VEEV", "PTON", "ZBH", "SYK",
            "MDT", "BSX", "EW", "HOLX", "VAR", "PKI", "A", "LH", "DGX",
            
            # Fintech & Digital
            "PYPL", "SQ", "V", "MA", "COIN", "HOOD", "SOFI", "AFRM", "LC",
            "UPST", "OPEN", "RBLX", "U", "PINS", "SNAP", "TWTR", "SPOT",
            
            # Clean Energy & EV
            "TSLA", "NIO", "XPEV", "LI", "RIVN", "LCID", "FSR", "NKLA",
            "PLUG", "FCEL", "BE", "BLNK", "CHPT", "ENPH", "SEDG", "RUN",
            
            # Cannabis & Alternative Investments
            "TLRY", "CGC", "ACB", "CRON", "SNDL", "OGI", "HEXO", "APHA",
            
            # Penny Stock Opportunities (Under $5)
            "SNDL", "NOK", "BB", "AMC", "GME", "BBBY", "CLOV", "WISH",
            "PLBY", "RIDE", "WKHS", "GOEV", "HYLN", "SPCE", "SKLZ", "DKNG",
            
            # REITs & Dividend Stocks
            "O", "PLD", "AMT", "CCI", "EQIX", "DLR", "PSA", "EXR", "AVB",
            "EQR", "UDR", "CPT", "MAA", "ESS", "AIV", "BXP", "VTR", "WELL",
            
            # Commodities & Materials
            "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "DBA", "DJP",
            "FCX", "NEM", "GOLD", "AUY", "KGC", "HL", "CDE", "AG",
        ]
        
        logger.info(f"🇺🇸 US market universe: {len(us_stocks)} stocks across all sectors")
        return us_stocks
    
    async def start_surveillance(self):
        """Start US market surveillance with Alpha Vantage"""
        
        logger.info("🚀 Starting US Market Surveillance (Alpha Vantage Paid)")
        
        # Perform Alpha Vantage scan
        scan_results = await self.perform_us_market_scan()
        
        if scan_results:
            # Update top 101 stocks
            await self.update_top_101_us_stocks(scan_results)
            
            # Generate surveillance report
            await self.generate_us_surveillance_report(scan_results)
            
            logger.info("✅ US market surveillance completed successfully")
        else:
            logger.warning("⚠️ No valid scan results obtained")
    
    async def perform_us_market_scan(self) -> Dict[str, USStockMetrics]:
        """Perform US market scan with Alpha Vantage paid subscription"""
        
        logger.info(f"🔍 Starting Alpha Vantage scan of {len(self.us_universe)} US stocks")
        
        estimated_time = len(self.us_universe) * self.config["call_delay"] / 60
        logger.info(f"⏱️ Estimated time: {estimated_time:.1f} minutes")
        
        scan_results = {}
        alpha_vantage_success = 0
        yahoo_fallback_used = 0
        failed_symbols = []
        
        # Reset rate limiting
        self.alpha_vantage_calls = 0
        self.minute_start = time.time()
        
        for i, symbol in enumerate(self.us_universe):
            try:
                # Rate limiting for Alpha Vantage
                await self._enforce_alpha_vantage_rate_limit()
                
                # Try Alpha Vantage first (PRIMARY - paid subscription)
                metrics = await self._scan_with_alpha_vantage(symbol)
                self.alpha_vantage_calls += 1
                
                if metrics:
                    scan_results[symbol] = metrics
                    alpha_vantage_success += 1
                    logger.debug(f"✅ Alpha Vantage: {symbol} - Score: {metrics.composite_score:.3f}")
                else:
                    # Fallback to Yahoo Finance (rare for US stocks)
                    logger.debug(f"🔄 Alpha Vantage failed for {symbol}, trying Yahoo fallback")
                    metrics = await self._scan_with_yahoo_fallback(symbol)
                    
                    if metrics:
                        scan_results[symbol] = metrics
                        yahoo_fallback_used += 1
                        logger.debug(f"✅ Yahoo Fallback: {symbol} - Score: {metrics.composite_score:.3f}")
                    else:
                        failed_symbols.append(symbol)
                        logger.debug(f"❌ Both APIs failed for {symbol}")
                
                # Progress updates
                if (i + 1) % self.config["progress_batch_size"] == 0:
                    progress = (i + 1) / len(self.us_universe) * 100
                    logger.info(f"📊 Progress: {progress:.1f}% ({i+1}/{len(self.us_universe)}) - "
                              f"Alpha Vantage: {alpha_vantage_success}, Yahoo: {yahoo_fallback_used}, Failed: {len(failed_symbols)}")
                
                # Delay between calls
                await asyncio.sleep(self.config["call_delay"])
                
            except Exception as e:
                failed_symbols.append(symbol)
                logger.warning(f"❌ Error scanning {symbol}: {e}")
        
        # Final summary
        total_success = len(scan_results)
        logger.info(f"✅ US market scan complete:")
        logger.info(f"   📊 Total analyzed: {total_success}/{len(self.us_universe)}")
        logger.info(f"   🔑 Alpha Vantage success: {alpha_vantage_success}")
        logger.info(f"   🔄 Yahoo fallback used: {yahoo_fallback_used}")
        logger.info(f"   ❌ Failed: {len(failed_symbols)}")
        
        return scan_results
    
    async def _enforce_alpha_vantage_rate_limit(self):
        """Enforce Alpha Vantage rate limiting"""
        
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
    
    async def _scan_with_alpha_vantage(self, symbol: str) -> Optional[USStockMetrics]:
        """Scan using Alpha Vantage (PRIMARY - paid subscription)"""
        
        try:
            # Alpha Vantage works great with US symbols
            data, source = self.alpha_vantage.fetch_daily_data(symbol)
            
            if data is None or data.empty or len(data) < 5:
                return None
            
            return self._calculate_us_metrics(symbol, data)
            
        except Exception as e:
            logger.debug(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    async def _scan_with_yahoo_fallback(self, symbol: str) -> Optional[USStockMetrics]:
        """Scan using Yahoo Finance (FALLBACK ONLY)"""
        
        try:
            data, source = self.yahoo_fallback.fetch_data(symbol, period="1mo", interval="1d")
            
            if data is None or data.empty or len(data) < 5:
                return None
            
            return self._calculate_us_metrics(symbol, data)
            
        except Exception as e:
            logger.debug(f"Yahoo fallback error for {symbol}: {e}")
            return None
    
    def _calculate_us_metrics(self, symbol: str, data: pd.DataFrame) -> USStockMetrics:
        """Calculate comprehensive US stock metrics"""
        
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
        sector = self._classify_us_sector(symbol)
        exchange = self._classify_us_exchange(symbol)
        
        # Composite scoring
        composite_score = self._calculate_composite_score(
            daily_return, weekly_return, monthly_return, volatility,
            volume_ratio, rsi, momentum_score, breakout_score,
            is_penny_stock, market_cap
        )
        
        return USStockMetrics(
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
        if market_cap < 1000000000:  # Under $1B
            opportunity_score += 0.2
        if volatility > 0.4:  # High volatility
            opportunity_score += 0.1
        
        score += opportunity_score * 0.15
        
        return score
    
    def _estimate_market_cap(self, symbol: str, price: float, volume: int) -> float:
        """Estimate market cap for US stocks"""
        # US stocks typically have higher volumes
        if volume > 10000000:  # 10M+ volume
            return price * volume * 0.05  # Large cap
        elif volume > 1000000:  # 1M+ volume
            return price * volume * 0.1   # Mid cap
        else:
            return price * volume * 0.5   # Small cap
    
    def _classify_us_sector(self, symbol: str) -> str:
        """Classify US stock sector"""
        if symbol in ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
            return "Mega Tech"
        elif symbol in ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"]:
            return "Banking"
        elif symbol in ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT"]:
            return "Healthcare"
        elif symbol in ["XOM", "CVX", "COP", "EOG", "SLB", "MPC"]:
            return "Energy"
        elif symbol in ["PLTR", "SNOW", "CRWD", "ZS", "OKTA", "DDOG"]:
            return "Growth Tech"
        elif symbol in ["MRNA", "BNTX", "NVAX", "TDOC", "VEEV"]:
            return "Biotech"
        elif symbol in ["PYPL", "SQ", "V", "MA", "COIN", "HOOD"]:
            return "Fintech"
        elif symbol in ["TSLA", "NIO", "XPEV", "LI", "RIVN", "LCID"]:
            return "EV/Clean Energy"
        elif symbol in ["TLRY", "CGC", "ACB", "CRON", "SNDL"]:
            return "Cannabis"
        elif symbol in ["AMC", "GME", "BBBY", "CLOV", "WISH"]:
            return "Meme Stocks"
        else:
            return "Other"
    
    def _classify_us_exchange(self, symbol: str) -> str:
        """Classify US exchange (simplified)"""
        # Most growth/tech stocks are NASDAQ, traditional companies NYSE
        if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
            return "NASDAQ"
        elif symbol in ["JPM", "BAC", "WFC", "GS", "XOM", "CVX", "JNJ", "PFE"]:
            return "NYSE"
        else:
            return "NASDAQ"  # Default for growth stocks
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name (simplified)"""
        return symbol  # In production, would use company name lookup
    
    async def update_top_101_us_stocks(self, scan_results: Dict[str, USStockMetrics]):
        """Update top 101 US stocks"""
        
        logger.info("🏆 Ranking US stocks and selecting top 101...")
        
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
        
        logger.info(f"📊 Top 101 US stocks updated: {len(added)} added, {len(removed)} removed")
        
        # Save results
        await self._save_us_results(top_101, scan_results)
    
    async def _save_us_results(self, top_101: List[USStockMetrics], all_results: Dict[str, USStockMetrics]):
        """Save US market surveillance results"""
        
        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "scan_summary": {
                "total_scanned": len(all_results),
                "top_101_count": len(top_101),
                "primary_source": "Alpha Vantage (Paid Subscription)",
                "fallback_source": "Yahoo Finance",
                "rate_limit": f"{self.config['alpha_vantage_calls_per_minute']} calls/minute"
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
        output_file = "us_market_surveillance_results.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save simple symbol list
        with open("top_101_us_symbols.txt", 'w') as f:
            f.write("\n".join(self.current_top_101))
        
        logger.info(f"💾 US market results saved to {output_file}")
    
    async def generate_us_surveillance_report(self, scan_results: Dict[str, USStockMetrics]):
        """Generate final US market surveillance report"""
        
        if not self.current_top_101:
            return
        
        logger.info("📊 Generating US market surveillance report...")
        
        # Get top stocks
        top_10 = self.current_top_101[:10]
        
        # Sector analysis
        sectors = {}
        for stock in scan_results.values():
            sectors[stock.sector] = sectors.get(stock.sector, 0) + 1
        
        logger.info("🏆 TOP 10 US OPPORTUNITIES:")
        for i, symbol in enumerate(top_10, 1):
            logger.info(f"   {i:2d}. {symbol}")
        
        logger.info(f"📊 Market Composition:")
        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"   {sector}: {count} stocks")
        
        logger.info(f"💰 Penny stocks in top 101: {len([s for s in scan_results.values() if s.is_penny_stock])}")
        logger.info(f"🚀 Total US opportunities: {len(self.current_top_101)}")
        logger.info("✅ US market surveillance complete!")

# Usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = USMarketSurveillanceAgent()
        await agent.start_surveillance()
    
    asyncio.run(main())