"""
Symbol Manager - TSX/TSXV Symbol Lists

Manages the 100 carefully selected Canadian stocks for data collection.
Prioritizes S&P/TSX 60 core holdings plus high-volume TSXV and growth stocks.
"""

import json
import logging
from typing import List, Dict, Set
from pathlib import Path
import yfinance as yf
from datetime import datetime

logger = logging.getLogger(__name__)

class SymbolManager:
    """Manages TSX/TSXV symbol lists with priority classification"""
    
    def __init__(self):
        self.symbols = self._load_symbols()
        self.verified_symbols = set()
        
        # Initialize enhanced collector for verification
        try:
            from .enhanced_collectors import MultiSourceDataCollector
            self.data_collector = MultiSourceDataCollector()
            self.use_enhanced_verification = True
            logger.info("✅ Using enhanced collectors for symbol verification")
        except ImportError:
            self.data_collector = None
            self.use_enhanced_verification = False
            logger.warning("⚠️ Enhanced collectors not available, using yfinance only")
        
    def _load_symbols(self) -> Dict[str, List[str]]:
        """Load the curated list of 100 TSX/TSXV symbols"""
        
        symbols = {
            # S&P/TSX 60 Core Holdings (30 symbols) - HIGHEST PRIORITY
            "tsx_core": [
                # Big 6 Banks
                "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
                
                # Energy Giants
                "CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO", "CVE.TO",
                
                # Tech & Growth
                "SHOP.TO", "CSU.TO", "LSPD.TO", "NVEI.TO",
                
                # Industrials & Materials
                "WCN.TO", "WSP.TO", "AEM.TO", "K.TO", "NTR.TO", "FNV.TO",
                
                # Consumer & Retail
                "ATD.TO", "DOL.TO", "QSR.TO", "MG.TO",
                
                # Financials & Insurance
                "MFC.TO", "SLF.TO", "IFC.TO", "PWF.TO",
                
                # Utilities & Infrastructure
                "FTS.TO", "EMA.TO"
            ],
            
            # High-Volume TSXV & Growth (25 symbols) - HIGH PRIORITY
            "growth_tech": [
                # Crypto/Blockchain
                "HUT.TO", "BITF.TO", "HIVE.TO", "DMGI.TO",
                
                # Psychedelics/Biotech
                "MMED.TO", "NUMI.TO", "CYBN.TO", "FTRP.TO",
                
                # Clean Energy
                "DCBO.TO", "PYR.TO", "HPQ.TO", "NOU.TO",
                
                # Tech/Software
                "TOI.TO", "WELL.TO", "CTS.TO", "QTRH.TO",
                
                # Mining/Resources
                "PMET.TO", "LAC.TO", "CRE.TO", "FL.TO",
                
                # Cannabis (select survivors)
                "WEED.TO", "ACB.TO", "HEXO.TO", "OGI.TO",
                
                # Other Growth
                "FOOD.TO"
            ],
            
            # Dividend & Utilities (25 symbols) - MEDIUM PRIORITY
            "dividend_utilities": [
                # Utilities
                "H.TO", "CU.TO", "AQN.TO", "BEP-UN.TO", "BIP-UN.TO",
                
                # REITs
                "CAR-UN.TO", "REI-UN.TO", "HR-UN.TO", "CHP-UN.TO", "FCR-UN.TO",
                
                # Telecom
                "T.TO", "BCE.TO", "RCI-B.TO", "TU.TO",
                
                # Consumer Staples
                "L.TO", "MRU.TO", "EMP-A.TO", "GIL.TO",
                
                # Infrastructure
                "BEI-UN.TO", "KMP-UN.TO", "IPL.TO", "PPL.TO",
                
                # Other Dividend
                "KEY.TO", "PXT.TO", "ARX.TO"
            ],
            
            # Cyclical & Commodity (20 symbols) - MEDIUM PRIORITY  
            "cyclical_commodity": [
                # Base Metals
                "FM.TO", "TKO.TO", "CS.TO", "HBM.TO", "LUN.TO",
                
                # Gold Miners
                "ABX.TO", "WPM.TO", "KL.TO", "ELD.TO", "YRI.TO",
                
                # Forestry
                "WFG.TO", "CFP.TO", "IFP.TO",
                
                # Transportation
                "CNR.TO", "CP.TO", "AC.TO", "CJT.TO",
                
                # Retail/Consumer Cyclical
                "CTC-A.TO", "BBD-B.TO", "GFL.TO"
            ]
        }
        
        return symbols
    
    def get_all_symbols(self) -> List[str]:
        """Get all 100 symbols as a flat list"""
        all_symbols = []
        for category in self.symbols.values():
            all_symbols.extend(category)
        return all_symbols
    
    def get_symbols_by_priority(self) -> Dict[str, List[str]]:
        """Get symbols organized by priority for collection"""
        return {
            "HIGHEST": self.symbols["tsx_core"],
            "HIGH": self.symbols["growth_tech"], 
            "MEDIUM": self.symbols["dividend_utilities"] + self.symbols["cyclical_commodity"]
        }
    
    def verify_symbol_availability(self, symbol: str) -> bool:
        """Verify a symbol is available using enhanced collectors or yfinance"""
        try:
            if self.use_enhanced_verification and self.data_collector:
                # Use enhanced multi-source collector (Alpha Vantage priority)
                data, source = self.data_collector.fetch_data(symbol, period="5d", interval="1d")
                if data is not None and not data.empty:
                    self.verified_symbols.add(symbol)
                    logger.info(f"✅ Verified symbol: {symbol} from {source}")
                    return True
                else:
                    logger.warning(f"⚠️ No data available for symbol: {symbol}")
                    return False
            else:
                # Fallback to yfinance only
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    self.verified_symbols.add(symbol)
                    logger.info(f"✅ Verified symbol: {symbol}")
                    return True
                else:
                    logger.warning(f"⚠️ No data available for symbol: {symbol}")
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to verify symbol {symbol}: {e}")
            return False
    
    def verify_all_symbols(self) -> Dict[str, bool]:
        """Verify all symbols and return results"""
        results = {}
        all_symbols = self.get_all_symbols()
        
        logger.info(f"🔍 Verifying {len(all_symbols)} TSX/TSXV symbols...")
        
        for i, symbol in enumerate(all_symbols, 1):
            logger.info(f"Verifying {i}/{len(all_symbols)}: {symbol}")
            results[symbol] = self.verify_symbol_availability(symbol)
            
            # Small delay to be respectful to yfinance
            import time
            time.sleep(0.5)
        
        # Summary
        verified_count = sum(results.values())
        logger.info(f"✅ Verification complete: {verified_count}/{len(all_symbols)} symbols available")
        
        return results
    
    def get_verified_symbols(self) -> List[str]:
        """Get only verified symbols"""
        return list(self.verified_symbols)
    
    def save_verification_results(self, results: Dict[str, bool], filepath: str = "data/symbol_verification.json"):
        """Save verification results to file"""
        verification_data = {
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(results),
            "verified_symbols": sum(results.values()),
            "verification_rate": sum(results.values()) / len(results),
            "results": results,
            "verified_list": [symbol for symbol, verified in results.items() if verified]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(verification_data, f, indent=2)
        
        logger.info(f"💾 Verification results saved to {filepath}")
    
    def load_verification_results(self, filepath: str = "data/symbol_verification.json") -> Dict[str, bool]:
        """Load previous verification results"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Update verified symbols set
            self.verified_symbols = set(data.get("verified_list", []))
            
            logger.info(f"📂 Loaded verification results: {len(self.verified_symbols)} verified symbols")
            return data.get("results", {})
            
        except FileNotFoundError:
            logger.info("No previous verification results found")
            return {}
        except Exception as e:
            logger.error(f"Failed to load verification results: {e}")
            return {}

# Convenience function
def get_tsx_symbols() -> List[str]:
    """Get all TSX/TSXV symbols (convenience function)"""
    manager = SymbolManager()
    return manager.get_all_symbols()

# For quick testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = SymbolManager()
    
    print("📊 TSX/TSXV Symbol Manager")
    print(f"Total symbols: {len(manager.get_all_symbols())}")
    
    # Show breakdown by category
    for priority, symbols in manager.get_symbols_by_priority().items():
        print(f"{priority} Priority: {len(symbols)} symbols")
        print(f"  Examples: {', '.join(symbols[:5])}")
    
    # Verify a few symbols as test
    test_symbols = ["RY.TO", "TD.TO", "SHOP.TO"]
    print(f"\n🔍 Testing symbol verification with: {test_symbols}")
    
    for symbol in test_symbols:
        result = manager.verify_symbol_availability(symbol)
        print(f"  {symbol}: {'✅ Available' if result else '❌ Not available'}")