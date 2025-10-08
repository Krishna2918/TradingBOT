"""
Options Chain and Greeks Processor
Handles options data, Greeks calculation, and delta-equivalent sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy.stats import norm
import math

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Individual option contract data"""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'C' or 'P'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class OptionsChain:
    """Complete options chain for an underlying"""
    underlying: str
    timestamp: datetime
    underlying_price: float
    calls: List[OptionContract]
    puts: List[OptionContract]
    expiry_dates: List[datetime]

class BlackScholesCalculator:
    """Black-Scholes options pricing and Greeks calculator"""
    
    @staticmethod
    def calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 for Black-Scholes formula"""
        if T <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    @staticmethod
    def calculate_d2(d1: float, sigma: float, T: float) -> float:
        """Calculate d2 for Black-Scholes formula"""
        return d1 - sigma * math.sqrt(T)
    
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call price"""
        if T <= 0:
            return max(S - K, 0)
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, sigma, T)
        
        call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put price"""
        if T <= 0:
            return max(K - S, 0)
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, sigma, T)
        
        put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option delta"""
        if T <= 0:
            if option_type == 'C':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        
        if option_type == 'C':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0
    
    @staticmethod
    def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma"""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    @staticmethod
    def calculate_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option theta (time decay)"""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, sigma, T)
        
        term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
        
        if option_type == 'C':
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        
        return term1 + term2
    
    @staticmethod
    def calculate_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega (volatility sensitivity)"""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T)
    
    @staticmethod
    def calculate_rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option rho (interest rate sensitivity)"""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, sigma, T)
        
        if option_type == 'C':
            return K * T * math.exp(-r * T) * norm.cdf(d2)
        else:
            return -K * T * math.exp(-r * T) * norm.cdf(-d2)

class OptionsChainProcessor:
    """Processes options chain data and calculates Greeks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.05)  # 5% default
        self.bs_calculator = BlackScholesCalculator()
        
        logger.info("Options Chain Processor initialized")
    
    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            # Initial guess
            sigma = 0.2
            
            for _ in range(100):  # Max iterations
                if option_type == 'C':
                    theoretical_price = self.bs_calculator.black_scholes_call(S, K, T, r, sigma)
                else:
                    theoretical_price = self.bs_calculator.black_scholes_put(S, K, T, r, sigma)
                
                # Calculate vega for Newton-Raphson
                vega = self.bs_calculator.calculate_vega(S, K, T, r, sigma)
                
                if vega == 0:
                    break
                
                # Newton-Raphson update
                price_diff = theoretical_price - market_price
                if abs(price_diff) < 0.001:  # Convergence
                    break
                
                sigma = sigma - price_diff / vega
                
                # Bounds check
                sigma = max(0.001, min(5.0, sigma))
            
            return sigma
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return 0.2  # Default fallback
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate all Greeks for an option"""
        try:
            greeks = {
                'delta': self.bs_calculator.calculate_delta(S, K, T, r, sigma, option_type),
                'gamma': self.bs_calculator.calculate_gamma(S, K, T, r, sigma),
                'theta': self.bs_calculator.calculate_theta(S, K, T, r, sigma, option_type),
                'vega': self.bs_calculator.calculate_vega(S, K, T, r, sigma),
                'rho': self.bs_calculator.calculate_rho(S, K, T, r, sigma, option_type)
            }
            return greeks
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    def create_option_contract(self, symbol: str, underlying: str, strike: float,
                             expiry: datetime, option_type: str, market_price: float,
                             underlying_price: float, volume: int = 0, 
                             open_interest: int = 0) -> OptionContract:
        """Create an option contract with calculated Greeks"""
        try:
            # Calculate time to expiry
            T = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
            T = max(0.001, T)  # Minimum time to avoid division by zero
            
            # Calculate implied volatility
            iv = self.calculate_implied_volatility(
                market_price, underlying_price, strike, T, self.risk_free_rate, option_type
            )
            
            # Calculate Greeks
            greeks = self.calculate_greeks(
                underlying_price, strike, T, self.risk_free_rate, iv, option_type
            )
            
            return OptionContract(
                symbol=symbol,
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                bid=market_price - 0.01,  # Simulate bid/ask
                ask=market_price + 0.01,
                last_price=market_price,
                volume=volume,
                open_interest=open_interest,
                implied_volatility=iv,
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                rho=greeks['rho']
            )
            
        except Exception as e:
            logger.error(f"Error creating option contract: {e}")
            return None
    
    def generate_options_chain(self, underlying: str, underlying_price: float,
                             expiry_dates: List[datetime]) -> OptionsChain:
        """Generate a complete options chain for an underlying"""
        try:
            calls = []
            puts = []
            
            for expiry in expiry_dates:
                # Generate strikes around current price
                atm_strike = round(underlying_price)
                strikes = [atm_strike + i * 5 for i in range(-10, 11)]  # ±50 points
                
                for strike in strikes:
                    # Skip if strike is too far from current price
                    if abs(strike - underlying_price) / underlying_price > 0.5:
                        continue
                    
                    # Generate call option
                    call_symbol = f"{underlying}{expiry.strftime('%y%m%d')}C{strike:05d}"
                    call_price = self._estimate_option_price(underlying_price, strike, expiry, 'C')
                    
                    if call_price > 0.01:  # Only include liquid options
                        call_contract = self.create_option_contract(
                            call_symbol, underlying, strike, expiry, 'C',
                            call_price, underlying_price
                        )
                        if call_contract:
                            calls.append(call_contract)
                    
                    # Generate put option
                    put_symbol = f"{underlying}{expiry.strftime('%y%m%d')}P{strike:05d}"
                    put_price = self._estimate_option_price(underlying_price, strike, expiry, 'P')
                    
                    if put_price > 0.01:  # Only include liquid options
                        put_contract = self.create_option_contract(
                            put_symbol, underlying, strike, expiry, 'P',
                            put_price, underlying_price
                        )
                        if put_contract:
                            puts.append(put_contract)
            
            return OptionsChain(
                underlying=underlying,
                timestamp=datetime.now(),
                underlying_price=underlying_price,
                calls=calls,
                puts=puts,
                expiry_dates=expiry_dates
            )
            
        except Exception as e:
            logger.error(f"Error generating options chain: {e}")
            return None
    
    def _estimate_option_price(self, S: float, K: float, expiry: datetime, option_type: str) -> float:
        """Estimate option price using Black-Scholes with reasonable volatility"""
        try:
            T = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
            T = max(0.001, T)
            
            # Use reasonable volatility estimate
            sigma = 0.25  # 25% annual volatility
            
            if option_type == 'C':
                return self.bs_calculator.black_scholes_call(S, K, T, self.risk_free_rate, sigma)
            else:
                return self.bs_calculator.black_scholes_put(S, K, T, self.risk_free_rate, sigma)
                
        except Exception as e:
            logger.error(f"Error estimating option price: {e}")
            return 0.0
    
    def calculate_delta_equivalent_position(self, option_contracts: List[OptionContract]) -> float:
        """Calculate delta-equivalent position size"""
        try:
            total_delta = 0.0
            
            for contract in option_contracts:
                # Delta represents shares of underlying
                total_delta += contract.delta * 100  # Options are typically 100 shares
            
            return total_delta
            
        except Exception as e:
            logger.error(f"Error calculating delta equivalent: {e}")
            return 0.0
    
    def find_atm_options(self, options_chain: OptionsChain, expiry: datetime) -> Tuple[OptionContract, OptionContract]:
        """Find at-the-money call and put options for a given expiry"""
        try:
            underlying_price = options_chain.underlying_price
            
            # Find closest strike to current price
            calls_for_expiry = [c for c in options_chain.calls if c.expiry.date() == expiry.date()]
            puts_for_expiry = [p for p in options_chain.puts if p.expiry.date() == expiry.date()]
            
            if not calls_for_expiry or not puts_for_expiry:
                return None, None
            
            # Find ATM call
            atm_call = min(calls_for_expiry, key=lambda x: abs(x.strike - underlying_price))
            
            # Find ATM put
            atm_put = min(puts_for_expiry, key=lambda x: abs(x.strike - underlying_price))
            
            return atm_call, atm_put
            
        except Exception as e:
            logger.error(f"Error finding ATM options: {e}")
            return None, None
    
    def calculate_put_call_ratio(self, options_chain: OptionsChain) -> float:
        """Calculate put/call ratio for sentiment analysis"""
        try:
            total_call_volume = sum(c.volume for c in options_chain.calls)
            total_put_volume = sum(p.volume for p in options_chain.puts)
            
            if total_call_volume == 0:
                return float('inf') if total_put_volume > 0 else 0.0
            
            return total_put_volume / total_call_volume
            
        except Exception as e:
            logger.error(f"Error calculating put/call ratio: {e}")
            return 0.0
    
    def get_volatility_smile(self, options_chain: OptionsChain, expiry: datetime) -> pd.DataFrame:
        """Extract volatility smile for a given expiry"""
        try:
            calls_for_expiry = [c for c in options_chain.calls if c.expiry.date() == expiry.date()]
            puts_for_expiry = [p for p in options_chain.puts if p.expiry.date() == expiry.date()]
            
            smile_data = []
            
            # Add calls
            for call in calls_for_expiry:
                smile_data.append({
                    'strike': call.strike,
                    'iv': call.implied_volatility,
                    'type': 'call',
                    'moneyness': call.strike / options_chain.underlying_price
                })
            
            # Add puts
            for put in puts_for_expiry:
                smile_data.append({
                    'strike': put.strike,
                    'iv': put.implied_volatility,
                    'type': 'put',
                    'moneyness': put.strike / options_chain.underlying_price
                })
            
            return pd.DataFrame(smile_data)
            
        except Exception as e:
            logger.error(f"Error extracting volatility smile: {e}")
            return pd.DataFrame()

class OptionsDataManager:
    """Manages options data collection and processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processor = OptionsChainProcessor(config)
        self.symbols = config.get('symbols', ['TD.TO', 'RY.TO', 'SHOP.TO'])
        self.expiry_dates = self._generate_expiry_dates()
        
        logger.info("Options Data Manager initialized")
    
    def _generate_expiry_dates(self) -> List[datetime]:
        """Generate standard options expiry dates"""
        today = datetime.now()
        expiry_dates = []
        
        # Add next 4 Fridays (weekly options)
        for i in range(1, 5):
            days_ahead = (4 - today.weekday()) % 7 + (i - 1) * 7
            if days_ahead == 0:
                days_ahead = 7
            expiry = today + timedelta(days=days_ahead)
            expiry_dates.append(expiry)
        
        # Add monthly expiries (3rd Friday of next 3 months)
        for i in range(1, 4):
            month = today.month + i
            year = today.year
            if month > 12:
                month -= 12
                year += 1
            
            # Find 3rd Friday of the month
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
            expiry_dates.append(third_friday)
        
        return sorted(expiry_dates)
    
    def get_options_chain(self, symbol: str, underlying_price: float) -> OptionsChain:
        """Get options chain for a symbol"""
        return self.processor.generate_options_chain(symbol, underlying_price, self.expiry_dates)
    
    def get_delta_equivalent_exposure(self, symbol: str, underlying_price: float) -> float:
        """Get delta-equivalent exposure for a symbol"""
        options_chain = self.get_options_chain(symbol, underlying_price)
        if not options_chain:
            return 0.0
        
        # For now, just return the underlying price (simplified)
        # In practice, this would sum up all option positions
        return underlying_price
    
    def get_put_call_ratio(self, symbol: str, underlying_price: float) -> float:
        """Get put/call ratio for sentiment analysis"""
        options_chain = self.get_options_chain(symbol, underlying_price)
        if not options_chain:
            return 0.0
        
        return self.processor.calculate_put_call_ratio(options_chain)
    
    def get_volatility_smile(self, symbol: str, underlying_price: float, expiry: datetime) -> pd.DataFrame:
        """Get volatility smile for a symbol and expiry"""
        options_chain = self.get_options_chain(symbol, underlying_price)
        if not options_chain:
            return pd.DataFrame()
        
        return self.processor.get_volatility_smile(options_chain, expiry)
