"""
Questrade Options API Integration

Integrates with Questrade API for:
- Real-time options chain data
- Options quotes and Greeks
- Options execution
- Position tracking
"""

import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Represents an option contract"""
    symbol: str
    underlying_symbol: str
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None


@dataclass
class OptionsChain:
    """Represents an options chain for a symbol"""
    symbol: str
    underlying_price: float
    expiration_dates: List[datetime]
    strikes: List[float]
    calls: List[OptionContract]
    puts: List[OptionContract]
    updated_at: datetime


class QuestradeOptionsAPI:
    """
    Questrade Options API client for live options data.
    
    Features:
    - Options chain retrieval
    - Real-time quotes
    - Greeks calculation
    - Options search and filtering
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Questrade API credentials
        self.refresh_token = config.get('questrade_refresh_token', '')
        self.api_url = config.get('questrade_api_url', 'https://api01.iq.questrade.com/v1')
        self.access_token = None
        self.token_expiry = None
        
        # Cache settings
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_ttl = config.get('cache_ttl_seconds', 60)  # 1 minute cache
        self.options_cache: Dict[str, Tuple[OptionsChain, datetime]] = {}
        
        logger.info("Questrade Options API initialized")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Questrade API and get access token.
        
        Returns:
            True if authentication successful
        """
        try:
            if not self.refresh_token or self.refresh_token == 'demo':
                logger.warning("No valid Questrade refresh token provided")
                return False
            
            # Check if current token is still valid
            if self.access_token and self.token_expiry:
                if datetime.now() < self.token_expiry - timedelta(minutes=5):
                    return True
            
            # Get new access token
            url = "https://login.questrade.com/oauth2/token"
            params = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            self.api_url = data['api_server']
            expires_in = data['expires_in']
            
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info("Questrade API authentication successful")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Questrade authentication failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False
    
    def get_options_chain(self, symbol: str) -> Optional[OptionsChain]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol (e.g., 'AAPL', 'TD.TO')
        
        Returns:
            OptionsChain object or None if failed
        """
        try:
            # Check cache
            if self.cache_enabled and symbol in self.options_cache:
                chain, cached_time = self.options_cache[symbol]
                if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                    logger.debug(f"Using cached options chain for {symbol}")
                    return chain
            
            # Authenticate if needed
            if not self.authenticate():
                logger.warning("Authentication failed, cannot retrieve options chain")
                return None
            
            # Get symbol ID
            symbol_id = self._get_symbol_id(symbol)
            if not symbol_id:
                logger.warning(f"Symbol ID not found for {symbol}")
                return None
            
            # Get options chain from Questrade
            headers = {'Authorization': f'Bearer {self.access_token}'}
            url = f"{self.api_url}/symbols/{symbol_id}/options"
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse options chain
            chain = self._parse_options_chain(symbol, data)
            
            # Cache the result
            if self.cache_enabled:
                self.options_cache[symbol] = (chain, datetime.now())
            
            logger.info(f"Retrieved options chain for {symbol}: {len(chain.calls)} calls, {len(chain.puts)} puts")
            return chain
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving options chain: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing options chain: {e}")
            return None
    
    def get_option_quote(self, option_symbol: str) -> Optional[OptionContract]:
        """
        Get real-time quote for a specific option.
        
        Args:
            option_symbol: Option symbol
        
        Returns:
            OptionContract with current data
        """
        try:
            if not self.authenticate():
                return None
            
            # Get option ID
            option_id = self._get_symbol_id(option_symbol)
            if not option_id:
                return None
            
            # Get quote
            headers = {'Authorization': f'Bearer {self.access_token}'}
            url = f"{self.api_url}/markets/quotes/{option_id}"
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse option contract
            option = self._parse_option_contract(data['quotes'][0])
            
            logger.debug(f"Retrieved quote for {option_symbol}: bid={option.bid}, ask={option.ask}")
            return option
            
        except Exception as e:
            logger.error(f"Error retrieving option quote: {e}")
            return None
    
    def search_options(self, underlying_symbol: str, option_type: str, 
                       min_strike: Optional[float] = None, max_strike: Optional[float] = None,
                       expiry_min_days: Optional[int] = None, expiry_max_days: Optional[int] = None,
                       min_delta: Optional[float] = None, max_delta: Optional[float] = None) -> List[OptionContract]:
        """
        Search for options matching specific criteria.
        
        Args:
            underlying_symbol: Underlying symbol
            option_type: 'CALL' or 'PUT'
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            expiry_min_days: Minimum days to expiry
            expiry_max_days: Maximum days to expiry
            min_delta: Minimum delta
            max_delta: Maximum delta
        
        Returns:
            List of matching OptionContract objects
        """
        try:
            # Get full options chain
            chain = self.get_options_chain(underlying_symbol)
            if not chain:
                return []
            
            # Select calls or puts
            options = chain.calls if option_type.upper() == 'CALL' else chain.puts
            
            # Filter by criteria
            filtered = []
            now = datetime.now()
            
            for option in options:
                # Strike filter
                if min_strike and option.strike < min_strike:
                    continue
                if max_strike and option.strike > max_strike:
                    continue
                
                # Expiry filter
                if expiry_min_days or expiry_max_days:
                    days_to_expiry = (option.expiry - now).days
                    if expiry_min_days and days_to_expiry < expiry_min_days:
                        continue
                    if expiry_max_days and days_to_expiry > expiry_max_days:
                        continue
                
                # Delta filter
                if option.delta:
                    if min_delta and abs(option.delta) < min_delta:
                        continue
                    if max_delta and abs(option.delta) > max_delta:
                        continue
                
                filtered.append(option)
            
            logger.info(f"Found {len(filtered)} options matching criteria for {underlying_symbol}")
            return filtered
            
        except Exception as e:
            logger.error(f"Error searching options: {e}")
            return []
    
    def get_atm_options(self, underlying_symbol: str, underlying_price: float) -> Dict[str, OptionContract]:
        """
        Get at-the-money (ATM) options for a symbol.
        
        Args:
            underlying_symbol: Underlying symbol
            underlying_price: Current price of underlying
        
        Returns:
            Dictionary with 'call' and 'put' keys
        """
        try:
            chain = self.get_options_chain(underlying_symbol)
            if not chain:
                return {}
            
            # Find closest strike to current price
            strikes = sorted(chain.strikes)
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
            
            # Find ATM call and put
            atm_call = next((c for c in chain.calls if c.strike == atm_strike), None)
            atm_put = next((p for p in chain.puts if p.strike == atm_strike), None)
            
            result = {}
            if atm_call:
                result['call'] = atm_call
            if atm_put:
                result['put'] = atm_put
            
            logger.info(f"ATM strike for {underlying_symbol} @ ${underlying_price:.2f}: ${atm_strike:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting ATM options: {e}")
            return {}
    
    def _get_symbol_id(self, symbol: str) -> Optional[int]:
        """Get Questrade symbol ID for a symbol"""
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            url = f"{self.api_url}/symbols/search"
            params = {'prefix': symbol}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['symbols']:
                return data['symbols'][0]['symbolId']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting symbol ID: {e}")
            return None
    
    def _parse_options_chain(self, symbol: str, data: Dict) -> OptionsChain:
        """Parse Questrade options chain response"""
        # This is a simplified parser - actual implementation would be more complex
        expiration_dates = []
        strikes = set()
        calls = []
        puts = []
        
        # Parse options from response
        for option_data in data.get('optionChain', []):
            option = self._parse_option_contract(option_data)
            
            if option.expiry not in expiration_dates:
                expiration_dates.append(option.expiry)
            strikes.add(option.strike)
            
            if option.option_type == 'CALL':
                calls.append(option)
            else:
                puts.append(option)
        
        chain = OptionsChain(
            symbol=symbol,
            underlying_price=data.get('underlyingPrice', 0.0),
            expiration_dates=sorted(expiration_dates),
            strikes=sorted(list(strikes)),
            calls=calls,
            puts=puts,
            updated_at=datetime.now()
        )
        
        return chain
    
    def _parse_option_contract(self, data: Dict) -> OptionContract:
        """Parse option contract from Questrade data"""
        return OptionContract(
            symbol=data.get('symbol', ''),
            underlying_symbol=data.get('underlying', ''),
            option_type='CALL' if 'C' in data.get('symbol', '') else 'PUT',
            strike=data.get('strikePrice', 0.0),
            expiry=datetime.fromisoformat(data.get('expiryDate', '2024-01-01')),
            bid=data.get('bidPrice', 0.0),
            ask=data.get('askPrice', 0.0),
            last=data.get('lastTradePriceTrHrs', 0.0),
            volume=data.get('volume', 0),
            open_interest=data.get('openInterest', 0),
            implied_volatility=data.get('volatility', 0.0),
            delta=data.get('delta'),
            gamma=data.get('gamma'),
            theta=data.get('theta'),
            vega=data.get('vega'),
            rho=data.get('rho')
        )
    
    def get_statistics(self) -> Dict:
        """Get API usage statistics"""
        return {
            'authenticated': self.access_token is not None,
            'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
            'cache_size': len(self.options_cache),
            'cache_enabled': self.cache_enabled
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize API
    config = {
        'questrade_refresh_token': 'demo',  # Replace with real token
        'cache_enabled': True
    }
    
    api = QuestradeOptionsAPI(config)
    
    # Test authentication
    print("\n1. Testing authentication...")
    success = api.authenticate()
    print(f"   Authentication: {'Success' if success else 'Failed'}")
    
    # This would work with a real token:
    # chain = api.get_options_chain('AAPL')
    # print(f"\n2. Options chain for AAPL:")
    # print(f"   Calls: {len(chain.calls)}")
    # print(f"   Puts: {len(chain.puts)}")
    
    print("\n3. API Statistics:")
    print(f"   {api.get_statistics()}")

