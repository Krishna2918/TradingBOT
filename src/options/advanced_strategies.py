"""
Advanced Options Strategies

Implements sophisticated options strategies:
- Butterfly spreads
- Iron butterflies
- Calendar spreads
- Diagonal spreads
- Ratio spreads
- Box spreads
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Advanced strategy types"""
    BUTTERFLY = "butterfly"
    IRON_BUTTERFLY = "iron_butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    RATIO_SPREAD = "ratio_spread"
    BOX_SPREAD = "box_spread"
    JADE_LIZARD = "jade_lizard"
    BIG_LIZARD = "big_lizard"


@dataclass
class StrategyLeg:
    """Represents one leg of a strategy"""
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    quantity: int
    action: str  # 'BUY' or 'SELL'


@dataclass
class StrategyDefinition:
    """Defines an options strategy"""
    name: str
    strategy_type: StrategyType
    legs: List[StrategyLeg]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    margin_requirement: float
    ideal_market_outlook: str
    risk_reward_ratio: float


class AdvancedStrategiesBuilder:
    """
    Builder for advanced options strategies.
    
    Creates complex multi-leg options strategies with:
    - Risk/reward calculations
    - Breakeven analysis
    - Margin requirements
    - Market outlook recommendations
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.commission_per_leg = self.config.get('commission_per_leg', 0.65)
        
        logger.info("Advanced Strategies Builder initialized")
    
    def create_butterfly_spread(self, underlying: str, center_strike: float, wing_width: float,
                                option_type: str, expiry: datetime, quantity: int = 1) -> StrategyDefinition:
        """
        Create a butterfly spread (buy-sell-sell-buy pattern).
        
        Args:
            underlying: Underlying symbol
            center_strike: Center strike price
            wing_width: Distance between strikes
            option_type: 'CALL' or 'PUT'
            expiry: Expiration date
            quantity: Number of butterflies
        
        Returns:
            StrategyDefinition
        
        Example:
            >>> # Long call butterfly: Buy 1x $45, Sell 2x $50, Buy 1x $55
            >>> builder.create_butterfly_spread('AAPL', 50.0, 5.0, 'CALL', expiry, 1)
        
        Characteristics:
            - Limited risk, limited reward
            - Profits if stock stays near center strike
            - Low cost, high potential ROI
            - Best for low volatility
        """
        try:
            lower_strike = center_strike - wing_width
            upper_strike = center_strike + wing_width
            
            legs = [
                StrategyLeg(option_type, lower_strike, expiry, quantity, 'BUY'),
                StrategyLeg(option_type, center_strike, expiry, quantity * 2, 'SELL'),
                StrategyLeg(option_type, upper_strike, expiry, quantity, 'BUY')
            ]
            
            # Estimate pricing (simplified)
            lower_price = 5.0  # Assume prices
            center_price = 3.0
            upper_price = 1.5
            
            net_debit = (lower_price + upper_price - (2 * center_price)) * 100 * quantity
            max_profit = (wing_width * 100 * quantity) - net_debit
            max_loss = net_debit
            
            breakeven_lower = center_strike - (net_debit / (100 * quantity))
            breakeven_upper = center_strike + (net_debit / (100 * quantity))
            
            strategy = StrategyDefinition(
                name=f"{option_type} Butterfly",
                strategy_type=StrategyType.BUTTERFLY,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                margin_requirement=max_loss,
                ideal_market_outlook="Neutral - expect stock to stay near center strike",
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
            logger.info(f"Created {option_type} butterfly: ${lower_strike}/${center_strike}/${upper_strike} | Max profit: ${max_profit:.2f}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating butterfly spread: {e}")
            raise
    
    def create_iron_butterfly(self, underlying: str, center_strike: float, wing_width: float,
                             expiry: datetime, quantity: int = 1) -> StrategyDefinition:
        """
        Create an iron butterfly (short straddle + long strangle).
        
        Args:
            underlying: Underlying symbol
            center_strike: Center strike (short straddle)
            wing_width: Distance to protective wings
            expiry: Expiration date
            quantity: Number of iron butterflies
        
        Returns:
            StrategyDefinition
        
        Example:
            >>> # Sell $50 straddle, Buy $45 put and $55 call
            >>> builder.create_iron_butterfly('AAPL', 50.0, 5.0, expiry, 1)
        
        Characteristics:
            - Limited risk, limited reward
            - Credit strategy
            - Profits if stock stays near center strike
            - Higher probability of profit than butterfly
        """
        try:
            lower_strike = center_strike - wing_width
            upper_strike = center_strike + wing_width
            
            legs = [
                StrategyLeg('PUT', lower_strike, expiry, quantity, 'BUY'),
                StrategyLeg('PUT', center_strike, expiry, quantity, 'SELL'),
                StrategyLeg('CALL', center_strike, expiry, quantity, 'SELL'),
                StrategyLeg('CALL', upper_strike, expiry, quantity, 'BUY')
            ]
            
            # Estimate pricing
            lower_put_price = 1.5
            center_put_price = 3.5
            center_call_price = 3.5
            upper_call_price = 1.5
            
            net_credit = (center_put_price + center_call_price - lower_put_price - upper_call_price) * 100 * quantity
            max_profit = net_credit
            max_loss = (wing_width * 100 * quantity) - net_credit
            
            breakeven_lower = center_strike - (net_credit / (100 * quantity))
            breakeven_upper = center_strike + (net_credit / (100 * quantity))
            
            strategy = StrategyDefinition(
                name="Iron Butterfly",
                strategy_type=StrategyType.IRON_BUTTERFLY,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                margin_requirement=max_loss,
                ideal_market_outlook="Neutral - expect low volatility",
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
            logger.info(f"Created iron butterfly: ${lower_strike}/${center_strike}/${upper_strike} | Max profit: ${max_profit:.2f}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating iron butterfly: {e}")
            raise
    
    def create_calendar_spread(self, underlying: str, strike: float, option_type: str,
                               near_expiry: datetime, far_expiry: datetime, quantity: int = 1) -> StrategyDefinition:
        """
        Create a calendar spread (time spread).
        
        Args:
            underlying: Underlying symbol
            strike: Strike price (same for both legs)
            option_type: 'CALL' or 'PUT'
            near_expiry: Near-term expiration
            far_expiry: Far-term expiration
            quantity: Number of spreads
        
        Returns:
            StrategyDefinition
        
        Example:
            >>> # Sell Feb $50 call, Buy Apr $50 call
            >>> builder.create_calendar_spread('AAPL', 50.0, 'CALL', feb_expiry, apr_expiry, 1)
        
        Characteristics:
            - Profits from time decay
            - Best when volatility increases
            - Limited risk
            - Theta positive
        """
        try:
            legs = [
                StrategyLeg(option_type, strike, near_expiry, quantity, 'SELL'),
                StrategyLeg(option_type, strike, far_expiry, quantity, 'BUY')
            ]
            
            # Estimate pricing
            near_price = 2.0
            far_price = 4.0
            
            net_debit = (far_price - near_price) * 100 * quantity
            max_loss = net_debit
            max_profit = net_debit * 2  # Estimate (actual depends on vol changes)
            
            strategy = StrategyDefinition(
                name=f"{option_type} Calendar Spread",
                strategy_type=StrategyType.CALENDAR_SPREAD,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[strike],
                margin_requirement=max_loss,
                ideal_market_outlook="Neutral - expect stock to stay near strike, volatility to increase",
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
            logger.info(f"Created {option_type} calendar spread: ${strike} | Near: {near_expiry.date()} | Far: {far_expiry.date()}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating calendar spread: {e}")
            raise
    
    def create_diagonal_spread(self, underlying: str, long_strike: float, short_strike: float,
                              option_type: str, near_expiry: datetime, far_expiry: datetime,
                              quantity: int = 1) -> StrategyDefinition:
        """
        Create a diagonal spread (different strikes and expirations).
        
        Args:
            underlying: Underlying symbol
            long_strike: Strike of long leg
            short_strike: Strike of short leg
            option_type: 'CALL' or 'PUT'
            near_expiry: Near-term expiration (short leg)
            far_expiry: Far-term expiration (long leg)
            quantity: Number of spreads
        
        Returns:
            StrategyDefinition
        
        Example:
            >>> # Buy Apr $50 call, Sell Feb $55 call (bullish diagonal)
            >>> builder.create_diagonal_spread('AAPL', 50.0, 55.0, 'CALL', feb_expiry, apr_expiry, 1)
        
        Characteristics:
            - Directional bias with time decay
            - Lower cost than vertical spread
            - Benefits from theta and delta
            - More complex risk profile
        """
        try:
            legs = [
                StrategyLeg(option_type, short_strike, near_expiry, quantity, 'SELL'),
                StrategyLeg(option_type, long_strike, far_expiry, quantity, 'BUY')
            ]
            
            # Estimate pricing
            short_price = 2.5
            long_price = 5.0
            
            net_debit = (long_price - short_price) * 100 * quantity
            max_loss = net_debit
            max_profit = abs(long_strike - short_strike) * 100 * quantity  # Estimate
            
            strategy = StrategyDefinition(
                name=f"{option_type} Diagonal Spread",
                strategy_type=StrategyType.DIAGONAL_SPREAD,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[long_strike],
                margin_requirement=max_loss,
                ideal_market_outlook=f"{'Bullish' if option_type == 'CALL' else 'Bearish'} - moderate move expected",
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
            logger.info(f"Created {option_type} diagonal spread: ${long_strike}/${short_strike}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating diagonal spread: {e}")
            raise
    
    def create_ratio_spread(self, underlying: str, long_strike: float, short_strike: float,
                           option_type: str, expiry: datetime, long_quantity: int = 1,
                           short_quantity: int = 2) -> StrategyDefinition:
        """
        Create a ratio spread (unbalanced spread).
        
        Args:
            underlying: Underlying symbol
            long_strike: Strike of long leg
            short_strike: Strike of short legs
            option_type: 'CALL' or 'PUT'
            expiry: Expiration date
            long_quantity: Quantity of long leg
            short_quantity: Quantity of short legs (usually 2:1 ratio)
        
        Returns:
            StrategyDefinition
        
        Example:
            >>> # Buy 1x $50 call, Sell 2x $55 calls (call ratio spread)
            >>> builder.create_ratio_spread('AAPL', 50.0, 55.0, 'CALL', expiry, 1, 2)
        
        Characteristics:
            - Can be credit or debit
            - Unlimited risk on short side
            - Profits from moderate move
            - Advanced strategy - use with caution
        """
        try:
            legs = [
                StrategyLeg(option_type, long_strike, expiry, long_quantity, 'BUY'),
                StrategyLeg(option_type, short_strike, expiry, short_quantity, 'SELL')
            ]
            
            # Estimate pricing
            long_price = 5.0
            short_price = 2.5
            
            net_debit_credit = (short_price * short_quantity - long_price * long_quantity) * 100
            
            if net_debit_credit > 0:  # Credit
                max_profit = (abs(long_strike - short_strike) * 100 * long_quantity) + net_debit_credit
                max_loss = float('inf')  # Unlimited on naked short
            else:  # Debit
                max_profit = abs(long_strike - short_strike) * 100 * long_quantity
                max_loss = abs(net_debit_credit)
            
            strategy = StrategyDefinition(
                name=f"{option_type} Ratio Spread",
                strategy_type=StrategyType.RATIO_SPREAD,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[short_strike],
                margin_requirement=10000.0,  # High margin for naked short
                ideal_market_outlook="Moderate move expected, not too large",
                risk_reward_ratio=max_profit / max_loss if max_loss != float('inf') and max_loss > 0 else 0
            )
            
            logger.warning(f"Created {option_type} ratio spread: ${long_strike}/${short_strike} | UNLIMITED RISK")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating ratio spread: {e}")
            raise
    
    def create_jade_lizard(self, underlying: str, short_call_strike: float, long_call_strike: float,
                          short_put_strike: float, expiry: datetime, quantity: int = 1) -> StrategyDefinition:
        """
        Create a jade lizard (short call spread + short put).
        
        Args:
            underlying: Underlying symbol
            short_call_strike: Short call strike
            long_call_strike: Long call strike (higher)
            short_put_strike: Short put strike (lower than calls)
            expiry: Expiration date
            quantity: Number of jade lizards
        
        Returns:
            StrategyDefinition
        
        Characteristics:
            - Credit strategy
            - No upside risk
            - Limited downside risk
            - Profits from neutral to bullish move
        """
        try:
            legs = [
                StrategyLeg('CALL', short_call_strike, expiry, quantity, 'SELL'),
                StrategyLeg('CALL', long_call_strike, expiry, quantity, 'BUY'),
                StrategyLeg('PUT', short_put_strike, expiry, quantity, 'SELL')
            ]
            
            # Estimate pricing
            short_call_price = 3.0
            long_call_price = 1.0
            short_put_price = 3.5
            
            net_credit = (short_call_price + short_put_price - long_call_price) * 100 * quantity
            max_profit = net_credit
            max_loss = (short_put_strike * 100 * quantity) - net_credit  # If put assigned
            
            strategy = StrategyDefinition(
                name="Jade Lizard",
                strategy_type=StrategyType.JADE_LIZARD,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[short_put_strike - (net_credit / (100 * quantity))],
                margin_requirement=max_loss,
                ideal_market_outlook="Neutral to bullish - no upside risk",
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
            logger.info(f"Created jade lizard: Calls ${short_call_strike}/${long_call_strike}, Put ${short_put_strike}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating jade lizard: {e}")
            raise
    
    def analyze_strategy(self, strategy: StrategyDefinition, underlying_price: float) -> Dict:
        """
        Analyze a strategy's current status and probabilities.
        
        Args:
            strategy: StrategyDefinition to analyze
            underlying_price: Current underlying price
        
        Returns:
            Analysis dictionary
        """
        try:
            # Determine if in profit zone
            in_profit_zone = False
            for breakeven in strategy.breakeven_points:
                if underlying_price > breakeven:
                    in_profit_zone = True
                    break
            
            # Calculate current P&L (simplified)
            distance_from_center = 0.0
            if strategy.breakeven_points:
                avg_breakeven = sum(strategy.breakeven_points) / len(strategy.breakeven_points)
                distance_from_center = abs(underlying_price - avg_breakeven)
            
            analysis = {
                'strategy_name': strategy.name,
                'underlying_price': underlying_price,
                'max_profit': strategy.max_profit,
                'max_loss': strategy.max_loss,
                'risk_reward_ratio': strategy.risk_reward_ratio,
                'breakeven_points': strategy.breakeven_points,
                'in_profit_zone': in_profit_zone,
                'distance_from_center': distance_from_center,
                'margin_requirement': strategy.margin_requirement,
                'ideal_outlook': strategy.ideal_market_outlook,
                'recommendation': self._get_recommendation(strategy, underlying_price)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategy: {e}")
            return {}
    
    def _get_recommendation(self, strategy: StrategyDefinition, underlying_price: float) -> str:
        """Get recommendation based on strategy and price"""
        if not strategy.breakeven_points:
            return "Monitor position"
        
        avg_breakeven = sum(strategy.breakeven_points) / len(strategy.breakeven_points)
        distance_pct = abs(underlying_price - avg_breakeven) / avg_breakeven * 100
        
        if distance_pct < 2:
            return "Price near optimal zone - hold position"
        elif distance_pct < 5:
            return "Price moving away - monitor closely"
        else:
            return "Price far from optimal - consider adjusting"
    
    def compare_strategies(self, strategies: List[StrategyDefinition]) -> Dict:
        """Compare multiple strategies"""
        comparison = {
            'strategies': [],
            'best_risk_reward': None,
            'highest_probability': None,
            'lowest_margin': None
        }
        
        best_rr = 0
        lowest_margin = float('inf')
        
        for strategy in strategies:
            strategy_data = {
                'name': strategy.name,
                'max_profit': strategy.max_profit,
                'max_loss': strategy.max_loss,
                'risk_reward_ratio': strategy.risk_reward_ratio,
                'margin_requirement': strategy.margin_requirement
            }
            
            comparison['strategies'].append(strategy_data)
            
            if strategy.risk_reward_ratio > best_rr:
                best_rr = strategy.risk_reward_ratio
                comparison['best_risk_reward'] = strategy.name
            
            if strategy.margin_requirement < lowest_margin:
                lowest_margin = strategy.margin_requirement
                comparison['lowest_margin'] = strategy.name
        
        return comparison


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize builder
    builder = AdvancedStrategiesBuilder()
    
    expiry = datetime(2025, 2, 21)
    
    # Create various strategies
    print("\n1. Long Call Butterfly:")
    butterfly = builder.create_butterfly_spread('AAPL', 150.0, 5.0, 'CALL', expiry, 1)
    print(f"   Max Profit: ${butterfly.max_profit:.2f}")
    print(f"   Max Loss: ${butterfly.max_loss:.2f}")
    print(f"   Breakevens: {butterfly.breakeven_points}")
    
    print("\n2. Iron Butterfly:")
    iron_butterfly = builder.create_iron_butterfly('AAPL', 150.0, 5.0, expiry, 1)
    print(f"   Max Profit: ${iron_butterfly.max_profit:.2f}")
    print(f"   Max Loss: ${iron_butterfly.max_loss:.2f}")
    
    print("\n3. Calendar Spread:")
    near_expiry = datetime(2025, 1, 17)
    far_expiry = datetime(2025, 3, 21)
    calendar = builder.create_calendar_spread('AAPL', 150.0, 'CALL', near_expiry, far_expiry, 1)
    print(f"   Max Profit: ${calendar.max_profit:.2f}")
    print(f"   Max Loss: ${calendar.max_loss:.2f}")
    
    print("\n4. Jade Lizard:")
    jade_lizard = builder.create_jade_lizard('AAPL', 155.0, 160.0, 145.0, expiry, 1)
    print(f"   Max Profit: ${jade_lizard.max_profit:.2f}")
    print(f"   Max Loss: ${jade_lizard.max_loss:.2f}")
    
    print("\n5. Strategy Analysis:")
    analysis = builder.analyze_strategy(butterfly, 150.0)
    print(f"   {analysis['recommendation']}")
    
    print("\n6. Strategy Comparison:")
    comparison = builder.compare_strategies([butterfly, iron_butterfly, calendar])
    print(f"   Best Risk/Reward: {comparison['best_risk_reward']}")
    print(f"   Lowest Margin: {comparison['lowest_margin']}")

