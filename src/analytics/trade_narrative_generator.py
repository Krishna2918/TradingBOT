"""
Trade Narrative Generator
Generates human-readable explanations for trading decisions and outcomes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import re

# Import local LLM integration
try:
    from src.ai.local_llm_integration import LocalLLMClient, LLMRequest, LLMResponse
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    logger.warning("Local LLM integration not available")

logger = logging.getLogger(__name__)

class NarrativeType(Enum):
    """Types of trade narratives"""
    TRADE_DECISION = "trade_decision"
    TRADE_EXECUTION = "trade_execution"
    TRADE_OUTCOME = "trade_outcome"
    PORTFOLIO_PERFORMANCE = "portfolio_performance"
    RISK_ANALYSIS = "risk_analysis"
    MARKET_ANALYSIS = "market_analysis"

class NarrativeTone(Enum):
    """Narrative tone options"""
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"

@dataclass
class TradeContext:
    """Context for trade narrative generation"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    price: float
    timestamp: datetime
    market_data: Dict[str, Any]
    ai_analysis: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]

@dataclass
class NarrativeTemplate:
    """Narrative template structure"""
    template_id: str
    narrative_type: NarrativeType
    tone: NarrativeTone
    template: str
    variables: List[str]
    conditions: Dict[str, Any]

class TradeNarrativeGenerator:
    """Generates human-readable trade narratives"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.narrative_templates = []
        self.narrative_history = []
        
        # Initialize local LLM client if available
        self.local_llm_client = None
        if LOCAL_LLM_AVAILABLE and config.get('use_local_llm', True):
            try:
                llm_config = config.get('local_llm_config', {})
                self.local_llm_client = LocalLLMClient(llm_config)
                logger.info("Local LLM client initialized for narrative generation")
            except Exception as e:
                logger.warning(f"Failed to initialize local LLM client: {e}")
                self.local_llm_client = None
        
        # Load default templates
        self._load_default_templates()
        
        logger.info("Trade Narrative Generator initialized")
    
    def _load_default_templates(self):
        """Load default narrative templates"""
        try:
            default_templates = [
                # Trade Decision Templates
                NarrativeTemplate(
                    template_id="trade_decision_buy_001",
                    narrative_type=NarrativeType.TRADE_DECISION,
                    tone=NarrativeTone.PROFESSIONAL,
                    template="Based on technical analysis showing {technical_signal} and fundamental factors indicating {fundamental_signal}, the AI system has decided to {action} {quantity} shares of {symbol} at ${price:.2f}. The decision is supported by {confidence_level}% confidence and {risk_level} risk assessment.",
                    variables=["technical_signal", "fundamental_signal", "action", "quantity", "symbol", "price", "confidence_level", "risk_level"],
                    conditions={"action": "BUY"}
                ),
                
                NarrativeTemplate(
                    template_id="trade_decision_sell_001",
                    narrative_type=NarrativeType.TRADE_DECISION,
                    tone=NarrativeTone.PROFESSIONAL,
                    template="Following {sell_reason} and considering the current market conditions showing {market_condition}, the AI system has decided to {action} {quantity} shares of {symbol} at ${price:.2f}. This decision aims to {sell_objective} while maintaining portfolio risk within acceptable limits.",
                    variables=["sell_reason", "market_condition", "action", "quantity", "symbol", "price", "sell_objective"],
                    conditions={"action": "SELL"}
                ),
                
                # Trade Execution Templates
                NarrativeTemplate(
                    template_id="trade_execution_001",
                    narrative_type=NarrativeType.TRADE_EXECUTION,
                    tone=NarrativeTone.TECHNICAL,
                    template="Trade executed successfully using {execution_method} algorithm. Execution details: {quantity} shares of {symbol} at average price of ${avg_price:.2f} with {slippage:.3f}% slippage and ${commission:.2f} commission. Total execution time: {execution_time}ms.",
                    variables=["execution_method", "quantity", "symbol", "avg_price", "slippage", "commission", "execution_time"],
                    conditions={}
                ),
                
                # Trade Outcome Templates
                NarrativeTemplate(
                    template_id="trade_outcome_profit_001",
                    narrative_type=NarrativeType.TRADE_OUTCOME,
                    tone=NarrativeTone.ANALYTICAL,
                    template="Trade outcome analysis for {symbol}: {action} position closed with {pnl_pct:.2f}% profit (${pnl_amount:.2f}). The trade performed {performance_vs_expectation} expectations, driven primarily by {key_factors}. Risk-adjusted return: {sharpe_ratio:.2f}.",
                    variables=["symbol", "action", "pnl_pct", "pnl_amount", "performance_vs_expectation", "key_factors", "sharpe_ratio"],
                    conditions={"pnl_pct": "> 0"}
                ),
                
                NarrativeTemplate(
                    template_id="trade_outcome_loss_001",
                    narrative_type=NarrativeType.TRADE_OUTCOME,
                    tone=NarrativeTone.ANALYTICAL,
                    template="Trade outcome analysis for {symbol}: {action} position closed with {pnl_pct:.2f}% loss (${pnl_amount:.2f}). The trade {performance_vs_expectation} expectations due to {key_factors}. Risk management protocols were {risk_management_status}.",
                    variables=["symbol", "action", "pnl_pct", "pnl_amount", "performance_vs_expectation", "key_factors", "risk_management_status"],
                    conditions={"pnl_pct": "< 0"}
                ),
                
                # Portfolio Performance Templates
                NarrativeTemplate(
                    template_id="portfolio_performance_001",
                    narrative_type=NarrativeType.PORTFOLIO_PERFORMANCE,
                    tone=NarrativeTone.PROFESSIONAL,
                    template="Portfolio performance summary: Total return of {total_return:.2f}% with Sharpe ratio of {sharpe_ratio:.2f}. Current portfolio value: ${portfolio_value:.2f}. Top performing positions: {top_performers}. Areas for improvement: {improvement_areas}.",
                    variables=["total_return", "sharpe_ratio", "portfolio_value", "top_performers", "improvement_areas"],
                    conditions={}
                ),
                
                # Risk Analysis Templates
                NarrativeTemplate(
                    template_id="risk_analysis_001",
                    narrative_type=NarrativeType.RISK_ANALYSIS,
                    tone=NarrativeTone.TECHNICAL,
                    template="Risk analysis update: Portfolio VaR (95%) at {var_95:.2f}%, Beta at {beta:.2f}, and maximum drawdown of {max_drawdown:.2f}%. Current risk level: {risk_level}. Risk management actions: {risk_actions}.",
                    variables=["var_95", "beta", "max_drawdown", "risk_level", "risk_actions"],
                    conditions={}
                ),
                
                # Market Analysis Templates
                NarrativeTemplate(
                    template_id="market_analysis_001",
                    narrative_type=NarrativeType.MARKET_ANALYSIS,
                    tone=NarrativeTone.ANALYTICAL,
                    template="Market analysis: Current regime identified as {market_regime} with {regime_confidence:.1f}% confidence. Key market factors: {key_factors}. Volatility level: {volatility_level}. Trading recommendations: {recommendations}.",
                    variables=["market_regime", "regime_confidence", "key_factors", "volatility_level", "recommendations"],
                    conditions={}
                )
            ]
            
            self.narrative_templates = default_templates
            logger.info(f"Loaded {len(default_templates)} narrative templates")
            
        except Exception as e:
            logger.error(f"Error loading default templates: {e}")
    
    def generate_trade_narrative(self, trade_context: TradeContext, 
                               narrative_type: NarrativeType,
                               tone: NarrativeTone = NarrativeTone.PROFESSIONAL,
                               use_llm: bool = True) -> str:
        """Generate a trade narrative based on context and type"""
        try:
            # Try local LLM first if available and requested
            if use_llm and self.local_llm_client:
                try:
                    llm_narrative = self._generate_llm_narrative(trade_context, narrative_type, tone)
                    if llm_narrative:
                        # Store narrative history
                        self._store_narrative(llm_narrative, trade_context, narrative_type, tone)
                        return llm_narrative
                except Exception as e:
                    logger.warning(f"LLM narrative generation failed, falling back to templates: {e}")
            
            # Fallback to template-based generation
            # Find appropriate template
            template = self._find_matching_template(narrative_type, tone, trade_context)
            
            if not template:
                return self._generate_fallback_narrative(trade_context, narrative_type)
            
            # Extract variables from context
            variables = self._extract_variables(trade_context, template)
            
            # Generate narrative
            narrative = self._apply_template(template.template, variables)
            
            # Store narrative history
            self._store_narrative(narrative, trade_context, narrative_type, tone)
            
            return narrative
            
        except Exception as e:
            logger.error(f"Error generating trade narrative: {e}")
            return self._generate_fallback_narrative(trade_context, narrative_type)
    
    def _generate_llm_narrative(self, trade_context: TradeContext, 
                               narrative_type: NarrativeType,
                               tone: NarrativeTone) -> Optional[str]:
        """Generate narrative using local LLM"""
        try:
            if not self.local_llm_client:
                return None
            
            # Build comprehensive prompt for LLM
            prompt = self._build_llm_prompt(trade_context, narrative_type, tone)
            
            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                task_type=f"narrative_{narrative_type.value}",
                temperature=0.7 if tone == NarrativeTone.CONVERSATIONAL else 0.5,
                max_tokens=1024,
                context={
                    'trade_context': {
                        'symbol': trade_context.symbol,
                        'action': trade_context.action,
                        'quantity': trade_context.quantity,
                        'price': trade_context.price,
                        'timestamp': trade_context.timestamp.isoformat()
                    },
                    'narrative_type': narrative_type.value,
                    'tone': tone.value
                }
            )
            
            # Get LLM response
            response = self.local_llm_client._make_request(request)
            
            if response and response.content:
                # Post-process the response
                processed_narrative = self._post_process_llm_response(response.content, trade_context)
                return processed_narrative
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating LLM narrative: {e}")
            return None
    
    def _build_llm_prompt(self, trade_context: TradeContext, 
                         narrative_type: NarrativeType,
                         tone: NarrativeTone) -> str:
        """Build comprehensive prompt for LLM narrative generation"""
        
        # Base context
        base_context = f"""
You are an expert trading analyst generating a {tone.value} narrative for a {narrative_type.value}.

Trade Details:
- Symbol: {trade_context.symbol}
- Action: {trade_context.action}
- Quantity: {trade_context.quantity}
- Price: ${trade_context.price:.2f}
- Timestamp: {trade_context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Add specific context based on narrative type
        if narrative_type == NarrativeType.TRADE_DECISION:
            prompt = base_context + f"""
Market Data:
- Technical Signals: {trade_context.market_data.get('technical_signal', 'Mixed signals')}
- Market Conditions: {trade_context.market_data.get('market_condition', 'Normal conditions')}
- Volatility: {trade_context.market_data.get('volatility_level', 'Moderate')}

AI Analysis:
- Confidence: {trade_context.ai_analysis.get('confidence', 75)}%
- Risk Level: {trade_context.ai_analysis.get('risk_level', 'Medium')}
- Key Factors: {trade_context.ai_analysis.get('key_factors', 'Market conditions')}

Generate a {tone.value} explanation of why this trade decision was made. Include:
1. Primary reasoning for the {trade_context.action} decision
2. Key market factors that influenced the decision
3. Risk considerations
4. Expected outcome

Keep it professional and under 200 words.
"""
        
        elif narrative_type == NarrativeType.TRADE_EXECUTION:
            prompt = base_context + f"""
Execution Details:
- Method: {trade_context.market_data.get('execution', {}).get('method', 'VWAP')}
- Slippage: {trade_context.market_data.get('execution', {}).get('slippage', 0.001):.3f}%
- Commission: ${trade_context.market_data.get('execution', {}).get('commission', 1.0):.2f}

Generate a {tone.value} summary of the trade execution. Include:
1. Execution method and performance
2. Slippage and cost analysis
3. Overall execution quality assessment

Keep it technical and under 150 words.
"""
        
        elif narrative_type == NarrativeType.TRADE_OUTCOME:
            prompt = base_context + f"""
Performance Metrics:
- P&L: {trade_context.performance_metrics.get('pnl_pct', 0.0):.2f}%
- Amount: ${trade_context.performance_metrics.get('pnl_amount', 0.0):.2f}
- Sharpe Ratio: {trade_context.performance_metrics.get('sharpe_ratio', 1.0):.2f}

Generate a {tone.value} analysis of the trade outcome. Include:
1. Performance vs expectations
2. Key factors that drove the result
3. Lessons learned
4. Risk management effectiveness

Keep it analytical and under 200 words.
"""
        
        elif narrative_type == NarrativeType.PORTFOLIO_PERFORMANCE:
            prompt = base_context + f"""
Portfolio Metrics:
- Total Return: {trade_context.performance_metrics.get('total_return', 0.0):.2f}%
- Portfolio Value: ${trade_context.performance_metrics.get('portfolio_value', 100000.0):,.2f}
- Sharpe Ratio: {trade_context.performance_metrics.get('sharpe_ratio', 1.0):.2f}

Generate a {tone.value} portfolio performance summary. Include:
1. Overall performance assessment
2. Key drivers of performance
3. Areas of strength and improvement
4. Strategic recommendations

Keep it professional and under 250 words.
"""
        
        elif narrative_type == NarrativeType.RISK_ANALYSIS:
            prompt = base_context + f"""
Risk Metrics:
- VaR (95%): {trade_context.risk_metrics.get('var_95', 0.03):.2f}%
- Beta: {trade_context.risk_metrics.get('beta', 1.0):.2f}
- Max Drawdown: {trade_context.risk_metrics.get('max_drawdown', 0.05):.2f}%
- Risk Level: {trade_context.risk_metrics.get('risk_level', 'Moderate')}

Generate a {tone.value} risk analysis. Include:
1. Current risk assessment
2. Key risk factors
3. Risk management effectiveness
4. Recommendations for risk optimization

Keep it technical and under 200 words.
"""
        
        elif narrative_type == NarrativeType.MARKET_ANALYSIS:
            prompt = base_context + f"""
Market Data:
- Regime: {trade_context.market_data.get('market_regime', 'Neutral')}
- Volatility: {trade_context.market_data.get('volatility_level', 'Moderate')}
- Trend: {trade_context.market_data.get('trend', 'Sideways')}

Generate a {tone.value} market analysis. Include:
1. Current market regime assessment
2. Key market drivers
3. Volatility and trend analysis
4. Trading environment implications

Keep it analytical and under 200 words.
"""
        
        else:
            prompt = base_context + f"""
Generate a {tone.value} narrative about this trading activity. Provide context and insights about the trade and its implications.
"""
        
        return prompt
    
    def _post_process_llm_response(self, response: str, trade_context: TradeContext) -> str:
        """Post-process LLM response to ensure quality and consistency"""
        try:
            # Clean up the response
            response = response.strip()
            
            # Remove any unwanted prefixes or suffixes
            unwanted_prefixes = ["Here's", "Here is", "The", "This", "Based on"]
            for prefix in unwanted_prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    if response.startswith(','):
                        response = response[1:].strip()
            
            # Ensure proper capitalization
            if response and not response[0].isupper():
                response = response[0].upper() + response[1:]
            
            # Ensure proper ending
            if response and not response.endswith(('.', '!', '?')):
                response += '.'
            
            # Limit length if too long
            if len(response) > 500:
                sentences = response.split('. ')
                response = '. '.join(sentences[:3])
                if not response.endswith('.'):
                    response += '.'
            
            return response
            
        except Exception as e:
            logger.error(f"Error post-processing LLM response: {e}")
            return response  # Return original if processing fails
    
    def _find_matching_template(self, narrative_type: NarrativeType, 
                              tone: NarrativeTone, 
                              trade_context: TradeContext) -> Optional[NarrativeTemplate]:
        """Find matching template based on type, tone, and conditions"""
        try:
            matching_templates = []
            
            for template in self.narrative_templates:
                if (template.narrative_type == narrative_type and 
                    template.tone == tone):
                    
                    # Check conditions
                    if self._check_template_conditions(template.conditions, trade_context):
                        matching_templates.append(template)
            
            # Return the first matching template
            return matching_templates[0] if matching_templates else None
            
        except Exception as e:
            logger.error(f"Error finding matching template: {e}")
            return None
    
    def _check_template_conditions(self, conditions: Dict[str, Any], 
                                 trade_context: TradeContext) -> bool:
        """Check if template conditions are met"""
        try:
            for key, expected_value in conditions.items():
                if key == "action":
                    if trade_context.action != expected_value:
                        return False
                elif key == "pnl_pct":
                    # This would need to be calculated from trade context
                    # For now, assume condition is met
                    pass
                # Add more condition checks as needed
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking template conditions: {e}")
            return True  # Default to allowing template
    
    def _extract_variables(self, trade_context: TradeContext, 
                         template: NarrativeTemplate) -> Dict[str, Any]:
        """Extract variables from trade context for template"""
        try:
            variables = {}
            
            # Basic trade variables
            variables.update({
                'symbol': trade_context.symbol,
                'action': trade_context.action,
                'quantity': trade_context.quantity,
                'price': trade_context.price,
                'timestamp': trade_context.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Market data variables
            market_data = trade_context.market_data
            variables.update({
                'technical_signal': market_data.get('technical_signal', 'mixed signals'),
                'fundamental_signal': market_data.get('fundamental_signal', 'neutral outlook'),
                'market_condition': market_data.get('market_condition', 'normal conditions'),
                'volatility_level': market_data.get('volatility_level', 'moderate'),
                'market_regime': market_data.get('market_regime', 'neutral')
            })
            
            # AI analysis variables
            ai_analysis = trade_context.ai_analysis
            variables.update({
                'confidence_level': ai_analysis.get('confidence', 75),
                'risk_level': ai_analysis.get('risk_level', 'medium'),
                'sell_reason': ai_analysis.get('sell_reason', 'profit taking'),
                'sell_objective': ai_analysis.get('sell_objective', 'lock in gains'),
                'key_factors': ai_analysis.get('key_factors', 'market conditions')
            })
            
            # Risk metrics variables
            risk_metrics = trade_context.risk_metrics
            variables.update({
                'var_95': risk_metrics.get('var_95', 0.03),
                'beta': risk_metrics.get('beta', 1.0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0.05),
                'risk_level': risk_metrics.get('risk_level', 'moderate'),
                'risk_actions': risk_metrics.get('risk_actions', 'monitoring')
            })
            
            # Performance metrics variables
            performance_metrics = trade_context.performance_metrics
            variables.update({
                'pnl_pct': performance_metrics.get('pnl_pct', 0.0),
                'pnl_amount': performance_metrics.get('pnl_amount', 0.0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 1.0),
                'total_return': performance_metrics.get('total_return', 0.0),
                'portfolio_value': performance_metrics.get('portfolio_value', 100000.0),
                'performance_vs_expectation': performance_metrics.get('performance_vs_expectation', 'met'),
                'risk_management_status': performance_metrics.get('risk_management_status', 'active')
            })
            
            # Execution variables (if available)
            execution_data = trade_context.market_data.get('execution', {})
            variables.update({
                'execution_method': execution_data.get('method', 'VWAP'),
                'avg_price': execution_data.get('avg_price', trade_context.price),
                'slippage': execution_data.get('slippage', 0.001),
                'commission': execution_data.get('commission', 1.0),
                'execution_time': execution_data.get('execution_time', 100)
            })
            
            # Additional calculated variables
            variables.update({
                'regime_confidence': ai_analysis.get('regime_confidence', 80.0),
                'top_performers': self._get_top_performers(performance_metrics),
                'improvement_areas': self._get_improvement_areas(performance_metrics),
                'recommendations': self._get_recommendations(ai_analysis)
            })
            
            return variables
            
        except Exception as e:
            logger.error(f"Error extracting variables: {e}")
            return {}
    
    def _apply_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Apply variables to template string"""
        try:
            # Replace variables in template
            narrative = template
            
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                if placeholder in narrative:
                    if isinstance(value, float):
                        # Handle float formatting
                        if ':.2f' in narrative:
                            narrative = narrative.replace(placeholder, f"{value:.2f}")
                        elif ':.1f' in narrative:
                            narrative = narrative.replace(placeholder, f"{value:.1f}")
                        elif ':.3f' in narrative:
                            narrative = narrative.replace(placeholder, f"{value:.3f}")
                        else:
                            narrative = narrative.replace(placeholder, str(value))
                    else:
                        narrative = narrative.replace(placeholder, str(value))
            
            return narrative
            
        except Exception as e:
            logger.error(f"Error applying template: {e}")
            return template
    
    def _get_top_performers(self, performance_metrics: Dict[str, Any]) -> str:
        """Get top performing positions"""
        try:
            top_performers = performance_metrics.get('top_performers', [])
            if top_performers:
                return ', '.join([f"{symbol} (+{pct:.1f}%)" for symbol, pct in top_performers[:3]])
            return "None identified"
            
        except Exception as e:
            logger.error(f"Error getting top performers: {e}")
            return "None identified"
    
    def _get_improvement_areas(self, performance_metrics: Dict[str, Any]) -> str:
        """Get areas for improvement"""
        try:
            improvement_areas = performance_metrics.get('improvement_areas', [])
            if improvement_areas:
                return ', '.join(improvement_areas[:3])
            return "Risk management, position sizing"
            
        except Exception as e:
            logger.error(f"Error getting improvement areas: {e}")
            return "Risk management, position sizing"
    
    def _get_recommendations(self, ai_analysis: Dict[str, Any]) -> str:
        """Get trading recommendations"""
        try:
            recommendations = ai_analysis.get('recommendations', [])
            if recommendations:
                return ', '.join(recommendations[:3])
            return "Continue monitoring, maintain risk controls"
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return "Continue monitoring, maintain risk controls"
    
    def _generate_fallback_narrative(self, trade_context: TradeContext, 
                                   narrative_type: NarrativeType) -> str:
        """Generate fallback narrative when no template matches"""
        try:
            if narrative_type == NarrativeType.TRADE_DECISION:
                return f"AI system executed {trade_context.action} order for {trade_context.quantity} shares of {trade_context.symbol} at ${trade_context.price:.2f} based on comprehensive market analysis."
            elif narrative_type == NarrativeType.TRADE_EXECUTION:
                return f"Trade executed: {trade_context.quantity} shares of {trade_context.symbol} at ${trade_context.price:.2f} on {trade_context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
            elif narrative_type == NarrativeType.TRADE_OUTCOME:
                return f"Trade outcome for {trade_context.symbol}: {trade_context.action} position with current market price of ${trade_context.price:.2f}."
            elif narrative_type == NarrativeType.PORTFOLIO_PERFORMANCE:
                return f"Portfolio performance update: Current value includes {trade_context.symbol} position valued at ${trade_context.price * trade_context.quantity:.2f}."
            elif narrative_type == NarrativeType.RISK_ANALYSIS:
                return f"Risk analysis for {trade_context.symbol}: Position size of {trade_context.quantity} shares represents appropriate risk exposure."
            elif narrative_type == NarrativeType.MARKET_ANALYSIS:
                return f"Market analysis: {trade_context.symbol} showing price of ${trade_context.price:.2f} with {trade_context.action} recommendation."
            else:
                return f"Trading activity: {trade_context.action} {trade_context.quantity} shares of {trade_context.symbol} at ${trade_context.price:.2f}."
                
        except Exception as e:
            logger.error(f"Error generating fallback narrative: {e}")
            return f"Trading activity recorded for {trade_context.symbol}."
    
    def _store_narrative(self, narrative: str, trade_context: TradeContext,
                        narrative_type: NarrativeType, tone: NarrativeTone):
        """Store narrative in history"""
        try:
            narrative_record = {
                'narrative_id': f"narr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now(),
                'narrative_type': narrative_type.value,
                'tone': tone.value,
                'symbol': trade_context.symbol,
                'action': trade_context.action,
                'narrative': narrative,
                'context': {
                    'quantity': trade_context.quantity,
                    'price': trade_context.price,
                    'timestamp': trade_context.timestamp.isoformat()
                }
            }
            
            self.narrative_history.append(narrative_record)
            
            # Keep only last 1000 narratives
            if len(self.narrative_history) > 1000:
                self.narrative_history = self.narrative_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error storing narrative: {e}")
    
    def generate_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate portfolio summary narrative"""
        try:
            total_value = portfolio_data.get('total_value', 0)
            total_return = portfolio_data.get('total_return', 0)
            sharpe_ratio = portfolio_data.get('sharpe_ratio', 0)
            positions = portfolio_data.get('positions', [])
            
            summary = f"Portfolio Summary: Total value of ${total_value:,.2f} with {total_return:.2f}% return and Sharpe ratio of {sharpe_ratio:.2f}. "
            
            if positions:
                summary += f"Current positions include {len(positions)} holdings. "
                top_position = max(positions, key=lambda x: x.get('value', 0))
                summary += f"Largest position: {top_position.get('symbol', 'Unknown')} valued at ${top_position.get('value', 0):,.2f}."
            else:
                summary += "No current positions."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return "Portfolio summary unavailable."
    
    def generate_risk_summary(self, risk_data: Dict[str, Any]) -> str:
        """Generate risk summary narrative"""
        try:
            var_95 = risk_data.get('var_95', 0)
            beta = risk_data.get('beta', 0)
            max_drawdown = risk_data.get('max_drawdown', 0)
            risk_level = risk_data.get('risk_level', 'moderate')
            
            summary = f"Risk Summary: Portfolio VaR (95%) at {var_95:.2f}%, Beta of {beta:.2f}, and maximum drawdown of {max_drawdown:.2f}%. "
            summary += f"Current risk level: {risk_level}. "
            
            if var_95 > 0.05:
                summary += "VaR exceeds normal thresholds - consider risk reduction."
            elif beta > 1.5:
                summary += "High beta exposure - monitor market sensitivity."
            elif max_drawdown > 0.15:
                summary += "Significant drawdown - review position sizing."
            else:
                summary += "Risk metrics within acceptable ranges."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return "Risk summary unavailable."
    
    def get_narrative_statistics(self) -> Dict:
        """Get narrative generation statistics"""
        try:
            if not self.narrative_history:
                return {}
            
            # Calculate statistics
            total_narratives = len(self.narrative_history)
            type_counts = {}
            tone_counts = {}
            symbol_counts = {}
            
            for narrative in self.narrative_history:
                narrative_type = narrative['narrative_type']
                tone = narrative['tone']
                symbol = narrative['symbol']
                
                type_counts[narrative_type] = type_counts.get(narrative_type, 0) + 1
                tone_counts[tone] = tone_counts.get(tone, 0) + 1
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            stats = {
                'total_narratives': total_narratives,
                'type_counts': type_counts,
                'tone_counts': tone_counts,
                'symbol_counts': symbol_counts,
                'active_templates': len(self.narrative_templates),
                'average_narrative_length': np.mean([len(n['narrative']) for n in self.narrative_history])
            }
            
            # Add LLM performance metrics if available
            if self.local_llm_client:
                try:
                    llm_metrics = self.local_llm_client.get_performance_metrics()
                    stats['llm_metrics'] = llm_metrics
                except Exception as e:
                    logger.warning(f"Could not get LLM metrics: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting narrative statistics: {e}")
            return {}
    
    def export_narratives(self, filepath: str, limit: int = 100):
        """Export recent narratives to file"""
        try:
            recent_narratives = self.narrative_history[-limit:] if limit else self.narrative_history
            
            with open(filepath, 'w') as f:
                json.dump(recent_narratives, f, indent=2, default=str)
            
            logger.info(f"Exported {len(recent_narratives)} narratives to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting narratives: {e}")
    
    def add_custom_template(self, template: NarrativeTemplate):
        """Add a custom narrative template"""
        try:
            self.narrative_templates.append(template)
            logger.info(f"Added custom template: {template.template_id}")
            
        except Exception as e:
            logger.error(f"Error adding custom template: {e}")
    
    def remove_template(self, template_id: str):
        """Remove a template by ID"""
        try:
            self.narrative_templates = [t for t in self.narrative_templates if t.template_id != template_id]
            logger.info(f"Removed template: {template_id}")
            
        except Exception as e:
            logger.error(f"Error removing template: {e}")
