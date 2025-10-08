"""
AI Ensemble Trading System
Integrates Grok, Kimi K2, and Claude for enhanced trading decisions
"""

import logging
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AIEnsemble:
    """
    Multi-AI ensemble for trading decisions
    Combines insights from Grok, Kimi K2, and Claude
    """
    
    def __init__(self, config_path: str = "config/ai_ensemble_config.yaml"):
        self.config = self._load_config(config_path)
        self.ai_models = {
            'grok': GrokAI(self.config['grok']),
            'kimi': KimiAI(self.config['kimi']),
            'claude': ClaudeAI(self.config['claude'])
        }
        
        # Ensemble weights (can be adjusted based on performance)
        self.weights = {
            'grok': 0.35,      # Grok: Real-time, creative analysis
            'kimi': 0.30,      # Kimi: Technical analysis specialist
            'claude': 0.35     # Claude: Risk management & reasoning
        }
        
        logger.info(" AI Ensemble initialized with Grok, Kimi K2, and Claude")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load AI ensemble configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f" Failed to load AI config: {e}")
            return {}
    
    def analyze_market(self, market_data: Dict, positions: Dict) -> Dict:
        """
        Get ensemble analysis from all AI models
        """
        logger.info(" Getting ensemble analysis from Grok, Kimi, and Claude...")
        
        # Prepare market context
        context = self._prepare_context(market_data, positions)
        
        # Get analysis from each AI
        analyses = {}
        for name, ai in self.ai_models.items():
            try:
                analysis = ai.analyze(context)
                analyses[name] = analysis
                logger.info(f" {name.capitalize()} analysis complete")
            except Exception as e:
                logger.error(f" {name.capitalize()} analysis failed: {e}")
                analyses[name] = {'error': str(e)}
        
        # Combine analyses using ensemble weights
        ensemble_decision = self._combine_analyses(analyses)
        
        return {
            'ensemble_decision': ensemble_decision,
            'individual_analyses': analyses,
            'weights': self.weights,
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_context(self, market_data: Dict, positions: Dict) -> Dict:
        """Prepare market context for AI analysis"""
        return {
            'market_data': market_data,
            'positions': positions,
            'timestamp': datetime.now().isoformat(),
            'market_conditions': self._assess_market_conditions(market_data)
        }
    
    def _assess_market_conditions(self, market_data: Dict) -> Dict:
        """Assess overall market conditions"""
        # Calculate market volatility, trend, etc.
        prices = list(market_data.values())
        if len(prices) > 1:
            volatility = np.std(prices) / np.mean(prices)
            trend = "bullish" if prices[-1] > prices[0] else "bearish"
        else:
            volatility = 0
            trend = "neutral"
        
        return {
            'volatility': volatility,
            'trend': trend,
            'market_hours': self._is_market_open(),
            'symbols_tracked': len(market_data)
        }
    
    def _is_market_open(self) -> bool:
        """Check if Canadian market is open"""
        now = datetime.now()
        # Canadian market: 9:30 AM - 4:00 PM EST, Monday-Friday
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= now <= market_close
        return False
    
    def _combine_analyses(self, analyses: Dict) -> Dict:
        """Combine individual AI analyses using weighted ensemble"""
        signals = []
        confidences = []
        reasons = []
        
        for ai_name, analysis in analyses.items():
            if 'error' not in analysis:
                weight = self.weights.get(ai_name, 0.33)
                
                # Extract signal (buy/sell/hold)
                signal = analysis.get('signal', 'hold')
                confidence = analysis.get('confidence', 0.5)
                reason = analysis.get('reason', 'No reason provided')
                
                # Convert signal to numeric for weighting
                signal_numeric = {'buy': 1, 'sell': -1, 'hold': 0}.get(signal, 0)
                
                signals.append(signal_numeric * weight)
                confidences.append(confidence * weight)
                reasons.append(f"{ai_name.capitalize()}: {reason}")
        
        # Calculate weighted ensemble decision
        if signals:
            ensemble_signal_numeric = sum(signals)
            ensemble_confidence = sum(confidences)
            
            # Convert back to signal
            if ensemble_signal_numeric > 0.3:
                final_signal = 'buy'
            elif ensemble_signal_numeric < -0.3:
                final_signal = 'sell'
            else:
                final_signal = 'hold'
            
            return {
                'signal': final_signal,
                'confidence': min(ensemble_confidence, 1.0),
                'reason': ' | '.join(reasons),
                'individual_signals': {name: analysis.get('signal', 'hold') for name, analysis in analyses.items()},
                'consensus': self._calculate_consensus(analyses)
            }
        else:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'reason': 'No AI analysis available',
                'individual_signals': {},
                'consensus': 'no_consensus'
            }
    
    def _calculate_consensus(self, analyses: Dict) -> str:
        """Calculate consensus among AI models"""
        signals = [analysis.get('signal', 'hold') for analysis in analyses.values() if 'error' not in analysis]
        
        if not signals:
            return 'no_consensus'
        
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        hold_count = signals.count('hold')
        
        total = len(signals)
        
        if buy_count / total >= 0.6:
            return 'strong_buy'
        elif sell_count / total >= 0.6:
            return 'strong_sell'
        elif buy_count > sell_count:
            return 'weak_buy'
        elif sell_count > buy_count:
            return 'weak_sell'
        else:
            return 'hold_consensus'


class GrokAI:
    """Grok AI integration for real-time market analysis"""
    
    def __init__(self, config: Dict):
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.grok.com')
        self.model = config.get('model', 'grok-beta')
    
    def analyze(self, context: Dict) -> Dict:
        """Get Grok's analysis of market conditions"""
        prompt = self._create_grok_prompt(context)
        
        try:
            response = self._call_grok_api(prompt)
            return self._parse_grok_response(response)
        except Exception as e:
            logger.error(f" Grok API error: {e}")
            return {'error': str(e)}
    
    def _create_grok_prompt(self, context: Dict) -> str:
        """Create prompt for Grok analysis"""
        market_data = context['market_data']
        positions = context['positions']
        market_conditions = context['market_conditions']
        
        prompt = f"""
        You are Grok, an AI trading analyst with real-time market awareness and creative insights.
        
        Current Canadian Market Data:
        {json.dumps(market_data, indent=2)}
        
        Current Positions:
        {json.dumps(positions, indent=2)}
        
        Market Conditions:
        - Volatility: {market_conditions['volatility']:.3f}
        - Trend: {market_conditions['trend']}
        - Market Open: {market_conditions['market_hours']}
        - Symbols Tracked: {market_conditions['symbols_tracked']}
        
        Analyze the current market situation and provide:
        1. Trading signal: buy, sell, or hold
        2. Confidence level (0.0 to 1.0)
        3. Reasoning based on real-time market dynamics
        4. Specific recommendations for Canadian stocks
        
        Focus on:
        - Real-time market sentiment
        - Creative pattern recognition
        - Unconventional insights
        - Canadian market specifics
        
        Respond in JSON format:
        {{
            "signal": "buy/sell/hold",
            "confidence": 0.0-1.0,
            "reason": "detailed reasoning",
            "recommendations": ["specific stock recommendations"],
            "market_insights": "unique market observations"
        }}
        """
        return prompt
    
    def _call_grok_api(self, prompt: str) -> Dict:
        """Call Grok API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        response = requests.post(
            f'{self.base_url}/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Grok API error: {response.status_code} - {response.text}")
    
    def _parse_grok_response(self, response: Dict) -> Dict:
        """Parse Grok's response"""
        try:
            content = response['choices'][0]['message']['content']
            # Try to extract JSON from response
            if '{' in content and '}' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return {
                    'signal': 'hold',
                    'confidence': 0.5,
                    'reason': content[:200],
                    'recommendations': [],
                    'market_insights': content
                }
        except Exception as e:
            logger.error(f" Error parsing Grok response: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.5,
                'reason': 'Error parsing response',
                'recommendations': [],
                'market_insights': 'Analysis unavailable'
            }


class KimiAI:
    """Kimi K2 AI integration for technical analysis"""
    
    def __init__(self, config: Dict):
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.moonshot.cn')
        self.model = config.get('model', 'moonshot-v1-8k')
    
    def analyze(self, context: Dict) -> Dict:
        """Get Kimi's technical analysis"""
        prompt = self._create_kimi_prompt(context)
        
        try:
            response = self._call_kimi_api(prompt)
            return self._parse_kimi_response(response)
        except Exception as e:
            logger.error(f" Kimi API error: {e}")
            return {'error': str(e)}
    
    def _create_kimi_prompt(self, context: Dict) -> str:
        """Create prompt for Kimi technical analysis"""
        market_data = context['market_data']
        positions = context['positions']
        
        prompt = f"""
        你是Kimi K2，专业的量化交易技术分析师，擅长技术指标分析和市场模式识别。
        
        当前加拿大市场数据:
        {json.dumps(market_data, indent=2)}
        
        当前持仓:
        {json.dumps(positions, indent=2)}
        
        请进行深度技术分析，包括:
        1. 技术指标分析 (RSI, MACD, 布林带, 移动平均线)
        2. 价格模式识别
        3. 支撑阻力位分析
        4. 成交量分析
        5. 趋势强度评估
        
        基于技术分析提供:
        1. 交易信号: buy, sell, hold
        2. 置信度 (0.0 到 1.0)
        3. 详细的技术分析理由
        4. 具体的加拿大股票建议
        
        重点关注:
        - 技术指标信号
        - 价格模式
        - 成交量确认
        - 风险控制
        
        请以JSON格式回复:
        {{
            "signal": "buy/sell/hold",
            "confidence": 0.0-1.0,
            "reason": "详细技术分析理由",
            "technical_indicators": {{
                "rsi": "RSI分析",
                "macd": "MACD分析",
                "bollinger": "布林带分析",
                "volume": "成交量分析"
            }},
            "recommendations": ["具体股票建议"],
            "risk_assessment": "风险评估"
        }}
        """
        return prompt
    
    def _call_kimi_api(self, prompt: str) -> Dict:
        """Call Kimi API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,  # Lower temperature for technical analysis
            'max_tokens': 1500
        }
        
        response = requests.post(
            f'{self.base_url}/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Kimi API error: {response.status_code} - {response.text}")
    
    def _parse_kimi_response(self, response: Dict) -> Dict:
        """Parse Kimi's response"""
        try:
            content = response['choices'][0]['message']['content']
            if '{' in content and '}' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {
                    'signal': 'hold',
                    'confidence': 0.5,
                    'reason': content[:200],
                    'technical_indicators': {},
                    'recommendations': [],
                    'risk_assessment': 'Analysis unavailable'
                }
        except Exception as e:
            logger.error(f" Error parsing Kimi response: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.5,
                'reason': 'Error parsing response',
                'technical_indicators': {},
                'recommendations': [],
                'risk_assessment': 'Error in analysis'
            }


class ClaudeAI:
    """Claude AI integration for risk management and reasoning"""
    
    def __init__(self, config: Dict):
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.anthropic.com')
        self.model = config.get('model', 'claude-3-sonnet-20240229')
    
    def analyze(self, context: Dict) -> Dict:
        """Get Claude's risk management analysis"""
        prompt = self._create_claude_prompt(context)
        
        try:
            response = self._call_claude_api(prompt)
            return self._parse_claude_response(response)
        except Exception as e:
            logger.error(f" Claude API error: {e}")
            return {'error': str(e)}
    
    def _create_claude_prompt(self, context: Dict) -> str:
        """Create prompt for Claude analysis"""
        market_data = context['market_data']
        positions = context['positions']
        market_conditions = context['market_conditions']
        
        prompt = f"""
        You are Claude, an AI assistant specializing in risk management and logical reasoning for trading decisions.
        
        Current Canadian Market Data:
        {json.dumps(market_data, indent=2)}
        
        Current Positions:
        {json.dumps(positions, indent=2)}
        
        Market Conditions:
        - Volatility: {market_conditions['volatility']:.3f}
        - Trend: {market_conditions['trend']}
        - Market Open: {market_conditions['market_hours']}
        - Symbols Tracked: {market_conditions['symbols_tracked']}
        
        Please provide a comprehensive risk management analysis including:
        1. Trading signal: buy, sell, or hold
        2. Confidence level (0.0 to 1.0)
        3. Detailed risk assessment and reasoning
        4. Portfolio risk analysis
        5. Position sizing recommendations
        
        Focus on:
        - Risk management principles
        - Portfolio diversification
        - Position sizing
        - Stop loss recommendations
        - Market risk assessment
        - Canadian market specifics
        
        Respond in JSON format:
        {{
            "signal": "buy/sell/hold",
            "confidence": 0.0-1.0,
            "reason": "detailed risk-based reasoning",
            "risk_assessment": {{
                "portfolio_risk": "overall portfolio risk level",
                "position_risk": "individual position risks",
                "market_risk": "current market risk level",
                "diversification": "diversification analysis"
            }},
            "recommendations": {{
                "position_sizing": "position sizing advice",
                "stop_loss": "stop loss recommendations",
                "risk_limits": "risk limit suggestions"
            }},
            "canadian_market_insights": "specific Canadian market observations"
        }}
        """
        return prompt
    
    def _call_claude_api(self, prompt: str) -> Dict:
        """Call Claude API"""
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': self.model,
            'max_tokens': 2000,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.2  # Low temperature for risk analysis
        }
        
        response = requests.post(
            f'{self.base_url}/v1/messages',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Claude API error: {response.status_code} - {response.text}")
    
    def _parse_claude_response(self, response: Dict) -> Dict:
        """Parse Claude's response"""
        try:
            content = response['content'][0]['text']
            if '{' in content and '}' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {
                    'signal': 'hold',
                    'confidence': 0.5,
                    'reason': content[:200],
                    'risk_assessment': {},
                    'recommendations': {},
                    'canadian_market_insights': content
                }
        except Exception as e:
            logger.error(f" Error parsing Claude response: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.5,
                'reason': 'Error parsing response',
                'risk_assessment': {},
                'recommendations': {},
                'canadian_market_insights': 'Analysis unavailable'
            }


if __name__ == "__main__":
    # Test the AI ensemble
    logging.basicConfig(level=logging.INFO)
    
    ensemble = AIEnsemble()
    
    # Mock market data
    market_data = {
        'RY.TO': 132.50,
        'TD.TO': 88.75,
        'SHOP.TO': 102.30
    }
    
    positions = {
        'RY.TO': {'quantity': 50, 'avg_price': 130.00}
    }
    
    # Get ensemble analysis
    analysis = ensemble.analyze_market(market_data, positions)
    print(json.dumps(analysis, indent=2))
