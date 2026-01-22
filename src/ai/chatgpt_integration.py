"""
ChatGPT Integration for Enhanced AI Trading Decisions
Uses OpenAI's GPT-4 for advanced market analysis and trading decisions
"""

import openai
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatGPTIntegration:
    """ChatGPT integration for enhanced AI trading decisions"""
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        self.api_key = api_key
        self.config = config or {}
        
        # OpenAI client will be initialized per request
        
        # Rate limiting
        self.last_request = 0
        self.request_count = 0
        self.rate_limits = self.config.get('rate_limits', {}).get('openai_api', {})
        self.requests_per_minute = self.rate_limits.get('requests_per_minute', 50)
        self.requests_per_hour = self.rate_limits.get('requests_per_hour', 3000)
        
        # Request tracking
        self.minute_requests = 0
        self.hour_requests = 0
        self.last_minute_reset = time.time()
        self.last_hour_reset = time.time()
        
        logger.info(" ChatGPT Integration initialized")
        logger.info(f" Rate limits: {self.requests_per_minute}/min, {self.requests_per_hour}/hour")
        logger.info(" Using GPT-4 Turbo (Most Advanced Available)")
    
    def _check_rate_limit(self) -> bool:
        """Check if we can make a request based on rate limits"""
        now = time.time()
        
        # Reset counters if needed
        if now - self.last_minute_reset > 60:
            self.minute_requests = 0
            self.last_minute_reset = now
        
        if now - self.last_hour_reset > 3600:
            self.hour_requests = 0
            self.last_hour_reset = now
        
        # Check limits
        if self.minute_requests >= self.requests_per_minute:
            logger.warning("Rate limit: Too many requests per minute")
            return False
        
        if self.hour_requests >= self.requests_per_hour:
            logger.warning("Rate limit: Too many requests per hour")
            return False
        
        return True
    
    def _update_rate_limit(self):
        """Update rate limit counters"""
        self.minute_requests += 1
        self.hour_requests += 1
        self.last_request = time.time()
    
    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions using ChatGPT"""
        if not self._check_rate_limit():
            return self._get_fallback_analysis(market_data)
        
        try:
            # Prepare market data for analysis
            symbols = list(market_data.get('basic_market_data', {}).keys())
            news_sentiment = market_data.get('news_sentiment', {})
            technical_indicators = market_data.get('technical_indicators', {})
            
            # Create prompt for ChatGPT
            prompt = self._create_market_analysis_prompt(
                symbols, market_data, news_sentiment, technical_indicators
            )
            
            # Call ChatGPT API
            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI trading analyst specializing in Canadian markets (TSX/TSXV). Provide detailed market analysis and trading recommendations based on the data provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.1,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            analysis = self._parse_analysis_response(analysis_text)
            
            self._update_rate_limit()
            
            logger.info(" ChatGPT market analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f" ChatGPT analysis error: {e}")
            return self._get_fallback_analysis(market_data)
    
    def _create_market_analysis_prompt(self, symbols: List[str], market_data: Dict, 
                                     news_sentiment: Dict, technical_indicators: Dict) -> str:
        """Create a comprehensive prompt for market analysis"""
        
        prompt = f"""
Analyze the following Canadian market data and provide trading recommendations:

**SYMBOLS TO ANALYZE:** {', '.join(symbols)}

**MARKET DATA:**
"""
        
        # Add basic market data
        for symbol in symbols:
            if symbol in market_data.get('basic_market_data', {}):
                data = market_data['basic_market_data'][symbol]
                if data is not None and hasattr(data, 'empty') and not data.empty:
                    latest = data.iloc[-1]
                    prompt += f"""
{symbol}:
- Price: ${latest.get('close', 0):.2f}
- Volume: {latest.get('volume', 0):,}
- High: ${latest.get('high', 0):.2f}
- Low: ${latest.get('low', 0):.2f}
"""
                else:
                    prompt += f"""
{symbol}:
- Price: $100.00 (demo data)
- Volume: 1,000,000 (demo data)
- High: $102.00 (demo data)
- Low: $98.00 (demo data)
"""
        
        # Add news sentiment
        if news_sentiment:
            prompt += f"""
**NEWS SENTIMENT:**
- Overall Sentiment: {news_sentiment.get('overall_sentiment', 0):.3f}
- Total Articles: {news_sentiment.get('total_articles', 0)}
"""
            for symbol, sentiment in news_sentiment.get('symbols', {}).items():
                prompt += f"- {symbol}: {sentiment.get('sentiment', 0):.3f} ({sentiment.get('article_count', 0)} articles)\n"
        
        # Add technical indicators
        if technical_indicators:
            prompt += "\n**TECHNICAL INDICATORS:**\n"
            for symbol, indicators in technical_indicators.items():
                prompt += f"""
{symbol}:
- SMA 20: {indicators.get('sma_20', 0):.2f}
- RSI 14: {indicators.get('rsi_14', 0):.2f}
- MACD: {indicators.get('macd', 0):.3f}
"""
        
        prompt += """
**ANALYSIS REQUIREMENTS:**
1. Market trend analysis (bullish/bearish/sideways)
2. Risk assessment for each symbol
3. Specific trading recommendations (BUY/SELL/HOLD)
4. Confidence level (1-10)
5. Reasoning for each recommendation
6. Key factors influencing the decision

**RESPONSE FORMAT:**
Provide a JSON response with the following structure:
{
  "market_trend": "bullish/bearish/sideways",
  "overall_confidence": 8,
  "recommendations": [
    {
      "symbol": "TD.TO",
      "action": "BUY/SELL/HOLD",
      "confidence": 8,
      "reasoning": "Detailed explanation",
      "key_factors": ["factor1", "factor2"]
    }
  ],
  "risk_assessment": "Low/Medium/High",
  "market_insights": "Overall market analysis"
}
"""
        
        return prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse ChatGPT response into structured data"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Add metadata
                analysis['timestamp'] = datetime.now().isoformat()
                analysis['source'] = 'chatgpt'
                analysis['model'] = 'gpt-4'
                
                return analysis
            else:
                # Fallback parsing
                return self._parse_text_response(response_text)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, using text parsing")
            return self._parse_text_response(response_text)
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback text parsing when JSON parsing fails"""
        return {
            'market_trend': 'neutral',
            'overall_confidence': 5,
            'recommendations': [],
            'risk_assessment': 'medium',
            'market_insights': response_text[:500],  # Truncate long responses
            'timestamp': datetime.now().isoformat(),
            'source': 'chatgpt',
            'model': 'gpt-4',
            'parsing_method': 'text_fallback'
        }
    
    def _get_fallback_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when ChatGPT is unavailable"""
        return {
            'market_trend': 'neutral',
            'overall_confidence': 3,
            'recommendations': [],
            'risk_assessment': 'medium',
            'market_insights': 'ChatGPT analysis unavailable - using fallback',
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback',
            'model': 'none'
        }
    
    def get_trading_decision(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get specific trading decision for a symbol"""
        if not self._check_rate_limit():
            return self._get_fallback_decision(symbol)
        
        try:
            # Create focused prompt for specific symbol
            prompt = self._create_trading_decision_prompt(symbol, market_data)
            
            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert AI trader. Provide a specific trading decision for {symbol} based on the market data provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.1,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            decision_text = response.choices[0].message.content
            decision = self._parse_trading_decision(decision_text, symbol)
            
            self._update_rate_limit()
            
            logger.info(f" ChatGPT trading decision for {symbol}")
            return decision
            
        except Exception as e:
            logger.error(f" ChatGPT decision error for {symbol}: {e}")
            return self._get_fallback_decision(symbol)
    
    def _create_trading_decision_prompt(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Create prompt for specific trading decision"""
        
        # Get symbol-specific data
        symbol_data = market_data.get('basic_market_data', {}).get(symbol)
        news_data = market_data.get('news_sentiment', {}).get('symbols', {}).get(symbol, {})
        tech_data = market_data.get('technical_indicators', {}).get(symbol, {})
        
        prompt = f"""
Provide a specific trading decision for {symbol}:

**CURRENT DATA:**
"""
        
        if symbol_data is not None and not symbol_data.empty:
            latest = symbol_data.iloc[-1]
            prompt += f"""
- Current Price: ${latest.get('close', 0):.2f}
- Volume: {latest.get('volume', 0):,}
- 24h High: ${latest.get('high', 0):.2f}
- 24h Low: ${latest.get('low', 0):.2f}
"""
        
        if news_data:
            prompt += f"""
- News Sentiment: {news_data.get('sentiment', 0):.3f}
- Articles: {news_data.get('article_count', 0)}
"""
        
        if tech_data:
            prompt += f"""
- SMA 20: {tech_data.get('sma_20', 0):.2f}
- RSI: {tech_data.get('rsi_14', 0):.2f}
- MACD: {tech_data.get('macd', 0):.3f}
"""
        
        prompt += f"""
**DECISION REQUIRED:**
Should I BUY, SELL, or HOLD {symbol}?

**RESPONSE FORMAT:**
{{
  "symbol": "{symbol}",
  "action": "BUY/SELL/HOLD",
  "confidence": 8,
  "reasoning": "Detailed explanation",
  "entry_price": 100.50,
  "stop_loss": 95.00,
  "take_profit": 110.00,
  "position_size": "small/medium/large"
}}
"""
        
        return prompt
    
    def _parse_trading_decision(self, response_text: str, symbol: str) -> Dict[str, Any]:
        """Parse trading decision response"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                decision = json.loads(json_str)
                
                # Add metadata
                decision['timestamp'] = datetime.now().isoformat()
                decision['source'] = 'chatgpt'
                decision['model'] = 'gpt-4'
                
                return decision
            else:
                return self._get_fallback_decision(symbol)
                
        except json.JSONDecodeError:
            return self._get_fallback_decision(symbol)
    
    def _get_fallback_decision(self, symbol: str) -> Dict[str, Any]:
        """Fallback decision when ChatGPT is unavailable"""
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 3,
            'reasoning': 'ChatGPT unavailable - using conservative HOLD',
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 'small',
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback',
            'model': 'none'
        }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get ChatGPT API status and usage"""
        return {
            'configured': bool(self.api_key and self.api_key != 'demo'),
            'rate_limit_status': self._check_rate_limit(),
            'requests_this_minute': self.minute_requests,
            'requests_this_hour': self.hour_requests,
            'last_request': self.last_request,
            'model': 'gpt-4'
        }


# Example usage
if __name__ == "__main__":
    # Test ChatGPT integration
    config = {
        'rate_limits': {
            'openai_api': {
                'requests_per_minute': 50,
                'requests_per_hour': 3000
            }
        }
    }
    
    # Initialize with your API key
    chatgpt = ChatGPTIntegration("your-api-key-here", config)
    
    # Test market analysis
    test_data = {
        'basic_market_data': {
            'TD.TO': None,  # Would contain actual market data
        },
        'news_sentiment': {
            'overall_sentiment': 0.2,
            'total_articles': 15
        },
        'technical_indicators': {
            'TD.TO': {
                'sma_20': 100.5,
                'rsi_14': 65.2,
                'macd': 0.5
            }
        }
    }
    
    # Get analysis
    analysis = chatgpt.analyze_market_conditions(test_data)
    print("Market Analysis:", json.dumps(analysis, indent=2))
    
    # Get trading decision
    decision = chatgpt.get_trading_decision('TD.TO', test_data)
    print("Trading Decision:", json.dumps(decision, indent=2))
