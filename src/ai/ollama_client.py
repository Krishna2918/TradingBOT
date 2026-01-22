"""
Ollama Client for Local AI Model Integration
Provides a unified interface to interact with local Ollama models
"""

import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    model: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server not available: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def generate(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> OllamaResponse:
        """Generate response from Ollama model"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        
        if system:
            payload["system"] = system
        if context:
            payload["context"] = context
        if options:
            payload["options"] = options
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return OllamaResponse(
                    model=data.get("model", model),
                    response=data.get("response", ""),
                    done=data.get("done", True),
                    context=data.get("context"),
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    prompt_eval_duration=data.get("prompt_eval_duration"),
                    eval_count=data.get("eval_count"),
                    eval_duration=data.get("eval_duration")
                )
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return OllamaResponse(model=model, response="", done=True)
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return OllamaResponse(model=model, response="", done=True)
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> OllamaResponse:
        """Chat with Ollama model using message format"""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return OllamaResponse(
                    model=data.get("model", model),
                    response=data.get("message", {}).get("content", ""),
                    done=data.get("done", True),
                    context=data.get("context"),
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    prompt_eval_duration=data.get("prompt_eval_duration"),
                    eval_count=data.get("eval_count"),
                    eval_duration=data.get("eval_duration")
                )
            else:
                logger.error(f"Ollama chat API error: {response.status_code} - {response.text}")
                return OllamaResponse(model=model, response="", done=True)
                
        except Exception as e:
            logger.error(f"Failed to chat with model: {e}")
            return OllamaResponse(model=model, response="", done=True)

class OllamaTradingAI:
    """Trading AI using Ollama models"""
    
    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self.available_models = self._get_available_models()
        
    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = self.client.list_models()
        return [model["name"] for model in models]
    
    def get_best_model(self) -> str:
        """Get the best available model for trading decisions"""
        # Priority order: qwen3-coder (coding), deepseek-v3.1 (reasoning), gpt-oss (general)
        preferred_models = [
            "qwen3-coder:480b-cloud",
            "deepseek-v3.1:671b-cloud", 
            "gpt-oss:120b",
            "qwen2.5:14b-instruct"
        ]
        
        for model in preferred_models:
            if model in self.available_models:
                return model
        
        # Fallback to first available model
        return self.available_models[0] if self.available_models else "qwen2.5:14b-instruct"
    
    def analyze_market_data(
        self, 
        symbol: str, 
        market_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze market data using Ollama model"""
        
        if not self.client.is_available():
            logger.warning("Ollama server not available, using fallback analysis")
            return self._fallback_analysis(symbol, market_data)
        
        model = self.get_best_model()
        
        # Prepare the prompt
        prompt = self._create_analysis_prompt(symbol, market_data, sentiment_data)
        
        # Generate analysis
        response = self.client.generate(
            model=model,
            prompt=prompt,
            system=self._get_trading_system_prompt(),
            options={
                "temperature": 0.3,  # Lower temperature for more consistent decisions
                "top_p": 0.9,
                "max_tokens": 1000
            }
        )
        
        # Parse response
        return self._parse_analysis_response(response.response, symbol)
    
    def _create_analysis_prompt(
        self, 
        symbol: str, 
        market_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create analysis prompt for the model"""
        
        prompt = f"""
Analyze the following market data for {symbol} and provide a trading recommendation:

MARKET DATA:
- Current Price: ${market_data.get('current_price', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}
- Price Change: {market_data.get('price_change', 'N/A')}
- Price Change %: {market_data.get('price_change_pct', 'N/A')}
- 52-Week High: ${market_data.get('52_week_high', 'N/A')}
- 52-Week Low: ${market_data.get('52_week_low', 'N/A')}
- Market Cap: ${market_data.get('market_cap', 'N/A')}
"""
        
        if sentiment_data:
            prompt += f"""
SENTIMENT DATA:
- News Sentiment: {sentiment_data.get('news_sentiment', 'N/A')}
- Social Sentiment: {sentiment_data.get('social_sentiment', 'N/A')}
- Overall Sentiment Score: {sentiment_data.get('overall_sentiment', 'N/A')}
"""
        
        prompt += """
Please provide your analysis in the following JSON format:
{
    "action": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "reasoning": ["reason1", "reason2", "reason3"],
    "target_price": price_target,
    "stop_loss": stop_loss_price,
    "position_size_pct": 0.01-0.20
}

Focus on:
1. Technical analysis of price movements
2. Volume analysis
3. Support/resistance levels
4. Market sentiment (if available)
5. Risk assessment
"""
        
        return prompt
    
    def _get_trading_system_prompt(self) -> str:
        """Get system prompt for trading AI"""
        return """
You are an expert quantitative trader and financial analyst. Your role is to analyze market data and provide objective, data-driven trading recommendations.

Key principles:
1. Always base decisions on concrete data and technical analysis
2. Consider risk management in every recommendation
3. Provide clear reasoning for your decisions
4. Be conservative with position sizing (1-20% of portfolio)
5. Always include stop-loss recommendations
6. Consider market volatility and current conditions

Your responses should be professional, analytical, and focused on maximizing risk-adjusted returns.
"""
    
    def _parse_analysis_response(self, response: str, symbol: str) -> Dict[str, Any]:
        """Parse the model's analysis response"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Validate and clean the analysis
                return {
                    "action": analysis.get("action", "HOLD"),
                    "confidence": float(analysis.get("confidence", 0.5)),
                    "reasoning": analysis.get("reasoning", ["AI analysis"]),
                    "target_price": float(analysis.get("target_price", 0)),
                    "stop_loss": float(analysis.get("stop_loss", 0)),
                    "position_size_pct": float(analysis.get("position_size_pct", 0.02)),
                    "model_used": "ollama",
                    "raw_response": response
                }
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
        
        # Fallback analysis
        return self._fallback_analysis(symbol, {})
    
    def _fallback_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when Ollama is not available"""
        return {
            "action": "HOLD",
            "confidence": 0.3,
            "reasoning": ["Ollama server not available - using fallback analysis"],
            "target_price": 0,
            "stop_loss": 0,
            "position_size_pct": 0.01,
            "model_used": "fallback"
        }

# Global instance
ollama_client = OllamaClient()
ollama_trading_ai = OllamaTradingAI(ollama_client)
