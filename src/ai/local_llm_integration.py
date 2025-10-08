"""
Local LLM Integration Module
Production-ready integration with Qwen2.5-14B-Instruct via Ollama
"""

import json
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structured response from local LLM"""
    content: str
    model: str
    timestamp: datetime
    tokens_used: int
    response_time: float
    confidence: float = 0.0
    reasoning: Optional[str] = None

@dataclass
class LLMRequest:
    """Structured request to local LLM"""
    prompt: str
    model: str = "qwen2.5:14b-instruct"
    max_tokens: int = 2048
    temperature: float = 0.7
    context: Optional[Dict[str, Any]] = None
    task_type: str = "general"

class LocalLLMClient:
    """
    Production-ready local LLM client for Qwen2.5-14B-Instruct
    Handles trading analysis, feature conflict detection, and narrative generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('ollama_url', 'http://localhost:11434')
        self.model_name = config.get('model_name', 'qwen2.5:14b-instruct')
        self.max_retries = config.get('max_retries', 3)
        self.timeout = config.get('timeout', 30)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.1)
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Cache for common responses
        self.response_cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
        logger.info(f"Local LLM Client initialized with model: {self.model_name}")
    
    def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make synchronous request to Ollama API"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._get_cache_key(request)
            if cache_key in self.response_cache:
                cached_response, timestamp = self.response_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug("Using cached response")
                    return cached_response
            
            # Prepare request payload
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    
                    # Parse response
                    result = response.json()
                    
                    # Calculate metrics
                    response_time = time.time() - start_time
                    tokens_used = len(result.get('response', '').split())
                    
                    # Create structured response
                    llm_response = LLMResponse(
                        content=result.get('response', ''),
                        model=request.model,
                        timestamp=datetime.now(),
                        tokens_used=tokens_used,
                        response_time=response_time,
                        confidence=self._calculate_confidence(result.get('response', '')),
                        reasoning=self._extract_reasoning(result.get('response', ''))
                    )
                    
                    # Update performance metrics
                    self._update_metrics(llm_response)
                    
                    # Cache response
                    self.response_cache[cache_key] = (llm_response, time.time())
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                    return llm_response
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            logger.error(f"Error making LLM request: {e}")
            self.error_count += 1
            raise
    
    async def _make_async_request(self, request: LLMRequest) -> LLMResponse:
        """Make asynchronous request to Ollama API"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._get_cache_key(request)
            if cache_key in self.response_cache:
                cached_response, timestamp = self.response_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug("Using cached response")
                    return cached_response
            
            # Prepare request payload
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
            
            # Make async request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    # Calculate metrics
                    response_time = time.time() - start_time
                    tokens_used = len(result.get('response', '').split())
                    
                    # Create structured response
                    llm_response = LLMResponse(
                        content=result.get('response', ''),
                        model=request.model,
                        timestamp=datetime.now(),
                        tokens_used=tokens_used,
                        response_time=response_time,
                        confidence=self._calculate_confidence(result.get('response', '')),
                        reasoning=self._extract_reasoning(result.get('response', ''))
                    )
                    
                    # Update performance metrics
                    self._update_metrics(llm_response)
                    
                    # Cache response
                    self.response_cache[cache_key] = (llm_response, time.time())
                    
                    return llm_response
                    
        except Exception as e:
            logger.error(f"Error making async LLM request: {e}")
            self.error_count += 1
            raise
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        key_data = f"{request.prompt}_{request.model}_{request.temperature}_{request.max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response characteristics"""
        if not response:
            return 0.0
        
        # Simple confidence calculation based on response length and structure
        confidence = 0.5  # Base confidence
        
        # Longer responses generally indicate more thoughtful analysis
        if len(response) > 100:
            confidence += 0.2
        if len(response) > 500:
            confidence += 0.1
        
        # Check for structured reasoning indicators
        reasoning_indicators = ['because', 'therefore', 'however', 'furthermore', 'analysis', 'indicates']
        for indicator in reasoning_indicators:
            if indicator.lower() in response.lower():
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from response if present"""
        # Look for reasoning sections
        reasoning_keywords = ['reasoning:', 'analysis:', 'because:', 'therefore:']
        
        for keyword in reasoning_keywords:
            if keyword.lower() in response.lower():
                # Extract text after the keyword
                parts = response.lower().split(keyword.lower())
                if len(parts) > 1:
                    return parts[1].strip()[:200]  # Limit length
        
        return None
    
    def _update_metrics(self, response: LLMResponse):
        """Update performance metrics"""
        self.request_count += 1
        self.total_tokens += response.tokens_used
        self.total_response_time += response.response_time
    
    def generate_trade_narrative(self, trade_data: Dict[str, Any]) -> LLMResponse:
        """Generate human-readable trade narrative"""
        prompt = self._build_trade_narrative_prompt(trade_data)
        
        request = LLMRequest(
            prompt=prompt,
            task_type="trade_narrative",
            temperature=0.7,
            max_tokens=1024,
            context=trade_data
        )
        
        return self._make_request(request)
    
    def detect_feature_conflicts(self, features: List[Dict[str, Any]]) -> LLMResponse:
        """Detect conflicts between trading features"""
        prompt = self._build_feature_conflict_prompt(features)
        
        request = LLMRequest(
            prompt=prompt,
            task_type="feature_conflict_detection",
            temperature=0.3,  # Lower temperature for more deterministic analysis
            max_tokens=512,
            context={"features": features}
        )
        
        return self._make_request(request)
    
    def analyze_market_regime(self, market_data: Dict[str, Any]) -> LLMResponse:
        """Analyze current market regime"""
        prompt = self._build_market_regime_prompt(market_data)
        
        request = LLMRequest(
            prompt=prompt,
            task_type="market_regime_analysis",
            temperature=0.5,
            max_tokens=768,
            context=market_data
        )
        
        return self._make_request(request)
    
    def generate_risk_assessment(self, portfolio_data: Dict[str, Any]) -> LLMResponse:
        """Generate comprehensive risk assessment"""
        prompt = self._build_risk_assessment_prompt(portfolio_data)
        
        request = LLMRequest(
            prompt=prompt,
            task_type="risk_assessment",
            temperature=0.4,
            max_tokens=1024,
            context=portfolio_data
        )
        
        return self._make_request(request)
    
    def _build_trade_narrative_prompt(self, trade_data: Dict[str, Any]) -> str:
        """Build prompt for trade narrative generation"""
        return f"""
You are an expert trading analyst. Generate a clear, concise narrative explaining the following trade decision:

Trade Data:
- Symbol: {trade_data.get('symbol', 'N/A')}
- Action: {trade_data.get('action', 'N/A')}
- Quantity: {trade_data.get('quantity', 'N/A')}
- Price: {trade_data.get('price', 'N/A')}
- Timestamp: {trade_data.get('timestamp', 'N/A')}
- AI Confidence: {trade_data.get('confidence', 'N/A')}
- Technical Signals: {trade_data.get('technical_signals', {})}
- Market Conditions: {trade_data.get('market_conditions', {})}

Please provide:
1. A brief explanation of why this trade was made
2. Key factors that influenced the decision
3. Risk considerations
4. Expected outcome

Keep the response professional and under 200 words.
"""
    
    def _build_feature_conflict_prompt(self, features: List[Dict[str, Any]]) -> str:
        """Build prompt for feature conflict detection"""
        feature_descriptions = []
        for i, feature in enumerate(features):
            feature_descriptions.append(f"{i+1}. {feature.get('name', 'Unknown')}: {feature.get('description', 'No description')}")
        
        return f"""
You are a quantitative analyst reviewing trading features for conflicts. Analyze the following features for potential conflicts:

Features:
{chr(10).join(feature_descriptions)}

Identify:
1. Any features that might be measuring the same underlying market phenomenon
2. Features that could lead to overfitting or redundant signals
3. Potential conflicts in feature logic or interpretation
4. Recommendations for feature selection or modification

Provide a structured analysis with specific examples and actionable recommendations.
"""
    
    def _build_market_regime_prompt(self, market_data: Dict[str, Any]) -> str:
        """Build prompt for market regime analysis"""
        return f"""
You are a market regime analyst. Analyze the current market conditions and determine the prevailing regime:

Market Data:
- Volatility: {market_data.get('volatility', 'N/A')}
- Trend Direction: {market_data.get('trend', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}
- Sector Performance: {market_data.get('sector_performance', {})}
- Economic Indicators: {market_data.get('economic_indicators', {})}
- News Sentiment: {market_data.get('news_sentiment', 'N/A')}

Classify the market regime as one of:
1. Bull Market (strong upward trend)
2. Bear Market (strong downward trend)
3. Sideways/Range-bound (consolidation)
4. High Volatility (uncertain/transitional)
5. Low Volatility (stable/calm)

Provide:
1. Regime classification with confidence level
2. Key characteristics supporting this classification
3. Expected duration and transition indicators
4. Trading strategy implications
"""
    
    def _build_risk_assessment_prompt(self, portfolio_data: Dict[str, Any]) -> str:
        """Build prompt for risk assessment"""
        return f"""
You are a risk management expert. Assess the current portfolio risk profile:

Portfolio Data:
- Total Value: {portfolio_data.get('total_value', 'N/A')}
- Positions: {portfolio_data.get('positions', {})}
- Sector Allocation: {portfolio_data.get('sector_allocation', {})}
- VaR (95%): {portfolio_data.get('var_95', 'N/A')}
- Max Drawdown: {portfolio_data.get('max_drawdown', 'N/A')}
- Beta: {portfolio_data.get('beta', 'N/A')}
- Correlation: {portfolio_data.get('correlation', 'N/A')}

Provide:
1. Overall risk assessment (Low/Medium/High)
2. Key risk factors and their impact
3. Concentration risks
4. Recommendations for risk mitigation
5. Portfolio optimization suggestions

Focus on actionable insights and specific recommendations.
"""
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        avg_tokens = self.total_tokens / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1)
        
        return {
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens,
            "average_response_time": avg_response_time,
            "average_tokens_per_request": avg_tokens,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "cache_size": len(self.response_cache),
            "model_name": self.model_name
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Ollama service"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_available = any(model.get('name', '').startswith(self.model_name) for model in models)
            
            return {
                "status": "healthy" if model_available else "unhealthy",
                "ollama_available": True,
                "model_available": model_available,
                "available_models": [model.get('name') for model in models],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "ollama_available": False,
                "model_available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Production configuration
DEFAULT_CONFIG = {
    "ollama_url": "http://localhost:11434",
    "model_name": "qwen2.5:14b-instruct",
    "max_retries": 3,
    "timeout": 30,
    "rate_limit_delay": 0.1,
    "cache_ttl": 300
}

def create_local_llm_client(config: Optional[Dict[str, Any]] = None) -> LocalLLMClient:
    """Factory function to create LocalLLMClient with default config"""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    return LocalLLMClient(config)
