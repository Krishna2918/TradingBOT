"""
Multi-Model Integration System

This module implements the integration of multiple AI models from Ollama,
each with specialized roles in the trading decision-making process.
"""

import logging
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time

from src.config.mode_manager import get_current_mode
from src.ai.adaptive_weights import get_adaptive_weight_manager, get_ensemble_weights
from src.ai.ollama_lifecycle import get_lifecycle_manager, ensure_ollama_healthy, cleanup_memory_if_needed

logger = logging.getLogger(__name__)

class ModelRole(Enum):
    """AI model role enumeration."""
    CHIEF_TECHNICAL_ANALYST = "CHIEF_TECHNICAL_ANALYST"
    RISK_MANAGEMENT_SPECIALIST = "RISK_MANAGEMENT_SPECIALIST"
    MARKET_SENTIMENT_ANALYST = "MARKET_SENTIMENT_ANALYST"
    FUNDAMENTAL_ANALYSIS_EXPERT = "FUNDAMENTAL_ANALYSIS_EXPERT"
    QUANTITATIVE_STRATEGY_DEVELOPER = "QUANTITATIVE_STRATEGY_DEVELOPER"

@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    name: str
    role: ModelRole
    model_id: str
    base_url: str
    temperature: float
    max_tokens: int
    system_prompt: str
    expertise_areas: List[str]
    weight: float

@dataclass
class ModelResponse:
    """Response from an AI model."""
    model_name: str
    role: ModelRole
    response: str
    confidence: float
    reasoning: str
    timestamp: datetime
    processing_time: float
    token_count: int

@dataclass
class ModelOpinion:
    """Opinion from a model in the collaborative discussion."""
    model_name: str
    role: ModelRole
    opinion: str
    confidence: float
    supporting_evidence: List[str]
    concerns: List[str]
    recommendation: str
    timestamp: datetime

class OllamaClient:
    """Client for interacting with Ollama models."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
        
        logger.info(f"Ollama client initialized with base URL: {base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def generate_response(self, model_id: str, prompt: str, 
                              system_prompt: str = "", temperature: float = 0.7,
                              max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response from an Ollama model."""
        try:
            if not self.session:
                raise RuntimeError("Ollama client not initialized. Use async context manager.")
            
            start_time = time.time()
            
            # Prepare request payload
            payload = {
                "model": model_id,
                "prompt": prompt,
                "system": system_prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }
            
            # Make request to Ollama
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    return {
                        "response": result.get("response", ""),
                        "processing_time": processing_time,
                        "token_count": len(result.get("response", "").split()),
                        "success": True
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    return {
                        "response": "",
                        "processing_time": 0.0,
                        "token_count": 0,
                        "success": False,
                        "error": f"API error {response.status}: {error_text}"
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout generating response from {model_id}")
            return {
                "response": "",
                "processing_time": 0.0,
                "token_count": 0,
                "success": False,
                "error": "Request timeout"
            }
        except Exception as e:
            logger.error(f"Error generating response from {model_id}: {e}")
            return {
                "response": "",
                "processing_time": 0.0,
                "token_count": 0,
                "success": False,
                "error": str(e)
            }
    
    async def check_model_availability(self, model_id: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            if not self.session:
                return False
            
            async with self.session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return model_id in models
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking model availability for {model_id}: {e}")
            return False

class MultiModelManager:
    """Multi-model AI manager (alias for MultiModelAI)."""
    pass

class MultiModelAI(MultiModelManager):
    """Multi-model AI class for compatibility."""
    """Manages multiple AI models and their interactions."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.models = {}
        self.ollama_client = OllamaClient()
        self.lifecycle_manager = get_lifecycle_manager()
        
        # Initialize model configurations
        self._initialize_model_configs()
        
        logger.info(f"Multi-Model Manager initialized for {mode} mode with lifecycle management")
    
    def _initialize_model_configs(self):
        """Initialize configurations for all AI models."""
        self.models = {
            "qwen2.5": ModelConfig(
                name="Qwen2.5",
                role=ModelRole.CHIEF_TECHNICAL_ANALYST,
                model_id="qwen2.5:14b-instruct",
                base_url="http://localhost:11434",
                temperature=0.7,
                max_tokens=1000,
                system_prompt=self._get_technical_analyst_prompt(),
                expertise_areas=["Technical Analysis", "Chart Patterns", "Market Trends", "Price Action"],
                weight=0.25
            ),
            "llama3.1": ModelConfig(
                name="Llama3.1",
                role=ModelRole.RISK_MANAGEMENT_SPECIALIST,
                model_id="gpt-oss:120b",
                base_url="http://localhost:11434",
                temperature=0.6,
                max_tokens=800,
                system_prompt=self._get_risk_management_prompt(),
                expertise_areas=["Risk Assessment", "Portfolio Management", "Position Sizing", "Drawdown Control"],
                weight=0.20
            ),
            "mistral": ModelConfig(
                name="Mistral",
                role=ModelRole.MARKET_SENTIMENT_ANALYST,
                model_id="deepseek-v3.1:671b-cloud",
                base_url="http://localhost:11434",
                temperature=0.8,
                max_tokens=900,
                system_prompt=self._get_sentiment_analyst_prompt(),
                expertise_areas=["Market Sentiment", "News Analysis", "Social Media", "Market Psychology"],
                weight=0.20
            ),
            "phi3": ModelConfig(
                name="Phi3",
                role=ModelRole.FUNDAMENTAL_ANALYSIS_EXPERT,
                model_id="qwen3-coder:480b-cloud",
                base_url="http://localhost:11434",
                temperature=0.5,
                max_tokens=700,
                system_prompt=self._get_fundamental_analyst_prompt(),
                expertise_areas=["Fundamental Analysis", "Financial Statements", "Valuation", "Economic Indicators"],
                weight=0.20
            ),
            "codellama": ModelConfig(
                name="CodeLlama",
                role=ModelRole.QUANTITATIVE_STRATEGY_DEVELOPER,
                model_id="gpt-oss:120b-cloud",
                base_url="http://localhost:11434",
                temperature=0.4,
                max_tokens=800,
                system_prompt=self._get_quantitative_developer_prompt(),
                expertise_areas=["Quantitative Analysis", "Algorithm Development", "Statistical Models", "Backtesting"],
                weight=0.15
            )
        }
        
        logger.info(f"Initialized {len(self.models)} AI models")
    
    def _get_technical_analyst_prompt(self) -> str:
        """Get system prompt for technical analyst."""
        return """You are a Chief Technical Analyst specializing in market analysis and trading decisions.

Your expertise includes:
- Technical analysis and chart patterns
- Market trends and price action
- Support and resistance levels
- Moving averages and indicators
- Volume analysis

When analyzing trading opportunities:
1. Focus on technical indicators and chart patterns
2. Identify key support and resistance levels
3. Assess trend strength and momentum
4. Consider volume confirmation
5. Provide clear buy/sell/hold recommendations with confidence levels

Always provide reasoning for your analysis and be specific about entry/exit points."""
    
    def _get_risk_management_prompt(self) -> str:
        """Get system prompt for risk management specialist."""
        return """You are a Risk Management Specialist focused on protecting capital and managing portfolio risk.

Your expertise includes:
- Risk assessment and measurement
- Portfolio management and diversification
- Position sizing and allocation
- Drawdown control and stop-losses
- Risk-adjusted returns

When evaluating trading opportunities:
1. Assess the risk-reward ratio
2. Evaluate position sizing based on volatility
3. Consider portfolio correlation and diversification
4. Identify potential risk factors
5. Recommend risk management strategies

Always prioritize capital preservation and provide specific risk management recommendations."""
    
    def _get_sentiment_analyst_prompt(self) -> str:
        """Get system prompt for market sentiment analyst."""
        return """You are a Market Sentiment Analyst specializing in understanding market psychology and sentiment.

Your expertise includes:
- Market sentiment analysis
- News and event impact assessment
- Social media sentiment
- Market psychology and behavioral finance
- Fear and greed indicators

When analyzing market conditions:
1. Assess overall market sentiment
2. Evaluate news and event impacts
3. Consider social media and retail sentiment
4. Identify sentiment extremes and reversals
5. Predict sentiment-driven price movements

Always consider the psychological aspects of trading and market behavior."""
    
    def _get_fundamental_analyst_prompt(self) -> str:
        """Get system prompt for fundamental analysis expert."""
        return """You are a Fundamental Analysis Expert specializing in company and market fundamentals.

Your expertise includes:
- Financial statement analysis
- Company valuation and metrics
- Economic indicators and macro trends
- Industry analysis and competitive positioning
- Long-term value assessment

When analyzing investment opportunities:
1. Evaluate company fundamentals and financial health
2. Assess valuation metrics and fair value
3. Consider industry trends and competitive position
4. Analyze economic factors and macro environment
5. Provide long-term value perspective

Always focus on fundamental value and long-term investment thesis."""
    
    def _get_quantitative_developer_prompt(self) -> str:
        """Get system prompt for quantitative strategy developer."""
        return """You are a Quantitative Strategy Developer specializing in algorithmic trading and statistical analysis.

Your expertise includes:
- Quantitative analysis and statistical models
- Algorithm development and optimization
- Backtesting and performance analysis
- Mathematical modeling and data analysis
- Systematic trading strategies

When developing trading strategies:
1. Use statistical and mathematical approaches
2. Develop systematic and rule-based strategies
3. Consider backtesting and historical performance
4. Optimize parameters and risk management
5. Provide quantitative justification for decisions

Always base recommendations on data, statistics, and systematic analysis."""
    
    async def check_model_availability(self) -> Dict[str, bool]:
        """Check availability of all models."""
        availability = {}
        
        async with self.ollama_client as client:
            for model_name, config in self.models.items():
                is_available = await client.check_model_availability(config.model_id)
                availability[model_name] = is_available
                logger.info(f"Model {model_name} ({config.model_id}): {'Available' if is_available else 'Not Available'}")
        
        return availability
    
    async def get_model_opinion(self, model_name: str, analysis_prompt: str, 
                              market_data: Dict[str, Any]) -> Optional[ModelOpinion]:
        """Get opinion from a specific model."""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return None
            
            config = self.models[model_name]
            
            # Prepare the full prompt with market data
            full_prompt = self._prepare_analysis_prompt(config, analysis_prompt, market_data)
            
            async with self.ollama_client as client:
                result = await client.generate_response(
                    model_id=config.model_id,
                    prompt=full_prompt,
                    system_prompt=config.system_prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
                
                if result["success"]:
                    # Parse the response to extract structured information
                    parsed_response = self._parse_model_response(result["response"], config.role)
                    
                    # Log model prediction for performance tracking
                    symbol = market_data.get("symbol", "UNKNOWN")
                    confidence = parsed_response["confidence"]
                    self.log_model_prediction(model_name, symbol, confidence)
                    
                    return ModelOpinion(
                        model_name=config.name,
                        role=config.role,
                        opinion=parsed_response["opinion"],
                        confidence=parsed_response["confidence"],
                        supporting_evidence=parsed_response["supporting_evidence"],
                        concerns=parsed_response["concerns"],
                        recommendation=parsed_response["recommendation"],
                        timestamp=datetime.now()
                    )
                else:
                    logger.error(f"Failed to get response from {model_name}: {result.get('error', 'Unknown error')}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting opinion from {model_name}: {e}")
            return None
    
    def _prepare_analysis_prompt(self, config: ModelConfig, analysis_prompt: str, 
                               market_data: Dict[str, Any]) -> str:
        """Prepare the full analysis prompt with market data."""
        prompt = f"""
Analysis Request: {analysis_prompt}

Market Data:
{json.dumps(market_data, indent=2)}

Please provide your analysis in the following format:

OPINION: [Your detailed analysis and opinion]

CONFIDENCE: [Confidence level from 0.0 to 1.0]

SUPPORTING_EVIDENCE:
- [Evidence point 1]
- [Evidence point 2]
- [Evidence point 3]

CONCERNS:
- [Concern 1]
- [Concern 2]

RECOMMENDATION: [BUY/SELL/HOLD with specific reasoning]

Focus on your expertise areas: {', '.join(config.expertise_areas)}
"""
        return prompt
    
    def _parse_model_response(self, response: str, role: ModelRole) -> Dict[str, Any]:
        """Parse model response to extract structured information."""
        try:
            # Initialize default values
            parsed = {
                "opinion": response,
                "confidence": 0.5,
                "supporting_evidence": [],
                "concerns": [],
                "recommendation": "HOLD"
            }
            
            # Try to extract structured information
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("OPINION:"):
                    parsed["opinion"] = line.replace("OPINION:", "").strip()
                    current_section = "opinion"
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence_str = line.replace("CONFIDENCE:", "").strip()
                        parsed["confidence"] = float(confidence_str)
                    except ValueError:
                        parsed["confidence"] = 0.5
                elif line.startswith("SUPPORTING_EVIDENCE:"):
                    current_section = "evidence"
                elif line.startswith("CONCERNS:"):
                    current_section = "concerns"
                elif line.startswith("RECOMMENDATION:"):
                    recommendation = line.replace("RECOMMENDATION:", "").strip()
                    if "BUY" in recommendation.upper():
                        parsed["recommendation"] = "BUY"
                    elif "SELL" in recommendation.upper():
                        parsed["recommendation"] = "SELL"
                    else:
                        parsed["recommendation"] = "HOLD"
                elif line.startswith("- ") and current_section:
                    item = line.replace("- ", "").strip()
                    if current_section == "evidence":
                        parsed["supporting_evidence"].append(item)
                    elif current_section == "concerns":
                        parsed["concerns"].append(item)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            return {
                "opinion": response,
                "confidence": 0.5,
                "supporting_evidence": [],
                "concerns": [],
                "recommendation": "HOLD"
            }
    
    async def get_all_model_opinions(self, analysis_prompt: str, 
                                   market_data: Dict[str, Any]) -> List[ModelOpinion]:
        """Get opinions from all available models."""
        opinions = []
        
        # Ensure models are ready before proceeding
        if not await self.ensure_models_ready():
            logger.error("Models are not ready for inference")
            return opinions
        
        # Check model availability first
        availability = await self.check_model_availability()
        
        # Get opinions from available models
        tasks = []
        for model_name, is_available in availability.items():
            if is_available:
                task = self.get_model_opinion(model_name, analysis_prompt, market_data)
                tasks.append(task)
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ModelOpinion):
                    opinions.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error getting model opinion: {result}")
        
        # Clean up memory if needed after inference
        await self.cleanup_memory_if_needed()
        
        logger.info(f"Collected {len(opinions)} model opinions")
        return opinions
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_all_model_configs(self) -> Dict[str, ModelConfig]:
        """Get all model configurations."""
        return self.models.copy()
    
    def get_models_by_role(self, role: ModelRole) -> List[ModelConfig]:
        """Get models by their role."""
        return [config for config in self.models.values() if config.role == role]
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get weights for all models."""
        return {name: config.weight for name, config in self.models.items()}
    
    def update_model_weight(self, model_name: str, weight: float):
        """Update weight for a specific model."""
        if model_name in self.models:
            self.models[model_name].weight = weight
            logger.info(f"Updated weight for {model_name} to {weight}")
        else:
            logger.error(f"Model {model_name} not found")
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get adaptive weights from the adaptive weight manager."""
        try:
            adaptive_weights = get_ensemble_weights()
            if adaptive_weights:
                # Map ensemble model names to our model names
                model_mapping = {
                    "technical_analyst": "qwen2.5",  # Chief Technical Analyst
                    "sentiment_analyst": "llama3.1",  # Market Sentiment Analyst
                    "fundamental_analyst": "gemma2",  # Fundamental Analysis Expert
                    "risk_analyst": "phi3",  # Risk Management Specialist
                    "market_regime_analyst": "mistral"  # Quantitative Strategy Developer
                }
                
                mapped_weights = {}
                for ensemble_name, weight in adaptive_weights.items():
                    if ensemble_name in model_mapping:
                        mapped_weights[model_mapping[ensemble_name]] = weight
                
                return mapped_weights
            else:
                # Return default weights if no adaptive weights available
                return self.get_model_weights()
                
        except Exception as e:
            logger.error(f"Error getting adaptive weights: {e}")
            return self.get_model_weights()
    
    async def ensure_models_ready(self) -> bool:
        """
        Ensure all required models are ready for inference.
        
        Returns:
            True if all models are ready, False otherwise
        """
        try:
            # Check Ollama health first
            if not await ensure_ollama_healthy():
                logger.error("Ollama is not healthy")
                return False
            
            # Get preferred models in order of preference
            preferred_models = list(self.models.keys())
            
            # Use lifecycle manager to get the best available model
            available_model = await self.lifecycle_manager.get_model_for_inference(preferred_models)
            
            if available_model:
                logger.info(f"Model {available_model} is ready for inference")
                return True
            else:
                logger.error("No models available for inference")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring models are ready: {e}")
            return False
    
    async def pre_warm_models(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Pre-warm specified models or all models.
        
        Args:
            model_names: List of model names to pre-warm, or None for all models
            
        Returns:
            Dictionary mapping model names to success status
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = {}
        for model_name in model_names:
            if model_name in self.models:
                success = await self.lifecycle_manager.pre_warm_model(model_name)
                results[model_name] = success
                if success:
                    logger.info(f"Model {model_name} pre-warmed successfully")
                else:
                    logger.warning(f"Failed to pre-warm model {model_name}")
            else:
                logger.warning(f"Model {model_name} not found in configuration")
                results[model_name] = False
        
        return results
    
    async def cleanup_memory_if_needed(self) -> bool:
        """
        Clean up memory if under pressure.
        
        Returns:
            True if cleanup was performed, False otherwise
        """
        return await cleanup_memory_if_needed()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including lifecycle information.
        
        Returns:
            Dictionary with system status
        """
        try:
            lifecycle_status = await self.lifecycle_manager.get_status()
            model_availability = await self.check_model_availability()
            
            return {
                'lifecycle_status': lifecycle_status,
                'model_availability': model_availability,
                'model_configs': {
                    name: {
                        'role': config.role.value,
                        'weight': config.weight,
                        'temperature': config.temperature,
                        'max_tokens': config.max_tokens
                    } for name, config in self.models.items()
                },
                'adaptive_weights': self.get_adaptive_weights(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def update_weights_from_performance(self) -> None:
        """Update model weights based on performance data."""
        try:
            adaptive_weights = self.get_adaptive_weights()
            
            # Update model weights with adaptive weights
            for model_name, adaptive_weight in adaptive_weights.items():
                if model_name in self.models:
                    # Blend adaptive weight with current weight (70% adaptive, 30% current)
                    current_weight = self.models[model_name].weight
                    blended_weight = (adaptive_weight * 0.7) + (current_weight * 0.3)
                    self.update_model_weight(model_name, blended_weight)
            
            logger.info("Updated model weights from performance data")
            
        except Exception as e:
            logger.error(f"Error updating weights from performance: {e}")
    
    def log_model_prediction(self, model_name: str, symbol: str, 
                           predicted_probability: float, actual_outcome: str = "PENDING") -> None:
        """Log a model prediction for performance tracking."""
        try:
            from ai.adaptive_weights import add_model_prediction
            
            # Map our model names to ensemble model names
            model_mapping = {
                "qwen2.5": "technical_analyst",
                "llama3.1": "sentiment_analyst", 
                "gemma2": "fundamental_analyst",
                "phi3": "risk_analyst",
                "mistral": "market_regime_analyst"
            }
            
            ensemble_name = model_mapping.get(model_name, model_name)
            
            add_model_prediction(
                model_name=ensemble_name,
                symbol=symbol,
                predicted_probability=predicted_probability,
                actual_outcome=actual_outcome,
                prediction_date=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error logging model prediction: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        try:
            from ai.adaptive_weights import get_performance_summary
            
            return get_performance_summary()
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def validate_models(self) -> bool:
        """Validate model configurations."""
        try:
            # Check if all required models are configured
            required_roles = [
                ModelRole.CHIEF_TECHNICAL_ANALYST,
                ModelRole.RISK_MANAGEMENT_SPECIALIST,
                ModelRole.MARKET_SENTIMENT_ANALYST,
                ModelRole.FUNDAMENTAL_ANALYSIS_EXPERT,
                ModelRole.QUANTITATIVE_STRATEGY_DEVELOPER
            ]
            
            configured_roles = [config.role for config in self.models.values()]
            
            for role in required_roles:
                if role not in configured_roles:
                    logger.error(f"Missing model for role: {role}")
                    return False
            
            # Check if weights sum to 1.0
            total_weight = sum(config.weight for config in self.models.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Model weights sum to {total_weight}, not 1.0")
            
            logger.info("Model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False

# Global multi-model manager instance
_multi_model_manager: Optional[MultiModelManager] = None

def get_multi_model_manager(mode: str = None) -> MultiModelManager:
    """Get the global multi-model manager instance."""
    global _multi_model_manager
    if _multi_model_manager is None:
        if mode is None:
            mode = get_current_mode()
        _multi_model_manager = MultiModelManager(mode)
    return _multi_model_manager

async def check_model_availability(mode: str = None) -> Dict[str, bool]:
    """Check availability of all models."""
    return await get_multi_model_manager(mode).check_model_availability()

async def get_model_opinion(model_name: str, analysis_prompt: str, 
                          market_data: Dict[str, Any], mode: str = None) -> Optional[ModelOpinion]:
    """Get opinion from a specific model."""
    return await get_multi_model_manager(mode).get_model_opinion(model_name, analysis_prompt, market_data)

async def get_all_model_opinions(analysis_prompt: str, market_data: Dict[str, Any], 
                               mode: str = None) -> List[ModelOpinion]:
    """Get opinions from all available models."""
    return await get_multi_model_manager(mode).get_all_model_opinions(analysis_prompt, market_data)

def get_model_config(model_name: str, mode: str = None) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    return get_multi_model_manager(mode).get_model_config(model_name)

def get_all_model_configs(mode: str = None) -> Dict[str, ModelConfig]:
    """Get all model configurations."""
    return get_multi_model_manager(mode).get_all_model_configs()

def validate_models(mode: str = None) -> bool:
    """Validate model configurations."""
    return get_multi_model_manager(mode).validate_models()

def get_adaptive_weights(mode: str = None) -> Dict[str, float]:
    """Get adaptive weights from the adaptive weight manager."""
    return get_multi_model_manager(mode).get_adaptive_weights()

def update_weights_from_performance(mode: str = None) -> None:
    """Update model weights based on performance data."""
    return get_multi_model_manager(mode).update_weights_from_performance()

def log_model_prediction(model_name: str, symbol: str, predicted_probability: float, 
                        actual_outcome: str = "PENDING", mode: str = None) -> None:
    """Log a model prediction for performance tracking."""
    return get_multi_model_manager(mode).log_model_prediction(
        model_name, symbol, predicted_probability, actual_outcome
    )

def get_performance_summary(mode: str = None) -> Dict[str, Any]:
    """Get performance summary for all models."""
    return get_multi_model_manager(mode).get_performance_summary()
