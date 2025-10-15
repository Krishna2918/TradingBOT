"""
AI Ensemble Module - Multi-Model AI Analysis

This module provides ensemble analysis using multiple Ollama models for stock selection.
Uses strict JSON schema validation and majority voting for final decisions.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import ollama
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of AI models."""
    STRATEGY = "strategy"
    MATHEMATICAL = "mathematical"
    DATA_PROCESSING = "data_processing"

@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    name: str
    model_type: ModelType
    description: str
    weight: float = 1.0
    enabled: bool = True

@dataclass
class AIResponse:
    """AI model response structure."""
    model_name: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    target_price: float
    stop_loss: float
    position_size_pct: float
    timestamp: datetime
    raw_response: str
    validation_errors: List[str] = None

@dataclass
class EnsembleResult:
    """Ensemble analysis result."""
    symbol: str
    final_action: str
    final_confidence: float
    model_responses: List[AIResponse]
    consensus_score: float
    reasoning: List[str]
    target_price: float
    stop_loss: float
    position_size_pct: float
    timestamp: datetime

class OllamaEnsemble:
    """Ensemble of Ollama models for stock analysis."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize Ollama ensemble."""
        self.host = host
        self.client = ollama.Client(host=self.host)
        
        # Model configurations - using only working models
        self.models = [
            ModelConfig(
                name="qwen2.5:14b-instruct",
                model_type=ModelType.STRATEGY,
                description="Strategy and execution decisions",
                weight=1.0
            ),
            ModelConfig(
                name="qwen3-coder:480b-cloud",
                model_type=ModelType.MATHEMATICAL,
                description="Mathematical analysis and risk calculations",
                weight=1.0
            )
        ]
        
        # JSON schema for validation
        self.response_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL", "HOLD"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reasoning": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                "target_price": {
                    "type": ["number", "null"],
                    "minimum": 0.0
                },
                "stop_loss": {
                    "type": ["number", "null"],
                    "minimum": 0.0
                },
                "position_size_pct": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["action", "confidence", "reasoning"],
            "additionalProperties": False
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _validate_response(self, response: str, model_name: str) -> Tuple[bool, List[str]]:
        """Validate AI response against JSON schema."""
        try:
            # Parse JSON
            response_data = json.loads(response)
            
            # Validate against schema
            validate(instance=response_data, schema=self.response_schema)
            
            return True, []
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON from {model_name}: {e}"
            logger.error(error_msg)
            return False, [error_msg]
            
        except ValidationError as e:
            error_msg = f"Schema validation error from {model_name}: {e.message}"
            logger.error(error_msg)
            return False, [error_msg]
            
        except Exception as e:
            error_msg = f"Unexpected validation error from {model_name}: {e}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def _create_prompt(self, symbol: str, market_data: Dict, sentiment_data: Dict, 
                      fundamental_data: Dict, model_type: ModelType) -> str:
        """Create model-specific prompt."""
        
        base_prompt = f"""
You are an expert financial analyst AI. Analyze the following market data for {symbol} and provide a trading decision.

Market Data:
{json.dumps(market_data, indent=2)}

Sentiment Data:
{json.dumps(sentiment_data, indent=2) if sentiment_data else "No sentiment data available"}

Fundamental Data:
{json.dumps(fundamental_data, indent=2) if fundamental_data else "No fundamental data available"}

Provide your response in JSON format with the following keys:
- "action": "BUY" | "SELL" | "HOLD"
- "confidence": float (0.0 to 1.0)
- "reasoning": list of strings (at least 3 reasons)
- "target_price": float (expected price target)
- "stop_loss": float (stop loss price)
- "position_size_pct": float (0.0 to 1.0, percentage of capital to allocate)
"""
        
        if model_type == ModelType.STRATEGY:
            return base_prompt + """

Focus on:
- Trading strategy and execution timing
- Market conditions and trends
- Risk-reward analysis
- Position sizing strategy
"""
        
        elif model_type == ModelType.MATHEMATICAL:
            return base_prompt + """

Focus on:
- Mathematical analysis of technical indicators
- Statistical probability calculations
- Risk metrics and volatility analysis
- Quantitative position sizing
"""
        
        elif model_type == ModelType.DATA_PROCESSING:
            return base_prompt + """

Focus on:
- Data quality and reliability
- Sentiment analysis and news impact
- Fundamental analysis and valuation
- Market sentiment and psychology
"""
        
        return base_prompt
    
    def analyze_stock(self, symbol: str, market_data: Dict, sentiment_data: Dict, 
                     fundamental_data: Dict) -> List[AIResponse]:
        """Analyze stock using all enabled models."""
        responses = []
        
        for model_config in self.models:
            if not model_config.enabled:
                continue
            
            try:
                self._rate_limit()
                
                # Create model-specific prompt with JSON serializable data
                serializable_market_data = self._make_json_serializable(market_data)
                serializable_sentiment_data = self._make_json_serializable(sentiment_data)
                serializable_fundamental_data = self._make_json_serializable(fundamental_data)
                
                prompt = self._create_prompt(
                    symbol, serializable_market_data, serializable_sentiment_data, 
                    serializable_fundamental_data, model_config.model_type
                )
                
                # Generate response
                response = self.client.generate(
                    model=model_config.name,
                    prompt=prompt,
                    format='json'
                )
                
                response_text = response['response']
                
                # Validate response
                is_valid, validation_errors = self._validate_response(response_text, model_config.name)
                
                if is_valid:
                    # Parse validated response
                    response_data = json.loads(response_text)
                    
                    ai_response = AIResponse(
                        model_name=model_config.name,
                        symbol=symbol,
                        action=response_data['action'],
                        confidence=response_data['confidence'],
                        reasoning=response_data['reasoning'],
                        target_price=response_data.get('target_price', None),
                        stop_loss=response_data.get('stop_loss', None),
                        position_size_pct=response_data.get('position_size_pct', None),
                        timestamp=datetime.now(),
                        raw_response=response_text
                    )
                    
                    responses.append(ai_response)
                    logger.info(f"[PASS] {model_config.name} analysis for {symbol}: {ai_response.action} (confidence: {ai_response.confidence:.2f})")
                
                else:
                    # Create fallback response for invalid data
                    fallback_response = AIResponse(
                        model_name=model_config.name,
                        symbol=symbol,
                        action="HOLD",
                        confidence=0.0,
                        reasoning=[f"Model response validation failed: {', '.join(validation_errors)}"],
                        target_price=market_data.get('current_price', 0.0),
                        stop_loss=market_data.get('current_price', 0.0) * 0.98,
                        position_size_pct=0.0,
                        timestamp=datetime.now(),
                        raw_response=response_text,
                        validation_errors=validation_errors
                    )
                    
                    responses.append(fallback_response)
                    logger.warning(f"[FAIL] {model_config.name} validation failed for {symbol}: {validation_errors}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol} with {model_config.name}: {e}")
                
                # Create error response
                error_response = AIResponse(
                    model_name=model_config.name,
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.0,
                    reasoning=[f"Model analysis failed: {str(e)}"],
                    target_price=market_data.get('current_price', 0.0),
                    stop_loss=market_data.get('current_price', 0.0) * 0.98,
                    position_size_pct=0.0,
                    timestamp=datetime.now(),
                    raw_response="",
                    validation_errors=[str(e)]
                )
                
                responses.append(error_response)
        
        return responses
    
    def create_ensemble_result(self, symbol: str, responses: List[AIResponse]) -> EnsembleResult:
        """Create ensemble result from individual model responses."""
        if not responses:
            return EnsembleResult(
                symbol=symbol,
                final_action="HOLD",
                final_confidence=0.0,
                model_responses=[],
                consensus_score=0.0,
                reasoning=["No model responses available"],
                target_price=0.0,
                stop_loss=0.0,
                position_size_pct=0.0,
                timestamp=datetime.now()
            )
        
        # Count actions
        action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_confidence = 0.0
        valid_responses = 0
        
        for response in responses:
            if response.validation_errors is None:  # Valid response
                action_counts[response.action] += 1
                total_confidence += response.confidence
                valid_responses += 1
        
        # Determine final action
        if valid_responses == 0:
            final_action = "HOLD"
            final_confidence = 0.0
        else:
            # Majority vote
            final_action = max(action_counts, key=action_counts.get)
            final_confidence = total_confidence / valid_responses
        
        # Calculate consensus score
        consensus_score = action_counts[final_action] / len(responses) if responses else 0.0
        
        # Aggregate reasoning
        all_reasoning = []
        for response in responses:
            all_reasoning.extend(response.reasoning)
        
        # Remove duplicates while preserving order
        unique_reasoning = []
        seen = set()
        for reason in all_reasoning:
            if reason not in seen:
                unique_reasoning.append(reason)
                seen.add(reason)
        
        # Calculate weighted averages for prices and position size
        if valid_responses > 0:
            total_target_price = 0.0
            total_stop_loss = 0.0
            total_position_size = 0.0
            
            for response in responses:
                if response.validation_errors is None:
                    total_target_price += response.target_price
                    total_stop_loss += response.stop_loss
                    total_position_size += response.position_size_pct
            
            avg_target_price = total_target_price / valid_responses
            avg_stop_loss = total_stop_loss / valid_responses
            avg_position_size = total_position_size / valid_responses
        else:
            avg_target_price = 0.0
            avg_stop_loss = 0.0
            avg_position_size = 0.0
        
        return EnsembleResult(
            symbol=symbol,
            final_action=final_action,
            final_confidence=final_confidence,
            model_responses=responses,
            consensus_score=consensus_score,
            reasoning=unique_reasoning,
            target_price=avg_target_price,
            stop_loss=avg_stop_loss,
            position_size_pct=avg_position_size,
            timestamp=datetime.now()
        )
    
    def analyze_multiple_stocks(self, stock_data: Dict[str, Dict]) -> List[EnsembleResult]:
        """Analyze multiple stocks using ensemble."""
        results = []
        
        for symbol, data in stock_data.items():
            market_data = data.get('market_data', {})
            sentiment_data = data.get('sentiment_data', {})
            fundamental_data = data.get('fundamental_data', {})
            
            # Analyze with all models
            responses = self.analyze_stock(symbol, market_data, sentiment_data, fundamental_data)
            
            # Create ensemble result
            result = self.create_ensemble_result(symbol, responses)
            results.append(result)
            
            logger.info(f"Ensemble analysis for {symbol}: {result.final_action} (confidence: {result.final_confidence:.2f}, consensus: {result.consensus_score:.2f})")
        
        # Sort by confidence and consensus
        results.sort(key=lambda x: (x.final_confidence * x.consensus_score), reverse=True)
        
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {}
        
        for model_config in self.models:
            try:
                # Test model availability
                test_response = self.client.generate(
                    model=model_config.name,
                    prompt="Test",
                    format='json'
                )
                
                status[model_config.name] = {
                    'enabled': model_config.enabled,
                    'type': model_config.model_type.value,
                    'weight': model_config.weight,
                    'status': 'available',
                    'description': model_config.description
                }
                
            except Exception as e:
                status[model_config.name] = {
                    'enabled': model_config.enabled,
                    'type': model_config.model_type.value,
                    'weight': model_config.weight,
                    'status': 'error',
                    'error': str(e),
                    'description': model_config.description
                }
        
        return status
    
    def _make_json_serializable(self, data: Dict) -> Dict:
        """Convert numpy types to Python native types for JSON serialization."""
        if not isinstance(data, dict):
            return data
        
        serializable_data = {}
        for key, value in data.items():
            if hasattr(value, 'item'):  # numpy scalar
                serializable_data[key] = value.item()
            elif isinstance(value, dict):
                serializable_data[key] = self._make_json_serializable(value)
            elif isinstance(value, list):
                serializable_data[key] = [
                    item.item() if hasattr(item, 'item') else item 
                    for item in value
                ]
            else:
                serializable_data[key] = value
        
        return serializable_data

# Global ensemble instance
ollama_ensemble = OllamaEnsemble()

def analyze_stock_ensemble(symbol: str, market_data: Dict, sentiment_data: Dict, 
                          fundamental_data: Dict) -> EnsembleResult:
    """Analyze a single stock using ensemble."""
    responses = ollama_ensemble.analyze_stock(symbol, market_data, sentiment_data, fundamental_data)
    return ollama_ensemble.create_ensemble_result(symbol, responses)

def analyze_stocks_ensemble(stock_data: Dict[str, Dict]) -> List[EnsembleResult]:
    """Analyze multiple stocks using ensemble."""
    return ollama_ensemble.analyze_multiple_stocks(stock_data)

def get_ensemble_status() -> Dict[str, Any]:
    """Get ensemble model status."""
    return ollama_ensemble.get_model_status()
