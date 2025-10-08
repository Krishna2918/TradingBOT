"""
Local Reasoner - Qwen2.5-14B-Instruct Integration
Offline LLM for routine explanations and feature proposals
Optimized for RTX 4080 16GB VRAM
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import subprocess
import os
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class ReasoningRequest:
    """Request for local reasoning"""
    request_type: str  # 'explain_conflict', 'propose_features', 'analyze_escalation'
    data: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: str

@dataclass
class ReasoningResponse:
    """Response from local reasoning"""
    reasoning: str
    confidence: float
    recommendations: List[str]
    feature_proposals: List[str]
    timestamp: str

class LocalReasoner:
    """
    Local Reasoner using Qwen2.5-14B-Instruct
    
    Optimized for:
    - RTX 4080 16GB VRAM
    - 4-bit quantization (GGUF)
    - Fast inference for intraday loops
    - Strong tool-use and reasoning capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Model configuration
        self.model_name = config.get('model_name', 'qwen2.5:14b-instruct')
        self.model_path = config.get('model_path', '~/ollama/models')
        self.quantization = config.get('quantization', 'q4_0')  # 4-bit quantization
        self.context_length = config.get('context_length', 8192)
        
        # Performance settings
        self.gpu_layers = config.get('gpu_layers', 35)  # Use most layers on GPU
        self.batch_size = config.get('batch_size', 512)
        self.threads = config.get('threads', 8)
        
        # Ollama configuration
        self.ollama_host = config.get('ollama_host', 'http://localhost:11434')
        self.ollama_timeout = config.get('ollama_timeout', 30)
        
        # Reasoning templates
        self.reasoning_templates = self._load_reasoning_templates()
        
        # Model status
        self.model_loaded = False
        self.last_inference_time = 0.0
        self.inference_count = 0
        
        logger.info(" Local Reasoner initialized")
        logger.info(f" Model: {self.model_name}")
        logger.info(f" Quantization: {self.quantization}")
        logger.info(f" GPU Layers: {self.gpu_layers}")
    
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load reasoning templates for different request types"""
        return {
            'explain_conflict': """
            Analyze the conflicting signals for {symbol}:
            
            Short-term (LSTM): {short_term_signal}
            Mid-term (GRU): {mid_term_signal}
            RL Agent: {rl_signal}
            
            Market Context:
            - Volatility: {volatility}
            - News Sentiment: {news_sentiment}
            - Liquidity: {liquidity}
            
            Provide:
            1. Explanation of the conflict
            2. Most likely scenario
            3. Recommended action
            4. Risk factors to monitor
            
            Response format: JSON with keys: explanation, scenario, action, risks
            """,
            
            'propose_features': """
            Analyze current feature performance and propose improvements:
            
            Current Features:
            {current_features}
            
            Performance Metrics:
            {performance_metrics}
            
            Market Conditions:
            {market_conditions}
            
            Propose:
            1. Features to disable (overfitting)
            2. New features to add
            3. Feature weight adjustments
            4. Market regime adaptations
            
            Response format: JSON with keys: disable_features, add_features, weight_adjustments, regime_adaptations
            """,
            
            'analyze_escalation': """
            Analyze escalation situation and provide recommendations:
            
            Escalation Type: {escalation_type}
            Portfolio State: {portfolio_state}
            Market Conditions: {market_conditions}
            
            Provide:
            1. Root cause analysis
            2. Immediate actions
            3. System adjustments
            4. Monitoring recommendations
            
            Response format: JSON with keys: root_cause, immediate_actions, system_adjustments, monitoring
            """
        }
    
    def initialize_model(self) -> bool:
        """Initialize the local model"""
        try:
            logger.info(" Initializing local model...")
            
            # Check if Ollama is running
            if not self._check_ollama_status():
                logger.error(" Ollama not running. Please start Ollama first.")
                return False
            
            # Pull model if not available
            if not self._check_model_availability():
                logger.info(" Pulling model from Ollama...")
                if not self._pull_model():
                    logger.error(" Failed to pull model")
                    return False
            
            # Test model with simple request
            test_response = self._test_model()
            if test_response:
                self.model_loaded = True
                logger.info(" Local model initialized successfully")
                return True
            else:
                logger.error(" Model test failed")
                return False
                
        except Exception as e:
            logger.error(f" Failed to initialize model: {e}")
            return False
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            import requests
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_model_availability(self) -> bool:
        """Check if model is available locally"""
        try:
            import requests
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'].startswith(self.model_name) for model in models)
            return False
        except:
            return False
    
    def _pull_model(self) -> bool:
        """Pull model from Ollama"""
        try:
            import requests
            data = {
                "name": self.model_name,
                "stream": False
            }
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json=data,
                timeout=300  # 5 minutes timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f" Failed to pull model: {e}")
            return False
    
    def _test_model(self) -> Optional[str]:
        """Test model with simple request"""
        try:
            test_request = ReasoningRequest(
                request_type='explain_conflict',
                data={
                    'symbol': 'TEST.TO',
                    'short_term_signal': 'BUY (0.7)',
                    'mid_term_signal': 'SELL (0.3)',
                    'rl_signal': 'HOLD (0.5)'
                },
                context={
                    'volatility': 'normal',
                    'news_sentiment': 'neutral',
                    'liquidity': 'high'
                },
                timestamp=datetime.now().isoformat()
            )
            
            response = self._call_model(test_request)
            return response.reasoning if response else None
            
        except Exception as e:
            logger.error(f" Model test failed: {e}")
            return None
    
    def reason(self, request: ReasoningRequest) -> Optional[ReasoningResponse]:
        """Main reasoning interface"""
        if not self.model_loaded:
            logger.warning(" Model not loaded, attempting to initialize...")
            if not self.initialize_model():
                return None
        
        try:
            start_time = datetime.now()
            response = self._call_model(request)
            end_time = datetime.now()
            
            # Update performance metrics
            self.last_inference_time = (end_time - start_time).total_seconds()
            self.inference_count += 1
            
            logger.info(f" Reasoning completed in {self.last_inference_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f" Reasoning failed: {e}")
            return None
    
    def _call_model(self, request: ReasoningRequest) -> Optional[ReasoningResponse]:
        """Call the local model"""
        try:
            # Get template for request type
            template = self.reasoning_templates.get(request.request_type)
            if not template:
                logger.error(f" Unknown request type: {request.request_type}")
                return None
            
            # Format prompt
            prompt = self._format_prompt(template, request)
            
            # Call Ollama API
            import requests
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1000,
                    "stop": ["</response>", "\n\n\n"]
                }
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=data,
                timeout=self.ollama_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Parse response
                return self._parse_response(response_text, request)
            else:
                logger.error(f" Model API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f" Model call failed: {e}")
            return None
    
    def _format_prompt(self, template: str, request: ReasoningRequest) -> str:
        """Format prompt with request data"""
        try:
            # Merge data and context
            format_data = {**request.data, **request.context}
            
            # Format template
            prompt = template.format(**format_data)
            
            # Add system instructions
            system_instructions = """
            You are an expert AI trading analyst. Provide clear, actionable insights.
            Always respond in valid JSON format as requested.
            Be concise but thorough in your analysis.
            """
            
            full_prompt = f"{system_instructions}\n\n{prompt}\n\n<response>"
            return full_prompt
            
        except Exception as e:
            logger.error(f" Prompt formatting failed: {e}")
            return template
    
    def _parse_response(self, response_text: str, request: ReasoningRequest) -> ReasoningResponse:
        """Parse model response"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                # Extract components based on request type
                if request.request_type == 'explain_conflict':
                    reasoning = parsed_data.get('explanation', response_text)
                    recommendations = [parsed_data.get('action', 'HOLD')]
                    feature_proposals = []
                    
                elif request.request_type == 'propose_features':
                    reasoning = "Feature analysis completed"
                    recommendations = parsed_data.get('disable_features', [])
                    feature_proposals = parsed_data.get('add_features', [])
                    
                elif request.request_type == 'analyze_escalation':
                    reasoning = parsed_data.get('root_cause', response_text)
                    recommendations = parsed_data.get('immediate_actions', [])
                    feature_proposals = parsed_data.get('system_adjustments', [])
                    
                else:
                    reasoning = response_text
                    recommendations = []
                    feature_proposals = []
                
                return ReasoningResponse(
                    reasoning=reasoning,
                    confidence=0.8,  # Default confidence
                    recommendations=recommendations if isinstance(recommendations, list) else [recommendations],
                    feature_proposals=feature_proposals if isinstance(feature_proposals, list) else [feature_proposals],
                    timestamp=datetime.now().isoformat()
                )
            else:
                # Fallback parsing
                return ReasoningResponse(
                    reasoning=response_text,
                    confidence=0.5,
                    recommendations=[],
                    feature_proposals=[],
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f" Response parsing failed: {e}")
            return ReasoningResponse(
                reasoning=response_text,
                confidence=0.3,
                recommendations=[],
                feature_proposals=[],
                timestamp=datetime.now().isoformat()
            )
    
    def explain_conflict(self, symbol: str, signals: Dict[str, Any], context: Dict[str, Any]) -> Optional[ReasoningResponse]:
        """Explain conflicting signals"""
        request = ReasoningRequest(
            request_type='explain_conflict',
            data={
                'symbol': symbol,
                'short_term_signal': signals.get('short_term', 'N/A'),
                'mid_term_signal': signals.get('mid_term', 'N/A'),
                'rl_signal': signals.get('rl', 'N/A')
            },
            context=context,
            timestamp=datetime.now().isoformat()
        )
        
        return self.reason(request)
    
    def propose_features(self, current_features: List[str], performance: Dict[str, float], market_conditions: Dict[str, Any]) -> Optional[ReasoningResponse]:
        """Propose feature improvements"""
        request = ReasoningRequest(
            request_type='propose_features',
            data={
                'current_features': ', '.join(current_features),
                'performance_metrics': json.dumps(performance),
                'market_conditions': json.dumps(market_conditions)
            },
            context={},
            timestamp=datetime.now().isoformat()
        )
        
        return self.reason(request)
    
    def analyze_escalation(self, escalation_type: str, portfolio_state: Dict[str, Any], market_conditions: Dict[str, Any]) -> Optional[ReasoningResponse]:
        """Analyze escalation situation"""
        request = ReasoningRequest(
            request_type='analyze_escalation',
            data={
                'escalation_type': escalation_type,
                'portfolio_state': json.dumps(portfolio_state),
                'market_conditions': json.dumps(market_conditions)
            },
            context={},
            timestamp=datetime.now().isoformat()
        )
        
        return self.reason(request)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'model_loaded': self.model_loaded,
            'model_name': self.model_name,
            'inference_count': self.inference_count,
            'last_inference_time': self.last_inference_time,
            'average_inference_time': self.last_inference_time if self.inference_count == 1 else 0.0
        }


# Example usage
if __name__ == "__main__":
    config = {
        'model_name': 'qwen2.5:14b-instruct',
        'ollama_host': 'http://localhost:11434',
        'gpu_layers': 35,
        'quantization': 'q4_0'
    }
    
    reasoner = LocalReasoner(config)
    
    # Initialize model
    if reasoner.initialize_model():
        print(" Local reasoner initialized")
        
        # Test conflict explanation
        signals = {
            'short_term': 'BUY (0.7)',
            'mid_term': 'SELL (0.3)',
            'rl': 'HOLD (0.5)'
        }
        
        context = {
            'volatility': 'high',
            'news_sentiment': 'negative',
            'liquidity': 'medium'
        }
        
        response = reasoner.explain_conflict('TD.TO', signals, context)
        if response:
            print(f"Reasoning: {response.reasoning}")
            print(f"Recommendations: {response.recommendations}")
            print(f"Confidence: {response.confidence}")
        
        # Get performance stats
        stats = reasoner.get_performance_stats()
        print(f"Performance: {stats}")
    else:
        print(" Failed to initialize local reasoner")
