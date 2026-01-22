"""
Unit tests for OllamaClient
============================

Tests for the refactored OllamaClient that inherits from BaseAPIClient.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.ai.ollama_client import (
    OllamaClient,
    OllamaResponse,
    OllamaTradingAI,
)
from src.utils.api_client import APIResponse


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_api_response_success():
    """Create a successful API response."""
    return APIResponse(
        success=True,
        data={"models": [{"name": "qwen2.5:14b"}, {"name": "llama2:7b"}]},
        status_code=200,
        headers={},
        elapsed_ms=100,
    )


@pytest.fixture
def mock_api_response_failure():
    """Create a failed API response."""
    return APIResponse(
        success=False,
        error="Connection refused",
        status_code=None,
        headers={},
        elapsed_ms=0,
    )


@pytest.fixture
def mock_generate_response():
    """Create a mock generate response."""
    return APIResponse(
        success=True,
        data={
            "model": "qwen2.5:14b",
            "response": "Hello! How can I help you today?",
            "done": True,
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500000000,
            "eval_count": 20,
            "eval_duration": 2000000000,
            "context": [1, 2, 3, 4, 5],
        },
        status_code=200,
        headers={},
        elapsed_ms=5000,
    )


@pytest.fixture
def mock_chat_response():
    """Create a mock chat response."""
    return APIResponse(
        success=True,
        data={
            "model": "qwen2.5:14b",
            "message": {"role": "assistant", "content": "I can help with that!"},
            "done": True,
            "total_duration": 3000000000,
        },
        status_code=200,
        headers={},
        elapsed_ms=3000,
    )


@pytest.fixture
def ollama_client():
    """Create an OllamaClient instance for testing."""
    return OllamaClient(base_url="http://localhost:11434")


# =============================================================================
# OllamaClient Tests
# =============================================================================

class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_init_default_url(self):
        """Test client initializes with default URL."""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.name == "Ollama"

    def test_init_custom_url(self):
        """Test client initializes with custom URL."""
        client = OllamaClient(base_url="http://192.168.1.100:11434")
        assert client.base_url == "http://192.168.1.100:11434"

    def test_init_config_settings(self, ollama_client):
        """Test client has correct config settings."""
        assert ollama_client.config.timeout == 60.0
        assert ollama_client.config.max_retries == 2
        assert ollama_client.config.circuit_breaker_threshold == 3


class TestOllamaClientHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_success(self, ollama_client, mock_api_response_success):
        """Test health check returns True when server is available."""
        with patch.object(ollama_client, '_get', return_value=mock_api_response_success):
            assert ollama_client.health_check() is True

    def test_health_check_failure(self, ollama_client, mock_api_response_failure):
        """Test health check returns False when server is unavailable."""
        with patch.object(ollama_client, '_get', return_value=mock_api_response_failure):
            assert ollama_client.health_check() is False

    def test_is_available_alias(self, ollama_client, mock_api_response_success):
        """Test is_available is an alias for health_check."""
        with patch.object(ollama_client, '_get', return_value=mock_api_response_success):
            assert ollama_client.is_available() is True


class TestOllamaClientListModels:
    """Tests for list_models functionality."""

    def test_list_models_success(self, ollama_client, mock_api_response_success):
        """Test listing models when server responds."""
        with patch.object(ollama_client, '_get', return_value=mock_api_response_success):
            models = ollama_client.list_models()
            assert len(models) == 2
            assert models[0]["name"] == "qwen2.5:14b"
            assert models[1]["name"] == "llama2:7b"

    def test_list_models_failure(self, ollama_client, mock_api_response_failure):
        """Test listing models returns empty list on failure."""
        with patch.object(ollama_client, '_get', return_value=mock_api_response_failure):
            models = ollama_client.list_models()
            assert models == []

    def test_list_models_empty(self, ollama_client):
        """Test listing models when no models available."""
        empty_response = APIResponse(
            success=True,
            data={"models": []},
            status_code=200,
        )
        with patch.object(ollama_client, '_get', return_value=empty_response):
            models = ollama_client.list_models()
            assert models == []


class TestOllamaClientGenerate:
    """Tests for generate functionality."""

    def test_generate_success(self, ollama_client, mock_generate_response):
        """Test successful text generation."""
        with patch.object(ollama_client, '_post', return_value=mock_generate_response):
            response = ollama_client.generate(
                model="qwen2.5:14b",
                prompt="Hello, world!",
            )

            assert isinstance(response, OllamaResponse)
            assert response.model == "qwen2.5:14b"
            assert response.response == "Hello! How can I help you today?"
            assert response.done is True
            assert response.context == [1, 2, 3, 4, 5]
            assert response.total_duration == 5000000000

    def test_generate_with_system_prompt(self, ollama_client, mock_generate_response):
        """Test generation with system prompt."""
        with patch.object(ollama_client, '_post', return_value=mock_generate_response) as mock_post:
            ollama_client.generate(
                model="qwen2.5:14b",
                prompt="Hello",
                system="You are a helpful assistant.",
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get('json', call_args[1].get('json'))
            assert payload["system"] == "You are a helpful assistant."

    def test_generate_with_options(self, ollama_client, mock_generate_response):
        """Test generation with custom options."""
        with patch.object(ollama_client, '_post', return_value=mock_generate_response) as mock_post:
            ollama_client.generate(
                model="qwen2.5:14b",
                prompt="Hello",
                options={"temperature": 0.7, "top_p": 0.9},
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get('json', call_args[1].get('json'))
            assert payload["options"]["temperature"] == 0.7
            assert payload["options"]["top_p"] == 0.9

    def test_generate_failure(self, ollama_client, mock_api_response_failure):
        """Test generation returns empty response on failure."""
        with patch.object(ollama_client, '_post', return_value=mock_api_response_failure):
            response = ollama_client.generate(
                model="qwen2.5:14b",
                prompt="Hello",
            )

            assert isinstance(response, OllamaResponse)
            assert response.response == ""
            assert response.done is True


class TestOllamaClientChat:
    """Tests for chat functionality."""

    def test_chat_success(self, ollama_client, mock_chat_response):
        """Test successful chat interaction."""
        messages = [
            {"role": "user", "content": "Hello!"},
        ]

        with patch.object(ollama_client, '_post', return_value=mock_chat_response):
            response = ollama_client.chat(
                model="qwen2.5:14b",
                messages=messages,
            )

            assert isinstance(response, OllamaResponse)
            assert response.response == "I can help with that!"
            assert response.done is True

    def test_chat_with_conversation(self, ollama_client, mock_chat_response):
        """Test chat with multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        with patch.object(ollama_client, '_post', return_value=mock_chat_response) as mock_post:
            ollama_client.chat(
                model="qwen2.5:14b",
                messages=messages,
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get('json', call_args[1].get('json'))
            assert len(payload["messages"]) == 3

    def test_chat_failure(self, ollama_client, mock_api_response_failure):
        """Test chat returns empty response on failure."""
        with patch.object(ollama_client, '_post', return_value=mock_api_response_failure):
            response = ollama_client.chat(
                model="qwen2.5:14b",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert isinstance(response, OllamaResponse)
            assert response.response == ""


class TestOllamaClientGetModelInfo:
    """Tests for get_model_info functionality."""

    def test_get_model_info_success(self, ollama_client):
        """Test getting model info."""
        model_info = {
            "modelfile": "FROM qwen2.5:14b",
            "parameters": "temperature 0.7",
            "template": "{{ .Prompt }}",
        }
        response = APIResponse(success=True, data=model_info, status_code=200)

        with patch.object(ollama_client, '_post', return_value=response):
            info = ollama_client.get_model_info("qwen2.5:14b")
            assert info == model_info

    def test_get_model_info_not_found(self, ollama_client, mock_api_response_failure):
        """Test getting model info for non-existent model."""
        with patch.object(ollama_client, '_post', return_value=mock_api_response_failure):
            info = ollama_client.get_model_info("nonexistent:model")
            assert info is None


# =============================================================================
# OllamaTradingAI Tests
# =============================================================================

class TestOllamaTradingAI:
    """Tests for OllamaTradingAI class."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock OllamaClient."""
        client = Mock(spec=OllamaClient)
        client.is_available.return_value = True
        client.list_models.return_value = [
            {"name": "qwen2.5:14b-instruct"},
            {"name": "llama2:7b"},
        ]
        return client

    @pytest.fixture
    def trading_ai(self, mock_ollama_client):
        """Create OllamaTradingAI with mocked client."""
        return OllamaTradingAI(client=mock_ollama_client)

    def test_init_with_client(self, trading_ai, mock_ollama_client):
        """Test initialization with provided client."""
        assert trading_ai.client == mock_ollama_client

    def test_get_best_model(self, trading_ai):
        """Test getting best available model."""
        # Should return qwen2.5:14b-instruct as it's in the preferred list
        model = trading_ai.get_best_model()
        assert model == "qwen2.5:14b-instruct"

    def test_get_best_model_fallback(self):
        """Test fallback when no preferred models available."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.list_models.return_value = [{"name": "custom:model"}]

        ai = OllamaTradingAI(client=mock_client)
        model = ai.get_best_model()
        assert model == "custom:model"

    def test_analyze_market_data_server_unavailable(self, mock_ollama_client):
        """Test fallback analysis when server unavailable."""
        mock_ollama_client.is_available.return_value = False
        ai = OllamaTradingAI(client=mock_ollama_client)

        result = ai.analyze_market_data(
            symbol="AAPL",
            market_data={"current_price": 150.0},
        )

        assert result["action"] == "HOLD"
        assert result["model_used"] == "fallback"
        assert result["confidence"] == 0.3

    def test_analyze_market_data_success(self, trading_ai, mock_ollama_client):
        """Test successful market analysis."""
        # Mock the generate response
        mock_response = OllamaResponse(
            model="qwen2.5:14b",
            response='{"action": "BUY", "confidence": 0.85, "reasoning": ["Strong momentum"], "target_price": 160.0, "stop_loss": 145.0, "position_size_pct": 0.05}',
            done=True,
        )
        mock_ollama_client.generate.return_value = mock_response

        result = trading_ai.analyze_market_data(
            symbol="AAPL",
            market_data={"current_price": 150.0, "volume": 1000000},
        )

        assert result["action"] == "BUY"
        assert result["confidence"] == 0.85
        assert result["target_price"] == 160.0
        assert result["model_used"] == "ollama"

    def test_fallback_analysis(self, trading_ai):
        """Test fallback analysis method."""
        result = trading_ai._fallback_analysis("AAPL", {})

        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.3
        assert result["model_used"] == "fallback"
        assert "Ollama server not available" in result["reasoning"][0]


# =============================================================================
# Integration Tests
# =============================================================================

class TestOllamaClientIntegration:
    """Integration tests for OllamaClient (require actual Ollama server)."""

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama server")
    def test_real_health_check(self):
        """Test real health check against Ollama server."""
        client = OllamaClient()
        # This would actually connect to Ollama
        result = client.health_check()
        assert isinstance(result, bool)

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama server")
    def test_real_list_models(self):
        """Test real model listing."""
        client = OllamaClient()
        models = client.list_models()
        assert isinstance(models, list)
