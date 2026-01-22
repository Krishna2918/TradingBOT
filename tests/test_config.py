"""
Tests for configuration management.
"""

import os
import pytest
from pathlib import Path
from src.adaptive_data_collection.config import CollectionConfig


def test_config_from_env():
    """Test configuration creation from environment variables."""
    # Set test environment variables
    os.environ["AV_API_KEY"] = "test_key"
    os.environ["AV_RPM"] = "100"
    
    config = CollectionConfig.from_env()
    
    assert config.alpha_vantage_api_key == "test_key"
    assert config.alpha_vantage_rpm == 100
    assert config.years_to_collect == 25  # default
    
    # Clean up
    del os.environ["AV_API_KEY"]
    del os.environ["AV_RPM"]


def test_config_validation():
    """Test configuration validation."""
    # Valid config
    config = CollectionConfig(
        alpha_vantage_api_key="test_key",
        alpha_vantage_rpm=74
    )
    
    # Should not raise for valid config (if symbols file exists)
    try:
        config.validate()
    except FileNotFoundError:
        # Expected if symbols file doesn't exist
        pass
    
    # Invalid config - empty API key
    config.alpha_vantage_api_key = ""
    with pytest.raises(ValueError, match="Alpha Vantage API key is required"):
        config.validate()


def test_config_to_dict():
    """Test configuration serialization."""
    config = CollectionConfig(
        alpha_vantage_api_key="secret_key",
        alpha_vantage_rpm=74
    )
    
    config_dict = config.to_dict()
    
    # API key should be redacted
    assert config_dict["alpha_vantage_api_key"] == "***REDACTED***"
    assert config_dict["alpha_vantage_rpm"] == 74


if __name__ == "__main__":
    pytest.main([__file__])