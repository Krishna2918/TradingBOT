"""
Test Market Transformer Implementation

This script tests the MarketTransformer neural network to ensure it works correctly
with the multi-model training system.
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.models.market_transformer import MarketTransformer, create_market_transformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_market_transformer_creation():
    """Test MarketTransformer model creation"""
    logger.info("Testing MarketTransformer creation...")
    
    try:
        # Test basic model creation
        model = MarketTransformer(
            input_dim=20,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_seq_length=100,
            num_classes=3,
            dropout=0.1
        )
        
        param_count = model.count_parameters()
        logger.info(f"Created MarketTransformer with {param_count} parameters")
        
        # Test factory function
        config = {
            'input_dim': 15,
            'd_model': 64,
            'num_heads': 2,
            'num_layers': 1,
            'num_classes': 3
        }
        
        model_from_config = create_market_transformer(config)
        config_param_count = model_from_config.count_parameters()
        logger.info(f"Created MarketTransformer from config with {config_param_count} parameters")
        
        logger.info("✓ MarketTransformer creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MarketTransformer creation test failed: {e}")
        return False

def test_market_transformer_forward_pass():
    """Test MarketTransformer forward pass"""
    logger.info("Testing MarketTransformer forward pass...")
    
    try:
        # Create model
        model = MarketTransformer(
            input_dim=10,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_length=50,
            num_classes=3,
            dropout=0.1
        )
        
        # Create sample data
        batch_size = 4
        seq_len = 20
        input_dim = 10
        
        # Input features
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Time features (day, month, hour, regime)
        time_features = torch.randint(0, 7, (batch_size, seq_len, 4))
        time_features[:, :, 1] = torch.randint(0, 12, (batch_size, seq_len))  # Month
        time_features[:, :, 2] = torch.randint(0, 24, (batch_size, seq_len))  # Hour
        time_features[:, :, 3] = torch.randint(0, 4, (batch_size, seq_len))   # Regime
        
        # Volatility
        volatility = torch.rand(batch_size, seq_len, 1) * 0.5 + 0.1
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, time_features, volatility)
        
        # Check output shapes
        logits = output['logits']
        attention_weights = output['attention_weights']
        hidden_states = output['hidden_states']
        
        assert logits.shape == (batch_size, 3), f"Expected logits shape {(batch_size, 3)}, got {logits.shape}"
        assert len(attention_weights) == 2, f"Expected 2 attention weight tensors, got {len(attention_weights)}"
        assert hidden_states.shape == (batch_size, seq_len, 64), f"Expected hidden states shape {(batch_size, seq_len, 64)}, got {hidden_states.shape}"
        
        logger.info(f"Forward pass successful:")
        logger.info(f"  - Logits shape: {logits.shape}")
        logger.info(f"  - Attention layers: {len(attention_weights)}")
        logger.info(f"  - Hidden states shape: {hidden_states.shape}")
        
        # Test attention maps
        attention_maps = model.get_attention_maps(x, time_features, volatility)
        logger.info(f"  - Attention maps: {list(attention_maps.keys())}")
        
        logger.info("✓ MarketTransformer forward pass test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MarketTransformer forward pass test failed: {e}")
        return False

def test_market_transformer_training_step():
    """Test MarketTransformer training step"""
    logger.info("Testing MarketTransformer training step...")
    
    try:
        # Create model
        model = MarketTransformer(
            input_dim=8,
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            max_seq_length=30,
            num_classes=3,
            dropout=0.1
        )
        
        # Create sample data
        batch_size = 2
        seq_len = 15
        input_dim = 8
        
        x = torch.randn(batch_size, seq_len, input_dim)
        targets = torch.randint(0, 3, (batch_size,))
        
        # Training setup
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output['logits'], targets)
        loss.backward()
        optimizer.step()
        
        logger.info(f"Training step successful:")
        logger.info(f"  - Loss: {loss.item():.4f}")
        logger.info(f"  - Gradients computed successfully")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            pred_output = model(x)
            predictions = torch.argmax(pred_output['logits'], dim=1)
            logger.info(f"  - Predictions: {predictions.tolist()}")
            logger.info(f"  - Targets: {targets.tolist()}")
        
        logger.info("✓ MarketTransformer training step test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MarketTransformer training step test failed: {e}")
        return False

def test_integration_with_existing_config():
    """Test integration with existing configuration system"""
    logger.info("Testing integration with existing configuration...")
    
    try:
        # Import configuration system
        from ai.multi_model_config import TransformerConfig
        
        # Create transformer config
        transformer_config = TransformerConfig(
            d_model=128,
            nhead=4,
            num_layers=3,
            batch_size=16,
            sequence_length=100
        )
        
        # Validate config
        transformer_config.validate()
        
        # Create model from config
        model_config = {
            'input_dim': 20,
            'd_model': transformer_config.d_model,
            'num_heads': transformer_config.nhead,
            'num_layers': transformer_config.num_layers,
            'max_seq_length': transformer_config.sequence_length,
            'num_classes': 3
        }
        
        model = create_market_transformer(model_config)
        param_count = model.count_parameters()
        
        logger.info(f"Created model from TransformerConfig with {param_count} parameters")
        logger.info(f"Config validation successful")
        
        logger.info("✓ Integration with existing configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration with existing configuration test failed: {e}")
        return False

def main():
    """Run all MarketTransformer tests"""
    logger.info("Starting MarketTransformer Tests")
    logger.info("=" * 50)
    
    tests = [
        test_market_transformer_creation,
        test_market_transformer_forward_pass,
        test_market_transformer_training_step,
        test_integration_with_existing_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
        
        logger.info("-" * 30)
    
    logger.info("=" * 50)
    logger.info(f"MarketTransformer Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All MarketTransformer tests passed! Ready for integration.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)