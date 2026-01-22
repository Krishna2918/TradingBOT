"""Quick test to verify transformer training fix"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.multi_model_orchestrator import MultiModelOrchestrator
from ai.multi_model_config import MultiModelConfig, TransformerConfig
import pandas as pd
import numpy as np
import asyncio

# Quick test
config = MultiModelConfig(
    enabled_models=['transformer'], 
    parallel_training=False,
    transformer_config=TransformerConfig(
        d_model=32,
        nhead=2,
        num_layers=1,
        epochs=2,
        sequence_length=20
    )
)
orchestrator = MultiModelOrchestrator(config)

# Create minimal data
data = pd.DataFrame({
    'feature_0': np.random.randn(50),
    'feature_1': np.random.randn(50),
    'direction_1d': np.random.randint(0, 3, 50)
})

# Test training
results = asyncio.run(orchestrator.train_all_models(data))
print(f'Training completed: {list(results.keys())}')
transformer_result = results['transformer']
print(f'Transformer accuracy: {transformer_result.validation_metrics["accuracy"]:.3f}')
print(f'Training logs: {len(transformer_result.training_logs)} entries')
print("✓ Transformer training fix verified!")