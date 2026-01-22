"""
GRU-Transformer Training - 10 Hours
Mid-term predictions (5-15 minutes)
"""

import sys
import os
from pathlib import Path
import torch
import pandas as pd
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).parent / "src"))

from ai.model_stack.gru_transformer_model import GRUTransformerPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/gru_transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 40% GPU
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.4, device=0)

logger.info("=" * 80)
logger.info("GRU-TRANSFORMER TRAINING - 10 HOURS")
logger.info("=" * 80)

# Load first available stock data
features_dir = Path("TrainingData/features")
feature_files = list(features_dir.glob("*_features.parquet"))

# Load multiple files and combine
dfs = []
for file in feature_files[:10]:
    df = pd.read_parquet(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
logger.info(f"Loaded {len(combined_df)} rows for training")

# Initialize and train
predictor = GRUTransformerPredictor(
    model_path="models/gru_transformer_10h.pth",
    sequence_length=100,
    hidden_size=256,
    num_gru_layers=2,
    num_attention_heads=8,
    dropout=0.2
)

# Train for many epochs to use 10 hours
logger.info("Starting 10-hour training session...")
predictor.train(combined_df, epochs=2000, batch_size=64, learning_rate=0.001)

logger.info("=" * 80)
logger.info("GRU-TRANSFORMER TRAINING COMPLETE")
logger.info("=" * 80)
