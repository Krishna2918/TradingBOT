"""Test imports to find hang location"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("1. Basic imports...")
import sys
import time
import numpy as np
import pandas as pd
print("   OK")

print("2. PyTorch imports...")
import torch
print(f"   OK - device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

print("3. Adding path...")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print("   OK")

print("4. Importing environment...")
from src.ai.rl.enhanced_trading_environment import EnhancedTradingEnvironment
print("   OK")

print("\nAll imports successful!")
