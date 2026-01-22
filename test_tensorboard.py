"""Test TensorBoard initialization"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("1. Importing torch...")
import torch
print("   OK")

print("2. Importing SummaryWriter...")
from torch.utils.tensorboard import SummaryWriter
print("   OK")

print("3. Creating SummaryWriter...")
from pathlib import Path
from datetime import datetime
log_dir = Path('logs/test') / datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir)
print(f"   OK - {log_dir}")

print("4. Writing test data...")
writer.add_scalar('test/value', 1.0, 0)
print("   OK")

print("5. Closing writer...")
writer.close()
print("   OK")

print("\nTensorBoard test successful!")
