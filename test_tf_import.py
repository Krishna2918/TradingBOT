"""Test what imports TensorFlow"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

# Track imports
original_import = __builtins__.__import__

def custom_import(name, *args, **kwargs):
    if 'tensorflow' in name.lower():
        print(f"[IMPORT DETECTED] {name}")
        import traceback
        traceback.print_stack()
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = custom_import

print("1. Importing torch...")
import torch
print("   OK")

print("2. Importing TensorBoard SummaryWriter...")
from torch.utils.tensorboard import SummaryWriter
print("   OK")

print("3. Creating SummaryWriter...")
writer = SummaryWriter('logs/test_tf')
print("   OK")

print("4. Adding scalar...")
writer.add_scalar('test', 1.0, 0)
print("   OK")

writer.close()
print("\nNo TensorFlow imports detected from TensorBoard!")
