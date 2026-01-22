"""
Cleanup old model checkpoints to free disk space
Keeps: best_model.pth and the 5 most recent epoch checkpoints
"""

import os
import re
from pathlib import Path

def cleanup_checkpoints(checkpoint_dir, keep_recent=5):
    """
    Clean up old checkpoint files, keeping only:
    - best_model.pth (always keep)
    - The N most recent epoch checkpoints

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_recent: Number of recent checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return

    # Find all epoch checkpoint files
    epoch_pattern = re.compile(r'epoch_(\d+)\.pth')
    epoch_checkpoints = []

    for file in checkpoint_dir.glob('*.pth'):
        if file.name == 'best_model.pth':
            print(f"Keeping: {file.name}")
            continue

        match = epoch_pattern.match(file.name)
        if match:
            epoch_num = int(match.group(1))
            epoch_checkpoints.append((epoch_num, file))

    # Sort by epoch number (descending)
    epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)

    print(f"\nFound {len(epoch_checkpoints)} epoch checkpoints")

    # Keep the most recent ones
    to_keep = epoch_checkpoints[:keep_recent]
    to_delete = epoch_checkpoints[keep_recent:]

    print(f"Keeping {len(to_keep)} most recent:")
    for epoch, file in to_keep:
        print(f"  - {file.name} (epoch {epoch})")

    print(f"\nDeleting {len(to_delete)} old checkpoints:")
    total_size = 0
    for epoch, file in to_delete:
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {file.name} (epoch {epoch}, {size_mb:.2f} MB)")
        file.unlink()

    print(f"\nFreed {total_size:.2f} MB of disk space")

if __name__ == '__main__':
    # Clean up main checkpoint directories
    base_dir = Path(__file__).parent
    dirs_to_clean = [
        base_dir / 'models/lstm_10h_checkpoint',
        base_dir / 'models/aggressive_lstm_daily_checkpoints',
        base_dir / 'models/transformer_checkpoints',
        base_dir / 'checkpoints/transformer'
    ]

    for dir_path in dirs_to_clean:
        print("\n" + "="*70)
        print(f"Cleaning: {dir_path}")
        print("="*70)
        cleanup_checkpoints(dir_path, keep_recent=5)

    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
