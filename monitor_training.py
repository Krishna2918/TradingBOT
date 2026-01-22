"""
TRAINING MONITOR DASHBOARD
==========================

Real-time monitoring of all AI model training processes:
- GPU utilization and memory
- Training progress per model
- Loss and accuracy metrics
- Estimated time remaining
- System resource usage
- Live log streaming

Usage:
    # Monitor all running training
    python monitor_training.py

    # Monitor specific model
    python monitor_training.py --model lstm

    # Refresh every 5 seconds
    python monitor_training.py --interval 5

Author: Trading Bot Team
Date: October 29, 2025
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None


class TrainingMonitor:
    """Real-time monitoring of training processes"""

    def __init__(self, refresh_interval: int = 2):
        self.refresh_interval = refresh_interval
        self.start_time = datetime.now()

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_gpu_info(self) -> Optional[Dict]:
        """Get GPU status"""
        if not GPUtil:
            return None

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu = gpus[0]
            return {
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'memory_free': gpu.memoryFree,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'utilization': gpu.load * 100,
                'temperature': gpu.temperature
            }
        except:
            return None

    def get_system_info(self) -> Dict:
        """Get system resource usage"""
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        return {
            'cpu_percent': cpu,
            'ram_total': ram.total / (1024**3),
            'ram_used': ram.used / (1024**3),
            'ram_percent': ram.percent,
            'disk_free': disk.free / (1024**3)
        }

    def find_training_processes(self) -> List[Dict]:
        """Find running Python training processes"""
        training_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline = proc.info['cmdline']
                if not cmdline:
                    continue

                # Check if it's a training script
                is_training = any(
                    'train' in str(cmd).lower() and '.py' in str(cmd).lower()
                    for cmd in cmdline
                )

                if is_training:
                    # Extract model type
                    model_type = "Unknown"
                    for cmd in cmdline:
                        if 'lstm' in str(cmd).lower():
                            model_type = "LSTM"
                        elif 'transformer' in str(cmd).lower():
                            model_type = "Transformer"
                        elif 'ppo' in str(cmd).lower():
                            model_type = "PPO"
                        elif 'dqn' in str(cmd).lower():
                            model_type = "DQN"

                    training_processes.append({
                        'pid': proc.info['pid'],
                        'model': model_type,
                        'command': ' '.join(cmdline[-3:]),  # Last 3 args
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / (1024**2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return training_processes

    def get_latest_logs(self, model: str = None) -> List[str]:
        """Get latest training log lines"""
        log_dir = Path('logs')
        if not log_dir.exists():
            return []

        # Find most recent log file
        log_files = list(log_dir.glob('*.log'))
        if model:
            log_files = [f for f in log_files if model.lower() in f.name.lower()]

        if not log_files:
            return []

        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                return lines[-10:]  # Last 10 lines
        except:
            return []

    def parse_training_status(self) -> Dict:
        """Parse training status from logs and checkpoints"""
        status = {
            'lstm': {'running': False, 'epoch': 0, 'loss': 0, 'accuracy': 0},
            'transformer': {'running': False, 'epoch': 0, 'loss': 0, 'accuracy': 0},
            'ppo': {'running': False, 'episode': 0, 'reward': 0},
            'dqn': {'running': False, 'episode': 0, 'reward': 0}
        }

        # Check for recent model checkpoints
        models_dir = Path('models')
        if models_dir.exists():
            for model_key in status.keys():
                checkpoint_files = list(models_dir.glob(f'{model_key}_*.pth'))
                if checkpoint_files:
                    latest = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
                    # Check if modified in last 5 minutes (likely training)
                    if time.time() - latest.stat().st_mtime < 300:
                        status[model_key]['running'] = True

                        # Try to extract epoch/episode from filename
                        try:
                            if 'epoch' in latest.name:
                                epoch_str = latest.name.split('epoch_')[1].split('.')[0]
                                status[model_key]['epoch'] = int(epoch_str)
                            elif 'episode' in latest.name:
                                ep_str = latest.name.split('episode_')[1].split('.')[0]
                                status[model_key]['episode'] = int(ep_str)
                        except:
                            pass

        return status

    def print_dashboard(self):
        """Print monitoring dashboard"""
        self.clear_screen()

        print("=" * 100)
        print(" " * 35 + "AI MODEL TRAINING MONITOR")
        print("=" * 100)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Monitoring started: {self.start_time.strftime('%H:%M:%S')}")
        print(f"Uptime: {str(datetime.now() - self.start_time).split('.')[0]}")
        print("=" * 100)

        # GPU Info
        gpu_info = self.get_gpu_info()
        if gpu_info:
            print(f"\n[GPU STATUS]")
            print(f"  Name: {gpu_info['name']}")
            print(f"  Memory: {gpu_info['memory_used']:.0f} MB / {gpu_info['memory_total']:.0f} MB ({gpu_info['memory_percent']:.1f}%)")
            print(f"  Utilization: {gpu_info['utilization']:.1f}%")
            print(f"  Temperature: {gpu_info['temperature']:.0f}°C")

            # Visual bar
            bar_length = 50
            used_length = int(bar_length * gpu_info['memory_percent'] / 100)
            bar = "[" + "#" * used_length + "-" * (bar_length - used_length) + "]"
            print(f"  VRAM: {bar}")
        else:
            print("\n[GPU STATUS] - Not available (nvidia-smi not found)")

        # System Info
        sys_info = self.get_system_info()
        print(f"\n[SYSTEM STATUS]")
        print(f"  CPU: {sys_info['cpu_percent']:.1f}%")
        print(f"  RAM: {sys_info['ram_used']:.1f} GB / {sys_info['ram_total']:.1f} GB ({sys_info['ram_percent']:.1f}%)")
        print(f"  Disk Free: {sys_info['disk_free']:.1f} GB")

        # Training Processes
        processes = self.find_training_processes()
        print(f"\n[ACTIVE TRAINING PROCESSES]")
        if processes:
            print(f"  {'PID':<8} {'Model':<15} {'CPU%':<8} {'Memory':<12} {'Command':<40}")
            print(f"  {'-'*8} {'-'*15} {'-'*8} {'-'*12} {'-'*40}")
            for proc in processes:
                print(f"  {proc['pid']:<8} {proc['model']:<15} {proc['cpu_percent']:<7.1f}% "
                      f"{proc['memory_mb']:<11.0f}MB {proc['command'][:40]:<40}")
        else:
            print("  No active training processes detected")

        # Training Status
        training_status = self.parse_training_status()
        print(f"\n[MODEL TRAINING STATUS]")
        print(f"  {'Model':<15} {'Status':<10} {'Progress':<15} {'Latest Checkpoint':<30}")
        print(f"  {'-'*15} {'-'*10} {'-'*15} {'-'*30}")

        for model_key, status in training_status.items():
            status_str = "[RUNNING]" if status['running'] else "[IDLE]"
            model_name = model_key.upper()

            if model_key in ['lstm', 'transformer']:
                progress = f"Epoch {status['epoch']}" if status['epoch'] > 0 else "-"
            else:
                progress = f"Episode {status['episode']}" if status['episode'] > 0 else "-"

            # Find latest checkpoint
            models_dir = Path('models')
            checkpoint = "-"
            if models_dir.exists():
                checkpoints = list(models_dir.glob(f'{model_key}_*.pth'))
                if checkpoints:
                    latest = max(checkpoints, key=lambda f: f.stat().st_mtime)
                    checkpoint = latest.name[:30]

            print(f"  {model_name:<15} {status_str:<10} {progress:<15} {checkpoint:<30}")

        # Latest Logs
        print(f"\n[LATEST LOG OUTPUT]")
        logs = self.get_latest_logs()
        if logs:
            for line in logs[-5:]:  # Last 5 lines
                print(f"  {line.rstrip()}")
        else:
            print("  No recent logs found")

        print("\n" + "=" * 100)
        print(f"Refreshing every {self.refresh_interval} seconds... (Press Ctrl+C to stop)")
        print("=" * 100)

    def run(self, duration: Optional[int] = None):
        """Run monitoring dashboard"""
        try:
            iterations = 0
            while True:
                self.print_dashboard()
                time.sleep(self.refresh_interval)
                iterations += 1

                if duration and iterations * self.refresh_interval >= duration:
                    break

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            return


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Monitor AI model training in real-time'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=2,
        help='Refresh interval in seconds (default: 2)'
    )

    parser.add_argument(
        '--model',
        choices=['lstm', 'transformer', 'ppo', 'dqn'],
        help='Monitor specific model only'
    )

    parser.add_argument(
        '--duration',
        type=int,
        help='Monitor for N seconds then exit'
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(refresh_interval=args.interval)
    monitor.run(duration=args.duration)


if __name__ == "__main__":
    main()
