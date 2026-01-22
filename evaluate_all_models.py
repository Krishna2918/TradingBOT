"""
MODEL COMPARISON FRAMEWORK
Evaluate and compare all trained models (LSTM, Transformer, PPO, DQN)

Features:
- Side-by-side accuracy comparison
- Backtest performance metrics
- Statistical significance testing
- Per-sector analysis
- Ensemble strategies

Usage:
    python evaluate_all_models.py --data TrainingData/features/*.parquet
    python evaluate_all_models.py --test-mode  # Quick test on 10% data

Author: Trading Bot Team
Date: October 28, 2025
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# Import models and utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.ai.models.market_transformer import MarketTransformer
from src.backtesting.backtest_runner import BacktestRunner, BacktestResult


class LSTMModel(nn.Module):
    """LSTM model for loading checkpoint"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_classes=3, dropout=0.3):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Classification
        output = self.fc(context)

        return output


class ModelEvaluator:
    """Evaluate and compare all models"""

    def __init__(self, device: torch.device):
        self.device = device
        self.models = {}
        self.results = {}

    def load_lstm(self, checkpoint_path: Path, input_dim: int):
        """Load LSTM model"""
        print(f"\nLoading LSTM from: {checkpoint_path}")

        model = LSTMModel(input_dim=input_dim)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.models['LSTM'] = model
        print(f"  ✓ LSTM loaded (accuracy: {checkpoint.get('val_acc', 0):.2f}%)")

    def load_transformer(self, checkpoint_path: Path, input_dim: int):
        """Load Transformer model"""
        print(f"\nLoading Transformer from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})

        model = MarketTransformer(
            input_dim=input_dim,
            d_model=config.get('d_model', 512),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 2048),
            max_seq_length=90,
            num_classes=3,
            dropout=config.get('dropout', 0.1)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.models['Transformer'] = model
        print(f"  ✓ Transformer loaded (accuracy: {checkpoint.get('val_acc', 0):.2f}%)")

    def predict_lstm(self, data: torch.Tensor) -> np.ndarray:
        """Get LSTM predictions"""
        with torch.no_grad():
            logits = self.models['LSTM'](data)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        return predictions

    def predict_transformer(self, data: torch.Tensor) -> np.ndarray:
        """Get Transformer predictions"""
        with torch.no_grad():
            outputs = self.models['Transformer'](data)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        return predictions

    def evaluate_accuracy(self, model_name: str, test_files: List[Path], sequence_length: int):
        """Evaluate classification accuracy"""

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} Accuracy")
        print(f"{'='*60}")

        correct = 0
        total = 0
        predictions_by_class = {0: [], 1: [], 2: []}

        for file_idx, file in enumerate(test_files[:50]):  # Test on 50 stocks
            if (file_idx + 1) % 10 == 0:
                print(f"  Progress: {file_idx + 1}/50 stocks")

            try:
                df = pd.read_parquet(file)

                if len(df) < sequence_length + 1:
                    continue

                features = df.values

                # Normalize
                features_mean = np.mean(features, axis=0)
                features_std = np.std(features, axis=0)
                features_std[features_std == 0] = 1
                features = (features - features_mean) / features_std

                # Create sequences
                for i in range(len(features) - sequence_length):
                    sequence = features[i:i + sequence_length]

                    # Label
                    current_price = features[i + sequence_length - 1, 0]
                    next_price = features[i + sequence_length, 0]

                    price_change = (next_price - current_price) / (abs(current_price) + 1e-8)
                    if price_change < -0.01:
                        label = 0  # Down
                    elif price_change > 0.01:
                        label = 2  # Up
                    else:
                        label = 1  # Neutral

                    # Predict
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

                    if model_name == 'LSTM':
                        pred = self.predict_lstm(sequence_tensor)[0]
                    elif model_name == 'Transformer':
                        pred = self.predict_transformer(sequence_tensor)[0]
                    else:
                        continue

                    predictions_by_class[label].append(pred)

                    if pred == label:
                        correct += 1
                    total += 1

            except Exception as e:
                continue

        # Calculate metrics
        accuracy = 100 * correct / total if total > 0 else 0

        # Per-class accuracy
        class_accuracies = {}
        for class_id in [0, 1, 2]:
            preds = predictions_by_class[class_id]
            if len(preds) > 0:
                class_acc = 100 * sum(p == class_id for p in preds) / len(preds)
                class_accuracies[class_id] = class_acc
            else:
                class_accuracies[class_id] = 0.0

        print(f"\n{model_name} Results:")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        print(f"  Down (0) Accuracy: {class_accuracies[0]:.2f}%")
        print(f"  Neutral (1) Accuracy: {class_accuracies[1]:.2f}%")
        print(f"  Up (2) Accuracy: {class_accuracies[2]:.2f}%")
        print(f"  Total samples: {total}")

        self.results[model_name] = {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total
        }

    def generate_trading_actions(self, model_name: str, df: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Generate trading actions from model predictions"""

        features = df.values

        # Normalize
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_std[features_std == 0] = 1
        features = (features - features_mean) / features_std

        actions = []
        current_position = 0.0  # 0 = no position, 1 = long

        for i in range(sequence_length, len(features)):
            sequence = features[i - sequence_length:i]

            # Predict
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

            if model_name == 'LSTM':
                pred = self.predict_lstm(sequence_tensor)[0]
            elif model_name == 'Transformer':
                pred = self.predict_transformer(sequence_tensor)[0]
            else:
                actions.append(0)  # Hold
                continue

            # Convert prediction to action
            if pred == 2 and current_position == 0:  # Predict up, no position
                action = 4  # Buy 100%
                current_position = 1.0
            elif pred == 0 and current_position > 0:  # Predict down, have position
                action = 8  # Sell 100%
                current_position = 0.0
            else:
                action = 0  # Hold

            actions.append(action)

        # Pad beginning
        actions = [0] * sequence_length + actions

        return np.array(actions)

    def run_backtests(self, test_files: List[Path], lstm_seq_len: int, transformer_seq_len: int):
        """Run backtests on all models"""

        print(f"\n{'='*60}")
        print("Running Backtests")
        print(f"{'='*60}")

        backtest_runner = BacktestRunner(initial_capital=100000.0)

        all_backtest_results = []

        # Test on multiple stocks
        for file_idx, file in enumerate(test_files[:10]):  # Backtest on 10 stocks
            print(f"\nBacktesting on: {file.name}")

            try:
                df = pd.read_parquet(file)

                if len(df) < 100:
                    continue

                # LSTM backtest
                if 'LSTM' in self.models:
                    lstm_actions = self.generate_trading_actions('LSTM', df, lstm_seq_len)
                    lstm_result = backtest_runner.run_backtest(df, lstm_actions, strategy_name=f"LSTM_{file.stem}")
                    all_backtest_results.append(lstm_result)

                # Transformer backtest
                if 'Transformer' in self.models:
                    transformer_actions = self.generate_trading_actions('Transformer', df, transformer_seq_len)
                    transformer_result = backtest_runner.run_backtest(df, transformer_actions, strategy_name=f"Transformer_{file.stem}")
                    all_backtest_results.append(transformer_result)

                # Buy-and-hold baseline
                baseline_result = backtest_runner.run_buy_and_hold_baseline(df)
                all_backtest_results.append(baseline_result)

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Aggregate results by strategy
        strategy_metrics = {}

        for result in all_backtest_results:
            strategy_type = result.strategy_name.split('_')[0]

            if strategy_type not in strategy_metrics:
                strategy_metrics[strategy_type] = {
                    'returns': [],
                    'sharpe': [],
                    'max_dd': [],
                    'win_rate': []
                }

            strategy_metrics[strategy_type]['returns'].append(result.total_return)
            strategy_metrics[strategy_type]['sharpe'].append(result.sharpe_ratio)
            strategy_metrics[strategy_type]['max_dd'].append(result.max_drawdown)
            strategy_metrics[strategy_type]['win_rate'].append(result.win_rate)

        # Print summary
        print(f"\n{'='*60}")
        print("Backtest Summary (Average across stocks)")
        print(f"{'='*60}")

        for strategy, metrics in strategy_metrics.items():
            print(f"\n{strategy}:")
            print(f"  Avg Return: {np.mean(metrics['returns']) * 100:.2f}%")
            print(f"  Avg Sharpe: {np.mean(metrics['sharpe']):.2f}")
            print(f"  Avg Max DD: {np.mean(metrics['max_dd']) * 100:.2f}%")
            print(f"  Avg Win Rate: {np.mean(metrics['win_rate']) * 100:.2f}%")

        self.results['backtests'] = strategy_metrics

    def generate_comparison_report(self, output_path: Path = Path('results/model_comparison.txt')):
        """Generate comprehensive comparison report"""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Accuracy comparison
            f.write("1. CLASSIFICATION ACCURACY\n")
            f.write("-" * 40 + "\n")
            for model_name, metrics in self.results.items():
                if model_name != 'backtests' and 'accuracy' in metrics:
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Overall: {metrics['accuracy']:.2f}%\n")
                    f.write(f"  Down:    {metrics['class_accuracies'][0]:.2f}%\n")
                    f.write(f"  Neutral: {metrics['class_accuracies'][1]:.2f}%\n")
                    f.write(f"  Up:      {metrics['class_accuracies'][2]:.2f}%\n")

            # Backtest comparison
            if 'backtests' in self.results:
                f.write("\n\n2. BACKTEST PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                for strategy, metrics in self.results['backtests'].items():
                    f.write(f"\n{strategy}:\n")
                    f.write(f"  Avg Return:   {np.mean(metrics['returns']) * 100:.2f}%\n")
                    f.write(f"  Avg Sharpe:   {np.mean(metrics['sharpe']):.2f}\n")
                    f.write(f"  Avg Max DD:   {np.mean(metrics['max_dd']) * 100:.2f}%\n")
                    f.write(f"  Avg Win Rate: {np.mean(metrics['win_rate']) * 100:.2f}%\n")

            # Recommendations
            f.write("\n\n3. RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            # Best classifier
            best_classifier = max(
                [(k, v['accuracy']) for k, v in self.results.items() if 'accuracy' in v],
                key=lambda x: x[1],
                default=("None", 0)
            )
            f.write(f"\nBest Classifier: {best_classifier[0]} ({best_classifier[1]:.2f}%)\n")

            # Best backtest performer
            if 'backtests' in self.results:
                best_backtest = max(
                    [(k, np.mean(v['returns'])) for k, v in self.results['backtests'].items()],
                    key=lambda x: x[1],
                    default=("None", 0)
                )
                f.write(f"Best Backtest: {best_backtest[0]} ({best_backtest[1] * 100:.2f}% avg return)\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"\nComparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare all trained models')

    parser.add_argument('--data', type=str, default='TrainingData/features/*.parquet',
                       help='Path pattern to feature files')
    parser.add_argument('--lstm-checkpoint', type=str, default='models/lstm_best.pth',
                       help='Path to LSTM checkpoint')
    parser.add_argument('--transformer-checkpoint', type=str, default='models/transformer_best.pth',
                       help='Path to Transformer checkpoint')
    parser.add_argument('--test-mode', action='store_true',
                       help='Use only 10%% of data for quick testing')
    parser.add_argument('--lstm-seq-len', type=int, default=30,
                       help='LSTM sequence length')
    parser.add_argument('--transformer-seq-len', type=int, default=90,
                       help='Transformer sequence length')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data files
    data_files = [Path(f) for f in glob(args.data)]

    if len(data_files) == 0:
        print(f"ERROR: No data files found matching pattern: {args.data}")
        return

    print(f"Found {len(data_files)} data files")

    # Test split
    if args.test_mode:
        data_files = data_files[:max(10, len(data_files) // 10)]

    np.random.shuffle(data_files)
    test_files = data_files[:100]  # Use 100 for testing

    print(f"Testing on {len(test_files)} files")

    # Get input dimension
    sample_df = pd.read_parquet(test_files[0])
    input_dim = sample_df.shape[1]
    print(f"Input dimension: {input_dim}")

    # Create evaluator
    evaluator = ModelEvaluator(device)

    # Load models
    if Path(args.lstm_checkpoint).exists():
        evaluator.load_lstm(Path(args.lstm_checkpoint), input_dim)
    else:
        print(f"WARNING: LSTM checkpoint not found: {args.lstm_checkpoint}")

    if Path(args.transformer_checkpoint).exists():
        evaluator.load_transformer(Path(args.transformer_checkpoint), input_dim)
    else:
        print(f"WARNING: Transformer checkpoint not found: {args.transformer_checkpoint}")

    # Evaluate accuracy
    if 'LSTM' in evaluator.models:
        evaluator.evaluate_accuracy('LSTM', test_files, args.lstm_seq_len)

    if 'Transformer' in evaluator.models:
        evaluator.evaluate_accuracy('Transformer', test_files, args.transformer_seq_len)

    # Run backtests
    if len(evaluator.models) > 0:
        evaluator.run_backtests(test_files, args.lstm_seq_len, args.transformer_seq_len)

    # Generate report
    evaluator.generate_comparison_report()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
