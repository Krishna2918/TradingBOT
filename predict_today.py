"""
Generate Today's Stock Predictions (Tuesday)
============================================
Uses the best trained LSTM model to predict stock movements for today.

Usage:
    python predict_today.py                    # Predict all stocks
    python predict_today.py --top 50           # Top 50 predictions only
    python predict_today.py --symbols AAPL MSFT GOOGL  # Specific symbols
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))


class LSTMModel(nn.Module):
    """LSTM Model Architecture - Matches train_lstm_production.py"""

    def __init__(self, input_dim, hidden_size=256, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary: UP/DOWN
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output


def load_model(model_path: Path, device: torch.device):
    """Load the best trained model"""
    print(f"Loading model from: {model_path}")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return None

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Determine input dimension from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Get input dim from first LSTM layer
    input_dim = state_dict['lstm.weight_ih_l0'].shape[1]

    print(f"Input dimension: {input_dim}")

    # Create and load model
    model = LSTMModel(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    return model


def load_latest_data(data_dir: Path):
    """Load latest market data for prediction"""
    print(f"\nLoading latest data from: {data_dir}")

    feature_files = list(data_dir.glob("*.parquet"))

    if not feature_files:
        print(f"ERROR: No feature files found in {data_dir}")
        return None

    print(f"Found {len(feature_files)} stock files")

    # Load recent data (last 30 days for sequence)
    all_predictions = []

    for file in feature_files[:]:  # Process all files
        try:
            df = pd.read_parquet(file)

            if len(df) < 30:
                continue

            # Get symbol from filename
            symbol = file.stem

            # Get last 30 days for prediction
            recent = df.tail(30).copy()

            # Get feature columns (exclude target columns)
            feature_cols = [col for col in recent.columns
                          if not col.startswith('target_') and
                          not col.startswith('future_') and
                          col not in ['date', 'symbol']]

            if len(feature_cols) == 0:
                continue

            features = recent[feature_cols].values

            # Normalize (simple z-score)
            features_mean = np.nanmean(features, axis=0)
            features_std = np.nanstd(features, axis=0) + 1e-8
            features = (features - features_mean) / features_std

            # Handle NaN
            features = np.nan_to_num(features, 0)

            all_predictions.append({
                'symbol': symbol,
                'features': features,
                'last_close': recent['close'].iloc[-1] if 'close' in recent.columns else None
            })

        except Exception as e:
            continue

    print(f"Loaded data for {len(all_predictions)} stocks")
    return all_predictions


def make_predictions(model, data_list, device):
    """Generate predictions for all stocks"""
    print(f"\nGenerating predictions...")

    predictions = []

    with torch.no_grad():
        for item in data_list:
            try:
                # Prepare input
                features = torch.FloatTensor(item['features']).unsqueeze(0).to(device)

                # Predict
                output = model(features)
                probs = torch.softmax(output, dim=1)

                # Get prediction (0=DOWN, 1=UP)
                pred_class = torch.argmax(output, dim=1).item()
                confidence = probs[0][pred_class].item()

                direction = ['DOWN', 'UP'][pred_class]

                predictions.append({
                    'Symbol': item['symbol'],
                    'Prediction': direction,
                    'Confidence': f"{confidence*100:.1f}%",
                    'Prob_DOWN': f"{probs[0][0].item()*100:.1f}%",
                    'Prob_UP': f"{probs[0][1].item()*100:.1f}%",
                    'Last_Close': f"${item['last_close']:.2f}" if item['last_close'] else "N/A"
                })

            except Exception as e:
                continue

    print(f"Generated {len(predictions)} predictions")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate Today's Stock Predictions")
    parser.add_argument("--model", type=str, default="models/lstm_best.pth",
                       help="Model checkpoint path")
    parser.add_argument("--data", type=str, default="TrainingData/features",
                       help="Feature data directory")
    parser.add_argument("--top", type=int, default=None,
                       help="Show only top N predictions")
    parser.add_argument("--symbols", nargs='+', help="Specific symbols to predict")
    parser.add_argument("--output", type=str, default="predictions_today.csv",
                       help="Output file for predictions")

    args = parser.parse_args()

    print("="*80)
    print("TUESDAY STOCK PREDICTIONS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %A')}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    model_path = Path(args.model)
    model = load_model(model_path, device)

    if model is None:
        return 1

    # Load data
    data_dir = Path(args.data)
    data_list = load_latest_data(data_dir)

    if data_list is None:
        return 1

    # Filter by symbols if specified
    if args.symbols:
        data_list = [d for d in data_list if d['symbol'] in args.symbols]
        print(f"Filtered to {len(data_list)} specified symbols")

    # Make predictions
    predictions = make_predictions(model, data_list, device)

    # Create DataFrame
    df = pd.DataFrame(predictions)

    # Sort by confidence
    df_sorted = df.sort_values('Confidence', ascending=False)

    # Apply top N filter if specified
    if args.top:
        df_sorted = df_sorted.head(args.top)

    # Display
    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)
    print(df_sorted.to_string(index=False))
    print("="*80)

    # Save to file
    output_path = Path(args.output)
    df_sorted.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total predictions: {len(predictions)}")
    print(f"UP predictions: {len(df[df['Prediction'] == 'UP'])}")
    print(f"DOWN predictions: {len(df[df['Prediction'] == 'DOWN'])}")
    print(f"FLAT predictions: {len(df[df['Prediction'] == 'FLAT'])}")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
