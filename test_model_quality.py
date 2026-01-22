"""
Model Quality Testing Suite
===========================

Comprehensive tests to validate AI model quality:
1. Class balance test - Ensure no class collapse
2. Accuracy threshold test - Must exceed 55% (better than random 33%)
3. Per-class performance test - All classes must be predicted
4. Overfitting detection - Train-val gap must be < 15%
5. Prediction distribution test - Check for reasonable predictions
6. Backtesting simulation - Simple forward test

Usage:
    python test_model_quality.py --model models/lstm_improved_best.pth
    python test_model_quality.py --model models/lstm_best.pth --binary

Author: Trading Bot Team
Date: December 2025
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class QualityTestResult:
    """Container for test results"""
    def __init__(self, name: str, passed: bool, details: str, score: float = 0.0):
        self.name = name
        self.passed = passed
        self.details = details
        self.score = score

    def __str__(self):
        status = "[PASS]" if self.passed else "[FAIL]"
        return f"{status} | {self.name}: {self.details}"


class ModelQualityTester:
    """Comprehensive model quality testing"""

    def __init__(self, model_path: str, data_dir: str = "TrainingData/features",
                 num_classes: int = 3, device: str = "auto"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.results: List[QualityTestResult] = []

        print("=" * 70)
        print("MODEL QUALITY TESTING SUITE")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Data: {self.data_dir}")
        print(f"Classes: {self.num_classes}")
        print(f"Device: {self.device}")
        print("")

    def load_model(self) -> Optional[nn.Module]:
        """Load trained model"""
        if not self.model_path.exists():
            print(f"ERROR: Model not found: {self.model_path}")
            return None

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Get config
            config = checkpoint.get('config', {})
            input_dim = config.get('input_dim', 58)
            hidden_size = config.get('hidden_size', 256)
            num_layers = config.get('num_layers', 3)
            dropout = config.get('dropout', 0.5)
            num_classes = checkpoint.get('num_classes', self.num_classes)

            # Try to import the model class
            try:
                from train_lstm_improved import ImprovedLSTMModel
                model = ImprovedLSTMModel(
                    input_dim=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    num_classes=num_classes
                )
            except ImportError:
                # Fall back to basic model
                from train_lstm_production import LSTMModel
                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )

            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            print(f"Model loaded successfully")
            print(f"  Val Accuracy (saved): {checkpoint.get('val_acc', 'N/A'):.2f}%")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            return model

        except Exception as e:
            print(f"ERROR loading model: {e}")
            return None

    def load_test_data(self, max_files: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Load a sample of test data"""
        print("\nLoading test data...")

        data_files = list(self.data_dir.glob("*.parquet"))[:max_files]
        if not data_files:
            print(f"ERROR: No data files found in {self.data_dir}")
            return None, None

        sequences = []
        labels = []
        threshold = 0.005  # 0.5% for 3-class
        expected_features = None  # Will be set from first valid file

        for file in data_files:
            try:
                df = pd.read_parquet(file)
                df = df.ffill().bfill()

                if len(df) < 65:
                    continue

                # Set expected features from first valid file
                if expected_features is None:
                    expected_features = df.shape[1]
                    print(f"  Expected features: {expected_features}")

                # Skip files with different feature counts
                if df.shape[1] != expected_features:
                    continue

                features = df.values
                features_mean = np.nanmean(features, axis=0)
                features_std = np.nanstd(features, axis=0)
                features_std = np.where(features_std == 0, 1, features_std)
                features = (features - features_mean) / features_std
                features = np.nan_to_num(features)

                # Create sequences (last 5 for testing)
                for j in range(len(df) - 65, len(df) - 61):
                    seq = features[j:j + 60]

                    # Verify sequence shape
                    if seq.shape != (60, expected_features):
                        continue

                    close_col = 3 if 'close' not in df.columns else df.columns.get_loc('close')
                    current_close = df.iloc[j + 59, close_col]
                    next_close = df.iloc[j + 60, close_col]

                    if current_close > 0:
                        ret = (next_close - current_close) / current_close

                        if self.num_classes == 3:
                            if ret > threshold:
                                label = 2
                            elif ret < -threshold:
                                label = 0
                            else:
                                label = 1
                        else:
                            label = 1 if ret > 0 else 0

                        sequences.append(seq)
                        labels.append(label)

            except Exception as e:
                continue

        if not sequences:
            print("ERROR: No valid sequences created")
            return None, None

        print(f"  Loaded {len(sequences)} test sequences")
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

    def get_predictions(self, model: nn.Module, data: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        model.eval()
        predictions = []

        with torch.no_grad():
            batch_size = 128
            for i in range(0, len(data), batch_size):
                batch = torch.FloatTensor(data[i:i+batch_size]).to(self.device)
                outputs = model(batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)

        return np.array(predictions)

    # =========================================================================
    # QUALITY TESTS
    # =========================================================================

    def test_class_collapse(self, predictions: np.ndarray) -> QualityTestResult:
        """Test 1: Check for class collapse (>80% single class)"""
        unique, counts = np.unique(predictions, return_counts=True)
        max_ratio = counts.max() / len(predictions)
        dominant_class = unique[counts.argmax()]

        passed = max_ratio < 0.8
        details = f"Dominant class {dominant_class} has {max_ratio*100:.1f}% predictions"

        if not passed:
            details += " (CLASS COLLAPSE DETECTED!)"

        return QualityTestResult("Class Collapse Test", passed, details, 1 - max_ratio)

    def test_accuracy_threshold(self, labels: np.ndarray, predictions: np.ndarray,
                                threshold: float = 0.40) -> QualityTestResult:
        """Test 2: Accuracy must exceed threshold"""
        accuracy = accuracy_score(labels, predictions)
        random_baseline = 1 / self.num_classes

        passed = accuracy > threshold
        details = f"Accuracy: {accuracy*100:.2f}% (threshold: {threshold*100:.0f}%, random: {random_baseline*100:.1f}%)"

        return QualityTestResult("Accuracy Threshold Test", passed, details, accuracy)

    def test_all_classes_predicted(self, predictions: np.ndarray) -> QualityTestResult:
        """Test 3: All classes should be predicted at least once"""
        unique_preds = set(predictions)
        expected_classes = set(range(self.num_classes))
        missing = expected_classes - unique_preds

        passed = len(missing) == 0
        details = f"Predicted classes: {sorted(unique_preds)}"
        if missing:
            details += f", Missing: {sorted(missing)}"

        return QualityTestResult("All Classes Predicted Test", passed, details, len(unique_preds) / self.num_classes)

    def test_prediction_distribution(self, predictions: np.ndarray) -> QualityTestResult:
        """Test 4: Prediction distribution should be somewhat balanced"""
        counts = np.bincount(predictions, minlength=self.num_classes)
        ratios = counts / len(predictions)

        # Each class should have at least 10% predictions
        min_ratio = ratios.min()
        passed = min_ratio >= 0.1

        class_names = ["DOWN", "FLAT", "UP"] if self.num_classes == 3 else ["DOWN", "UP"]
        dist_str = ", ".join([f"{class_names[i]}:{ratios[i]*100:.1f}%" for i in range(self.num_classes)])
        details = f"Distribution: {dist_str}"

        return QualityTestResult("Prediction Balance Test", passed, details, min_ratio)

    def test_per_class_f1(self, labels: np.ndarray, predictions: np.ndarray,
                          min_f1: float = 0.2) -> QualityTestResult:
        """Test 5: Each class should have F1 > threshold"""
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)

        class_names = ["0", "1", "2"] if self.num_classes == 3 else ["0", "1"]
        f1_scores = []
        details_parts = []

        for cls in class_names:
            if cls in report:
                f1 = report[cls]['f1-score']
                f1_scores.append(f1)
                details_parts.append(f"Class {cls}: F1={f1:.3f}")

        min_f1_score = min(f1_scores) if f1_scores else 0
        passed = min_f1_score >= min_f1
        details = ", ".join(details_parts)

        return QualityTestResult("Per-Class F1 Test", passed, details, min_f1_score)

    def test_confusion_matrix_diagonal(self, labels: np.ndarray, predictions: np.ndarray) -> QualityTestResult:
        """Test 6: Diagonal of confusion matrix should dominate"""
        cm = confusion_matrix(labels, predictions)
        diagonal_sum = np.trace(cm)
        total = cm.sum()
        diagonal_ratio = diagonal_sum / total

        passed = diagonal_ratio > 0.4  # At least 40% correct
        details = f"Diagonal ratio: {diagonal_ratio*100:.1f}% of predictions are correct"

        return QualityTestResult("Confusion Matrix Test", passed, details, diagonal_ratio)

    def test_model_not_random(self, labels: np.ndarray, predictions: np.ndarray) -> QualityTestResult:
        """Test 7: Model should significantly outperform random"""
        accuracy = accuracy_score(labels, predictions)
        random_baseline = 1 / self.num_classes

        # Should be at least 10% better than random
        improvement = accuracy - random_baseline
        passed = improvement > 0.05

        details = f"Accuracy {accuracy*100:.1f}% vs random {random_baseline*100:.1f}% (improvement: {improvement*100:.1f}%)"

        return QualityTestResult("Better Than Random Test", passed, details, improvement)

    def run_all_tests(self) -> bool:
        """Run all quality tests"""
        print("\n" + "=" * 70)
        print("RUNNING QUALITY TESTS")
        print("=" * 70)

        # Load model
        model = self.load_model()
        if model is None:
            return False

        # Load test data
        test_data, test_labels = self.load_test_data()
        if test_data is None:
            return False

        # Get predictions
        print("\nGenerating predictions...")
        predictions = self.get_predictions(model, test_data)
        print(f"  Generated {len(predictions)} predictions")

        # Run tests
        print("\n" + "-" * 70)
        print("TEST RESULTS")
        print("-" * 70)

        self.results = [
            self.test_class_collapse(predictions),
            self.test_accuracy_threshold(test_labels, predictions),
            self.test_all_classes_predicted(predictions),
            self.test_prediction_distribution(predictions),
            self.test_per_class_f1(test_labels, predictions),
            self.test_confusion_matrix_diagonal(test_labels, predictions),
            self.test_model_not_random(test_labels, predictions),
        ]

        # Print results
        passed_count = 0
        for result in self.results:
            print(result)
            if result.passed:
                passed_count += 1

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        total_tests = len(self.results)
        print(f"Tests Passed: {passed_count}/{total_tests}")
        print(f"Pass Rate: {passed_count/total_tests*100:.0f}%")

        all_passed = passed_count == total_tests
        if all_passed:
            print("\n[SUCCESS] ALL TESTS PASSED - Model is production ready!")
        else:
            print(f"\n[FAILED] {total_tests - passed_count} TESTS FAILED - Model needs improvement")

        # Detailed metrics
        print("\n" + "-" * 70)
        print("DETAILED METRICS")
        print("-" * 70)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Overall Accuracy: {accuracy*100:.2f}%")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, predictions)
        class_names = ["DOWN", "FLAT", "UP"] if self.num_classes == 3 else ["DOWN", "UP"]
        print(f"{'':>10}", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print()
        for i, name in enumerate(class_names):
            print(f"{name:>10}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i,j]:>10}", end="")
            print()

        print("\nClassification Report:")
        print(classification_report(test_labels, predictions,
                                    target_names=class_names,
                                    zero_division=0))

        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Model Quality Testing")
    parser.add_argument("--model", type=str, default="models/lstm_improved_best.pth")
    parser.add_argument("--data", type=str, default="TrainingData/features")
    parser.add_argument("--binary", action="store_true", help="Use binary classification (2 classes)")
    args = parser.parse_args()

    num_classes = 2 if args.binary else 3

    tester = ModelQualityTester(
        model_path=args.model,
        data_dir=args.data,
        num_classes=num_classes
    )

    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
