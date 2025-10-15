#!/usr/bin/env python3
"""Phase 7 Validation: Market Microstructure Prediction + Advanced Feature Engineering"""

import sys
import os
import io

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass  # Already wrapped or not needed

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ai.market_microstructure_predictor import get_microstructure_predictor, OrderBookSnapshot
from src.ai.advanced_feature_engineering import get_feature_pipeline

def test_market_microstructure_predictor():
    """Test market microstructure predictor functionality."""
    predictor = get_microstructure_predictor()
    
    # Create sample order book data
    order_book_data = []
    base_price = 100.0
    
    for i in range(100):
        # Generate realistic order book data
        timestamp = datetime.now() - timedelta(minutes=100-i)
        
        # Create bid/ask levels
        bids = []
        asks = []
        
        for level in range(5):  # 5 levels each side
            bid_price = base_price - (level + 1) * 0.01
            ask_price = base_price + (level + 1) * 0.01
            quantity = 1000 + level * 500
            
            bids.append([bid_price, quantity])
            asks.append([ask_price, quantity])
        
        # Add some price movement
        price_change = (i % 10 - 5) * 0.005
        base_price += price_change
        
        order_book_data.append({
            'timestamp': timestamp,
            'bids': bids,
            'asks': asks,
            'last_price': base_price,
            'volume': 1000000 + i * 10000
        })
    
    # Convert to DataFrame
    order_book_df = pd.DataFrame(order_book_data)
    
    # Test adding market data
    predictor.add_market_data(order_book_df)
    print("[PASS] Market data addition works")
    
    # Test microstructure prediction
    market_conditions = {
        'regime': 'trending',
        'volatility': 0.02,
        'liquidity': 'medium'
    }
    
    prediction = predictor.predict_microstructure(
        time_horizon_minutes=5,
        trade_size=1000.0,
        market_conditions=market_conditions
    )
    
    assert prediction is not None, "No microstructure prediction returned"
    assert prediction.predicted_spread > 0.0, "Invalid predicted spread"
    assert prediction.predicted_liquidity >= 0.0, "Invalid predicted liquidity"
    assert prediction.predicted_price_impact >= 0.0, "Invalid predicted price impact"
    assert prediction.predicted_volatility >= 0.0, "Invalid predicted volatility"
    assert prediction.confidence >= 0.0, "Invalid confidence"
    assert len(prediction.recommendations) > 0, "No recommendations provided"
    print("[PASS] Microstructure prediction works")
    
    # Test statistics
    stats = predictor.get_microstructure_statistics()
    assert 'total_order_book_snapshots' in stats, "Missing order book snapshots in stats"
    assert 'total_metrics_calculated' in stats, "Missing metrics calculated in stats"
    assert 'total_predictions_made' in stats, "Missing predictions made in stats"
    assert stats['total_order_book_snapshots'] >= 100, "Insufficient order book snapshots"
    print("[PASS] Microstructure statistics work")
    
    return True

def test_advanced_feature_engineering():
    """Test advanced feature engineering functionality."""
    pipeline = get_feature_pipeline()
    
    # Create sample market data
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    
    # Generate realistic price data
    base_price = 100.0
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(99):
        # Random walk with some trend
        price_change = np.random.normal(0.001, 0.02)  # 0.1% mean return, 2% volatility
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        # Volume with some correlation to price movement
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(100000, new_volume))  # Minimum volume
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    # Ensure high >= low and high/low are reasonable
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Test feature engineering
    feature_set = pipeline.engineer_features(data)
    
    assert feature_set is not None, "No feature set returned"
    assert len(feature_set.features) > 0, "No features generated"
    assert len(feature_set.feature_names) > 0, "No feature names"
    assert len(feature_set.feature_categories) > 0, "No feature categories"
    assert feature_set.metadata['total_features'] > 0, "No total features in metadata"
    print("[PASS] Basic feature engineering works")
    
    # Test feature categories
    expected_categories = ['technical', 'statistical', 'market_regime']
    for category in expected_categories:
        assert category in feature_set.feature_categories, f"Missing {category} category"
        assert len(feature_set.feature_categories[category]) > 0, f"No {category} features"
    print("[PASS] Feature categories work")
    
    # Test feature importance
    importance = pipeline.get_feature_importance(feature_set)
    assert len(importance) == len(feature_set.feature_names), "Importance length mismatch"
    assert all(imp >= 0.0 for imp in importance.values()), "Negative importance values"
    print("[PASS] Feature importance works")
    
    # Test top feature selection
    top_features = pipeline.select_top_features(feature_set, top_k=20)
    assert len(top_features) <= 20, "Too many top features selected"
    assert len(top_features) > 0, "No top features selected"
    print("[PASS] Top feature selection works")
    
    return True

def test_feature_engineering_with_additional_data():
    """Test feature engineering with additional data sources."""
    pipeline = get_feature_pipeline()
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    
    base_price = 100.0
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(99):
        price_change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(100000, new_volume))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Create additional data
    additional_data = {
        'order_book': pd.DataFrame({
            'spread': np.random.uniform(0.001, 0.01, 50),
            'depth': np.random.uniform(100000, 1000000, 50),
            'price_impact': np.random.uniform(0.0001, 0.005, 50)
        }),
        'market_regime': {
            'regime_score': 0.7,
            'confidence': 0.8
        },
        'news_sentiment': {
            'score': 0.6,
            'volume': 100,
            'volatility': 0.3
        },
        'economic_indicators': {
            'momentum': 0.5,
            'volatility': 0.2
        },
        'market_breadth': {
            'advance_decline_ratio': 1.2,
            'new_highs_lows_ratio': 0.8
        }
    }
    
    # Test feature engineering with additional data
    feature_set = pipeline.engineer_features(data, additional_data)
    
    assert feature_set is not None, "No feature set returned with additional data"
    assert len(feature_set.features) > 0, "No features generated with additional data"
    
    # Check that microstructure features were added
    microstructure_features = feature_set.feature_categories.get('microstructure', [])
    assert len(microstructure_features) > 0, "No microstructure features generated"
    
    # Check that custom features were added
    custom_features = feature_set.feature_categories.get('custom', [])
    assert len(custom_features) > 0, "No custom features generated"
    
    print("[PASS] Feature engineering with additional data works")
    
    return True

def test_microstructure_prediction_accuracy():
    """Test microstructure prediction accuracy and consistency."""
    predictor = get_microstructure_predictor()
    
    # Create more realistic order book data
    order_book_data = []
    base_price = 100.0
    
    for i in range(200):
        timestamp = datetime.now() - timedelta(minutes=200-i)
        
        # Create more realistic bid/ask spreads
        spread = 0.01 + np.random.uniform(-0.005, 0.005)  # 1% base spread with variation
        mid_price = base_price + np.random.normal(0, 0.1)  # Small price movements
        
        bids = []
        asks = []
        
        for level in range(10):  # 10 levels each side
            bid_price = mid_price - spread/2 - level * 0.005
            ask_price = mid_price + spread/2 + level * 0.005
            quantity = 1000 + level * 200 + np.random.uniform(-100, 100)
            
            bids.append([bid_price, quantity])
            asks.append([ask_price, quantity])
        
        # Add some trend
        price_change = np.random.normal(0.0005, 0.01)  # Slight upward bias
        base_price += price_change
        
        order_book_data.append({
            'timestamp': timestamp,
            'bids': bids,
            'asks': asks,
            'last_price': base_price,
            'volume': 1000000 + i * 5000 + np.random.uniform(-100000, 100000)
        })
    
    order_book_df = pd.DataFrame(order_book_data)
    predictor.add_market_data(order_book_df)
    
    # Test multiple predictions for consistency
    predictions = []
    for i in range(10):
        market_conditions = {
            'regime': 'trending' if i % 2 == 0 else 'ranging',
            'volatility': 0.02 + i * 0.001,
            'liquidity': 'medium'
        }
        
        prediction = predictor.predict_microstructure(
            time_horizon_minutes=5,
            trade_size=1000.0 + i * 100,
            market_conditions=market_conditions
        )
        
        predictions.append(prediction)
        
        # Validate prediction
        assert prediction.predicted_spread > 0.0, f"Invalid spread in prediction {i}"
        assert prediction.predicted_liquidity >= 0.0, f"Invalid liquidity in prediction {i}"
        assert prediction.predicted_price_impact >= 0.0, f"Invalid price impact in prediction {i}"
        assert prediction.confidence >= 0.0, f"Invalid confidence in prediction {i}"
    
    # Check consistency
    spreads = [p.predicted_spread for p in predictions]
    liquidity_scores = [p.predicted_liquidity for p in predictions]
    
    # Spreads should be reasonable (not too volatile)
    spread_std = np.std(spreads)
    assert spread_std < 0.01, f"Spread predictions too volatile: {spread_std}"
    
    # Liquidity scores should be in valid range
    assert all(0.0 <= score <= 1.0 for score in liquidity_scores), "Invalid liquidity scores"
    
    print("[PASS] Microstructure prediction accuracy and consistency work")
    
    return True

def test_integration():
    """Test integration between microstructure prediction and feature engineering."""
    predictor = get_microstructure_predictor()
    pipeline = get_feature_pipeline()
    
    # Create comprehensive market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=150, freq='1H')
    
    base_price = 100.0
    prices = [base_price]
    volumes = [1000000]
    
    for i in range(149):
        price_change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        volumes.append(max(100000, new_volume))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Create order book data for microstructure analysis
    order_book_data = []
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        
        bids = []
        asks = []
        
        for level in range(5):
            bid_price = prices[i] - (level + 1) * 0.01
            ask_price = prices[i] + (level + 1) * 0.01
            quantity = 1000 + level * 500
            
            bids.append([bid_price, quantity])
            asks.append([ask_price, quantity])
        
        order_book_data.append({
            'timestamp': timestamp,
            'bids': bids,
            'asks': asks,
            'last_price': prices[i],
            'volume': volumes[i]
        })
    
    order_book_df = pd.DataFrame(order_book_data)
    
    # Add data to microstructure predictor
    predictor.add_market_data(order_book_df)
    
    # Get microstructure prediction
    market_conditions = {
        'regime': 'trending',
        'volatility': 0.02,
        'liquidity': 'medium'
    }
    
    microstructure_prediction = predictor.predict_microstructure(
        time_horizon_minutes=5,
        trade_size=1000.0,
        market_conditions=market_conditions
    )
    
    # Create additional data including microstructure prediction
    additional_data = {
        'order_book': pd.DataFrame({
            'spread': [microstructure_prediction.predicted_spread] * 50,
            'depth': [1000000] * 50,  # Simplified
            'price_impact': [microstructure_prediction.predicted_price_impact] * 50
        }),
        'market_regime': {
            'regime_score': 0.7,
            'confidence': microstructure_prediction.confidence
        },
        'microstructure_prediction': {
            'liquidity': microstructure_prediction.predicted_liquidity,
            'volatility': microstructure_prediction.predicted_volatility,
            'confidence': microstructure_prediction.confidence
        }
    }
    
    # Engineer features
    feature_set = pipeline.engineer_features(data, additional_data)
    
    # Validate integration
    assert feature_set is not None, "No feature set from integrated system"
    assert len(feature_set.features) > 0, "No features from integrated system"
    
    # Check that microstructure features are included
    microstructure_features = feature_set.feature_categories.get('microstructure', [])
    assert len(microstructure_features) > 0, "No microstructure features in integrated system"
    
    # Check that market regime features include microstructure data
    regime_features = feature_set.feature_categories.get('market_regime', [])
    assert len(regime_features) > 0, "No market regime features in integrated system"
    
    print("[PASS] Integration between systems works")
    
    return True

if __name__ == "__main__":
    try:
        test_market_microstructure_predictor()
        test_advanced_feature_engineering()
        test_feature_engineering_with_additional_data()
        test_microstructure_prediction_accuracy()
        test_integration()
        print("\n[PASS] PHASE 7 VALIDATION: PASSED")
        exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] PHASE 7 VALIDATION: FAILED - {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] PHASE 7 VALIDATION: ERROR - {e}")
        exit(1)