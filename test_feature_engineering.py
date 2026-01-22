"""
Test Feature Engineering

Test the feature engineering process on a small subset of symbols.
"""

import sys
sys.path.append('src')

from src.ai.feature_engineering.comprehensive_feature_engineer import ComprehensiveFeatureEngineer

def main():
    """Test feature engineering on a few symbols"""
    print("Testing Feature Engineering")
    print("=" * 50)
    
    # Initialize feature engineer
    engineer = ComprehensiveFeatureEngineer()
    
    # Test on first 5 symbols
    results = engineer.process_all_symbols(max_symbols=5)
    
    # Show results
    print("\nTest Results:")
    for symbol, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {symbol}: {status}")
    
    # Generate summary
    summary = engineer.get_feature_summary()
    if not summary.empty:
        print("\nFeature Summary:")
        print(summary.to_string(index=False))
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nOverall: {successful}/{total} symbols successful ({successful/total*100:.1f}%)")

if __name__ == "__main__":
    main()