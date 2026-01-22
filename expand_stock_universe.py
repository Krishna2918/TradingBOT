#!/usr/bin/env python3
"""
Expand stock universe to 1,400+ stocks by adding more comprehensive lists.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_existing_stocks():
    """Load existing stock list."""
    stock_file = Path("data/production/full_stock_list.txt")
    if stock_file.exists():
        with open(stock_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def generate_common_tickers():
    """Generate common ticker patterns."""
    tickers = []
    
    # Single letter + numbers (A, AA, AAA, etc.)
    for letter in string.ascii_uppercase:
        tickers.extend([
            letter,
            letter * 2,
            letter * 3,
            f"{letter}A",
            f"{letter}B",
            f"{letter}C",
            f"A{letter}",
            f"B{letter}",
            f"C{letter}"
        ])
    
    # Common patterns
    common_patterns = [
        # Major companies by sector
        'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM', 'MU', 'AMAT', 'LRCX', 'KLAC',
        'MRVL', 'SWKS', 'MCHP', 'ADI', 'TXN', 'XLNX', 'NXPI', 'ON', 'MPWR',
        
        # Biotech
        'MRNA', 'BNTX', 'NVAX', 'GILD', 'REGN', 'VRTX', 'BIIB', 'AMGN', 'CELG',
        'ILMN', 'INCY', 'BMRN', 'SGEN', 'ALXN', 'TECH', 'RARE', 'BLUE', 'FOLD',
        
        # Finance
        'V', 'MA', 'PYPL', 'SQ', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC',
        
        # Consumer
        'AMZN', 'SHOP', 'ETSY', 'EBAY', 'BABA', 'JD', 'PDD', 'MELI', 'SE', 'GRAB',
        
        # Energy & Materials
        'XOM', 'CVX', 'COP', 'EOG', 'DVN', 'FANG', 'MRO', 'OXY', 'SLB', 'HAL',
        
        # REITs
        'O', 'REIT', 'VNO', 'BXP', 'KIM', 'REG', 'FRT', 'TCO', 'UE', 'WPC',
        
        # ETFs (some trade like stocks)
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'LQD'
    ]
    
    tickers.extend(common_patterns)
    
    # Add numbered variations
    for base in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        for num in range(1, 10):
            tickers.append(f"{base}{num}")
            tickers.append(f"{base}A{num}")
            tickers.append(f"{base}B{num}")
    
    return list(set(tickers))  # Remove duplicates

def validate_tickers(tickers, batch_size=50):
    """Validate tickers by checking if they exist and have data."""
    valid_tickers = []
    
    logger.info(f"Validating {len(tickers)} potential tickers...")
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
        
        for ticker in batch:
            try:
                # Quick validation - just check if ticker exists
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Check if it's a valid stock (has basic info)
                if info and 'symbol' in info and info.get('regularMarketPrice'):
                    valid_tickers.append(ticker)
                    if len(valid_tickers) % 50 == 0:
                        logger.info(f"Found {len(valid_tickers)} valid tickers so far...")
                        
            except Exception:
                continue  # Skip invalid tickers
                
        # Don't overwhelm the API
        if i < len(tickers) - batch_size:
            import time
            time.sleep(1)
    
    return valid_tickers

def create_expanded_stock_list():
    """Create expanded stock list with 1,000+ stocks."""
    logger.info("Creating expanded stock universe...")
    
    # Start with existing stocks
    existing_stocks = get_existing_stocks()
    logger.info(f"Starting with {len(existing_stocks)} existing stocks")
    
    # Generate potential tickers
    potential_tickers = generate_common_tickers()
    
    # Remove already existing ones
    new_candidates = [t for t in potential_tickers if t not in existing_stocks]
    logger.info(f"Generated {len(new_candidates)} new candidate tickers")
    
    # Validate new tickers (this will take a while)
    logger.info("This may take 10-15 minutes to validate all tickers...")
    valid_new_tickers = validate_tickers(new_candidates[:500])  # Limit to 500 for time
    
    # Combine all stocks
    all_stocks = existing_stocks.union(set(valid_new_tickers))
    final_list = sorted(list(all_stocks))
    
    logger.info(f"Final stock universe: {len(final_list)} stocks")
    
    # Save expanded list
    output_file = Path("data/production/expanded_stock_list.txt")
    with open(output_file, 'w') as f:
        for symbol in final_list:
            f.write(f"{symbol}\n")
    
    logger.info(f"Expanded stock list saved to: {output_file}")
    return final_list

if __name__ == "__main__":
    stocks = create_expanded_stock_list()
    print(f"\n🎯 Expanded stock universe to {len(stocks)} symbols")
    print(f"📁 Saved to: data/production/expanded_stock_list.txt")
    print(f"🚀 Ready for large-scale data collection!")