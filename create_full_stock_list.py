#!/usr/bin/env python3
"""
Create comprehensive stock list for full production data collection.
This script generates a list of 1,400+ stocks from major indices.
"""

import pandas as pd
import yfinance as yf
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sp500_stocks():
    """Get S&P 500 stock list from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_df = tables[0]
        symbols = sp500_df['Symbol'].tolist()
        logger.info(f"Retrieved {len(symbols)} S&P 500 stocks")
        return symbols
    except Exception as e:
        logger.error(f"Failed to get S&P 500 list: {e}")
        return []

def get_nasdaq100_stocks():
    """Get NASDAQ 100 stocks."""
    # Major NASDAQ 100 stocks
    nasdaq100 = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
        'AVGO', 'ORCL', 'COST', 'NFLX', 'ADBE', 'PEP', 'TMUS', 'CSCO',
        'CMCSA', 'TXN', 'QCOM', 'AMGN', 'HON', 'INTU', 'AMAT', 'BKNG',
        'ADP', 'SBUX', 'GILD', 'MDLZ', 'ISRG', 'ADI', 'VRTX', 'FISV',
        'CSX', 'ATVI', 'PYPL', 'CHTR', 'MRNA', 'NXPI', 'KLAC', 'MELI',
        'ORLY', 'LRCX', 'CTAS', 'DXCM', 'SNPS', 'CDNS', 'MAR', 'MCHP',
        'FTNT', 'ASML', 'WDAY', 'MNST', 'TEAM', 'ADSK', 'AEP', 'EXC',
        'KDP', 'ROST', 'VRSK', 'NTES', 'LULU', 'XEL', 'ODFL', 'CTSH',
        'PCAR', 'PAYX', 'FAST', 'CPRT', 'SGEN', 'VRSN', 'CSGP', 'ANSS',
        'BIIB', 'DLTR', 'SIRI', 'SWKS', 'MTCH', 'TCOM', 'ALGN', 'BMRN',
        'INCY', 'DOCU', 'SPLK', 'OKTA', 'FOXA', 'FOX', 'WBA', 'ILMN',
        'JD', 'PDD', 'BIDU', 'ZM', 'CRWD', 'DDOG', 'ZS', 'SNOW'
    ]
    logger.info(f"Added {len(nasdaq100)} NASDAQ 100 stocks")
    return nasdaq100

def get_additional_stocks():
    """Get additional popular stocks from various sectors."""
    additional = [
        # Banking & Finance
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
        'AXP', 'BLK', 'SCHW', 'CB', 'MMC', 'AON', 'PGR', 'TRV', 'ALL', 'MET',
        
        # Healthcare & Pharma
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
        'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'CELG', 'MYL', 'CVS', 'WBA', 'MCK',
        
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
        'WMB', 'EPD', 'ET', 'MPLX', 'PAA', 'BKR', 'HAL', 'DVN', 'FANG', 'MRO',
        
        # Consumer Goods
        'PG', 'KO', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'DIS',
        'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB', 'CAG', 'SJM', 'HSY', 'MDLZ',
        
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
        'GD', 'DE', 'EMR', 'ETN', 'PH', 'CMI', 'ITW', 'ROK', 'DOV', 'XYL',
        
        # Technology (Additional)
        'IBM', 'CRM', 'NOW', 'SNOW', 'PLTR', 'U', 'TWLO', 'OKTA', 'ZS', 'CRWD',
        'DDOG', 'NET', 'FSLY', 'ESTC', 'MDB', 'WORK', 'ZM', 'DOCU', 'PTON', 'ROKU',
        
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED',
        'FE', 'ES', 'AWK', 'WEC', 'DTE', 'PPL', 'CMS', 'NI', 'LNT', 'EVRG',
        
        # Real Estate & REITs
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR',
        'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'ESS', 'UDR', 'CPT', 'FRT', 'REG',
        
        # Materials
        'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF',
        'ALB', 'CE', 'FMC', 'LYB', 'CF', 'MOS', 'NUE', 'STLD', 'X', 'CLF',
        
        # Communication Services
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'DISH',
        'TWTR', 'SNAP', 'PINS', 'SPOT', 'ROKU', 'FUBO', 'SIRI', 'LBRDA', 'LBRDK', 'FWONA'
    ]
    
    logger.info(f"Added {len(additional)} additional stocks")
    return additional

def create_full_stock_list():
    """Create comprehensive stock list."""
    logger.info("Creating comprehensive stock list...")
    
    all_stocks = set()  # Use set to avoid duplicates
    
    # Add S&P 500
    sp500 = get_sp500_stocks()
    all_stocks.update(sp500)
    
    # Add NASDAQ 100
    nasdaq100 = get_nasdaq100_stocks()
    all_stocks.update(nasdaq100)
    
    # Add additional stocks
    additional = get_additional_stocks()
    all_stocks.update(additional)
    
    # Convert to sorted list
    final_list = sorted(list(all_stocks))
    
    logger.info(f"Total unique stocks: {len(final_list)}")
    
    # Save to file
    output_dir = Path("data/production")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "full_stock_list.txt"
    with open(output_file, 'w') as f:
        for symbol in final_list:
            f.write(f"{symbol}\n")
    
    logger.info(f"Stock list saved to: {output_file}")
    
    # Create sample lists for testing
    sample_100 = final_list[:100]
    sample_file = output_dir / "sample_100_stocks.txt"
    with open(sample_file, 'w') as f:
        for symbol in sample_100:
            f.write(f"{symbol}\n")
    
    logger.info(f"Sample 100 stocks saved to: {sample_file}")
    
    return final_list

if __name__ == "__main__":
    stocks = create_full_stock_list()
    print(f"\n🎯 Created stock list with {len(stocks)} symbols")
    print(f"📁 Files saved to: data/production/")
    print(f"📊 Ready for full production data collection!")