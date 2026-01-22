"""
Create comprehensive stock lists for massive data collection
Downloads S&P 500, S&P 400, and other indices from Wikipedia
Total target: 1,400 new stocks
"""

import pandas as pd
import requests
from pathlib import Path

def download_sp500():
    """Download S&P 500 list from Wikipedia"""
    print("Downloading S&P 500 list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df['Symbol'].tolist()
        # Clean symbols (some have dots that need conversion)
        symbols = [s.replace('.', '-') for s in symbols]
        print(f"  Found {len(symbols)} S&P 500 stocks")
        return symbols
    except Exception as e:
        print(f"  Error: {e}")
        print("  Will use backup list")
        return []

def download_sp400():
    """Download S&P 400 MidCap list from Wikipedia"""
    print("Downloading S&P 400 MidCap list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

    try:
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df['Ticker symbol'].tolist() if 'Ticker symbol' in df.columns else df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols]
        print(f"  Found {len(symbols)} S&P 400 stocks")
        return symbols
    except Exception as e:
        print(f"  Error: {e}")
        print("  Will use backup list")
        return []

def download_sp600():
    """Download S&P 600 SmallCap list from Wikipedia"""
    print("Downloading S&P 600 SmallCap list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

    try:
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df['Ticker symbol'].tolist() if 'Ticker symbol' in df.columns else df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols]
        print(f"  Found {len(symbols)} S&P 600 stocks")
        return symbols[:300]  # Take top 300
    except Exception as e:
        print(f"  Error: {e}")
        return []

def get_etf_list():
    """Get comprehensive ETF list"""
    print("Creating ETF list...")

    etfs = [
        # Major Indices
        'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'VEA', 'VWO',
        # Sector ETFs
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE', 'XLC',
        # Technology
        'VGT', 'FTEC', 'IGV', 'SKYY', 'HACK', 'CIBR', 'ARKK', 'ARKQ', 'ARKW', 'ARKG',
        # International
        'EFA', 'EEM', 'FXI', 'EWJ', 'EWZ', 'EWH', 'EWG', 'EWU', 'EWT', 'EWY',
        # Bonds
        'AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'JNK', 'MBB', 'MUB',
        # Commodities
        'GLD', 'SLV', 'GDX', 'GDXJ', 'USO', 'UNG', 'DBA', 'DBC', 'PALL', 'PPLT',
        # Real Estate
        'VNQ', 'IYR', 'XLRE', 'REM', 'MORT', 'RWO', 'ICF', 'SCHH',
        # Dividend
        'VYM', 'DVY', 'SCHD', 'SDY', 'VIG', 'DGRO', 'HDV', 'NOBL',
        # Growth/Value
        'VUG', 'IWF', 'VONG', 'VTV', 'IWD', 'VONV', 'RPG', 'RPV',
        # Small/Mid Cap
        'IJH', 'MDY', 'IJR', 'VB', 'VO', 'VBK', 'VBR',
        # Emerging Markets
        'VWO', 'IEMG', 'SCHE', 'DEM', 'EMQQ', 'EEMV',
        # China/Asia
        'FXI', 'MCHI', 'KWEB', 'GXC', 'EWY', 'EWT', 'EWH', 'EWS',
        # Europe
        'VGK', 'EWG', 'EWU', 'EWI', 'EWQ', 'EWP',
        # Leveraged (for advanced strategies)
        'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'UDOW', 'SDOW', 'TNA', 'TZA',
        # Volatility
        'VXX', 'UVXY', 'SVXY',
        # Clean Energy
        'ICLN', 'TAN', 'FAN', 'QCLN', 'PBW',
        # Thematic
        'ARKF', 'FINX', 'ESPO', 'GAMR', 'HERO', 'UFO', 'ROBO', 'BOTZ',
        # Cannabis
        'MJ', 'YOLO', 'THCX',
        # Crypto-related
        'BITO', 'BITI',
    ]

    print(f"  Created {len(etfs)} ETF symbols")
    return etfs

def get_growth_stocks():
    """Get popular growth stocks"""
    print("Creating growth stock list...")

    growth = [
        # Recent IPOs & Hot Stocks (2020-2024)
        'ABNB', 'COIN', 'RIVN', 'LCID', 'DASH', 'SNOW', 'PLTR', 'RBLX',
        'U', 'DKNG', 'OPEN', 'AFRM', 'SOFI', 'HOOD', 'PATH', 'UPST',
        # Fintech
        'SQ', 'PYPL', 'SHOP', 'ROKU', 'Z', 'ZG', 'BILL', 'FOUR',
        # Cloud/SaaS
        'NET', 'DDOG', 'MDB', 'ESTC', 'ZS', 'OKTA', 'CRWD', 'S',
        # E-commerce
        'ETSY', 'W', 'CHWY', 'CVNA', 'RH',
        # Healthcare/Biotech
        'TDOC', 'PTON', 'EXAS', 'CRSP', 'BEAM', 'EDIT', 'NTLA', 'VERV',
        # EVs & Clean Energy
        'NIO', 'XPEV', 'LI', 'FSR', 'CHPT', 'BLNK', 'PLUG', 'FCEL',
        # Semiconductors
        'MPWR', 'MRVL', 'SWKS', 'QRVO', 'ON', 'WOLF',
        # Gaming
        'RBLX', 'U', 'PLTK', 'DKNG', 'PENN',
        # Other
        'ZM', 'DOCU', 'TWLO', 'SNAP', 'PINS', 'UBER', 'LYFT', 'SPOT',
    ]

    print(f"  Created {len(growth)} growth stock symbols")
    return growth

def load_existing_stocks():
    """Load existing 164 stocks to avoid duplicates"""
    print("Loading existing stocks...")
    try:
        with open('stock_symbols_list.txt', 'r') as f:
            existing = [line.strip() for line in f if line.strip()]
        print(f"  Found {len(existing)} existing stocks")
        return set(existing)
    except FileNotFoundError:
        print("  No existing list found, starting fresh")
        return set()

def main():
    """Main function to create all stock lists"""
    print("=" * 80)
    print("CREATING COMPREHENSIVE STOCK LISTS FOR MASSIVE DATA COLLECTION")
    print("=" * 80)
    print()

    # Create lists directory if it doesn't exist
    Path('lists').mkdir(exist_ok=True)

    # Load existing stocks to avoid duplicates
    existing = load_existing_stocks()
    print(f"Existing stocks: {len(existing)}")
    print()

    # Download/create all lists
    sp500 = download_sp500()
    sp400 = download_sp400()
    sp600 = download_sp600()
    etfs = get_etf_list()
    growth = get_growth_stocks()

    print()
    print("=" * 80)
    print("CREATING FILTERED LISTS (EXCLUDING EXISTING STOCKS)")
    print("=" * 80)
    print()

    # Filter out existing stocks
    sp500_new = [s for s in sp500 if s not in existing]
    sp400_new = [s for s in sp400 if s not in existing]
    sp600_new = [s for s in sp600 if s not in existing]
    etfs_new = [s for s in etfs if s not in existing]
    growth_new = [s for s in growth if s not in existing]

    print(f"S&P 500 new: {len(sp500_new)} (original: {len(sp500)})")
    print(f"S&P 400 new: {len(sp400_new)} (original: {len(sp400)})")
    print(f"S&P 600 new: {len(sp600_new)} (original: {len(sp600)})")
    print(f"ETFs new: {len(etfs_new)} (original: {len(etfs)})")
    print(f"Growth new: {len(growth_new)} (original: {len(growth)})")
    print()

    # Save individual lists
    with open('lists/sp500_remaining.txt', 'w') as f:
        f.write('\n'.join(sp500_new))

    with open('lists/sp400_midcap.txt', 'w') as f:
        f.write('\n'.join(sp400_new))

    with open('lists/sp600_smallcap.txt', 'w') as f:
        f.write('\n'.join(sp600_new))

    with open('lists/etfs_comprehensive.txt', 'w') as f:
        f.write('\n'.join(etfs_new))

    with open('lists/growth_stocks.txt', 'w') as f:
        f.write('\n'.join(growth_new))

    # Create combined list
    all_new = list(set(sp500_new + sp400_new + sp600_new + etfs_new + growth_new))
    all_new.sort()

    with open('lists/additional_1400_stocks.txt', 'w') as f:
        f.write('\n'.join(all_new))

    # Create master list (existing + new)
    master = list(existing) + all_new
    master.sort()

    with open('lists/master_all_stocks.txt', 'w') as f:
        f.write('\n'.join(master))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"New stocks to collect: {len(all_new)}")
    print(f"Total stocks (existing + new): {len(master)}")
    print()
    print("Files created:")
    print("  lists/sp500_remaining.txt")
    print("  lists/sp400_midcap.txt")
    print("  lists/sp600_smallcap.txt")
    print("  lists/etfs_comprehensive.txt")
    print("  lists/growth_stocks.txt")
    print("  lists/additional_1400_stocks.txt  <- Use this for collection")
    print("  lists/master_all_stocks.txt       <- Final master list")
    print()

    if len(all_new) < 1000:
        print(f"⚠ Warning: Only found {len(all_new)} new stocks (target: 1,400)")
        print("   This is normal if:")
        print("   1. Many stocks already in existing list")
        print("   2. Wikipedia downloads failed (check internet connection)")
        print("   3. Can manually add more stocks to lists if needed")
    else:
        print(f"✓ Excellent: Found {len(all_new)} new stocks!")

    print()
    print("Next step: Run data collection script")
    print("  python collect_massive_nyse_data.py --symbols lists/additional_1400_stocks.txt")
    print()

if __name__ == '__main__':
    main()
