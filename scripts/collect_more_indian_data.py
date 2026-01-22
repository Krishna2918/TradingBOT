"""
Extended Indian Market Data Collector

Collects additional Indian stocks beyond NIFTY 50/Next 50:
- NIFTY Midcap 150
- NIFTY Smallcap 250
- PSU Stocks
- F&O Active Stocks
- Sectoral Indices
"""

import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Already collected stocks (from first run)
ALREADY_COLLECTED = {
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'ITC', 'LT', 'HCLTECH', 'AXISBANK', 'ASIANPAINT',
    'MARUTI', 'SUNPHARMA', 'TITAN', 'BAJFINANCE', 'DMART', 'ULTRACEMCO', 'NTPC',
    'ONGC', 'WIPRO', 'POWERGRID', 'M&M', 'ADANIENT', 'ADANIPORTS', 'COALINDIA',
    'JSWSTEEL', 'TATASTEEL', 'NESTLEIND', 'TECHM', 'INDUSINDBK', 'BAJAJFINSV',
    'GRASIM', 'HINDALCO', 'CIPLA', 'BRITANNIA', 'EICHERMOT', 'DRREDDY',
    'APOLLOHOSP', 'DIVISLAB', 'BPCL', 'HEROMOTOCO', 'TATACONSUM', 'SBILIFE',
    'HDFCLIFE', 'UPL', 'SHREECEM', 'ADANIGREEN', 'AMBUJACEM', 'AUROPHARMA',
    'BAJAJ-AUTO', 'BANKBARODA', 'BERGEPAINT', 'BIOCON', 'BOSCHLTD', 'CANBK',
    'CHOLAFIN', 'COLPAL', 'CONCOR', 'DLF', 'GAIL', 'GODREJCP', 'HAVELLS',
    'HINDPETRO', 'ICICIGI', 'ICICIPRULI', 'IDEA', 'IGL', 'INDHOTEL', 'INDUSTOWER',
    'IOC', 'IRCTC', 'JINDALSTEL', 'JUBLFOOD', 'LUPIN', 'MARICO', 'MUTHOOTFIN',
    'NAUKRI', 'NMDC', 'OBEROIRLTY', 'OFSS', 'PAGEIND', 'PETRONET', 'PFC',
    'PIDILITIND', 'PIIND', 'PNB', 'RECLTD', 'SAIL', 'SRF', 'TORNTPHARM', 'TRENT',
    'VEDL', 'IRFC', 'ADANIPOWER', 'PAYTM', 'NYKAA', 'POLICYBZR', 'ZYDUSLIFE',
    'PERSISTENT', 'COFORGE', 'MPHASIS', 'LTTS', 'TATAELXSI', 'TATAPOWER',
    'TATACHEM', 'TATACOMM', 'VOLTAS', 'GODREJPROP', 'PRESTIGE', 'LODHA',
    'PHOENIXLTD', 'BRIGADE', 'INDIGO', 'SPICEJET', 'MAXHEALTH', 'FORTIS', 'METROPOLIS'
}

# NIFTY Midcap 150 (stocks not already collected)
MIDCAP_150 = [
    'AARTIIND', 'ABB', 'ACC', 'ABCAPITAL', 'ABFRL', 'AJANTPHARM', 'ALKEM',
    'APLLTD', 'ASHOKLEY', 'ASTRAL', 'ATUL', 'AUBANK', 'AFFLE', 'BAJAJELEC',
    'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BATAINDIA', 'BEL', 'BHEL',
    'BSE', 'CANFINHOME', 'CARBORUNIV', 'CASTROLIND', 'CDSL', 'CENTRALBK',
    'CESC', 'CGPOWER', 'CHAMBLFERT', 'CIEINDIA', 'CLEAN', 'COCHINSHIP',
    'CROMPTON', 'CUB', 'CUMMINSIND', 'CYIENT', 'DALBHARAT', 'DCMSHRIRAM',
    'DEEPAKNTR', 'DELHIVERY', 'DEVYANI', 'DIXON', 'ELGIEQUIP', 'EMAMILTD',
    'ENDURANCE', 'ENGINERSIN', 'EQUITASBNK', 'ESCORTS', 'EXIDEIND',
    'FEDERALBNK', 'FINEORG', 'FLUOROCHEM', 'FSL', 'GICRE', 'GLAXO',
    'GLENMARK', 'GMRINFRA', 'GNFC', 'GRANULES', 'GRAPHITE', 'GSFC',
    'GSPL', 'GUJGASLTD', 'HAL', 'HAPPSTMNDS', 'HATSUN', 'HFCL', 'HONAUT',
    'IBREALEST', 'IDFCFIRSTB', 'IEX', 'IIFL', 'INDIANB', 'INDIAMART',
    'IPCALAB', 'IRB', 'ISEC', 'ITI', 'JBCHEPHARM', 'JINDALSAW', 'JKCEMENT',
    'JKLAKSHMI', 'JSWENERGY', 'KAJARIACER', 'KANSAINER', 'KARURVYSYA',
    'KEI', 'KIMS', 'KPITTECH', 'KRBL', 'KSB', 'LALPATHLAB', 'LATENTVIEW',
    'LAURUSLABS', 'LICHSGFIN', 'LINDEINDIA', 'LTF', 'LTIM', 'LUXIND',
    'MANAPPURAM', 'MANYAVAR', 'MAPMYINDIA', 'MCX', 'MEDANTA', 'MOTILALOFS',
    'MRF', 'MRPL', 'NATCOPHARM', 'NATIONALUM', 'NAVINFLUOR', 'NBCC', 'NCC',
    'NHPC', 'NLCINDIA', 'NUVOCO', 'OIL', 'OLECTRA', 'PGHH', 'POLYCAB',
    'POONAWALLA', 'PRINCEPIPE', 'PVRINOX', 'RADICO', 'RAIN', 'RAJESHEXPO',
    'RAMCOCEM', 'RATNAMANI', 'RAYMOND', 'RELAXO', 'RITES', 'RVNL', 'SANOFI',
    'SCHAEFFLER', 'SHRIRAMFIN', 'SJVN', 'SKFINDIA', 'SOBHA', 'SONACOMS',
    'SOLARINDS', 'SONATSOFTW', 'STARHEALTH', 'SUMICHEM', 'SUNDARMFIN',
    'SUNDRMFAST', 'SUNTV', 'SUPREMEIND', 'SWANENERGY', 'SYMPHONY', 'TANLA',
    'TATATECH', 'THERMAX', 'TIMKEN', 'TITAGARH', 'TORNTPOWER', 'TRIDENT',
    'TVSMOTOR', 'UBL', 'UNIONBANK', 'UNOMINDA', 'UCOBANK', 'UJJIVANSFB',
    'VBL', 'VGUARD', 'VINATIORGA', 'WELCORP', 'WELSPUNIND', 'WHIRLPOOL',
    'YESBANK', 'ZEEL', 'ZENSARTECH'
]

# PSU Stocks (Public Sector Undertakings)
PSU_STOCKS = [
    'IOB', 'MAHABANK', 'BEML', 'BDL', 'MIDHANI', 'GRSE', 'MAZAGON',
    'RAILTEL', 'IRCON', 'HUDCO', 'RVNL', 'SJVN', 'POWERGRID', 'GAIL',
    'HINDPETRO', 'IOC', 'BPCL', 'ONGC', 'OIL', 'MRPL', 'CHENNPETRO',
    'FACT', 'NFL', 'RCF', 'GSFC', 'GNFC', 'HINDZINC', 'MOIL', 'VEDL',
    'NALCO', 'NATIONALUM', 'HINDALCO', 'NTPC', 'NHPC', 'NLCINDIA',
    'SJVN', 'JSWENERGY', 'TATAPOWER', 'CESC', 'TORNTPOWER'
]

# Smallcap 250 Popular Stocks
SMALLCAP_250 = [
    'AARTIDRUGS', 'AAVAS', 'ACCELYA', 'ADANITRANS', 'AEGISCHEM', 'AETHER',
    'AGARIND', 'AIAENG', 'AJMERA', 'AKZOINDIA', 'AMBER', 'ANANTRAJ',
    'ANGELONE', 'APARINDS', 'APTUS', 'ARVIND', 'ASAHIINDIA', 'ASHOKA',
    'AURIONPRO', 'AVANTIFEED', 'AXISCADES', 'BAJAJCON', 'BALAMINES',
    'BASF', 'BBTC', 'BCG', 'BEPL', 'BLUESTARCO', 'BLS', 'BOMBAYBURMAH',
    'BORNOIL', 'BORORENEW', 'CAMPUS', 'CAPACITE', 'CAPLIPOINT', 'CCL',
    'CENTURYPLY', 'CENTURYTEX', 'CERA', 'CHALET', 'CHEMCON', 'CHEVIOT',
    'CMSINFO', 'CRAFTSMAN', 'CREDITACC', 'CSBBANK', 'CYIENTDLM', 'DATAMATICS',
    'DBCORP', 'DCAL', 'DELTACORP', 'DHANI', 'DODLA', 'DREAMFOLKS',
    'EASEMYTRIP', 'ECLERX', 'EDELWEISS', 'EIDPARRY', 'ELECON', 'EMUDHRA',
    'EPIGRAL', 'ERIS', 'ESABINDIA', 'ETHOSLTD', 'FIEMIND', 'FINOPB',
    'FIVESTAR', 'FUSION', 'GALAXYSURF', 'GATEWAY', 'GENUSPOWER', 'GEPIL',
    'GHCL', 'GLAND', 'GLOBUSSPR', 'GLS', 'GMM', 'GOCOLORS', 'GODFRYPHLP',
    'GOLDIAM', 'GPIL', 'GRINDWELL', 'GRINFRA', 'GSHIP', 'GUFICBIO',
    'HBLPOWER', 'HCC', 'HCG', 'HGINFRA', 'HIKAL', 'HIMATSEIDE', 'HLE',
    'HOMEFIRST', 'HSCL', 'HUDCO', 'IBREALEST', 'ICIL', 'IFBIND', 'IIFLSEC',
    'IMFA', 'INDIAGLYCO', 'INDIGOPNTS', 'INTELLECT', 'IONEXCHANGE', 'JAMNAAUTO',
    'JAYNECOIND', 'JBMA', 'JKTYRE', 'JMFINANCIL', 'JPASSOCIAT', 'JPPOWER',
    'JSLHISAR', 'JUBLINGREA', 'JUSTDIAL', 'KALYANKJIL', 'KARDA', 'KAYNES',
    'KFINTECH', 'KIRLOSBROS', 'KIRLOSENG', 'KITEX', 'KNRCON', 'KOKUYOCMLN',
    'KOPRAN', 'KPRMILL', 'KSCL', 'KTK', 'KUNSTSTOF', 'LAOPALA', 'LAXMIMACH',
    'LEMONTREE', 'LGBBROSLTD', 'LIKHITHA', 'LUMAXTECH', 'MAHABANK',
    'MAHINDCIE', 'MAHLIFE', 'MAHLOG', 'MAHSEAMLES', 'MAITHANALL', 'MANALIPETC',
    'MANGCHEFER', 'MARATHON', 'MASFIN', 'MASTEK', 'MATRIMONY', 'MAXINDIA',
    'MAYURUNIQ', 'MAZDA', 'MHRIL', 'MIRZAINT', 'MMTC', 'MOLDTECH',
    'MONTECARLO', 'MOTHERSON', 'MSTCLTD', 'MUKANDLTD', 'MUKTAARTS', 'MUNJALAU',
    'NAVNETEDUL', 'NAZARA', 'NELCAST', 'NELCO', 'NETWEB', 'NEWGEN',
    'NIACL', 'NIITLTD', 'NRBBEARING', 'NUCLEUS', 'ONMOBILE', 'ORIENTELEC',
    'ORIENTPPR', 'PAISALO', 'PARADEEP', 'PARAS', 'PATELENG', 'PATINTL',
    'PCBL', 'PDSL', 'PENIND', 'PFIZER', 'PNCINFRA', 'POLYMED', 'PPLPHARMA',
    'PRAXIS', 'PRICOLLTD', 'PRIVISCL', 'PTCIL', 'PUNJABCHEM', 'QUESS',
    'RAGHAVPRO', 'RAIAGRO', 'RAJRATAN', 'RALLIS', 'RANEENGINE', 'RANEHOLDIN',
    'RATEGAIN', 'RBA', 'RCF', 'REDINGTON', 'RESPONIND', 'RHIM', 'RIIL',
    'RKFORGE', 'ROLEXRINGS', 'ROSSARI', 'RPGLIFE', 'RPSGVENT', 'RPOWER',
    'RSTL', 'RTNINDIA', 'RTNPOWER', 'SAGCEM', 'SAKAR', 'SAMMAANCAP',
    'SANDHAR', 'SANGAMIND', 'SANGHIIND', 'SANGHVIMOV', 'SAPPHIRE', 'SATIN',
    'SBCL', 'SCHNEIDER', 'SENCO', 'SEQUENT', 'SHAKTIPUMP', 'SHANTIGEAR',
    'SHARDACROP', 'SHAREINDIA', 'SHILPAMED', 'SHOPERSTOP', 'SHYAMMETL',
    'SIEMENS', 'SIGACHI', 'SKIPPER', 'SMLISUZU', 'SMSPHARMA', 'SNOWMAN',
    'SOLARA', 'SONACOMS', 'SOUTHBANK', 'SPARC', 'SPLPETRO', 'SPMLINFRA',
    'SRHHYPOLTD', 'SSWL', 'STAR', 'STCINDIA', 'STLTECH', 'SUBEXLTD',
    'SUDARSCHEM', 'SUNDARMHLD', 'SUNFLAG', 'SUNPHARMA', 'SUPRAJIT', 'SUPRIYA',
    'SURYAROSNI', 'SUVEN', 'SUVENPHAR', 'SWARAJENG', 'SYMPHONY', 'SYRMA',
    'TATAINVEST', 'TCNSBRANDS', 'TDPOWERSYS', 'TEAMLEASE', 'TECHNOE',
    'TEGA', 'THOMASCOOK', 'TIDEWATER', 'TIIL', 'TINPLATE', 'TIPSINDLTD',
    'TMB', 'TNPETRO', 'TPLPLASTEH', 'TRANSCORP', 'TREJHARA', 'TRF',
    'TRITURBINE', 'TRIVENI', 'TTKHLTCARE', 'TTKPRESTIG', 'TV18BRDCST',
    'TVSSCS', 'TVTODAY', 'UGROCAP', 'UNICHEMLAB', 'UNIPARTS', 'UNIVCABLES',
    'VADILALIND', 'VAIBHAVGBL', 'VAKRANGEE', 'VALUEIND', 'VARROC', 'VENKEYS',
    'VESUVIUS', 'VIMTALABS', 'VINDHYATEL', 'VINYLINDIA', 'VIPIND', 'VISAKAIND',
    'VSTIND', 'VSTRILLERS', 'WABAG', 'WABCOINDIA', 'WATERBASE', 'WELENT',
    'WENDT', 'WOCKPHARMA', 'WONDERLA', 'XPROINDIA', 'YATHARTH', 'ZAGGLE',
    'ZFCVINDIA', 'ZOMATO', 'ZOPPER'
]

# Combine all new symbols (excluding already collected)
ALL_NEW_SYMBOLS = sorted(list(
    (set(MIDCAP_150) | set(PSU_STOCKS) | set(SMALLCAP_250)) - ALREADY_COLLECTED
))


def download_stock(symbol, suffix=".NS", start_date="2000-01-01"):
    """Download a single stock's data"""
    full_symbol = f"{symbol}{suffix}"
    try:
        ticker = yf.Ticker(full_symbol)
        df = ticker.history(start=start_date, end=None, interval="1d")

        if df.empty and suffix == ".NS":
            # Try BSE
            return download_stock(symbol, ".BO", start_date)

        if df.empty:
            return None, symbol

        df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df, symbol

    except Exception as e:
        logger.error(f"Error {symbol}: {e}")
        return None, symbol


def main():
    print("=" * 60)
    print("EXTENDED INDIAN MARKET DATA COLLECTOR")
    print("=" * 60)
    print(f"\nNew symbols to collect: {len(ALL_NEW_SYMBOLS)}")
    print(f"Categories:")
    print(f"  - Midcap 150 additions")
    print(f"  - PSU Stocks")
    print(f"  - Smallcap 250")
    print()

    output_path = Path("TrainingData/daily")
    output_path.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    total_rows = 0
    failed_symbols = []

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_stock, sym): sym for sym in ALL_NEW_SYMBOLS}

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df, sym = future.result()

                if df is not None and not df.empty:
                    # Save to parquet
                    filename = f"{sym}.NS_daily.parquet"
                    filepath = output_path / filename
                    df.to_parquet(filepath, compression='snappy')

                    success += 1
                    total_rows += len(df)
                    years = (df.index.max() - df.index.min()).days / 365
                    print(f"[{success}/{len(ALL_NEW_SYMBOLS)}] {sym}: {len(df):,} rows ({years:.1f} years)")
                else:
                    failed += 1
                    failed_symbols.append(symbol)

            except Exception as e:
                failed += 1
                failed_symbols.append(symbol)
                logger.error(f"Error processing {symbol}: {e}")

            time.sleep(0.1)  # Rate limiting

    print()
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Success: {success}/{len(ALL_NEW_SYMBOLS)}")
    print(f"Failed: {failed}")
    print(f"Total new rows: {total_rows:,}")
    print()

    if failed_symbols and len(failed_symbols) <= 30:
        print(f"Failed symbols: {failed_symbols}")


if __name__ == "__main__":
    main()
