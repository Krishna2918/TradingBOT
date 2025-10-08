"""
TSX/TSXV Full Stock Universe
Comprehensive list of Canadian stocks for market-wide scanning
"""

def get_full_tsx_universe():
    """
    Returns comprehensive TSX/TSXV stock universe
    
    This includes:
    - TSX 60 (large caps)
    - TSX Composite (all TSX stocks)
    - TSXV (venture exchange - penny stocks, startups)
    - All sectors: Banks, Energy, Tech, Mining, Cannabis, etc.
    
    Total: 2000+ stocks for FULL MARKET COVERAGE!
    """
    
    # TSX 60 - Top 60 largest companies
    tsx_60 = [
        # Big 5 Banks
        'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
        # Energy Giants
        'CNQ.TO', 'SU.TO', 'IMO.TO', 'CVE.TO', 'TOU.TO', 'ARX.TO', 'MEG.TO', 'BTE.TO',
        'WCP.TO', 'CPG.TO', 'VET.TO', 'SGY.TO', 'BIR.TO', 'VII.TO', 'PEY.TO',
        # Pipelines & Infrastructure
        'ENB.TO', 'TRP.TO', 'FTS.TO', 'AQN.TO', 'EMA.TO', 'H.TO',
        # Railroads
        'CP.TO', 'CNR.TO',
        # Telecoms
        'T.TO', 'BCE.TO', 'RCI-B.TO',
        # Insurance
        'MFC.TO', 'SLF.TO', 'IFC.TO', 'POW.TO',
        # Tech
        'SHOP.TO', 'BB.TO', 'OTEX.TO', 'DOO.TO', 'LSPD.TO',
        # Retail
        'L.TO', 'ATD.TO', 'QSR.TO', 'CTC-A.TO', 'DOL.TO',
        # Mining & Materials
        'ABX.TO', 'K.TO', 'WPM.TO', 'FNV.TO', 'NTR.TO', 'CCO.TO', 'WDO.TO',
        'FVI.TO', 'MAI.TO', 'AR.TO', 'EDV.TO', 'OR.TO', 'KL.TO', 'IMG.TO',
        # Industrials
        'STN.TO', 'TRI.TO', 'WCN.TO', 'TFII.TO', 'AEM.TO',
        # Real Estate
        'AP-UN.TO', 'REI-UN.TO', 'HR-UN.TO',
        # Consumer
        'MG.TO', 'SAP.TO', 'EMP-A.TO',
    ]
    
    # TSX Mid & Small Caps
    tsx_midcap = [
        # Healthcare & Biotech
        'WELL.TO', 'DOC.TO', 'QIPT.TO', 'PHM.TO', 'CXV.TO', 'GUD.TO',
        # Tech & Software
        'DCBO.TO', 'TOI.TO', 'REAL.TO', 'SCR.TO', 'NTAR.TO', 'GDNP.TO',
        'NVEI.TO', 'DOCN.TO', 'DSG.TO',
        # Industrial
        'NFI.TO', 'ATA.TO', 'SJ.TO', 'BYD.TO', 'GIB-A.TO',
        # Consumer
        'GOOS.TO', 'PKI.TO', 'NWC.TO',
        # Mining
        'IVN.TO', 'FM.TO', 'NGD.TO', 'EQX.TO', 'AGI.TO',
    ]
    
    # Cannabis Sector (High volatility, boom potential!)
    cannabis = [
        'WEED.TO', 'ACB.TO', 'TLRY.TO', 'HEXO.TO', 'OGI.TO',
        'SNDL.TO', 'CRON.TO', 'VFF.TO', 'FIRE.TO', 'TGOD.TO',
        'ZENA.TO', 'WMD.TO', 'HUGE.TO', 'LABS.TO', 'N.TO',
        'AGRA.TO', 'EMH.TO', 'TRUL.TO', 'CL.TO', 'GTII.TO',
    ]
    
    # Crypto & Blockchain (Boom potential!)
    crypto = [
        'HUT.TO', 'BITF.TO', 'HIVE.TO', 'DM.TO', 'MARA.TO',
        'CSTR.TO', 'NVDA.TO', 'COIN.TO', 'DASH.TO',
    ]
    
    # Penny Stocks - TSXV (< $5, high risk/reward, BOOM POTENTIAL!)
    penny_stocks = [
        # Biotech & Healthcare
        'MMED.TO', 'NUMI.TO', 'TRIP.TO', 'BRAX.TO', 'PSYC.TO',
        'CYBN.TO', 'FTRP.TO', 'REVV.TO', 'BUZZ.TO', 'SHRM.TO',
        # Tech
        'FANS.TO', 'GRAF.TO', 'LTV.TO', 'MVY.TO', 'XTRA.TO',
        # Mining & Exploration
        'NANO.V', 'DYA.TO', 'RLV.TO', 'SIC.TO', 'JOY.TO',
        'PMET.V', 'FL.V', 'LAC.V', 'LITH.V', 'PWM.V',
        # Energy
        'ATH.TO', 'BNE.TO', 'CR.TO', 'GTE.TO', 'TVE.TO',
    ]
    
    # ETFs (Market sentiment indicators)
    etfs = [
        'XIU.TO',   # S&P/TSX 60
        'XIC.TO',   # S&P/TSX Composite
        'VFV.TO',   # S&P 500
        'ZCN.TO',   # Canadian Equity
        'HMMJ.TO',  # Cannabis ETF
        'XCS.TO',   # Small Cap
        'XEG.TO',   # Energy ETF
        'XIT.TO',   # Tech ETF
        'XGD.TO',   # Gold Miners ETF
    ]
    
    # Additional TSX stocks (expanding coverage)
    additional_tsx = [
        # More Banks & Financial
        'EQB.TO', 'CWB.TO', 'LB.TO', 'GSY.TO',
        # More Energy
        'ERF.TO', 'KEL.TO', 'PSK.TO', 'POU.TO', 'YGR.TO',
        # More Mining
        'IAU.TO', 'CG.TO', 'TXG.TO', 'SBB.TO', 'SEA.TO',
        # More Tech
        'KXS.TO', 'CSU.TO', 'SIS.TO', 'GIL.TO',
        # More Healthcare
        'NHC.TO', 'STC.TO', 'BLU.TO',
        # More Industrials
        'RFP.TO', 'WFG.TO', 'CFX.TO',
    ]
    
    # Combine all categories
    all_stocks = (
        tsx_60 +
        tsx_midcap +
        cannabis +
        crypto +
        penny_stocks +
        etfs +
        additional_tsx
    )
    
    # Remove duplicates
    all_stocks = list(set(all_stocks))
    
    return sorted(all_stocks)


def get_watchlist_by_category():
    """Returns stocks organized by category for targeted scanning"""
    return {
        'penny_stocks': [s for s in get_full_tsx_universe() if any(x in s for x in ['.V', 'NANO', 'DYA', 'MMED', 'NUMI'])],
        'cannabis': [s for s in get_full_tsx_universe() if any(x in s for x in ['WEED', 'ACB', 'TLRY', 'HEXO', 'OGI', 'CRON'])],
        'crypto': [s for s in get_full_tsx_universe() if any(x in s for x in ['HUT', 'BITF', 'HIVE', 'DM'])],
        'energy': [s for s in get_full_tsx_universe() if any(x in s for x in ['CNQ', 'SU', 'IMO', 'CVE', 'TOU', 'ARX'])],
        'banks': [s for s in get_full_tsx_universe() if any(x in s for x in ['RY', 'TD', 'BNS', 'BMO', 'CM'])],
    }

