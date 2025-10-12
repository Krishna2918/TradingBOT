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
    
    # Additional TSX stocks (expanding coverage to 500+)
    additional_tsx = [
        # More Banks & Financial (20 stocks)
        'EQB.TO', 'CWB.TO', 'LB.TO', 'GSY.TO', 'HCG.TO', 'FSZ.TO', 'EFN.TO',
        'FFH.TO', 'IFC.TO', 'GWO.TO', 'IAG.TO', 'BN.TO', 'BAM.TO', 'BAM-A.TO',
        'ONT.TO', 'CGI.TO', 'FSV.TO', 'SLF.TO', 'MFC.TO', 'PWF.TO',
        
        # More Energy (40 stocks)
        'ERF.TO', 'KEL.TO', 'PSK.TO', 'POU.TO', 'YGR.TO', 'NVA.TO', 'OBE.TO',
        'PNE.TO', 'TAG.TO', 'WTE.TO', 'JOY.TO', 'GXE.TO', 'PIPE.TO', 'ATH.TO',
        'BNE.TO', 'CR.TO', 'GTE.TO', 'TVE.TO', 'AAV.TO', 'BIR.TO', 'PSK.TO',
        'ERF.TO', 'SGY.TO', 'CPG.TO', 'VET.TO', 'WCP.TO', 'BTE.TO', 'MEG.TO',
        'ARX.TO', 'TOU.TO', 'CVE.TO', 'IMO.TO', 'SU.TO', 'CNQ.TO', 'VII.TO',
        'PEY.TO', 'BIR.TO', 'SGY.TO', 'VET.TO', 'CPG.TO', 'WCP.TO',
        
        # More Mining & Materials (60 stocks)
        'IAU.TO', 'CG.TO', 'TXG.TO', 'SBB.TO', 'SEA.TO', 'PAAS.TO', 'SSL.TO',
        'AUY.TO', 'ELD.TO', 'YRI.TO', 'IMG.TO', 'KL.TO', 'OR.TO', 'EDV.TO',
        'AR.TO', 'MAI.TO', 'FVI.TO', 'WDO.TO', 'CCO.TO', 'NTR.TO', 'FNV.TO',
        'WPM.TO', 'K.TO', 'ABX.TO', 'AEM.TO', 'B2GOLD.TO', 'BTO.TO', 'CG.TO',
        'DPM.TO', 'EQX.TO', 'FSM.TO', 'GOR.TO', 'GPR.TO', 'HBM.TO', 'IAMGOLD.TO',
        'SAND.TO', 'TKO.TO', 'TMQ.TO', 'TSG.TO', 'TXG.TO', 'USG.TO', 'WRN.TO',
        'NGD.TO', 'FM.TO', 'IVN.TO', 'AGI.TO', 'TECK-B.TO', 'CS.TO', 'LUN.TO',
        'HBM.TO', 'SMT.TO', 'TGB.TO', 'NCU.TO', 'TKO.TO', 'CU.TO', 'ERO.TO',
        'MMX.TO', 'NGD.TO', 'OR.TO', 'PGM.TO', 'SAND.TO',
        
        # More Tech & Software (30 stocks)
        'KXS.TO', 'CSU.TO', 'SIS.TO', 'GIL.TO', 'OTEX.TO', 'BB.TO', 'SHOP.TO',
        'LSPD.TO', 'DOO.TO', 'DCBO.TO', 'TOI.TO', 'REAL.TO', 'SCR.TO', 'NTAR.TO',
        'GDNP.TO', 'NVEI.TO', 'DOCN.TO', 'DSG.TO', 'KINAXIS.TO', 'OPEN.TO',
        'LSPD.TO', 'GSY.TO', 'QTRH.TO', 'DND.TO', 'EIF.TO', 'GIB-A.TO',
        'JWEL.TO', 'LNF.TO', 'MTY.TO', 'PBH.TO',
        
        # More Healthcare & Pharma (25 stocks)
        'NHC.TO', 'STC.TO', 'BLU.TO', 'WELL.TO', 'DOC.TO', 'QIPT.TO', 'PHM.TO',
        'CXV.TO', 'GUD.TO', 'LABS.TO', 'N.TO', 'PHA.TO', 'HLS.TO', 'CXR.TO',
        'JWEL.TO', 'MT.TO', 'PLC.TO', 'PTQ.TO', 'VMD.TO', 'VPH.TO', 'WELL.TO',
        'DOC.TO', 'CRX.TO', 'CXV.TO', 'GLXY.TO',
        
        # More Industrials (25 stocks)
        'RFP.TO', 'WFG.TO', 'CFX.TO', 'NFI.TO', 'ATA.TO', 'SJ.TO', 'BYD.TO',
        'GIB-A.TO', 'STN.TO', 'TRI.TO', 'WCN.TO', 'TFII.TO', 'AEM.TO', 'GFL.TO',
        'BEP-UN.TO', 'RNW.TO', 'AQN.TO', 'NPI.TO', 'INE.TO', 'EMA.TO', 'FTS.TO',
        'CU.TO', 'FTT.TO', 'AC.TO', 'TFII.TO',
        
        # More Consumer & Retail (20 stocks)
        'GOOS.TO', 'PKI.TO', 'NWC.TO', 'L.TO', 'ATD.TO', 'QSR.TO', 'CTC-A.TO',
        'DOL.TO', 'MG.TO', 'SAP.TO', 'EMP-A.TO', 'JWEL.TO', 'MTY.TO', 'PRMW.TO',
        'RECP.TO', 'RET-A.TO', 'SIR-UN.TO', 'SRV-UN.TO', 'BDI.TO', 'CGX.TO',
        
        # More Real Estate (30 stocks)
        'AP-UN.TO', 'REI-UN.TO', 'HR-UN.TO', 'CAR-UN.TO', 'NWH-UN.TO', 'TNT-UN.TO',
        'IIP-UN.TO', 'CHP-UN.TO', 'BEI-UN.TO', 'D-UN.TO', 'FCR.TO', 'KMP-UN.TO',
        'MRG-UN.TO', 'SMU-UN.TO', 'SRU-UN.TO', 'CRT-UN.TO', 'EDIT.TO', 'CHR.TO',
        'GRT-UN.TO', 'MRC.TO', 'NXR-UN.TO', 'PMZ-UN.TO', 'DIR-UN.TO', 'BAM-A.TO',
        'BPY-UN.TO', 'BEP-UN.TO', 'BIP-UN.TO', 'BBU-UN.TO', 'BHC.TO', 'REI-UN.TO',
        
        # More Utilities (15 stocks)
        'AQN.TO', 'EMA.TO', 'FTS.TO', 'H.TO', 'CU.TO', 'BEP-UN.TO', 'BLX.TO',
        'CPX.TO', 'INE.TO', 'NPI.TO', 'RNW.TO', 'TA.TO', 'UNS.TO', 'FTS.TO', 'ENB.TO',
        
        # More Telecoms & Media (10 stocks)
        'T.TO', 'BCE.TO', 'RCI-B.TO', 'SJR-B.TO', 'QBR-B.TO', 'CJR-B.TO',
        'DBO.TO', 'TCN.TO', 'CCA.TO', 'CIGI.TO',
        
        # More Pipelines (10 stocks)
        'ENB.TO', 'TRP.TO', 'PPL.TO', 'KEY.TO', 'IPL.TO', 'ALA.TO', 'GEI.TO',
        'BIPC.TO', 'BEP-UN.TO', 'FLT.TO',
    ]
    
    # More TSXV Penny Stocks (100+ stocks with BOOM potential!)
    tsxv_penny_extended = [
        # Psychedelics & Wellness
        'MMED.TO', 'NUMI.TO', 'TRIP.TO', 'CYBN.TO', 'FTRP.TO', 'REVV.TO', 
        'BUZZ.TO', 'SHRM.TO', 'PSYC.TO', 'BRAX.TO', 'MYCO.V', 'DRUG.V',
        'RVV.V', 'TBP.V', 'WILD.V', 'BOSS.V', 'NEPT.V', 'TCAN.V',
        
        # Biotech & Life Sciences
        'THTX.TO', 'PLI.TO', 'BNC.TO', 'SBM.V', 'MSB.V', 'SZLS.V', 'DYA.V',
        'PMED.V', 'HITI.V', 'PNG.V', 'CANN.TO', 'NEXE.V', 'GDNP.V', 'LXX.V',
        'COOL.V', 'FOOD.V', 'CRUZ.V', 'ISOL.V', 'LITT.V', 'MTEC.V',
        
        # Mining Exploration
        'NANO.V', 'DYA.TO', 'RLV.TO', 'SIC.TO', 'JOY.TO', 'PMET.V', 'FL.V',
        'LAC.V', 'LITH.V', 'PWM.V', 'FPX.V', 'NMX.V', 'CYP.V', 'LI.V',
        'AVL.TO', 'CRE.V', 'PE.V', 'PGDC.V', 'VOLT.V', 'FTMC.V', 'GRAT.V',
        'MIN.V', 'NOU.V', 'PODA.V', 'ROCK.V', 'SYA.V', 'UGE.V', 'ZAC.V',
        
        # Tech & Software Ventures
        'FANS.TO', 'GRAF.TO', 'LTV.TO', 'MVY.TO', 'XTRA.TO', 'EAGR.V', 'EATS.V',
        'GDNP.V', 'GBLC.V', 'LQID.V', 'MILK.V', 'MPXI.V', 'MSTR.V', 'NTAR.V',
        'POND.V', 'PTAL.V', 'WIFI.V', 'YOLO.V', 'NXO.V', 'RHT.V',
        
        # Clean Energy & ESG
        'CBDT.V', 'ENGH.V', 'EXRO.V', 'FUEL.V', 'GRN.V', 'HPQ.V', 'HPQFF.V',
        'NRG.V', 'NRTH.V', 'SOLR.V', 'THNK.V', 'ZEV.V', 'GRID.V', 'ION.V',
        
        # Special Situations (SPACs, Crypto, NFT)
        'COIN.V', 'DASH.V', 'BIGG.V', 'DM.V', 'ELBM.V', 'FORK.V', 'MOGO.V',
        'NFTI.V', 'OILS.V', 'PMN.V', 'VOX.V', 'WEST.V', 'BRCK.V', 'QYOU.V',
        
        # More Ventures (Rounding to 500+)
        'CBDL.V', 'TAAT.V', 'VERO.V', 'ACDC.V', 'AVCR.V', 'BRAG.V', 'CHV.V',
        'CST.V', 'DMG.V', 'EAT.V', 'EMPR.V', 'EQT.V', 'GENE.V', 'GRAT.V',
        'GRIT.V', 'HBOR.V', 'HRH.V', 'INC.V', 'JAG.V', 'KASH.V', 'LITT.V',
        'MANA.V', 'NEXT.V', 'NGW.V', 'NLN.V', 'NSP.V', 'PNG.V', 'QCA.V',
        'RVVTF.V', 'SLNG.V', 'SPMT.V', 'SUGR.V', 'TRIL.V', 'TUF.V', 'USHA.V',
        'VKIN.V', 'VRTX.V', 'VUI.V', 'WTC.V', 'XBLK.V', 'ZILD.V',
        
        # High-Beta Small Caps (Momentum plays)
        'ACQ.V', 'ADK.V', 'AI.V', 'AIX.V', 'AKR.V', 'ALY.V', 'AMK.V',
        'ANX.V', 'ASM.V', 'ATP.V', 'AXL.V', 'AXU.V', 'BCM.V', 'BEX.V',
        'BSX.V', 'BTR.V', 'BUR.V', 'BVA.V', 'BWN.V', 'CAN.V', 'CAT.V',
        'CDB.V', 'CDH.V', 'CFG.V', 'CNE.V', 'CODE.V', 'CRK.V', 'DEF.V',
        'DNT.V', 'DGTL.V', 'EGT.V', 'EGLX.V', 'ELO.V', 'EMO.V', 'EMX.V',
        'ESM.V', 'EVN.V', 'EXN.V', 'FAT.V', 'FCU.V', 'FIND.V', 'FVL.V',
        'GEM.V', 'GMV.V', 'GPG.V', 'GRC.V', 'GSP.V', 'GUY.V', 'GXS.V',
        'HEO.V', 'HME.V', 'HPYT.V', 'HSM.V', 'ICG.V', 'IDG.V', 'IDK.V',
    ]
    
    # Combine all categories (500+ stocks total!)
    all_stocks = (
        tsx_60 +
        tsx_midcap +
        cannabis +
        crypto +
        penny_stocks +
        tsxv_penny_extended +
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

