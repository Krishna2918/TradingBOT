"""
Technical Analysis Page - Comprehensive Technical Indicators
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

COLORS = {
    'primary': '#00D09C',
    'secondary': '#44475B',
    'background': '#FAFAFA',
    'card': '#FFFFFF',
    'text': '#44475B',
    'success': '#00D09C',
    'danger': '#EB5B3C',
    'warning': '#FDB022',
    'info': '#5367FE',
    'muted': '#8B8B8B'
}

def create_technical_analysis_page():
    """Create comprehensive technical analysis page"""
    
    return html.Div([
        html.H2("Technical Analysis", className="mb-4"),
        
        # Advanced Filters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-filter me-2", style={'color': COLORS['primary']}),
                        html.Strong("Advanced Filters", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Symbol Selection"),
                                dcc.Dropdown(
                                    id='ta-symbol-filter',
                                    options=[
                                        {'label': 'All Symbols', 'value': 'all'},
                                        {'label': 'Penny Stocks (AI.TO, HUT.TO, BITF.TO)', 'value': 'penny'},
                                        {'label': 'Core Holdings (RY.TO, TD.TO, SHOP.TO)', 'value': 'core'},
                                        {'label': 'F&O (XIU.TO, XSP.TO, XEG.TO)', 'value': 'futures_options'},
                                        {'label': 'Custom Selection', 'value': 'custom'}
                                    ],
                                    value='core',
                                    multi=False
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Custom Symbols"),
                                dcc.Dropdown(
                                    id='ta-custom-symbols',
                                    options=[
                                        {'label': 'RY.TO - Royal Bank', 'value': 'RY.TO'},
                                        {'label': 'TD.TO - TD Bank', 'value': 'TD.TO'},
                                        {'label': 'SHOP.TO - Shopify', 'value': 'SHOP.TO'},
                                        {'label': 'CNR.TO - CN Railway', 'value': 'CNR.TO'},
                                        {'label': 'ENB.TO - Enbridge', 'value': 'ENB.TO'},
                                        {'label': 'AI.TO - AI Stock', 'value': 'AI.TO'},
                                        {'label': 'HUT.TO - Hut 8 Mining', 'value': 'HUT.TO'},
                                        {'label': 'XIU.TO - TSX 60 ETF', 'value': 'XIU.TO'}
                                    ],
                                    value=['RY.TO', 'TD.TO'],
                                    multi=True,
                                    disabled=True
                                )
                            ], width=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Timeframe"),
                                dcc.Dropdown(
                                    id='ta-timeframe',
                                    options=[
                                        {'label': '1 Minute', 'value': '1m'},
                                        {'label': '5 Minutes', 'value': '5m'},
                                        {'label': '15 Minutes', 'value': '15m'},
                                        {'label': '1 Hour', 'value': '1h'},
                                        {'label': 'Daily', 'value': '1d'}
                                    ],
                                    value='5m'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Period"),
                                dcc.Dropdown(
                                    id='ta-period',
                                    options=[
                                        {'label': 'Last 24 Hours', 'value': '1d'},
                                        {'label': 'Last 3 Days', 'value': '3d'},
                                        {'label': 'Last Week', 'value': '1wk'},
                                        {'label': 'Last Month', 'value': '1mo'},
                                        {'label': 'Last 3 Months', 'value': '3mo'}
                                    ],
                                    value='1wk'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Indicators"),
                                dcc.Dropdown(
                                    id='ta-indicators',
                                    options=[
                                        {'label': 'All Indicators', 'value': 'all'},
                                        {'label': 'Momentum (RSI, MACD, Stochastic)', 'value': 'momentum'},
                                        {'label': 'Trend (MA, EMA, Bollinger)', 'value': 'trend'},
                                        {'label': 'Volume (VWAP, OBV)', 'value': 'volume'},
                                        {'label': 'Volatility (ATR, Bollinger)', 'value': 'volatility'},
                                        {'label': 'Custom Selection', 'value': 'custom'}
                                    ],
                                    value='all'
                                )
                            ], width=4)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Custom Indicators"),
                                dcc.Checklist(
                                    id='ta-custom-indicators',
                                    options=[
                                        {'label': 'RSI (14)', 'value': 'rsi'},
                                        {'label': 'MACD (12,26,9)', 'value': 'macd'},
                                        {'label': 'Bollinger Bands (20,2)', 'value': 'bollinger'},
                                        {'label': 'Moving Averages (20,50,200)', 'value': 'ma'},
                                        {'label': 'VWAP', 'value': 'vwap'},
                                        {'label': 'ATR (14)', 'value': 'atr'},
                                        {'label': 'Stochastic (14,3)', 'value': 'stochastic'},
                                        {'label': 'Volume Ratio', 'value': 'volume_ratio'}
                                    ],
                                    value=['rsi', 'macd', 'bollinger'],
                                    inline=True,
                                    disabled=True
                                )
                            ], width=12)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Technical Analysis Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-line me-2", style={'color': COLORS['primary']}),
                        html.Strong("Price Chart with Technical Indicators", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='ta-main-chart',
                            config={'displayModeBar': True, 'modeBarButtonsToAdd': ['drawline', 'eraseshape']},
                            style={'height': '600px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-bar me-2", style={'color': COLORS['primary']}),
                        html.Strong("Volume Analysis", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='ta-volume-chart',
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4"),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-area me-2", style={'color': COLORS['primary']}),
                        html.Strong("RSI Analysis", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='ta-rsi-chart',
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=4)
        ]),
        
        # Technical Indicators Summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-table me-2", style={'color': COLORS['primary']}),
                        html.Strong("Technical Indicators Summary", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='ta-indicators-table')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ])
        ]),
        
        # Signal Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-signal me-2", style={'color': COLORS['primary']}),
                        html.Strong("Trading Signals", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='ta-signals-analysis')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-pie me-2", style={'color': COLORS['primary']}),
                        html.Strong("Signal Distribution", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='ta-signals-chart',
                            config={'displayModeBar': False}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6)
        ]),
        
        # Pattern Recognition
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-search me-2", style={'color': COLORS['primary']}),
                        html.Strong("Pattern Recognition", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='ta-patterns-analysis')
                    ])
                ], className="shadow-sm border-0")
            ])
        ])
    ])

def create_technical_indicators_summary(data):
    """Create technical indicators summary table"""
    if not data:
        return html.P("No data available", className="text-muted")
    
    # Calculate technical indicators for each symbol
    summary_data = []
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
        latest = df.iloc[-1]
        
        # RSI analysis
        rsi = latest.get('rsi', 50)
        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        rsi_color = COLORS['danger'] if rsi > 70 else COLORS['success'] if rsi < 30 else COLORS['muted']
        
        # MACD analysis
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_hist = latest.get('macd_histogram', 0)
        macd_trend = "Bullish" if macd > macd_signal else "Bearish"
        macd_color = COLORS['success'] if macd > macd_signal else COLORS['danger']
        
        # Bollinger Bands analysis
        bb_percent = latest.get('bb_percent', 0.5)
        bb_signal = "Upper Band" if bb_percent > 0.8 else "Lower Band" if bb_percent < 0.2 else "Middle"
        bb_color = COLORS['danger'] if bb_percent > 0.8 else COLORS['success'] if bb_percent < 0.2 else COLORS['muted']
        
        # Volume analysis
        volume_ratio = latest.get('volume_ratio', 1)
        volume_signal = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
        volume_color = COLORS['warning'] if volume_ratio > 1.5 else COLORS['info'] if volume_ratio < 0.5 else COLORS['muted']
        
        summary_data.append({
            'Symbol': symbol,
            'Price': f"${latest['close']:.2f}",
            'RSI': f"{rsi:.1f}",
            'RSI Signal': rsi_signal,
            'MACD': f"{macd:.3f}",
            'MACD Trend': macd_trend,
            'BB %': f"{bb_percent:.2f}",
            'BB Signal': bb_signal,
            'Volume Ratio': f"{volume_ratio:.2f}",
            'Volume Signal': volume_signal
        })
    
    if not summary_data:
        return html.P("No data available", className="text-muted")
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create styled table
    table = dbc.Table.from_dataframe(
        df_summary,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size='sm'
    )
    
    return table

def create_trading_signals_analysis(data):
    """Create trading signals analysis"""
    if not data:
        return html.P("No data available", className="text-muted")
    
    signals = []
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
        latest = df.iloc[-1]
        
        # Calculate signals
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        bb_percent = latest.get('bb_percent', 0.5)
        volume_ratio = latest.get('volume_ratio', 1)
        
        # Determine overall signal
        buy_signals = 0
        sell_signals = 0
        
        if rsi < 30:  # Oversold
            buy_signals += 1
        elif rsi > 70:  # Overbought
            sell_signals += 1
        
        if macd > macd_signal:  # Bullish MACD
            buy_signals += 1
        else:  # Bearish MACD
            sell_signals += 1
        
        if bb_percent < 0.2:  # Lower Bollinger Band
            buy_signals += 1
        elif bb_percent > 0.8:  # Upper Bollinger Band
            sell_signals += 1
        
        if volume_ratio > 1.5:  # High volume confirms signal
            if buy_signals > sell_signals:
                buy_signals += 1
            elif sell_signals > buy_signals:
                sell_signals += 1
        
        # Overall signal
        if buy_signals > sell_signals:
            overall_signal = "BUY"
            signal_color = COLORS['success']
        elif sell_signals > buy_signals:
            overall_signal = "SELL"
            signal_color = COLORS['danger']
        else:
            overall_signal = "HOLD"
            signal_color = COLORS['muted']
        
        confidence = max(buy_signals, sell_signals) / (buy_signals + sell_signals) if (buy_signals + sell_signals) > 0 else 0
        
        signals.append({
            'symbol': symbol,
            'signal': overall_signal,
            'confidence': confidence,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'color': signal_color
        })
    
    if not signals:
        return html.P("No signals available", className="text-muted")
    
    # Create signal cards
    signal_cards = []
    for signal in signals:
        signal_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5(signal['symbol'], className="mb-1"),
                        html.H4(signal['signal'], style={'color': signal['color'], 'fontWeight': 'bold'}, className="mb-1"),
                        html.P(f"Confidence: {signal['confidence']:.1%}", className="mb-1"),
                        html.Small(f"Buy: {signal['buy_signals']} | Sell: {signal['sell_signals']}", className="text-muted")
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(signal_cards)

def create_pattern_recognition_analysis(data):
    """Create pattern recognition analysis"""
    if not data:
        return html.P("No data available", className="text-muted")
    
    patterns = []
    
    for symbol, df in data.items():
        if len(df) < 20:  # Need enough data for pattern recognition
            continue
        
        # Simple pattern detection (in real implementation, use more sophisticated algorithms)
        recent_prices = df['close'].tail(20).values
        
        # Trend analysis
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if price_change > 0.05:  # 5% increase
            trend = "Strong Uptrend"
            trend_color = COLORS['success']
        elif price_change > 0.02:  # 2% increase
            trend = "Uptrend"
            trend_color = COLORS['success']
        elif price_change < -0.05:  # 5% decrease
            trend = "Strong Downtrend"
            trend_color = COLORS['danger']
        elif price_change < -0.02:  # 2% decrease
            trend = "Downtrend"
            trend_color = COLORS['danger']
        else:
            trend = "Sideways"
            trend_color = COLORS['muted']
        
        # Volatility analysis
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        if volatility > 0.03:  # 3% volatility
            vol_level = "High Volatility"
            vol_color = COLORS['warning']
        elif volatility > 0.015:  # 1.5% volatility
            vol_level = "Medium Volatility"
            vol_color = COLORS['info']
        else:
            vol_level = "Low Volatility"
            vol_color = COLORS['muted']
        
        patterns.append({
            'symbol': symbol,
            'trend': trend,
            'trend_color': trend_color,
            'volatility': vol_level,
            'vol_color': vol_color,
            'price_change': price_change,
            'volatility_value': volatility
        })
    
    if not patterns:
        return html.P("No patterns detected", className="text-muted")
    
    # Create pattern cards
    pattern_cards = []
    for pattern in patterns:
        pattern_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5(pattern['symbol'], className="mb-2"),
                        html.Div([
                            html.Strong("Trend: ", className="me-2"),
                            html.Span(pattern['trend'], style={'color': pattern['trend_color']})
                        ], className="mb-1"),
                        html.Div([
                            html.Strong("Volatility: ", className="me-2"),
                            html.Span(pattern['volatility'], style={'color': pattern['vol_color']})
                        ], className="mb-1"),
                        html.Small(f"Price Change: {pattern['price_change']:.2%}", className="text-muted")
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(pattern_cards)
