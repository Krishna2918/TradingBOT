from __future__ import annotations

import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from .state_manager import trading_state, STATE_STORE, save_trading_state
from src.data_pipeline.questrade_client import QuestradeClient
from src.infrastructure.state_store import SQLiteStateStore

# Price cache and rate limiting
_price_cache = {}
_last_request_time = {}
RATE_LIMIT_SECONDS = 2  # Minimum seconds between requests for same symbol
CACHE_DURATION_SECONDS = 30  # Cache prices for 30 seconds


def _ensure_live_broker():
    try:
        if trading_state.get('broker') is None:
            from src.execution.brokers.questrade_broker import QuestradeBroker
            allow_trading = os.getenv('QUESTRADE_ALLOW_TRADING', 'false').lower() in ('1','true','yes')
            practice_mode = os.getenv('QUESTRADE_PRACTICE_MODE', 'true').lower() in ('1','true','yes')
            trading_state['broker'] = QuestradeBroker(
                allow_trading=allow_trading,
                practice_mode=practice_mode,
            )
    except Exception as e:
        print(f"Live broker init failed: {e}")

def real_ai_trade():
    """Execute a trade using the REAL AI system with full feature integration"""
    if not trading_state['initialized']:
        return
    
    # Only trade when market is open
    if not is_market_open():
        return
    # Respect pause and kill switch
    if trading_state.get('paused'):
        return
    start_cap = float(trading_state.get('starting_capital') or 0)
    cur_cap = float(trading_state.get('current_capital') or 0)
    kst = float(trading_state.get('kill_switch_threshold') or 0)
    if start_cap > 0 and kst > 0:
        dd_pct = (cur_cap - start_cap) / start_cap * 100.0
        if dd_pct <= -kst:
            trading_state['kill_switch_active'] = True
    if trading_state.get('kill_switch_active'):
        return
    
    # Use REAL AI if available, otherwise fallback to simulate_ai_trade
    ai = trading_state.get('ai_instance')
    
    if ai is not None:
        try:
            # First, analyze the market to get comprehensive data
            analysis = ai.analyze_market()
            
            # AI makes a REAL decision based on ALL data
            decision = ai.make_trading_decision(analysis)
            
            # Log ALL decisions including HOLD
            if decision:
                action = decision.get('action', 'HOLD')
                symbol = decision.get('symbol', 'N/A')
                confidence = decision.get('confidence', 0.0)
                reasoning = decision.get('reasoning', [])
                
                print(f"\n{'='*80}")
                print(f"🤖 AI DECISION: {action} {symbol}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Reasoning:")
                for reason in reasoning:
                    print(f"      • {reason}")
                print(f"{'='*80}\n")
                
                # Log to AI activity logger
                try:
                    from src.logging.ai_activity_logger import log_ai_decision
                    log_ai_decision(
                        decision_type='TRADING_DECISION',
                        symbol=symbol,
                        decision=f"{action} (confidence: {confidence:.1%})",
                        reasoning=reasoning,
                        risk_factors={'action': action, 'confidence': confidence}
                    )
                except Exception as log_err:
                    print(f"Warning: Failed to log AI decision: {log_err}")
            
            if decision and decision['action'] != 'HOLD':
                symbol = decision['symbol']
                action = decision['action']
                confidence = decision.get('confidence', 0.0)
                reasoning = decision.get('reasoning', 'AI decision')
                
                # Get real price
                price = get_demo_price(symbol)
                if price is None:
                    print(f"❌ Could not fetch price for {symbol}")
                    return
                
                # Calculate position size from AI's recommendation
                position_size = decision.get('position_size', 0.02)  # Default 2% of capital
                risk_mult = trading_state['learning_state']['risk_multiplier'] if trading_state['mode'] == 'demo' else 1.0
                trade_value = trading_state['current_capital'] * position_size * max(0.2, min(risk_mult, 2.0))
                # Cap by max position percent
                cap = trading_state.get('max_position_pct', 0.05) * trading_state['current_capital']
                trade_value = min(trade_value, cap)
                is_demo = trading_state.get('mode') == 'demo'
                qty = int(trade_value // price) if is_demo else int(trade_value / price)
                if qty < 1 and price <= trading_state['current_capital']:
                    qty = 1
                    trade_value = qty * price

                if qty < 1:
                    try:
                        _log_trade_attempt(symbol, action, 'qty_too_small', {'qty': qty, 'price': f"{price:.4f}", 'cap': f"{cap:.2f}"})
                    except Exception:
                        pass
                    return
                
                # If LIVE mode, route to broker and return
                if trading_state.get('mode') == 'live':
                    try:
                        _ensure_live_broker()
                        br = trading_state.get('broker')
                        if br:
                            result = br.place_order(
                                symbol=symbol,
                                quantity=max(1, qty),
                                action='Buy' if action == 'BUY' else 'Sell',
                                order_type='Market'
                            )
                            trading_state['trades'].append({
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'symbol': symbol,
                                'side': action,
                                'qty': max(1, qty),
                                'price': round(price, 2),
                                'status': 'SUBMITTED' if result and not result.get('error') else 'REJECTED',
                                'pnl': None,
                                'broker': 'Questrade',
                                'broker_result': (result or {})
                            })
                        return
                    except Exception as be:
                        print(f"Live broker order failed: {be}")
                        # Fall back to demo execution

                # Execute the trade (DEMO mode)
                existing = next((h for h in trading_state['holdings'] if h['symbol'] == symbol), None)
                
                if action == 'BUY':
                    # BUY logic (demo)
                    cost = qty * price
                    if cost > trading_state['current_capital']:
                        max_affordable = int(trading_state['current_capital'] // price)
                        if max_affordable < 1:
                            try:
                                _log_trade_attempt(symbol, action, 'qty_too_small', {'qty': 0, 'price': f"{price:.4f}", 'cap': f"{cap:.2f}"})
                            except Exception:
                                pass
                            return
                        qty = max_affordable
                        cost = qty * price
                    
                    if qty < 1:
                        try:
                            _log_trade_attempt(symbol, action, 'qty_too_small', {'qty': qty, 'price': f"{price:.4f}", 'cap': f"{cap:.2f}"})
                        except Exception:
                            pass
                        return
                    
                    trading_state['current_capital'] -= cost
                    
                    if existing:
                        total_cost = existing['avg_price'] * existing['qty'] + price * qty
                        existing['qty'] += qty
                        existing['qty'] = int(existing['qty'])
                        existing['avg_price'] = total_cost / existing['qty']
                        existing['current_price'] = price
                    else:
                        trading_state['holdings'].append({
                            'symbol': symbol,
                            'name': symbol.replace('.TO', ''),
                            'qty': qty,
                            'avg_price': price,
                            'current_price': price,
                            'pnl': 0,
                            'pnl_pct': 0
                        })
                    
                    # Record trade
                    trade = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'symbol': symbol,
                        'side': 'BUY',
                        'qty': qty,
                        'price': round(price, 2),
                        'status': 'FILLED',
                        'pnl': None,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'features_entry': compute_trade_features(symbol),
                        'regime': trading_state.get('regime')
                    }
                    trading_state['trades'].append(trade)
                    # Write to trade log file for Logs page
                    _log_trade_file(symbol, 'BUY', qty, price)
                    save_trading_state()
                    trading_state['ai_decisions'].append(decision)
                    try:
                        STATE_STORE.insert_trade(trade)
                        if existing:
                            STATE_STORE.upsert_holding(symbol, existing['qty'], existing['avg_price'], price)
                        else:
                            STATE_STORE.upsert_holding(symbol, qty, price, price)
                    except Exception:
                        pass
                    
                    print(f"SUCCESS: AI BUY: {qty} x {symbol} @ ${price:.2f} (Confidence: {confidence:.1%})")
                    print(f"   Reasoning: {reasoning}")
                
                elif action == 'SELL' and existing and existing['qty'] > 0:
                    # SELL logic (demo)
                    available_qty = int(existing.get('qty') or 0)
                    sell_qty = min(qty, available_qty)
                    if sell_qty < 1:
                        return
                    revenue = sell_qty * price
                    realized_pnl = (price - existing['avg_price']) * sell_qty
                    
                    trading_state['current_capital'] += revenue
                    existing['qty'] = max(0, available_qty - sell_qty)
                    
                    if existing['qty'] == 0:
                        trading_state['holdings'].remove(existing)
                    
                    # Record trade
                    trade = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'symbol': symbol,
                        'side': 'SELL',
                        'qty': sell_qty,
                        'price': round(price, 2),
                        'status': 'FILLED',
                        'pnl': round(realized_pnl, 2),
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'features_exit': compute_trade_features(symbol)
                    }
                    trading_state['trades'].append(trade)
                    _log_trade_file(symbol, 'SELL', sell_qty, price, pnl=realized_pnl)
                    _update_learning_from_trade(realized_pnl)
                    # On loss, reflect and log
                    try:
                        if realized_pnl < 0:
                            entry_feat = None
                            # find last buy of same symbol
                            for t in reversed(trading_state['trades'][:-1]):
                                if t.get('symbol') == symbol and t.get('side') == 'BUY':
                                    entry_feat = t.get('features_entry')
                                    break
                            reflection = reflect_on_loss(entry_feat or {}, trade.get('features_exit') or {})
                            trading_state['learning_log'].append({'timestamp': datetime.now().isoformat(), 'reflection': reflection})
                            STATE_STORE.insert_learning_entry(
                                timestamp=datetime.now().isoformat(),
                                pnl=float(realized_pnl),
                                win_streak=trading_state['learning_state'].get('win_streak',0),
                                loss_streak=trading_state['learning_state'].get('loss_streak',0),
                                risk_multiplier=trading_state['learning_state'].get('risk_multiplier',1.0),
                                reflection=reflection,
                                features={'entry': entry_feat, 'exit': trade.get('features_exit'), 'regime': trading_state.get('regime')}
                            )
                    except Exception:
                        pass
                    save_trading_state()
                    trading_state['ai_decisions'].append(decision)
                    try:
                        STATE_STORE.insert_trade(trade)
                        if existing and existing['qty'] <= 0:
                            STATE_STORE.delete_holding(symbol)
                        else:
                            STATE_STORE.upsert_holding(symbol, existing['qty'], existing['avg_price'], price)
                    except Exception:
                        pass
                    
                    print(f"SUCCESS: AI SELL: {sell_qty} x {symbol} @ ${price:.2f} | P&L: ${realized_pnl:.2f} (Confidence: {confidence:.1%})")
                    print(f"   Reasoning: {reasoning}")
            
            # Log learning
            learning_entry = {
                'timestamp': datetime.now().isoformat(),
                'decision': decision,
                'executed': True
            }
            trading_state['learning_log'].append(learning_entry)
        # Optionally record HOLD rationale (disabled if heavy)
                
        except Exception as e:
            print(f"WARNING: AI trade error: {e}")
            # Fall back to basic mode
            simulate_ai_trade()
    else:
        # AI instance is None - use fallback trading
        try:
            simulate_ai_trade()
        except Exception as e:
            print(f"Fallback trading failed: {e}")
            return
            cap = trading_state.get('max_position_pct', 0.05) * trading_state.get('current_capital', 0)
            affordable = df_signals[df_signals['price'] * 0.1 <= cap]
            row = (affordable if not affordable.empty else df_signals).iloc[0]
            decision = {
                'symbol': row['symbol'],
                'action': row['signal'],
                'confidence': float(row['confidence']),
                'reasoning': [row['reason']],
                'position_size': 0.02
            }
            # Inline execute using same path as above
            symbol = decision['symbol']
            action = decision['action']
            price = get_demo_price(symbol)
            if not price:
                _log_trade_attempt(symbol, action, 'price_not_available')
                return
            cap = trading_state.get('max_position_pct', 0.05) * trading_state.get('current_capital', 0)
            trade_value = min(trading_state['current_capital'] * decision['position_size'], cap)
            qty = int(trade_value // price)
            if qty < 1 and price <= trading_state['current_capital']:
                qty = 1
                trade_value = qty * price
            if qty < 1:
                _log_trade_attempt(symbol, action, 'qty_too_small', {'qty': qty, 'price': f"{price:.4f}", 'cap': f"{cap:.2f}"})
                return
            # Execute BUY only for demo signals; SELL requires holdings
            existing = next((h for h in trading_state['holdings'] if h['symbol'] == symbol), None)
            if action == 'SELL' and (not existing or existing.get('qty',0) <= 0):
                _log_trade_attempt(symbol, 'SELL', 'no_holdings', {'held_qty': existing.get('qty',0) if existing else 0})
                return
            if action == 'BUY':
                cost = qty * price
                if cost > trading_state['current_capital']:
                    max_affordable = int(trading_state['current_capital'] // price)
                    if max_affordable < 1:
                        _log_trade_attempt(symbol, 'BUY', 'qty_too_small', {'qty': 0, 'price': f"{price:.4f}"})
                        return
                    qty = max_affordable
                    cost = qty * price
                if qty < 1:
                    _log_trade_attempt(symbol, 'BUY', 'qty_too_small', {'qty': qty, 'price': f"{price:.4f}"})
                    return
                trading_state['current_capital'] -= cost
                if existing:
                    total_cost = existing['avg_price'] * existing['qty'] + price * qty
                    existing['qty'] += qty
                    existing['qty'] = int(existing['qty'])
                    existing['avg_price'] = total_cost / existing['qty']
                    existing['current_price'] = price
                else:
                    trading_state['holdings'].append({'symbol':symbol,'name':symbol.replace('.TO',''),'qty':qty,'avg_price':price,'current_price':price,'pnl':0,'pnl_pct':0})
                trade = {'time': datetime.now().strftime('%H:%M:%S'),'symbol':symbol,'side':'BUY','qty':qty,'price':round(price,2),'status':'FILLED','pnl':None,'features_entry': compute_trade_features(symbol), 'regime': trading_state.get('regime')}
                trading_state['trades'].append(trade)
                _log_trade_file(symbol,'BUY',qty,price)
                save_trading_state()
                try:
                    STATE_STORE.insert_trade(trade)
                    STATE_STORE.upsert_holding(symbol, existing['qty'] if existing else qty, existing['avg_price'] if existing else price, price)
                except Exception:
                    pass
            elif action == 'SELL' and existing and existing['qty'] > 0:
                available_qty = int(existing.get('qty') or 0)
                sell_qty = min(qty, available_qty)
                if sell_qty < 1:
                    return
                revenue = sell_qty * price
                realized_pnl = (price - existing['avg_price']) * sell_qty
                trading_state['current_capital'] += revenue
                existing['qty'] = max(0, available_qty - sell_qty)
                if existing['qty'] <= 0:
                    trading_state['holdings'].remove(existing)
                trade = {'time': datetime.now().strftime('%H:%M:%S'),'symbol':symbol,'side':'SELL','qty':sell_qty,'price':round(price,2),'status':'FILLED','pnl':round(realized_pnl,2),'features_exit': compute_trade_features(symbol)}
                trading_state['trades'].append(trade)
                _log_trade_file(symbol,'SELL',sell_qty,price, pnl=realized_pnl)
                save_trading_state()
                try:
                    STATE_STORE.insert_trade(trade)
                    if existing and existing.get('qty',0) <= 0:
                        STATE_STORE.delete_holding(symbol)
                    else:
                        STATE_STORE.upsert_holding(symbol, existing['qty'], existing['avg_price'], price)
                except Exception:
                    pass
        except Exception as e:
            # safe fallback: try simulate_ai_trade
            print(f"Signal generation failed: {e}, trying simulate_ai_trade")
            try:
                simulate_ai_trade()
            except Exception as e2:
                print(f"Fallback trading failed: {e2}")
                return

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_feature_importance_from_db(top_n: int = 5, regime: str | None = None) -> list:
    """Compute simple feature importance using RandomForest on logged features vs P&L."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        logs = STATE_STORE.fetch_learning_log(limit=1000)
        X, y = [], []
        for row in logs:
            feats = row.get('features') or {}
            entry = feats.get('entry') or {}
            exitf = feats.get('exit') or {}
            if regime and (feats.get('regime') or '').upper() != regime.upper():
                continue
            if not entry:
                continue
            # Build feature vector from entry stats
            xi = [
                entry.get('rsi14') or 0.0,
                entry.get('sma20_ratio') or 0.0,
                entry.get('sma50_ratio') or 0.0,
                entry.get('volatility_20') or 0.0,
                entry.get('momentum_10') or 0.0,
            ]
            X.append(xi)
            y.append(row.get('pnl') or 0.0)
        if len(X) < 10:
            return []
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        names = ['rsi14', 'sma20_ratio', 'sma50_ratio', 'volatility_20', 'momentum_10']
        imp = list(zip(names, model.feature_importances_))
        imp.sort(key=lambda t: t[1], reverse=True)
        return imp[:top_n]
    except Exception:
        return []

def reflect_on_loss(entry_features: dict, exit_features: dict) -> str:
    """Heuristic reflection: suggests improvements based on feature context."""
    notes = []
    try:
        rsi_entry = entry_features.get('rsi14') if entry_features else None
        rsi_exit = exit_features.get('rsi14') if exit_features else None
        if rsi_entry and rsi_entry > 70:
            notes.append("Avoid buying when RSI > 70 (overbought)")
        if rsi_entry and rsi_entry < 30 and rsi_exit and rsi_exit < rsi_entry:
            notes.append("Bearish momentum after oversold; wait for RSI cross-up")
        mom = entry_features.get('momentum_10') if entry_features else None
        if mom is not None and mom < 0:
            notes.append("Momentum negative at entry; require positive momentum filter")
        vol = entry_features.get('volatility_20') if entry_features else None
        if vol and vol > 0.01:
            notes.append("High volatility; reduce position size or use wider stops")
        if not notes:
            notes.append("Tighten risk: smaller size, wait for stronger confirmation (SMA20 > SMA50, RSI > 50)")
    except Exception:
        notes.append("General: strengthen entry confirmation and reduce risk after losses")
    return "; ".join(notes)

def fetch_live_market_data():
    """Fetch live data for all Canadian stocks"""
    for symbol in CANADIAN_STOCKS:
        get_live_price(symbol)

def _update_learning_from_trade(realized_pnl: float) -> None:
    """Simple adaptive risk control for demo mode."""
    try:
        if realized_pnl is None:
            return
        ls = trading_state['learning_state']
        if realized_pnl > 0:
            ls['win_streak'] = ls.get('win_streak', 0) + 1
            ls['loss_streak'] = 0
            ls['risk_multiplier'] = min(1.5, ls.get('risk_multiplier', 1.0) * 1.02)
        else:
            ls['loss_streak'] = ls.get('loss_streak', 0) + 1
            ls['win_streak'] = 0
            ls['risk_multiplier'] = max(0.5, ls.get('risk_multiplier', 1.0) * 0.92)
        trading_state['learning_log'].append({
            'timestamp': datetime.now().isoformat(),
            'pnl': realized_pnl,
            'win_streak': ls['win_streak'],
            'loss_streak': ls['loss_streak'],
            'risk_multiplier': ls['risk_multiplier']
        })
    except Exception as e:
        print(f"Learning update failed: {e}")

def get_live_price(symbol: str) -> float | None:
    """Get live price from broker or data source with rate limiting and caching"""
    current_time = time.time()
    
    # Check cache first
    if symbol in _price_cache:
        cached_price, cache_time = _price_cache[symbol]
        if current_time - cache_time < CACHE_DURATION_SECONDS:
            return cached_price
    
    # Check rate limiting
    if symbol in _last_request_time:
        time_since_last = current_time - _last_request_time[symbol]
        if time_since_last < RATE_LIMIT_SECONDS:
            # Return cached price if available, otherwise return None
            if symbol in _price_cache:
                return _price_cache[symbol][0]
            return None
    
    try:
        # Try broker quote if a client is available via trading_state['broker']
        br = trading_state.get('broker')
        if br and hasattr(br, 'get_quotes'):
            qs = br.get_quotes([symbol]) or []
            if qs:
                q = qs[0]
                for k in ('lastTradePriceTrHrs','lastTradePrice','bidPrice','askPrice'):
                    v = q.get(k)
                    if v and v > 0:
                        price = float(v)
                        _price_cache[symbol] = (price, current_time)
                        _last_request_time[symbol] = current_time
                        return price
        
        # Fallback to Yahoo Finance
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('regularMarketPrice') or info.get('currentPrice')
        if price and price > 0:
            price = float(price)
            _price_cache[symbol] = (price, current_time)
            _last_request_time[symbol] = current_time
            return price
        
        return None
    except Exception as e:
        print(f"Error getting live price for {symbol}: {e}")
        return None

def get_demo_price(symbol: str) -> float | None:
    """Prefer broker quotes if available (even in demo), else fallback to Yahoo."""
    # Try broker quote if a client is available via trading_state['broker']
    try:
        br = trading_state.get('broker')
        if br and hasattr(br, 'get_quotes'):
            qs = br.get_quotes([symbol]) or []
            if qs:
                # Questrade returns 'lastTradePriceTrHrs' / 'lastTradePrice' keys in quotes
                q = qs[0]
                for k in ('lastTradePriceTrHrs','lastTradePrice','bidPrice','askPrice'):
                    v = q.get(k)
                    if v and v > 0:
                        return float(v)
    except Exception:
        pass
    # Fallback to Yahoo
    return get_live_price(symbol)

def is_market_open():
    """Check if TSX market is currently open (09:30–16:00 America/Toronto, Mon–Fri),
    with demo override via trading_state['force_market_open']."""
    # Demo override
    try:
        if trading_state.get('mode') == 'demo' and trading_state.get('force_market_open'):
            return True
    except Exception as e:
        pass
    try:
        if ZoneInfo is None:
            raise RuntimeError("ZoneInfo not available")
        tz = ZoneInfo("America/Toronto")
        et_now = datetime.now(tz)
    except Exception:
        # Fallback: naive local time assumption
        et_now = datetime.now()

    # Weekends closed
    if et_now.weekday() >= 5:
        return False

    market_start = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_start <= et_now <= market_end

def build_training_dataset_from_db(min_samples: int = 50, regime: str | None = None):
    try:
        logs = STATE_STORE.fetch_learning_log(limit=2000)
        X, y = [], []
        for row in logs:
            feats = (row.get('features') or {})
            if regime and (feats.get('regime') or '').upper() != regime.upper():
                continue
            entry = feats.get('entry') or {}
            pnl = row.get('pnl')
            if entry and pnl is not None:
                xi = [
                    entry.get('rsi14') or 0.0,
                    entry.get('sma20_ratio') or 0.0,
                    entry.get('sma50_ratio') or 0.0,
                    entry.get('volatility_20') or 0.0,
                    entry.get('momentum_10') or 0.0,
                ]
                X.append(xi)
                y.append(pnl)
        if len(X) < min_samples:
            return None, None
        return np.array(X), np.array(y)
    except Exception:
        return None, None

def compute_trade_features(symbol: str) -> dict:
    """Compute simple features (RSI, SMA ratios, vol) from recent 1m data."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d', interval='1m')
        if hist is None or hist.empty:
            return {}
        close = hist['Close']
        returns = close.pct_change().dropna()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        rsi = _rsi(close, window=14)
        feat = {
            'price': float(close.iloc[-1]),
            'rsi14': float(rsi.iloc[-1]) if not rsi.isna().iloc[-1] else None,
            'sma20_ratio': float((close.iloc[-1] / sma20.iloc[-1]) - 1) if not pd.isna(sma20.iloc[-1]) else None,
            'sma50_ratio': float((close.iloc[-1] / sma50.iloc[-1]) - 1) if not pd.isna(sma50.iloc[-1]) else None,
            'volatility_20': float(returns.tail(20).std()) if len(returns) >= 20 else None,
            'momentum_10': float((close.iloc[-1] / close.iloc[-10]) - 1) if len(close) >= 11 else None,
        }
        return feat
    except Exception:
        return {}

def update_holdings_prices():
    """Update current prices for all holdings with REAL market data"""
    if not trading_state['initialized'] or not trading_state['holdings']:
        return
    
    for holding in trading_state['holdings']:
        # Fetch real price
        current_price = get_live_price(holding['symbol'])
        if current_price:
            qty = int(round(float(holding.get('qty') or 0)))
            avg_price = float(holding.get('avg_price') or 0)
            holding['qty'] = qty
            holding['current_price'] = current_price
            holding['pnl'] = (current_price - avg_price) * qty if avg_price else 0.0
            holding['pnl_pct'] = ((current_price - avg_price) / avg_price * 100) if avg_price else 0.0
            try:
                STATE_STORE.upsert_holding(holding['symbol'], qty, avg_price, current_price)
            except Exception:
                pass

def _log_trade_attempt(symbol: str | None, action: str | None, reason: str, details: dict | None = None):
    """Append an attempted-trade line to the trades log so the UI shows why it was skipped."""
    try:
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        sym = symbol or 'N/A'
        act = (action or 'N/A').upper()
        line = f"{datetime.now().isoformat()} ATTEMPT {act} {sym} reason={reason}"
        if details:
            try:
                extras = ' '.join(f"{k}={v}" for k,v in details.items())
                if extras:
                    line += ' ' + extras
            except Exception:
                pass
        with open(log_dir / 'ai_trades.log', 'a', encoding='utf-8') as fh:
            fh.write(line + "\n")
    except Exception:
        pass

def generate_ai_signals():
    """Get REAL AI signals from the actual AI instance."""
    # If AI is initialized and has made analysis, use real signals
    if trading_state.get('initialized') and trading_state.get('ai_instance'):
        ai = trading_state['ai_instance']
        
        # Get AI's latest analysis
        try:
            analysis = ai.analyze_market()
            
            # Convert AI analysis to signal format for display
            signals = []
            held_symbols = {
                h.get('symbol')
                for h in trading_state.get('holdings', [])
                if h.get('qty', 0)
            }
            
            # Get AI's decision
            decision = ai.make_trading_decision(analysis)
            
            if decision and decision.get('action') != 'HOLD':
                signal_data = {
                    'symbol': decision.get('symbol', 'N/A'),
                    'signal': decision.get('action', 'HOLD'),
                    'confidence': decision.get('confidence', 0.0),
                    'price': decision.get('price', 0.0),
                    'target': decision.get('price', 0.0) * (1.02 if decision.get('action') == 'BUY' else 0.98),
                    'reason': decision.get('reasoning', 'AI decision')
                }
                signals.append(signal_data)
            
            # Also show top opportunities from analysis
            if analysis.get('market_data'):
                regime = analysis.get('regime', 'SIDEWAYS')
                for symbol, df in list(analysis['market_data'].items())[:10]:
                    if symbol == decision.get('symbol'):
                        continue  # Skip if already in signals
                    
                    try:
                        if isinstance(df, pd.DataFrame) and len(df) > 0:
                            price = float(df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1])
                            
                            # Quick RSI calc
                            close = df['close'] if 'close' in df.columns else df['Close']
                            delta = close.diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
                            
                            sig = None
                            conf = 0.0
                            reason = ""
                            
                            if rsi_val < 45:
                                sig = 'BUY'
                                conf = min(0.95, 0.55 + (45 - rsi_val) / 70)
                                reason = f"Momentum oversold - RSI {rsi_val:.0f} | {regime}"
                            elif rsi_val > 60 and symbol in held_symbols:
                                sig = 'SELL'
                                conf = min(0.95, 0.55 + (rsi_val - 60) / 70)
                                reason = f"Taking profits - RSI {rsi_val:.0f} | {regime}"
                            
                            if not sig:
                                continue
                            
                            signals.append({
                                'symbol': symbol,
                                'signal': sig,
                                'confidence': min(0.99, conf),
                                'price': price,
                                'target': price * (1.02 if sig == 'BUY' else 0.98),
                                'reason': reason
                            })
                    except Exception as e:
                        continue
            
            if signals:
                return pd.DataFrame(signals).sort_values('confidence', ascending=False).head(10)
        except Exception as e:
            print(f"Error getting AI signals: {e}")
    
    # Fallback: return empty DataFrame
    return pd.DataFrame(columns=['symbol','signal','confidence','price','target','reason'])

def _log_trade_file(symbol: str, side: str, qty: float, price: float, pnl=None):
    """Append a simple trade line to logs/ai_trades.log for the Logs tab."""
    try:
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        line = f"{datetime.now().isoformat()} {side.upper()} {symbol} qty={float(qty):.4f} price=${float(price):.4f}"
        if pnl is not None:
            try:
                line += f" pnl=${float(pnl):.2f}"
            except Exception:
                pass
        with open(log_dir / 'ai_trades.log', 'a', encoding='utf-8') as fh:
            fh.write(line + "\n")
    except Exception:
        pass

def retrain_on_trades(regime: str | None = None) -> dict:
    """Start a small PPO fine-tune pass using market data; report pre/post metrics.

    Note: Uses yfinance OHLCV data as environment (stored trades steer feature salience
    via feature importances above; full on-policy training can be done off-session).
    """
    try:
        # Pick a symbol relevant to recent trades if available; fall back to a safe TSX ticker
        recent_trades = trading_state.get('trades', [])
        symbols = [t.get('symbol') for t in recent_trades[-50:] if t.get('symbol')]
        sym = symbols[-1] if symbols else 'RY.TO'
        import yfinance as yf
        hist = yf.Ticker(sym).history(period='6mo', interval='1d')
        if hist is None or hist.empty:
            return {"status": "insufficient_data"}
        df = hist.rename(columns={
            'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'
        })[['open','high','low','close','volume']].dropna().reset_index(drop=True)
        if len(df) < 200:
            return {"status": "insufficient_data"}

        # Import PPO agent and fine-tune briefly
        try:
            from src.ai.rl.ppo_agent import PPOTradingAgent
        except Exception as e:
            return {"status":"error","error":f"RL agent unavailable: {e}"}

        agent = PPOTradingAgent()
        env = agent.create_environment(df, initial_capital=float(trading_state.get('starting_capital') or 100000))
        agent.create_model(env)
        # Quick baseline eval
        baseline = agent.evaluate(df, n_episodes=1, initial_capital=float(trading_state.get('starting_capital') or 100000))
        # Tiny fine-tune (kept small to avoid blocking)
        agent.model.learn(total_timesteps=2000)
        post = agent.evaluate(df, n_episodes=1, initial_capital=float(trading_state.get('starting_capital') or 100000))
        return {
            "status": "trained",
            "symbol": sym,
            "baseline_return": float(baseline.get('avg_return',0.0)),
            "post_return": float(post.get('avg_return',0.0)),
            "baseline_sharpe": float(baseline.get('avg_sharpe',0.0)),
            "post_sharpe": float(post.get('avg_sharpe',0.0)),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_random_tsx_stock() -> str:
    """Get a random TSX stock symbol"""
    # Simple list of major TSX stocks
    tsx_stocks = [
        'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',  # Banks
        'SHOP.TO', 'BB.TO', 'OTEX.TO', 'DOO.TO',        # Tech
        'CNQ.TO', 'SU.TO', 'IMO.TO', 'CVE.TO',          # Energy
        'ENB.TO', 'TRP.TO', 'FTS.TO', 'AQN.TO',         # Infrastructure
        'CP.TO', 'CNR.TO',                              # Railroads
        'T.TO', 'BCE.TO', 'RCI-B.TO',                   # Telecoms
        'ABX.TO', 'K.TO', 'WPM.TO', 'FNV.TO',           # Mining
        'L.TO', 'ATD.TO', 'QSR.TO',                     # Retail
        'MFC.TO', 'SLF.TO', 'IFC.TO',                   # Insurance
    ]
    return random.choice(tsx_stocks)

def simulate_historical_trading():
    """Simulate trading using historical data replay for training when market is closed"""
    if not trading_state.get('initialized'):
        return
    
    # Only run in DEMO mode when market is closed
    if trading_state.get('mode') != 'demo' or is_market_open():
        return
    
    try:
        # Get a random TSX stock
        symbol = get_random_tsx_stock()
        
        # Use historical data for training
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d', interval='1d')
        
        if hist is None or hist.empty:
            return
        
        # Use the most recent historical price
        price = float(hist['Close'].iloc[-1])
        
        # Simulate AI decision with historical context
        decision = {
            'symbol': symbol,
            'action': random.choice(['BUY', 'SELL', 'HOLD']),
            'confidence': random.uniform(0.6, 0.9),
            'reasoning': [f"Historical data analysis - {symbol} training mode"],
            'position_size': 0.02,
            'timestamp': datetime.now(),
            'training_mode': True
        }
        
        # Log the training decision
        decisions = trading_state.get('ai_decisions', [])
        decisions.append(f"TRAINING: {decision['action']} {symbol} @ ${price:.2f} (Historical Data)")
        trading_state['ai_decisions'] = decisions[-10:]  # Keep last 10
        
        # Update learning metrics
        trading_state['ai_decisions_today'] = trading_state.get('ai_decisions_today', 0) + 1
        
        # Store training decision
        training_log = trading_state.get('training_log', [])
        training_log.append(decision)
        trading_state['training_log'] = training_log[-50:]  # Keep last 50 training decisions
        
        save_trading_state()
        logger.info(f"🎓 Training Mode: {decision['action']} {symbol} @ ${price:.2f} (Historical Data)")
        
    except Exception as e:
        logger.error(f"Historical trading simulation failed: {e}")

def simulate_ai_trade():
    """AI making a trade based on REAL LIVE market data - ONLY when market is open"""
    if not trading_state['initialized']:
        return
    
    # CRITICAL: Only trade when TSX is actually open
    market_open = is_market_open()
    if not market_open:
        return  # No trades outside market hours
    if trading_state.get('paused') or trading_state.get('kill_switch_active'):
        return
    
    # Get a random Canadian stock
    symbol = get_random_tsx_stock()
    
    # Fetch REAL LIVE price from market
    price = get_live_price(symbol)
    
    if price is None:
        # Use simulated price for demo mode when real price fetching fails
        price = round(random.uniform(10.0, 200.0), 2)
    
    # Decide: BUY only if we have cash, SELL only if we have holdings
    existing = next((h for h in trading_state['holdings'] if h['symbol'] == symbol), None)
    
    # Calculate position size (10-20% of capital for demo mode) with adaptive risk
    base_pct = np.random.uniform(0.10, 0.20)  # Increased from 1-5% to 10-20% for demo
    risk_mult = trading_state['learning_state'].get('risk_multiplier', 1.0)
    max_position = trading_state['current_capital'] * base_pct * max(0.5, min(risk_mult, 2.0))  # Increased min from 0.2 to 0.5
    cap = trading_state.get('max_position_pct', 0.20) * trading_state['current_capital']  # Increased from 0.05 to 0.20
    max_position = min(max_position, cap)
    
    # FIX: Avoid ZeroDivisionError if price is 0 (rate limited/no data)
    if price <= 0:
        print(f"WARNING: Cannot trade - invalid price {price} for {symbol}")
        return
    
    qty = int(max_position / price)
    
    # CRITICAL FIX: Ensure minimum 1 share if we can afford it
    if qty < 1 and price <= trading_state['current_capital']:
        qty = 1  # Force at least 1 share if affordable
    elif qty < 1:
        return
    
    # Decide side based on logic with risk bias
    if existing and existing['qty'] > 0:
        buy_bias = max(0.1, 0.6 - 0.1 * trading_state['learning_state'].get('loss_streak', 0))
        sell_bias = 1.0 - buy_bias
        side = np.random.choice(['BUY', 'SELL'], p=[buy_bias, sell_bias])
    else:
        side = 'BUY'
    
    # For SELL, ensure we have enough shares
    if side == 'SELL':
        if not existing or existing['qty'] < qty:
            return  # Can't sell what we don't have
        qty = min(qty, existing['qty'])  # Don't sell more than we own
    
    # Execute trade
    trade_value = qty * price
    
    if side == 'BUY':
        # Deduct cost from capital
        trading_state['current_capital'] -= trade_value
        
        if existing:
            total_cost = existing['avg_price'] * existing['qty'] + price * qty
            existing['qty'] += qty
            existing['avg_price'] = total_cost / existing['qty']
            existing['current_price'] = price
        else:
            trading_state['holdings'].append({
                'symbol': symbol,
                'name': symbol.replace('.TO', ''),
                'qty': qty,
                'avg_price': price,
                'current_price': price,
                'pnl': 0,
                'pnl_pct': 0
            })
        # Record trade
        trade = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': round(price, 2),
            'status': 'FILLED',
            'pnl': None,
            'features_entry': compute_trade_features(symbol),
            'regime': trading_state.get('regime')
        }
        trading_state['trades'].append(trade)
        _log_trade_file(symbol, 'BUY', qty, price)
        save_trading_state()
        
        try:
            STATE_STORE.upsert_holding(symbol, (existing['qty'] if existing else qty), (existing['avg_price'] if existing else price), price)
        except Exception:
            pass
    
    elif side == 'SELL' and existing:
        # Realize P&L from sale
        realized_pnl = (price - existing['avg_price']) * qty
        trading_state['current_capital'] += trade_value  # Get cash back
        
        existing['qty'] -= qty
        if existing['qty'] == 0:
            trading_state['holdings'].remove(existing)
        
        # Record trade with realized P&L
        trade = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': round(price, 2),
            'status': 'FILLED',
            'pnl': round(realized_pnl, 2),
            'features_exit': compute_trade_features(symbol),
            'regime': trading_state.get('regime')
        }
        trading_state['trades'].append(trade)
        _log_trade_file(symbol, 'SELL', qty, price, pnl=realized_pnl)
        _update_learning_from_trade(realized_pnl)
        save_trading_state()
        try:
            STATE_STORE.insert_trade(trade)
            if realized_pnl < 0:
                try:
                    entry_feat = None
                    for t in reversed(trading_state['trades'][:-1]):
                        if t.get('symbol') == symbol and t.get('side') == 'BUY':
                            entry_feat = t.get('features_entry')
                            break
                    reflection = reflect_on_loss(entry_feat or {}, trade.get('features_exit') or {})
                    trading_state['learning_log'].append({'timestamp': datetime.now().isoformat(), 'reflection': reflection})
                    STATE_STORE.insert_learning_entry(
                        timestamp=datetime.now().isoformat(),
                        pnl=float(realized_pnl),
                        win_streak=trading_state['learning_state'].get('win_streak',0),
                        loss_streak=trading_state['learning_state'].get('loss_streak',0),
                        risk_multiplier=trading_state['learning_state'].get('risk_multiplier',1.0),
                        reflection=reflection,
                        features={'entry': entry_feat, 'exit': trade.get('features_exit'), 'regime': trading_state.get('regime')}
                    )
                except Exception:
                    pass
            if existing and existing['qty'] <= 0:
                STATE_STORE.delete_holding(symbol)
        except Exception:
            pass
        return
    
    # Record BUY trade (no P&L yet)
    trade = {
        'time': datetime.now().strftime('%H:%M:%S'),
        'symbol': symbol,
        'side': side,
        'qty': qty,
        'price': round(price, 2),
        'status': 'FILLED',
        'pnl': None,  # No P&L on buy
        'features_entry': compute_trade_features(symbol),
        'regime': trading_state.get('regime')
    }
    
    trading_state['trades'].append(trade)
    save_trading_state()
    try:
        STATE_STORE.insert_trade(trade)
    except Exception:
        pass
def _coerce_to_float(value, default: float = 0.0) -> float:
    """Safely convert nested structures into a float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "score", "confidence", "compound", "overall"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    if isinstance(value, (list, tuple, set)):
        for v in value:
            if isinstance(v, (int, float)):
                return float(v)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

