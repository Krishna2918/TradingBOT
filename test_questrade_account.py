"""
Test Questrade API Connection and Account Information
Verifies updated account access and retrieves current data
"""

import os
import sys
import json
from datetime import datetime
from src.data_pipeline.questrade_client import QuestradeClient

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_questrade_account():
    """Test Questrade API connection and retrieve account info"""
    
    print_section("QUESTRADE API CONNECTION TEST")
    
    # Check for refresh token
    refresh_token = os.getenv("QUESTRADE_REFRESH_TOKEN")
    if not refresh_token:
        print("ERROR: QUESTRADE_REFRESH_TOKEN environment variable not set")
        print("\nTo set it, run:")
        print('  $env:QUESTRADE_REFRESH_TOKEN = "your_refresh_token_here"')
        return
    
    print(f"Refresh Token: {refresh_token[:10]}...{refresh_token[-10:]} (masked)")
    print(f"Token Length: {len(refresh_token)} characters")
    
    # Initialize client (trading disabled for safety)
    print("\nInitializing Questrade client...")
    try:
        client = QuestradeClient(
            allow_trading=False,  # Safety: trading disabled
            practice_mode=True    # Safety: practice mode
        )
        print("SUCCESS: Client initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize client: {e}")
        return
    
    # Test 1: Authenticate and get server info
    print_section("TEST 1: AUTHENTICATION & SERVER INFO")
    try:
        client._ensure_authenticated()
        print(f"API Server: {client.api_server}")
        print(f"Access Token: {client.access_token[:10]}...{client.access_token[-10:] if client.access_token else 'None'} (masked)")
        print(f"Token Expiry: {client.token_expiry}")
        print("SUCCESS: Authentication successful")
    except Exception as e:
        print(f"ERROR: Authentication failed: {e}")
        return
    
    # Test 2: Get account information
    print_section("TEST 2: ACCOUNT INFORMATION")
    try:
        accounts = client.get_accounts()
        if accounts:
            print(f"Number of Accounts: {len(accounts)}")
            for idx, account in enumerate(accounts, 1):
                print(f"\nAccount {idx}:")
                print(f"  Account Number: {account.get('number', 'N/A')}")
                print(f"  Type: {account.get('type', 'N/A')}")
                print(f"  Status: {account.get('status', 'N/A')}")
                print(f"  Is Primary: {account.get('isPrimary', False)}")
                print(f"  Is Billing: {account.get('isBilling', False)}")
                print(f"  Client Account Type: {account.get('clientAccountType', 'N/A')}")
        else:
            print("WARNING: No accounts found")
    except Exception as e:
        print(f"ERROR: Failed to get accounts: {e}")
        return
    
    # Test 3: Get account balances
    print_section("TEST 3: ACCOUNT BALANCES")
    try:
        if client.account_id:
            balances = client.get_balances()
            if balances:
                combined = balances.get('combinedBalances', [{}])[0]
                per_currency = balances.get('perCurrencyBalances', [])
                
                print("Combined Balances:")
                print(f"  Currency: {combined.get('currency', 'N/A')}")
                print(f"  Cash: ${combined.get('cash', 0):,.2f}")
                print(f"  Market Value: ${combined.get('marketValue', 0):,.2f}")
                print(f"  Total Equity: ${combined.get('totalEquity', 0):,.2f}")
                print(f"  Buying Power: ${combined.get('buyingPower', 0):,.2f}")
                print(f"  Maintenance Excess: ${combined.get('maintenanceExcess', 0):,.2f}")
                
                if per_currency:
                    print("\nPer Currency Balances:")
                    for currency_balance in per_currency:
                        currency = currency_balance.get('currency', 'N/A')
                        cash = currency_balance.get('cash', 0)
                        market_value = currency_balance.get('marketValue', 0)
                        print(f"  {currency}: Cash ${cash:,.2f}, Market Value ${market_value:,.2f}")
            else:
                print("WARNING: No balance data available")
        else:
            print("WARNING: No account ID available")
    except Exception as e:
        print(f"ERROR: Failed to get balances: {e}")
    
    # Test 4: Get positions
    print_section("TEST 4: CURRENT POSITIONS")
    try:
        if client.account_id:
            positions = client.get_positions()
            if positions:
                print(f"Number of Positions: {len(positions)}")
                for idx, position in enumerate(positions, 1):
                    symbol = position.get('symbol', 'N/A')
                    quantity = position.get('openQuantity', 0)
                    avg_price = position.get('averageEntryPrice', 0)
                    current_price = position.get('currentPrice', 0)
                    market_value = position.get('currentMarketValue', 0)
                    pnl = position.get('openPnl', 0)
                    
                    print(f"\nPosition {idx}: {symbol}")
                    print(f"  Quantity: {quantity}")
                    print(f"  Avg Entry Price: ${avg_price:,.2f}")
                    print(f"  Current Price: ${current_price:,.2f}")
                    print(f"  Market Value: ${market_value:,.2f}")
                    print(f"  Open P&L: ${pnl:,.2f}")
            else:
                print("No open positions")
        else:
            print("WARNING: No account ID available")
    except Exception as e:
        print(f"ERROR: Failed to get positions: {e}")
    
    # Test 5: Get market quotes for TSX stocks
    print_section("TEST 5: MARKET QUOTES (TSX)")
    try:
        test_symbols = ['TD', 'RY', 'SHOP']
        quotes = client.get_quotes(test_symbols)
        if quotes:
            print(f"Retrieved quotes for {len(quotes)} symbols:")
            for quote in quotes:
                symbol = quote.get('symbol', 'N/A')
                last_price = quote.get('lastTradePrice', 0)
                bid = quote.get('bidPrice', 0)
                ask = quote.get('askPrice', 0)
                volume = quote.get('volume', 0)
                
                print(f"\n{symbol}:")
                print(f"  Last Price: ${last_price:,.2f}")
                print(f"  Bid: ${bid:,.2f}")
                print(f"  Ask: ${ask:,.2f}")
                print(f"  Volume: {volume:,}")
        else:
            print("WARNING: No quote data available")
    except Exception as e:
        print(f"ERROR: Failed to get quotes: {e}")
    
    # Test 6: Verify trading is disabled
    print_section("TEST 6: TRADING CONTROLS VERIFICATION")
    print(f"Trading Allowed: {client.allow_trading}")
    print(f"Practice Mode: {client.practice_mode}")
    
    if not client.allow_trading:
        print("\nSUCCESS: Trading is disabled (safety control active)")
        print("To enable: Set QUESTRADE_ALLOW_TRADING=true")
    else:
        print("\nWARNING: Trading is enabled!")
    
    # Summary
    print_section("SUMMARY")
    print("Connection Status: SUCCESS")
    print(f"API Server: {client.api_server}")
    print(f"Account ID: {client.account_id or 'Not set'}")
    print(f"Trading Enabled: {client.allow_trading}")
    print(f"Practice Mode: {client.practice_mode}")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*60)
    print("  TEST COMPLETE - Your Questrade API is connected!")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        test_questrade_account()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

