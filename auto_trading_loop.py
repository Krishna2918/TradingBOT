#!/usr/bin/env python3
"""
Automated Trading Loop
Runs continuous trading cycles during market hours
"""
import sys
sys.path.insert(0, '.')

import time
import logging
from datetime import datetime
from src.demo.demo_trading_engine import DemoTradingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def is_market_hours():
    """Check if within TSX market hours (9:30 AM - 4:00 PM EST, Mon-Fri)"""
    now = datetime.now()
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    return market_open <= now <= market_close

def main():
    # Initialize engine
    engine = DemoTradingEngine()

    print("=" * 60)
    print("  AUTOMATED PAPER TRADING - CANADIAN MARKETS")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Starting Capital: ${engine.account.starting_capital:,.2f} CAD")
    print(f"  Tracking: {len(engine.symbols)} TSX stocks")
    print(f"  Cycle Interval: 30 seconds")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    cycle_count = 0
    cycle_interval = 30  # seconds between cycles

    try:
        while True:
            cycle_count += 1
            now = datetime.now()

            # Check market hours (optional - can trade anytime for demo)
            market_status = "OPEN" if is_market_hours() else "CLOSED"

            print(f"\n{'='*60}")
            print(f"CYCLE {cycle_count} | {now.strftime('%H:%M:%S')} | Market: {market_status}")
            print(f"{'='*60}")

            # Run trading cycle
            engine.run_demo_cycle()

            # Get current state
            prices = engine.current_prices or {}
            summary = engine.account.get_summary(prices)

            # Display status
            pnl_color = "+" if summary['total_pnl'] >= 0 else ""
            print(f"\n  Portfolio: ${summary['total_value']:,.2f}")
            print(f"  Cash:      ${summary['cash']:,.2f}")
            print(f"  Invested:  ${summary['positions_value']:,.2f}")
            print(f"  P&L:       {pnl_color}${summary['total_pnl']:,.2f} ({pnl_color}{summary['total_return_pct']:.2f}%)")
            print(f"  Positions: {summary['num_positions']}")
            print(f"  Trades:    {summary['num_trades']}")

            # Show recent trade if any new ones
            if engine.account.trade_history:
                last_trade = engine.account.trade_history[-1]
                # Check if trade was in this cycle (within last cycle_interval seconds)
                trade_age = (now - last_trade['timestamp']).total_seconds()
                if trade_age < cycle_interval:
                    pnl_str = f" | P&L: ${last_trade['pnl']:+,.2f}" if last_trade.get('pnl') else ""
                    print(f"\n  >> NEW TRADE: {last_trade['side']} {last_trade['quantity']} {last_trade['symbol']} @ ${last_trade['price']:.2f}{pnl_str}")

            # Show positions summary
            if engine.account.positions:
                print(f"\n  Open Positions:")
                for symbol, pos in list(engine.account.positions.items())[:5]:
                    current_price = prices.get(symbol, pos['avg_price'])
                    pnl = (current_price - pos['avg_price']) * pos['quantity']
                    print(f"    {symbol}: {pos['quantity']} @ ${current_price:.2f} | P&L: ${pnl:+,.2f}")
                if len(engine.account.positions) > 5:
                    print(f"    ... and {len(engine.account.positions) - 5} more")

            print(f"\n  Next cycle in {cycle_interval} seconds...")
            time.sleep(cycle_interval)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("TRADING STOPPED")
        print("=" * 60)

        # Final summary
        prices = engine.current_prices or {}
        final = engine.account.get_summary(prices)

        print(f"\nFinal Results:")
        print(f"  Total Cycles: {cycle_count}")
        print(f"  Final Portfolio: ${final['total_value']:,.2f}")
        print(f"  Total P&L: ${final['total_pnl']:+,.2f} ({final['total_return_pct']:+.2f}%)")
        print(f"  Total Trades: {final['num_trades']}")
        print(f"  Open Positions: {final['num_positions']}")
        print("=" * 60)

if __name__ == '__main__':
    main()
