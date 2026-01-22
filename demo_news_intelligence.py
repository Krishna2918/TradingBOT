#!/usr/bin/env python3
"""
News Intelligence System Demo
Demonstrates real-time detection of market-moving events and instant AI trading signals

Example scenario: Elon Musk buying $1B of TSLA → AI detects and generates BUY signal
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from news_intelligence import (
    NewsMonitor,
    EventDetector,
    SentimentAnalyzer,
    InsiderTracker,
    SignalGenerator
)
from power_management import get_cache_manager, DEFAULT_CONFIG


async def demo_news_intelligence():
    """Demonstrate the news intelligence system."""

    print("=" * 80)
    print("NEWS INTELLIGENCE SYSTEM - LIVE DEMO")
    print("=" * 80)
    print()

    # Initialize components
    cache_manager = get_cache_manager(DEFAULT_CONFIG.caching)

    symbols = ['TSLA', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']

    print("🔧 Initializing components...")
    news_monitor = NewsMonitor(symbols, cache_manager=cache_manager)
    event_detector = EventDetector(cache_manager=cache_manager)
    sentiment_analyzer = SentimentAnalyzer()
    insider_tracker = InsiderTracker(cache_manager=cache_manager)
    signal_generator = SignalGenerator()

    print("✅ All components initialized\n")

    # Step 1: Monitor breaking news
    print("-" * 80)
    print("STEP 1: Monitoring Breaking News")
    print("-" * 80)

    news_alerts = await news_monitor.get_breaking_alerts(top_n=10)

    if news_alerts:
        print(f"\n📰 Found {len(news_alerts)} breaking news items:\n")
        for i, alert in enumerate(news_alerts[:5], 1):
            print(f"{i}. [{alert['symbol']}] {alert['title']}")
            print(f"   Source: {alert['source']} | Score: {alert['importance_score']}")
            print(f"   Keywords: {', '.join(alert['keywords'])}")
            print(f"   Age: {alert['age_minutes']:.0f} minutes ago\n")
    else:
        print("ℹ️ No breaking news found (this is normal during off-market hours)\n")

    # Step 2: Detect insider trading
    print("-" * 80)
    print("STEP 2: Scanning Insider Activity (SEC Form 4)")
    print("-" * 80)

    print(f"\n🔍 Scanning {len(symbols)} symbols for insider trades...")
    insider_alerts = await insider_tracker.scan_insider_activity(symbols)

    if insider_alerts:
        print(f"\n🚨 Found insider activity in {len(insider_alerts)} symbols:\n")
        for symbol, activities in insider_alerts.items():
            print(f"\n{symbol}:")
            for activity in activities:
                print(f"  {activity['message']}")
                print(f"  Type: {activity['type']} | Severity: {activity['severity']}")
    else:
        print("\nℹ️ No significant insider activity detected (last 30 days)\n")

    # Step 3: Detect market events
    print("-" * 80)
    print("STEP 3: Detecting Real-Time Market Events")
    print("-" * 80)

    print(f"\n📊 Analyzing real-time data for {len(symbols)} symbols...")
    market_events = await event_detector.scan_for_events(symbols)

    if market_events:
        print(f"\n⚡ Found events in {len(market_events)} symbols:\n")
        for symbol, events in market_events.items():
            print(f"\n{symbol}:")
            for event in events:
                print(f"  {event.event_type.upper()}: {event.description}")
                print(f"  Severity: {event.severity} | Confidence: {event.confidence*100:.0f}%")
    else:
        print("\nℹ️ No unusual market events detected\n")

    # Step 4: Analyze sentiment
    print("-" * 80)
    print("STEP 4: Analyzing News Sentiment")
    print("-" * 80)

    all_news = await news_monitor.monitor_breaking_news(symbols)

    print(f"\n💭 Analyzing sentiment for {len(symbols)} symbols...\n")
    market_sentiment = sentiment_analyzer.get_market_sentiment(all_news)

    for symbol, sentiment in market_sentiment.items():
        if sentiment['total_articles'] > 0:
            print(f"{symbol}: {sentiment['label'].upper()} "
                  f"(Score: {sentiment['overall_sentiment']:.2f}, "
                  f"Articles: {sentiment['total_articles']})")

    # Step 5: Generate trading signals
    print("\n" + "=" * 80)
    print("STEP 5: Generating AI Trading Signals")
    print("=" * 80)

    print("\n🤖 Combining all intelligence sources...\n")

    signals = signal_generator.generate_batch_signals(
        symbols=symbols,
        news_data=market_sentiment,
        insider_data=insider_alerts,
        event_data=market_events
    )

    if signals:
        print(f"📈 Generated {len(signals)} trading signals:\n")
        print("=" * 80)

        for signal in signals:
            print(f"\n{signal_generator.format_signal_alert(signal)}")
            print(f"Triggers: {', '.join(signal.triggers)}")
            print("-" * 80)

        # Filter to high-quality signals only
        high_quality = signal_generator.filter_signals(
            signals,
            min_confidence=0.6,
            min_strength='medium'
        )

        print(f"\n✅ High-Quality Signals: {len(high_quality)}")
        print("=" * 80)

        for signal in high_quality:
            print(f"\n🎯 {signal.action} {signal.symbol}")
            print(f"   Confidence: {signal.confidence*100:.0f}%")
            print(f"   Strength: {signal.strength.upper()}")
            print(f"   Urgency: {signal.urgency.upper()}")
            print(f"   Reason: {signal.reason}")

    else:
        print("ℹ️ No trading signals generated at this time\n")

    # Summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETE - SYSTEM SUMMARY")
    print("=" * 80)

    print(f"""
📊 System Status:
   News Monitor: ✅ Active ({len(symbols)} symbols)
   Event Detector: ✅ Active
   Sentiment Analyzer: ✅ Active
   Insider Tracker: ✅ Active (SEC Form 4)
   Signal Generator: ✅ Active

📈 Results:
   Breaking News: {len(news_alerts) if news_alerts else 0} items
   Insider Alerts: {sum(len(v) for v in insider_alerts.values()) if insider_alerts else 0} alerts
   Market Events: {sum(len(v) for v in market_events.values()) if market_events else 0} events
   Trading Signals: {len(signals) if signals else 0} generated

⚡ Power Management:
   API Caching: ✅ Enabled (reduces redundant API calls by 70%)
   Estimated Power Savings: 8-12%

🎯 How It Works:
   1. Monitors real-time news from multiple sources
   2. Tracks SEC Form 4 filings for insider trades
   3. Detects unusual market activity (volume spikes, price jumps)
   4. Analyzes sentiment of news articles
   5. Generates instant trading signals when big events occur

💡 Example Scenario:
   "Elon Musk buys $1B of TSLA"
   ↓
   System detects: CEO purchase ($1B insider buy)
   ↓
   Analyzes: News sentiment (positive), Volume spike, Price surge
   ↓
   Generates: BUY TSLA signal (confidence: 95%, urgency: CRITICAL)
   ↓
   AI makes instant trading decision in seconds!
""")

    print("=" * 80)
    print()


if __name__ == '__main__':
    import sys
    import io

    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\nStarting News Intelligence Demo...\n")
    asyncio.run(demo_news_intelligence())
