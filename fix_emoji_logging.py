#!/usr/bin/env python3
"""
Fix emoji logging issues in autonomous_trading_ai.py
"""

import re

def fix_emoji_logging():
    """Remove emojis from logging statements"""
    
    # Read the file
    with open('src/ai/autonomous_trading_ai.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove emojis from logger statements
    emoji_patterns = [
        (r'logger\.info\("🤖 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("📊 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("✅ ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("🎯 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("💰 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("🧠 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("📅 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("🔍 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("🎲 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("📈 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("💵 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("📂 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("🏗️ ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("📦 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("💎 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("🚀 ([^"]*)"', r'logger.info("\1"'),
        (r'logger\.info\("⚠️ ([^"]*)"', r'logger.warning("\1"'),
        (r'logger\.error\("❌ ([^"]*)"', r'logger.error("\1"'),
        (r'logger\.warning\("⚠️ ([^"]*)"', r'logger.warning("\1"'),
    ]
    
    # Apply all patterns
    for pattern, replacement in emoji_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write back to file
    with open('src/ai/autonomous_trading_ai.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed emoji logging in autonomous_trading_ai.py")

if __name__ == "__main__":
    fix_emoji_logging()
