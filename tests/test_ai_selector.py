#!/usr/bin/env python3
"""
Test AI Selector with Ollama Integration
"""

from dotenv import load_dotenv
load_dotenv(".env")

from src.agents.ai_selector import AISelectorAgent

print("Testing AI Selector with Ollama...")

try:
    agent = AISelectorAgent()
    results = agent.run(limit=5)  # smaller to avoid API throttles
    print("Top 5:", [r.symbol for r in results[:5]])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
