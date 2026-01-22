#!/usr/bin/env python3
"""
Test Maximum Power AI Engine
"""
import sys
import asyncio
import logging
import time
import psutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_maximum_power_ai():
    """Test the maximum power AI engine"""
    try:
        logger.info("🚀 Testing Maximum Power AI Engine...")
        
        # Test 1: Initialize Maximum Power AI Engine
        logger.info("Test 1: Initializing Maximum Power AI Engine...")
        from src.dashboard.maximum_power_ai_engine import max_power_ai_engine
        
        if max_power_ai_engine.initialize():
            logger.info("✅ Maximum Power AI Engine initialized successfully")
        else:
            logger.error("❌ Maximum Power AI Engine initialization failed")
            return False
        
        # Test 2: Check initial resource usage
        logger.info("Test 2: Checking initial resource usage...")
        initial_cpu = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory().percent
        
        logger.info(f"Initial CPU usage: {initial_cpu:.1f}%")
        logger.info(f"Initial Memory usage: {initial_memory:.1f}%")
        
        # Test 3: Run one maximum power cycle
        logger.info("Test 3: Running one maximum power cycle...")
        start_time = time.time()
        
        await max_power_ai_engine.execute_maximum_power_cycle()
        
        cycle_time = time.time() - start_time
        logger.info(f"Maximum power cycle completed in {cycle_time:.2f} seconds")
        
        # Test 4: Check resource usage after cycle
        logger.info("Test 4: Checking resource usage after cycle...")
        final_cpu = psutil.cpu_percent(interval=1)
        final_memory = psutil.virtual_memory().percent
        
        logger.info(f"Final CPU usage: {final_cpu:.1f}%")
        logger.info(f"Final Memory usage: {final_memory:.1f}%")
        
        # Test 5: Run multiple cycles to see sustained usage
        logger.info("Test 5: Running 3 cycles to test sustained resource usage...")
        
        for i in range(3):
            logger.info(f"Running cycle {i+1}/3...")
            cycle_start = time.time()
            
            await max_power_ai_engine.execute_maximum_power_cycle()
            
            cycle_time = time.time() - cycle_start
            current_cpu = psutil.cpu_percent(interval=0.5)
            current_memory = psutil.virtual_memory().percent
            
            logger.info(f"  Cycle {i+1} completed in {cycle_time:.2f}s")
            logger.info(f"  CPU: {current_cpu:.1f}%, Memory: {current_memory:.1f}%")
            
            # Small delay between cycles
            await asyncio.sleep(1)
        
        # Test 6: Final resource check
        logger.info("Test 6: Final resource usage check...")
        final_cpu = psutil.cpu_percent(interval=2)
        final_memory = psutil.virtual_memory().percent
        
        logger.info(f"Final CPU usage: {final_cpu:.1f}%")
        logger.info(f"Final Memory usage: {final_memory:.1f}%")
        
        # Determine success
        if final_cpu >= 50:  # At least 50% CPU usage
            logger.info("✅ SUCCESS: Maximum Power AI Engine is using significant resources!")
            logger.info(f"   CPU Usage: {final_cpu:.1f}% (Target: 80%+)")
            logger.info(f"   Memory Usage: {final_memory:.1f}% (Target: 70%+)")
            return True
        else:
            logger.warning("⚠️ WARNING: AI Engine is not using enough resources")
            logger.warning(f"   CPU Usage: {final_cpu:.1f}% (Target: 80%+)")
            logger.warning(f"   Memory Usage: {final_memory:.1f}% (Target: 70%+)")
            return False
        
    except Exception as e:
        logger.error(f"Maximum Power AI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_maximum_power_ai())
    if success:
        print("\n🚀 SUCCESS: Maximum Power AI Engine is working at full capacity!")
        print("The AI is now consuming significant system resources!")
    else:
        print("\n⚠️ WARNING: Maximum Power AI Engine needs optimization")
        print("Resource usage is below target levels")
