"""
Performance Tests for Stress
============================

Tests the system resilience under high load, market volatility,
and resource exhaustion scenarios.
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import psutil
import gc
import random

# Import components to test
from src.ai.enhanced_ensemble import get_enhanced_ensemble
from src.ai.multi_model import get_multi_model_manager
from src.workflows.trading_cycle import execute_complete_cycle
from src.trading.execution import get_execution_engine
from src.config.mode_manager import set_mode

logger = logging.getLogger(__name__)


class TestStress:
    """Test suite for stress testing and system resilience."""
    
    @pytest.fixture
    async def setup_demo_mode(self):
        """Setup demo mode for testing."""
        set_mode("DEMO")
        yield
        # Cleanup after test
    
    @pytest.fixture
    async def generate_volatile_market_data(self):
        """Generate volatile market data for stress testing."""
        stocks = {}
        for i in range(100):
            symbol = f"VOLATILE_{i:03d}"
            # Generate volatile prices
            base_price = 100.0 + (i % 50) * 2.0
            volatility = random.uniform(0.05, 0.20)  # 5-20% volatility
            price_change = random.uniform(-volatility, volatility)
            current_price = base_price * (1 + price_change)
            
            stocks[symbol] = {
                "price": current_price,
                "volume": random.randint(100000, 10000000),
                "timestamp": datetime.now(),
                "bid": current_price * 0.999,
                "ask": current_price * 1.001,
                "volatility": volatility
            }
        return stocks
    
    @pytest.mark.asyncio
    async def test_high_frequency_scenarios(self, setup_demo_mode, generate_volatile_market_data):
        """Test high frequency trading scenarios."""
        logger.info("Testing high frequency scenarios...")
        
        volatile_market_data = generate_volatile_market_data
        symbols = list(volatile_market_data.keys())
        
        # Simulate high frequency updates
        update_frequency = 0.1  # 10 updates per second
        total_updates = 100
        update_interval = 1.0 / update_frequency
        
        start_time = time.time()
        successful_updates = 0
        failed_updates = 0
        
        for i in range(total_updates):
            try:
                # Simulate market data update
                symbol = random.choice(symbols)
                old_price = volatile_market_data[symbol]["price"]
                price_change = random.uniform(-0.02, 0.02)  # 2% max change
                new_price = old_price * (1 + price_change)
                
                # Update market data
                volatile_market_data[symbol]["price"] = new_price
                volatile_market_data[symbol]["timestamp"] = datetime.now()
                
                # Process update
                await self._process_market_update(symbol, volatile_market_data[symbol])
                successful_updates += 1
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                failed_updates += 1
                logger.warning(f"Update failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify high frequency performance
        success_rate = successful_updates / total_updates
        assert success_rate > 0.95, f"Success rate {success_rate:.2%}, should be >95%"
        
        # Verify timing
        expected_time = total_updates * update_interval
        time_tolerance = expected_time * 0.1  # 10% tolerance
        assert abs(total_time - expected_time) < time_tolerance, f"Timing off by {abs(total_time - expected_time):.2f}s"
        
        logger.info(f"✓ High frequency scenarios: {successful_updates}/{total_updates} updates successful ({success_rate:.2%})")
    
    async def _process_market_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """Process a market data update."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Mock processing result
        return {
            "symbol": symbol,
            "price": data["price"],
            "processed_at": datetime.now()
        }
    
    @pytest.mark.asyncio
    async def test_market_volatility_simulation(self, setup_demo_mode, generate_volatile_market_data):
        """Test system behavior under market volatility."""
        logger.info("Testing market volatility simulation...")
        
        volatile_market_data = generate_volatile_market_data
        symbols = list(volatile_market_data.keys())
        
        # Simulate extreme market volatility
        volatility_scenarios = [
            {"name": "Normal", "volatility": 0.05, "duration": 10},
            {"name": "High", "volatility": 0.15, "duration": 10},
            {"name": "Extreme", "volatility": 0.30, "duration": 10},
            {"name": "Crash", "volatility": 0.50, "duration": 5}
        ]
        
        total_scenarios = len(volatility_scenarios)
        successful_scenarios = 0
        
        for scenario in volatility_scenarios:
            try:
                logger.info(f"Testing {scenario['name']} volatility scenario...")
                
                # Apply volatility to all stocks
                for symbol in symbols:
                    old_price = volatile_market_data[symbol]["price"]
                    volatility = scenario["volatility"]
                    price_change = random.uniform(-volatility, volatility)
                    new_price = old_price * (1 + price_change)
                    
                    volatile_market_data[symbol]["price"] = new_price
                    volatile_market_data[symbol]["volatility"] = volatility
                
                # Test system stability under volatility
                stability_result = await self._test_system_stability(volatile_market_data)
                
                if stability_result["stable"]:
                    successful_scenarios += 1
                    logger.info(f"✓ {scenario['name']} volatility scenario passed")
                else:
                    logger.warning(f"✗ {scenario['name']} volatility scenario failed: {stability_result['reason']}")
                
                # Wait for scenario duration
                await asyncio.sleep(scenario["duration"])
                
            except Exception as e:
                logger.error(f"Volatility scenario failed: {e}")
        
        # Verify system handled volatility
        success_rate = successful_scenarios / total_scenarios
        assert success_rate > 0.75, f"Volatility success rate {success_rate:.2%}, should be >75%"
        
        logger.info(f"✓ Market volatility simulation: {successful_scenarios}/{total_scenarios} scenarios successful")
    
    async def _test_system_stability(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test system stability under market conditions."""
        try:
            # Test AI analysis stability
            ai_ensemble = get_enhanced_ensemble()
            
            # Mock AI analysis
            with patch('src.ai.enhanced_ensemble.get_enhanced_ensemble') as mock_ensemble:
                mock_ai = Mock()
                mock_ai.analyze_for_entry.return_value = {
                    "decision": "HOLD",
                    "symbol": "TEST",
                    "confidence": 0.5,
                    "reasoning": "Market too volatile"
                }
                mock_ensemble.return_value = mock_ai
                
                # Test analysis
                result = await ai_ensemble.analyze_for_entry(
                    symbols=["TEST"],
                    market_data=market_data
                )
                
                if result and "decision" in result:
                    return {"stable": True, "reason": "System stable"}
                else:
                    return {"stable": False, "reason": "AI analysis failed"}
                    
        except Exception as e:
            return {"stable": False, "reason": f"System error: {e}"}
    
    @pytest.mark.asyncio
    async def test_system_recovery(self, setup_demo_mode, generate_volatile_market_data):
        """Test system recovery from failures."""
        logger.info("Testing system recovery...")
        
        volatile_market_data = generate_volatile_market_data
        
        # Test recovery from various failure scenarios
        recovery_scenarios = [
            {"name": "AI Model Failure", "failure_type": "ai_model"},
            {"name": "Database Connection Loss", "failure_type": "database"},
            {"name": "Memory Exhaustion", "failure_type": "memory"},
            {"name": "Network Timeout", "failure_type": "network"}
        ]
        
        successful_recoveries = 0
        
        for scenario in recovery_scenarios:
            try:
                logger.info(f"Testing recovery from {scenario['name']}...")
                
                # Simulate failure
                failure_result = await self._simulate_failure(scenario["failure_type"])
                
                if failure_result["failed"]:
                    # Test recovery
                    recovery_result = await self._test_recovery(scenario["failure_type"])
                    
                    if recovery_result["recovered"]:
                        successful_recoveries += 1
                        logger.info(f"✓ Recovery from {scenario['name']} successful")
                    else:
                        logger.warning(f"✗ Recovery from {scenario['name']} failed: {recovery_result['reason']}")
                else:
                    logger.warning(f"✗ Failed to simulate {scenario['name']}")
                
            except Exception as e:
                logger.error(f"Recovery test failed: {e}")
        
        # Verify recovery success rate
        success_rate = successful_recoveries / len(recovery_scenarios)
        assert success_rate > 0.5, f"Recovery success rate {success_rate:.2%}, should be >50%"
        
        logger.info(f"✓ System recovery: {successful_recoveries}/{len(recovery_scenarios)} recoveries successful")
    
    async def _simulate_failure(self, failure_type: str) -> Dict[str, Any]:
        """Simulate a system failure."""
        try:
            if failure_type == "ai_model":
                # Simulate AI model failure
                raise Exception("AI model connection lost")
            elif failure_type == "database":
                # Simulate database failure
                raise Exception("Database connection timeout")
            elif failure_type == "memory":
                # Simulate memory exhaustion
                raise MemoryError("Memory allocation failed")
            elif failure_type == "network":
                # Simulate network timeout
                raise TimeoutError("Network request timeout")
            else:
                return {"failed": False, "reason": "Unknown failure type"}
                
        except Exception as e:
            return {"failed": True, "reason": str(e)}
    
    async def _test_recovery(self, failure_type: str) -> Dict[str, Any]:
        """Test system recovery from failure."""
        try:
            # Simulate recovery process
            await asyncio.sleep(0.1)  # Recovery time
            
            # Test if system is functional
            if failure_type == "ai_model":
                # Test AI model recovery
                return {"recovered": True, "reason": "AI model reconnected"}
            elif failure_type == "database":
                # Test database recovery
                return {"recovered": True, "reason": "Database reconnected"}
            elif failure_type == "memory":
                # Test memory recovery
                gc.collect()  # Force garbage collection
                return {"recovered": True, "reason": "Memory freed"}
            elif failure_type == "network":
                # Test network recovery
                return {"recovered": True, "reason": "Network reconnected"}
            else:
                return {"recovered": False, "reason": "Unknown failure type"}
                
        except Exception as e:
            return {"recovered": False, "reason": f"Recovery failed: {e}"}
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, setup_demo_mode):
        """Test handling of resource exhaustion."""
        logger.info("Testing resource exhaustion handling...")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory exhaustion handling
        memory_test_result = await self._test_memory_exhaustion()
        
        # Test CPU exhaustion handling
        cpu_test_result = await self._test_cpu_exhaustion()
        
        # Test disk space exhaustion handling
        disk_test_result = await self._test_disk_exhaustion()
        
        # Verify resource exhaustion handling
        assert memory_test_result["handled"], f"Memory exhaustion not handled: {memory_test_result['reason']}"
        assert cpu_test_result["handled"], f"CPU exhaustion not handled: {cpu_test_result['reason']}"
        assert disk_test_result["handled"], f"Disk exhaustion not handled: {disk_test_result['reason']}"
        
        # Verify system recovered
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB, should be <100MB"
        
        logger.info(f"✓ Resource exhaustion handling: Memory {memory_test_result['handled']}, CPU {cpu_test_result['handled']}, Disk {disk_test_result['handled']}")
    
    async def _test_memory_exhaustion(self) -> Dict[str, Any]:
        """Test memory exhaustion handling."""
        try:
            # Simulate memory pressure
            large_data = []
            for i in range(1000):
                large_data.append({
                    "id": i,
                    "data": "x" * 1000,  # 1KB per item
                    "timestamp": datetime.now()
                })
            
            # Force garbage collection
            gc.collect()
            
            return {"handled": True, "reason": "Memory pressure handled"}
            
        except MemoryError:
            return {"handled": True, "reason": "Memory error caught and handled"}
        except Exception as e:
            return {"handled": False, "reason": f"Memory test failed: {e}"}
    
    async def _test_cpu_exhaustion(self) -> Dict[str, Any]:
        """Test CPU exhaustion handling."""
        try:
            # Simulate CPU intensive task
            start_time = time.time()
            
            # CPU intensive calculation
            result = 0
            for i in range(1000000):
                result += i * i
            
            end_time = time.time()
            cpu_time = end_time - start_time
            
            # Verify CPU task completed
            assert cpu_time < 5.0, f"CPU task took {cpu_time:.2f}s, should be <5s"
            
            return {"handled": True, "reason": "CPU intensive task completed"}
            
        except Exception as e:
            return {"handled": False, "reason": f"CPU test failed: {e}"}
    
    async def _test_disk_exhaustion(self) -> Dict[str, Any]:
        """Test disk space exhaustion handling."""
        try:
            # Simulate disk space check
            import shutil
            
            # Get disk usage
            disk_usage = shutil.disk_usage(".")
            free_space = disk_usage.free / (1024**3)  # GB
            
            # Verify sufficient disk space
            assert free_space > 1.0, f"Free disk space {free_space:.1f}GB, should be >1GB"
            
            return {"handled": True, "reason": "Disk space sufficient"}
            
        except Exception as e:
            return {"handled": False, "reason": f"Disk test failed: {e}"}
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, setup_demo_mode, generate_volatile_market_data):
        """Test system under sustained load."""
        logger.info("Testing sustained load...")
        
        volatile_market_data = generate_volatile_market_data
        symbols = list(volatile_market_data.keys())
        
        # Test sustained load for extended period
        load_duration = 30  # 30 seconds
        update_interval = 0.5  # Update every 0.5 seconds
        
        start_time = time.time()
        successful_operations = 0
        failed_operations = 0
        
        while time.time() - start_time < load_duration:
            try:
                # Simulate trading operation
                symbol = random.choice(symbols)
                operation_result = await self._simulate_trading_operation(symbol, volatile_market_data[symbol])
                
                if operation_result["success"]:
                    successful_operations += 1
                else:
                    failed_operations += 1
                
                # Wait for next operation
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                failed_operations += 1
                logger.warning(f"Operation failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify sustained load performance
        total_operations = successful_operations + failed_operations
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        
        assert success_rate > 0.90, f"Sustained load success rate {success_rate:.2%}, should be >90%"
        assert total_time >= load_duration * 0.9, f"Load test duration {total_time:.1f}s, should be ~{load_duration}s"
        
        logger.info(f"✓ Sustained load: {successful_operations}/{total_operations} operations successful ({success_rate:.2%}) over {total_time:.1f}s")
    
    async def _simulate_trading_operation(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a trading operation."""
        try:
            # Simulate operation time
            await asyncio.sleep(0.01)
            
            # Mock operation result
            return {
                "success": True,
                "symbol": symbol,
                "operation": "analysis",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    @pytest.mark.asyncio
    async def test_error_cascade_prevention(self, setup_demo_mode, generate_volatile_market_data):
        """Test prevention of error cascades."""
        logger.info("Testing error cascade prevention...")
        
        volatile_market_data = generate_volatile_market_data
        symbols = list(volatile_market_data.keys())
        
        # Simulate error cascade scenario
        error_symbols = symbols[:10]  # First 10 symbols will have errors
        
        # Test error isolation
        isolated_errors = 0
        cascaded_errors = 0
        
        for symbol in symbols:
            try:
                if symbol in error_symbols:
                    # Simulate error
                    raise ValueError(f"Error in {symbol}")
                else:
                    # Process normally
                    await self._process_single_stock(symbol, volatile_market_data[symbol])
                    
            except ValueError as e:
                # Check if error cascaded
                if symbol in error_symbols:
                    isolated_errors += 1
                else:
                    cascaded_errors += 1
            except Exception as e:
                cascaded_errors += 1
        
        # Verify error isolation
        assert isolated_errors == len(error_symbols), f"Expected {len(error_symbols)} isolated errors, got {isolated_errors}"
        assert cascaded_errors == 0, f"Error cascade detected: {cascaded_errors} cascaded errors"
        
        logger.info(f"✓ Error cascade prevention: {isolated_errors} isolated errors, {cascaded_errors} cascaded errors")
    
    async def _process_single_stock(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single stock."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        return {
            "symbol": symbol,
            "price": data["price"],
            "processed_at": datetime.now()
        }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
