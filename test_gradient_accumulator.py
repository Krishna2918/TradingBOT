"""
Test Gradient Accumulator

Test the gradient accumulation functionality.
"""

import sys
sys.path.append('src')

from src.ai.models.gradient_accumulator import GradientAccumulator
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTestModel(nn.Module):
    """Simple model for testing gradient accumulation"""
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def test_gradient_accumulator():
    """Test gradient accumulator functionality"""
    print("Testing Gradient Accumulator")
    print("=" * 50)
    
    # Setup test model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTestModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test 1: Basic gradient accumulation
    print("\n1. Basic Gradient Accumulation")
    print("-" * 30)
    
    accumulator = GradientAccumulator(accumulation_steps=4, gradient_clip_norm=1.0)
    
    # Simulate training steps
    batch_sizes = [8, 8, 8, 8]  # 4 steps of batch size 8
    total_loss = 0.0
    
    for step, batch_size in enumerate(batch_sizes):
        # Create dummy data
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randint(0, 3, (batch_size,), device=device)
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Accumulate gradients
        accumulation_info = accumulator.accumulate_gradients(model, loss, batch_size)
        
        print(f"Step {step+1}: Loss={loss.item():.4f}, "
              f"Gradient norm={accumulation_info['gradient_norm']:.4f}, "
              f"Complete={accumulation_info['is_complete']}")
        
        total_loss += loss.item()
        
        # Update optimizer if accumulation is complete
        if accumulator.should_update_optimizer():
            update_info = accumulator.update_optimizer(optimizer)
            print(f"✅ Optimizer updated: Accumulated loss={update_info['accumulated_loss']:.4f}, "
                  f"Effective batch size={update_info['effective_batch_size']}")
    
    # Test 2: Mixed precision training
    print("\n2. Mixed Precision Training")
    print("-" * 30)
    
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        accumulator_amp = GradientAccumulator(accumulation_steps=2)
        
        for step in range(2):
            x = torch.randn(16, 10, device=device)
            y = torch.randint(0, 3, (16,), device=device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            
            # Accumulate with scaler
            accumulation_info = accumulator_amp.accumulate_gradients(model, loss, 16, scaler)
            
            print(f"AMP Step {step+1}: Loss={loss.item():.4f}, Complete={accumulation_info['is_complete']}")
            
            if accumulator_amp.should_update_optimizer():
                update_info = accumulator_amp.update_optimizer(optimizer, scaler)
                print(f"✅ AMP Optimizer updated: Final gradient norm={update_info['final_gradient_norm']:.4f}")
    else:
        print("⚠️  CUDA not available - skipping mixed precision test")
    
    # Test 3: Gradient health monitoring
    print("\n3. Gradient Health Monitoring")
    print("-" * 30)
    
    # Run several accumulation cycles to build history
    health_accumulator = GradientAccumulator(accumulation_steps=2)
    
    for cycle in range(3):
        for step in range(2):
            x = torch.randn(8, 10, device=device)
            y = torch.randint(0, 3, (8,), device=device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            health_accumulator.accumulate_gradients(model, loss, 8)
            
            if health_accumulator.should_update_optimizer():
                health_accumulator.update_optimizer(optimizer)
        
        # Check gradient health
        is_healthy, message = health_accumulator.is_gradient_healthy()
        status_emoji = "✅" if is_healthy else "⚠️"
        print(f"Cycle {cycle+1}: {status_emoji} {message}")
    
    # Test 4: Accumulation statistics
    print("\n4. Accumulation Statistics")
    print("-" * 30)
    
    stats = health_accumulator.get_accumulation_statistics()
    
    print(f"Total steps: {stats['total_steps']}")
    print(f"Total accumulations: {stats['total_accumulations']}")
    print(f"Current progress: {stats['current_accumulation_progress']}/{stats['accumulation_steps']}")
    print(f"Average gradient norm: {stats['average_gradient_norm']:.4f}")
    print(f"Average loss: {stats['average_loss']:.4f}")
    
    if 'gradient_norm_stats' in stats:
        norm_stats = stats['gradient_norm_stats']
        print(f"Gradient norm range: {norm_stats['min']:.4f} - {norm_stats['max']:.4f}")
    
    print(f"Recent steps: {stats['recent_steps']['count']} steps, "
          f"total batch size: {stats['recent_steps']['total_batch_size']}")
    
    # Test 5: Dynamic accumulation steps
    print("\n5. Dynamic Accumulation Steps")
    print("-" * 30)
    
    dynamic_accumulator = GradientAccumulator(accumulation_steps=2)
    
    # Test different accumulation step values
    test_steps = [1, 2, 4, 8]
    
    for steps in test_steps:
        dynamic_accumulator.set_accumulation_steps(steps)
        print(f"Set accumulation steps to: {steps}")
        
        # Run one complete accumulation cycle
        for step in range(steps):
            x = torch.randn(4, 10, device=device)
            y = torch.randint(0, 3, (4,), device=device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            accumulation_info = dynamic_accumulator.accumulate_gradients(model, loss, 4)
            
            if dynamic_accumulator.should_update_optimizer():
                update_info = dynamic_accumulator.update_optimizer(optimizer)
                print(f"  ✅ Completed {steps}-step accumulation, "
                      f"effective batch size: {update_info['effective_batch_size']}")
                break
    
    # Test 6: Memory optimization
    print("\n6. Memory Optimization")
    print("-" * 30)
    
    memory_accumulator = GradientAccumulator()
    
    # Test accumulation step optimization
    test_scenarios = [
        ("High memory", 4.0, 32, 0.1),
        ("Medium memory", 2.0, 32, 0.1),
        ("Low memory", 1.0, 32, 0.1),
        ("Very low memory", 0.5, 32, 0.1)
    ]
    
    for scenario, target_memory, batch_size, memory_per_sample in test_scenarios:
        optimal_steps = memory_accumulator.optimize_accumulation_steps(
            target_memory, batch_size, memory_per_sample
        )
        print(f"{scenario}: Optimal accumulation steps = {optimal_steps}")
    
    # Test 7: Reset functionality
    print("\n7. Reset Functionality")
    print("-" * 30)
    
    reset_accumulator = GradientAccumulator(accumulation_steps=3)
    
    # Partially accumulate
    x = torch.randn(8, 10, device=device)
    y = torch.randint(0, 3, (8,), device=device)
    outputs = model(x)
    loss = criterion(outputs, y)
    
    reset_accumulator.accumulate_gradients(model, loss, 8)
    print(f"Before reset: Current step = {reset_accumulator.current_step}")
    
    reset_accumulator.reset()
    print(f"After reset: Current step = {reset_accumulator.current_step}")
    
    print("\n" + "=" * 50)
    print("Gradient Accumulator Test Complete")
    
    return accumulator

if __name__ == "__main__":
    test_gradient_accumulator()