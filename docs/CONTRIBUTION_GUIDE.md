# AI Trading System - Contribution Guide

## Overview

Thank you for your interest in contributing to the AI Trading System! This guide provides comprehensive information on how to contribute to the project, including development setup, coding standards, and the contribution process.

## Getting Started

### Prerequisites
- **Python 3.11+**: [Download from python.org](https://www.python.org/downloads/)
- **Git**: [Download from git-scm.com](https://git-scm.com/downloads)
- **IDE**: VS Code, PyCharm, or similar
- **Docker** (Optional): [Download from docker.com](https://www.docker.com/products/docker-desktop)

### Development Setup

#### 1. Fork and Clone Repository
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/TradingBOT.git
cd TradingBOT

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/TradingBOT.git
```

#### 2. Create Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 3. Setup Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

#### 4. Run Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
pytest tests/security/
pytest tests/e2e/
```

## Contribution Process

### 1. Issue Reporting

#### Before Creating an Issue
- Check existing issues to avoid duplicates
- Search closed issues for similar problems
- Verify the issue is reproducible

#### Issue Template
```markdown
## Bug Report / Feature Request

### Description
Brief description of the issue or feature request.

### Steps to Reproduce (for bugs)
1. Step 1
2. Step 2
3. Step 3

### Expected Behavior
What should happen?

### Actual Behavior
What actually happens?

### Environment
- OS: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g., 3.11.0]
- System Version: [e.g., v1.0.0]

### Additional Context
Any additional information, screenshots, or logs.
```

### 2. Development Workflow

#### Branch Strategy
- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: Feature development branches
- **bugfix/**: Bug fix branches
- **hotfix/**: Critical bug fixes
- **release/**: Release preparation branches

#### Creating a Branch
```bash
# Update local main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

#### Development Process
```bash
# Make changes to code
# Add tests for new functionality
# Update documentation

# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat(trading): add position sizing algorithm"

# Push to your fork
git push origin feature/your-feature-name
```

### 3. Pull Request Process

#### Before Submitting PR
- [ ] Code follows style guidelines
- [ ] Tests are added/updated
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Security considerations addressed
- [ ] Performance impact assessed

#### PR Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass
- [ ] Security tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Security considerations addressed

## Related Issues
Closes #issue_number
```

#### PR Review Process
1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer review required
3. **Testing**: All tests must pass
4. **Documentation**: Documentation must be updated
5. **Approval**: Maintainer approval required for merge

## Coding Standards

### Code Style

#### Python Style Guide
Follow PEP 8 with these additional guidelines:

```python
# Good example
class TradingEngine:
    """Trading engine for executing orders and managing positions."""
    
    def __init__(self, mode: str = 'DEMO'):
        """Initialize trading engine with specified mode.
        
        Args:
            mode: Trading mode ('LIVE' or 'DEMO')
        """
        self.mode = mode
        self._initialize_components()
    
    def execute_order(self, order: Order) -> ExecutionResult:
        """Execute a trading order.
        
        Args:
            order: Order to execute
            
        Returns:
            ExecutionResult with execution details
            
        Raises:
            InsufficientFundsError: If account has insufficient funds
            InvalidOrderError: If order parameters are invalid
        """
        if not self._validate_order(order):
            raise InvalidOrderError("Invalid order parameters")
        
        return self._process_order(order)
```

#### Type Hints
Always use type hints for better code clarity:

```python
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float

def get_positions(mode: str = None) -> List[Position]:
    """Get all positions for specified mode."""
    pass

def calculate_pnl(position: Position) -> float:
    """Calculate profit/loss for position."""
    return (position.current_price - position.entry_price) * position.quantity
```

#### Documentation
Use docstrings for all public methods and classes:

```python
def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stock_price: float,
    stop_loss_percent: float
) -> int:
    """Calculate optimal position size based on risk management.
    
    Args:
        account_balance: Total account balance
        risk_percent: Maximum risk percentage (0.01 = 1%)
        stock_price: Current stock price
        stop_loss_percent: Stop loss percentage (0.05 = 5%)
    
    Returns:
        Number of shares to purchase
        
    Example:
        >>> calculate_position_size(10000, 0.02, 100, 0.05)
        40
    """
    risk_amount = account_balance * risk_percent
    stop_loss_amount = stock_price * stop_loss_percent
    position_size = risk_amount / stop_loss_amount
    return int(position_size)
```

### Commit Message Convention

#### Format
```
type(scope): description

[optional body]

[optional footer]
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build process or auxiliary tool changes
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes

#### Examples
```
feat(trading): add position sizing algorithm
fix(ai): resolve model loading timeout issue
docs(api): update API documentation
test(integration): add trading cycle tests
perf(database): optimize query performance
refactor(monitoring): improve logging structure
```

## Testing Guidelines

### Test Structure

#### Unit Tests
```python
# Example: Unit test
import unittest
from unittest.mock import Mock, patch
from src.trading.positions import PositionManager

class TestPositionManager(unittest.TestCase):
    def setUp(self):
        self.position_manager = PositionManager('DEMO')
    
    def test_create_position(self):
        position = self.position_manager.create_position('AAPL', 100, 150.0)
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, 'AAPL')
        self.assertEqual(position.quantity, 100)
    
    @patch('src.config.database.get_connection')
    def test_create_position_database_error(self, mock_get_connection):
        mock_get_connection.side_effect = Exception("Database error")
        with self.assertRaises(Exception):
            self.position_manager.create_position('AAPL', 100, 150.0)
```

#### Integration Tests
```python
# Example: Integration test
import unittest
from src.workflows.trading_cycle import TradingCycle
from src.trading.positions import PositionManager
from src.trading.execution import ExecutionEngine

class TestTradingCycleIntegration(unittest.TestCase):
    def setUp(self):
        self.trading_cycle = TradingCycle()
    
    def test_complete_trading_cycle(self):
        result = self.trading_cycle.execute_complete_cycle()
        self.assertTrue(result.success)
        self.assertIsNotNone(result.positions_created)
```

#### Performance Tests
```python
# Example: Performance test
import unittest
import time
from src.ai.multi_model import MultiModelManager

class TestPerformance(unittest.TestCase):
    def test_ai_decision_latency(self):
        manager = MultiModelManager()
        start_time = time.time()
        
        result = manager.get_analysis('AAPL', 'entry_signal')
        
        end_time = time.time()
        latency = end_time - start_time
        
        self.assertLess(latency, 2.0)  # Should complete in under 2 seconds
        self.assertIsNotNone(result)
```

### Test Coverage
- **Unit Tests**: Minimum 80% coverage
- **Integration Tests**: Cover all major workflows
- **Performance Tests**: Critical path performance
- **Security Tests**: Security vulnerabilities
- **E2E Tests**: Complete user scenarios

## Documentation Guidelines

### Code Documentation

#### Inline Comments
```python
def calculate_risk_metrics(positions: List[Position]) -> RiskMetrics:
    """Calculate portfolio risk metrics."""
    # Calculate portfolio value
    total_value = sum(pos.quantity * pos.current_price for pos in positions)
    
    # Calculate individual position weights
    weights = [pos.quantity * pos.current_price / total_value for pos in positions]
    
    # Calculate portfolio variance using weights and correlations
    portfolio_variance = 0
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            correlation = get_correlation(pos1.symbol, pos2.symbol)
            portfolio_variance += weights[i] * weights[j] * correlation
    
    return RiskMetrics(
        portfolio_variance=portfolio_variance,
        portfolio_std=math.sqrt(portfolio_variance)
    )
```

#### API Documentation
```python
@app.route('/api/v1/positions', methods=['POST'])
def create_position():
    """Create a new trading position.
    
    This endpoint creates a new trading position with the specified parameters.
    The position will be tracked and managed by the system.
    
    Request Body:
        {
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.00
        }
    
    Returns:
        201: Position created successfully
        400: Invalid request parameters
        401: Unauthorized
        500: Internal server error
    
    Example:
        POST /api/v1/positions
        Content-Type: application/json
        
        {
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.00
        }
    """
    pass
```

### README Updates
When adding new features, update the README:

```markdown
## New Feature: Position Sizing Algorithm

### Description
Added intelligent position sizing algorithm that calculates optimal position sizes based on risk management principles.

### Usage
```python
from src.trading.position_sizing import calculate_position_size

position_size = calculate_position_size(
    account_balance=10000,
    risk_percent=0.02,
    stock_price=100,
    stop_loss_percent=0.05
)
```

### Configuration
Add to `config/trading_config.yaml`:
```yaml
position_sizing:
  enabled: true
  risk_percent: 0.02
  max_position_size: 10000
```
```

## Security Guidelines

### Security Best Practices

#### Input Validation
```python
# Good: Input validation
def validate_order_input(data: dict) -> bool:
    """Validate order input data."""
    required_fields = ['symbol', 'quantity', 'price']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate symbol format
    if not re.match(r'^[A-Za-z0-9]{1,10}$', data['symbol']):
        return False
    
    # Validate quantity
    try:
        quantity = int(data['quantity'])
        if not 1 <= quantity <= 10000:
            return False
    except (ValueError, TypeError):
        return False
    
    # Validate price
    try:
        price = float(data['price'])
        if not 0.01 <= price <= 10000.0:
            return False
    except (ValueError, TypeError):
        return False
    
    return True
```

#### Secure Coding Practices
- Always validate input data
- Use parameterized queries for database operations
- Implement proper authentication and authorization
- Log security events
- Use secure communication protocols
- Regular security audits

## Performance Guidelines

### Performance Best Practices

#### Efficient Algorithms
```python
# Good: Efficient algorithm
def calculate_portfolio_metrics(positions: List[Position]) -> PortfolioMetrics:
    """Calculate portfolio metrics efficiently."""
    # Use vectorized operations where possible
    quantities = np.array([pos.quantity for pos in positions])
    prices = np.array([pos.current_price for pos in positions])
    
    # Calculate total value
    total_value = np.sum(quantities * prices)
    
    # Calculate weights
    weights = (quantities * prices) / total_value
    
    # Calculate returns
    returns = np.array([pos.return_percent for pos in positions])
    portfolio_return = np.sum(weights * returns)
    
    return PortfolioMetrics(
        total_value=total_value,
        portfolio_return=portfolio_return,
        weights=weights.tolist()
    )
```

#### Caching Strategies
```python
# Good: Caching implementation
from functools import lru_cache
import redis

class CachedDataManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    @lru_cache(maxsize=1000)
    def get_stock_info(self, symbol: str) -> dict:
        """Get stock information with local caching."""
        # Check Redis cache first
        cached_data = self.redis_client.get(f"stock:{symbol}")
        if cached_data:
            return json.loads(cached_data)
        
        # Fetch from API
        data = self._fetch_from_api(symbol)
        
        # Cache in Redis
        self.redis_client.setex(f"stock:{symbol}", 300, json.dumps(data))
        
        return data
```

## Code Review Guidelines

### Review Checklist

#### For Reviewers
- [ ] Code follows style guidelines
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact is acceptable
- [ ] No breaking changes
- [ ] Error handling is appropriate
- [ ] Logging is adequate

#### For Authors
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Performance tested
- [ ] Security reviewed
- [ ] Breaking changes documented

### Review Process
1. **Automated Checks**: CI/CD pipeline runs
2. **Code Review**: Peer review required
3. **Testing**: All tests must pass
4. **Documentation**: Must be updated
5. **Approval**: Maintainer approval required

## Release Process

### Version Numbering
Follow Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared
- [ ] Security audit completed
- [ ] Performance benchmarks met

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow project guidelines
- Report issues appropriately

### Getting Help
- **Documentation**: Check existing docs first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions
- **Discord**: Join community Discord
- **Email**: Contact maintainers

### Recognition
Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Community highlights

## Conclusion

Thank you for contributing to the AI Trading System! Following these guidelines ensures high-quality contributions and a smooth development process.

For questions or clarifications, please:
1. Check existing documentation
2. Search existing issues
3. Create a new issue if needed
4. Contact maintainers

Happy coding! 🚀
