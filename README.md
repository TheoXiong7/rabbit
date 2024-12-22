# Rabbit - An python algo trade project

## Overview
A Python-based framework for developing, testing, and optimizing algorithmic trading strategies. The project includes multiple strategy implementations, parameter optimization, and comprehensive performance analysis tools.

## Features

### Trading Strategies
- **Trend Following**: EMA and MACD-based trend detection with volume confirmation
- **Mean Reversion**: Bollinger Bands and RSI-based mean reversion
- **Breakout**: Price channel breakouts with volume confirmation
- **Adaptive**: Dynamic strategy switching based on market conditions

### Parameter Optimization
- Parameter space exploration
- Performance metrics calculation
- Robustness analysis
- Stock-specific optimization

### Data Handling
- Historical price data retrieval
- Data cleaning and preprocessing
- Cache management for efficient testing

## Dependencies
```python
pandas
numpy
yfinance
```

## Usage

### Basic Strategy Testing
```python
from data import Data
from strategies import TrendFollow

# Initialize data retriever
data = Data()

# Get historical data
stock_data = data.get_historical_data("AAPL", period="1y", interval="1d")

# Create and test strategy
strategy = TrendFollow()
signals = strategy.generate_signals(stock_data)
metrics = strategy.calculate_portfolio_metrics(signals, initial_capital=100000)
```

### Parameter Optimization
```python
from trend_parameter_tester import TrendParameterTester

# Initialize tester
tester = TrendParameterTester()

# Run optimization
results = tester.test_all_combinations()
analysis = tester.analyze_results(results)
```

## Strategy Parameters

### Trend Following Strategy
- `fast_ema`: Fast EMA period (default: 12)
- `slow_ema`: Slow EMA period (default: 26)
- `volume_threshold`: Volume surge threshold (default: 1.5)
- `profit_target`: Take profit level (default: 15%)
- `stop_loss`: Stop loss level (default: 5%)

### Mean Reversion Strategy
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std`: Standard deviation multiplier (default: 2.0)
- `rsi_period`: RSI calculation period (default: 14)

## Performance Metrics
- Total Return
- Sharpe Ratio
- Win Rate
- Maximum Drawdown
- Trade Count
- Risk-adjusted Returns

## Future Enhancements
- Additional strategy implementations
- Machine learning integration
- Real-time trading capabilities
- Portfolio optimization tools
- Risk management enhancements

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This software is for educational purposes only. Do not use it for live trading without proper testing and risk management.