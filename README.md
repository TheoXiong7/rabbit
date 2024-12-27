# Rabbit

## Overview
A Python-based framework for developing, testing, optimizing, and deploying algorithmic trading strategies. The project includes multiple strategy implementations, parameter optimization, performance analysis tools, and Aplaca API integration.

## Features

### Strategies & Features
- Adaptive EMA
- Adaptive MACD
- Dynamic stop losses
- Dynaic position sizing
- Tiered profit targets
- Auto adjustments for high/medium/low volatility and volume regimes

### Parameter Optimization
- Gridsearch for optimal parameters
- Performance metrics calculation
- Robustness analysis
- Stock-specific optimizatio

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

## Performance Metrics
- Total Return
- Sharpe Ratio
- Win Rate
- Maximum Drawdown
- Trade Count
- Risk-adjusted Returns


## Sample Backtest Returns
```
Developmental Strategy:

==================== 1Y Period ====================

Mean Dollar Return: $2,611.07
Mean Percent Return: 52.22%
Total Entries: 2,070
Total Exits: 1,755
Total Actions: 3,825
Mean Sharpe Ratio: 1.15
Win Rate: 86.03%
Return Std Dev: 115.02%

==================== 2Y Period ====================

Developmental Strategy:
Mean Dollar Return: $9,228.58
Mean Percent Return: 184.57%
Total Entries: 4,253
Total Exits: 3,427
Total Actions: 7,680
Mean Sharpe Ratio: 1.13
Win Rate: 92.65%
Return Std Dev: 532.48%

==================== 5Y Period ====================

Developmental Strategy:
Mean Dollar Return: $1,453,675.07
Mean Percent Return: 29073.50%
Total Entries: 12,283
Total Exits: 9,536
Total Actions: 21,819
Mean Sharpe Ratio: 1.48
Win Rate: 98.53%
Return Std Dev: 195330.89%

===================================================
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This software is for educational purposes only. Do not use it for live trading without proper testing and risk management.