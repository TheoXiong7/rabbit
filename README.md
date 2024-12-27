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


## Backtest Returns
![alt text](https://github.com/TheoXiong7/rabbit/blob/main/images/carbon.png?raw=true)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This software is for educational purposes only. Do not use it for live trading without proper testing and risk management.