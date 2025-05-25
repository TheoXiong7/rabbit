# DEPRECATED STRATEGIES - DO NOT USE
# These strategies performed poorly in testing and are kept for reference only

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        strategy_returns = df['Signal'].shift(1) * df['Returns']
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades': 0,
                'win_rate': 0
            }
        
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        signal_changes = df['Signal'].diff().abs()
        trades = signal_changes.sum() / 2
        
        profitable_trades = len(strategy_returns[strategy_returns > 0])
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'win_rate': win_rate
        }

# DEPRECATED: Over-complex trend following strategies that failed in testing