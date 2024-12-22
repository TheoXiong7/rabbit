import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class BaseStrategy(ABC):
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def calculate_portfolio_metrics(self, df: pd.DataFrame, initial_capital: float) -> Dict:
        df = df.copy()
        
        # Calculate positions and returns
        df['Position'] = df['Signal'].fillna(0)
        last_position = 0
        for i in range(len(df)):
            if df['Position'].iloc[i] == 0:
                df.loc[df.index[i], 'Position'] = last_position
            else:
                last_position = df['Position'].iloc[i]
        
        df['Returns'] = df['Close'].pct_change() * df['Position']
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
        
        # Calculate metrics
        total_return = df['Cumulative_Returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        
        daily_returns = df['Returns'].dropna()
        sharpe_ratio = (np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if len(daily_returns) and daily_returns.std() != 0 else 0
        max_drawdown = (df['Cumulative_Returns'] / df['Cumulative_Returns'].cummax() - 1).min()
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Final Capital': initial_capital * (1 + total_return)
        }

class TrendFollow(BaseStrategy):
    def __init__(
        self,
        fast_ema: int = 10,
        slow_ema: int = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        volume_threshold: float = 1.5,
        profit_target: float = 0.15,
        stop_loss: float = 0.05
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volume_threshold = volume_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Core trend indicators
        df['EMA_fast'] = df['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['EMA_diff'] = (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow'] * 100
        
        # MACD
        df['MACD'] = (df['Close'].ewm(span=self.macd_fast, adjust=False).mean() - 
                      df['Close'].ewm(span=self.macd_slow, adjust=False).mean())
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Trend strength
        df['Trend_Strength'] = df['EMA_diff'].abs()
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        df['Entry_Price'] = 0.0
        last_signal = 0
        entry_price = 0.0
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            
            # Track position and check for exit conditions if in a trade
            if last_signal != 0:
                price_change = (current_price - entry_price) / entry_price
                
                # Check stop loss and profit target
                if (last_signal == 1 and (price_change <= -self.stop_loss or price_change >= self.profit_target)) or \
                   (last_signal == -1 and (-price_change <= -self.stop_loss or -price_change >= self.profit_target)):
                    df.loc[df.index[i], 'Signal'] = -last_signal  # Exit the position
                    last_signal = 0
                    continue
            
            # Core trend conditions
            trend_up = (df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and 
                       df['MACD_Hist'].iloc[i] > 0 and
                       df['MACD_Hist'].iloc[i] > df['MACD_Hist'].iloc[i-1])
            
            trend_down = (df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and 
                         df['MACD_Hist'].iloc[i] < 0 and
                         df['MACD_Hist'].iloc[i] < df['MACD_Hist'].iloc[i-1])
            
            # Volume confirmation
            volume_confirmed = df['Volume_Ratio'].iloc[i] > self.volume_threshold
            
            # Trend strength filter
            strong_trend = df['Trend_Strength'].iloc[i] > 0.5  # 0.5% minimum trend strength
            
            # Generate signals with confirmation
            if last_signal == 0:  # Not in a position
                if trend_up and volume_confirmed and strong_trend:
                    df.loc[df.index[i], 'Signal'] = 1
                    last_signal = 1
                    entry_price = current_price
                elif trend_down and volume_confirmed and strong_trend:
                    df.loc[df.index[i], 'Signal'] = -1
                    last_signal = -1
                    entry_price = current_price
            
            # Track entry prices for position management
            if df['Signal'].iloc[i] != 0:
                df.loc[df.index[i], 'Entry_Price'] = current_price
        
        return df

class MeanReversion(BaseStrategy):
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=self.bb_period).mean()
        rolling_std = df['Close'].rolling(window=self.bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * self.bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * self.bb_std)
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        
        for i in range(1, len(df)):
            # Enhanced oversold conditions with price confirmation
            oversold = (
                df['Close'].iloc[i] < df['BB_Lower'].iloc[i] and 
                df['RSI'].iloc[i] < 30 and
                df['Close'].iloc[i] > df['Close'].iloc[i-1]  # Price starting to recover
            )
            
            # Enhanced overbought conditions with price confirmation
            overbought = (
                df['Close'].iloc[i] > df['BB_Upper'].iloc[i] and 
                df['RSI'].iloc[i] > 70 and
                df['Close'].iloc[i] < df['Close'].iloc[i-1]  # Price starting to fall
            )
            
            if oversold:
                df.loc[df.index[i], 'Signal'] = 1
            elif overbought:
                df.loc[df.index[i], 'Signal'] = -1
                
            # Add exit conditions
            elif df['Signal'].iloc[i-1] == 1 and df['RSI'].iloc[i] > 50:
                df.loc[df.index[i], 'Signal'] = -1  # Exit long position
            elif df['Signal'].iloc[i-1] == -1 and df['RSI'].iloc[i] < 50:
                df.loc[df.index[i], 'Signal'] = 1  # Exit short position
                
        return df

class Breakout(BaseStrategy):
    def __init__(self, channel_period: int = 20, volume_factor: float = 2.0):
        self.channel_period = channel_period
        self.volume_factor = volume_factor

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate price channels
        df['Upper_Channel'] = df['High'].rolling(window=self.channel_period).max()
        df['Lower_Channel'] = df['Low'].rolling(window=self.channel_period).min()
        
        # Calculate volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        
        for i in range(1, len(df)):
            # Breakout conditions
            breakout_up = (df['Close'].iloc[i] > df['Upper_Channel'].iloc[i-1] and 
                         df['Volume_Ratio'].iloc[i] > self.volume_factor)
            
            breakout_down = (df['Close'].iloc[i] < df['Lower_Channel'].iloc[i-1] and 
                           df['Volume_Ratio'].iloc[i] > self.volume_factor)
            
            if breakout_up:
                df.loc[df.index[i], 'Signal'] = 1
            elif breakout_down:
                df.loc[df.index[i], 'Signal'] = -1
                
        return df

class Adaptive(BaseStrategy):
    def __init__(self, volatility_window: int = 20, volume_window: int = 20):
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.trend_strategy = TrendFollow()
        self.mean_rev_strategy = MeanReversion()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=self.volatility_window).std()
        df['Volatility_MA'] = df['Volatility'].rolling(window=self.volatility_window).mean()
        
        # Calculate volume trend
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_window).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Get signals from both strategies
        df_trend = self.trend_strategy.generate_signals(df)
        df_mean_rev = self.mean_rev_strategy.generate_signals(df)
        
        df['Trend_Signal'] = df_trend['Signal']
        df['MeanRev_Signal'] = df_mean_rev['Signal']
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        
        for i in range(1, len(df)):
            high_volatility = df['Volatility'].iloc[i] > df['Volatility_MA'].iloc[i]
            rising_volume = (df['Volume'].iloc[i] > df['Volume'].iloc[i-1] and 
                           df['Volume_Ratio'].iloc[i] > 1.2)
            
            # Calculate trend strength
            price_trend = (df['Close'].iloc[i] - df['Close'].iloc[max(0, i-5)]) / df['Close'].iloc[max(0, i-5)]
            strong_trend = abs(price_trend) > 0.02  # 2% price movement in 5 days
            
            if strong_trend and rising_volume:
                # Use trend following in strong trends with rising volume
                df.loc[df.index[i], 'Signal'] = df['Trend_Signal'].iloc[i]
            elif not strong_trend and not high_volatility:
                # Use mean reversion in ranging markets
                df.loc[df.index[i], 'Signal'] = df['MeanRev_Signal'].iloc[i]
            elif df['Trend_Signal'].iloc[i] == df['MeanRev_Signal'].iloc[i] and df['Trend_Signal'].iloc[i] != 0:
                # When both strategies agree, take the signal
                df.loc[df.index[i], 'Signal'] = df['Trend_Signal'].iloc[i]
                
        return df