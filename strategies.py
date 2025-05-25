# strategies.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

# Base Strategy
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

# 3.0
# even more optimized, and more features.
class TrendFollow3(BaseStrategy):
    def __init__(
        self,
        volatility_window: int = 20,
        volume_window: int = 20,
        high_vol_threshold: float = 0.30,
        low_vol_threshold: float = 0.15,
        market_cap: str = 'mid',  # 'mid' or 'large'
        # High volatility parameters
        high_vol_fast_ema: int = 7,  # Faster for mid-caps
        high_vol_slow_ema: int = 22,  # Shorter trend window
        high_vol_volume_threshold: float = 2.0,  # More aggressive volume filter
        high_vol_profit_target: float = 0.25,  # Higher profit target
        high_vol_stop_loss: float = 0.08,  # Wider stop for volatility
        # Medium volatility parameters
        med_vol_fast_ema: int = 10,
        med_vol_slow_ema: int = 30,
        med_vol_volume_threshold: float = 1.5,
        med_vol_profit_target: float = 0.15,
        med_vol_stop_loss: float = 0.05,
        # Low volatility parameters
        low_vol_fast_ema: int = 12,
        low_vol_slow_ema: int = 35,
        low_vol_volume_threshold: float = 1.2,
        low_vol_profit_target: float = 0.10,
        low_vol_stop_loss: float = 0.03,
        # MACD parameters
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        super().__init__()
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.market_cap = market_cap
        
        # Adjust parameters based on market cap
        if market_cap == 'mid':
            # More aggressive parameters for mid-caps
            high_vol_fast_ema = 7
            high_vol_slow_ema = 22
            high_vol_profit_target = 0.25
            high_vol_volume_threshold = 2.0
        
        # Store volatility-based parameters
        self.vol_params = {
            'high': {
                'fast_ema': high_vol_fast_ema,
                'slow_ema': high_vol_slow_ema,
                'volume_threshold': high_vol_volume_threshold,
                'profit_target': high_vol_profit_target,
                'stop_loss': high_vol_stop_loss
            },
            'medium': {
                'fast_ema': med_vol_fast_ema,
                'slow_ema': med_vol_slow_ema,
                'volume_threshold': med_vol_volume_threshold,
                'profit_target': med_vol_profit_target,
                'stop_loss': med_vol_stop_loss
            },
            'low': {
                'fast_ema': low_vol_fast_ema,
                'slow_ema': low_vol_slow_ema,
                'volume_threshold': low_vol_volume_threshold,
                'profit_target': low_vol_profit_target,
                'stop_loss': low_vol_stop_loss
            }
        }
        
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def _calculate_market_conditions(self, df: pd.DataFrame) -> tuple:
        """Calculate market conditions including volatility and volume regimes"""
        # Calculate annualized volatility
        volatility = df['Returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Calculate volume regime
        volume_ma = df['Volume'].rolling(window=self.volume_window).mean()
        volume_ratio = df['Volume'] / volume_ma
        
        # Determine volume regime (prefer extremes, avoid medium)
        if volume_ratio.iloc[-1] > 1.8:
            volume_regime = 'high'
        elif volume_ratio.iloc[-1] < 0.7:
            volume_regime = 'low'
        else:
            volume_regime = 'medium'
            
        # Determine volatility regime
        if volatility.iloc[-1] >= self.high_vol_threshold:
            vol_regime = 'high'
        elif volatility.iloc[-1] <= self.low_vol_threshold:
            vol_regime = 'low'
        else:
            vol_regime = 'medium'
            
        return vol_regime, volume_regime, volatility.iloc[-1]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate market conditions
        vol_regime, volume_regime, current_vol = self._calculate_market_conditions(df)
        params = self.vol_params[vol_regime]
        
        # Adjust parameters based on volume regime
        if volume_regime != 'medium':  # Prefer trading in high/low volume
            params['volume_threshold'] *= 0.9  # More lenient volume threshold
        else:
            params['volume_threshold'] *= 1.2  # Stricter threshold in medium volume
        
        # Calculate core indicators
        df['EMA_fast'] = df['Close'].ewm(span=params['fast_ema'], adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=params['slow_ema'], adjust=False).mean()
        df['EMA_diff'] = (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow'] * 100
        
        # Enhanced MACD with market cap adjustment
        if self.market_cap == 'mid':
            # More responsive MACD for mid-caps
            df['MACD'] = (df['Close'].ewm(span=10, adjust=False).mean() - 
                         df['Close'].ewm(span=20, adjust=False).mean())
        else:
            df['MACD'] = (df['Close'].ewm(span=self.macd_fast, adjust=False).mean() - 
                         df['Close'].ewm(span=self.macd_slow, adjust=False).mean())
        
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_window).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Store current parameters and conditions
        df['Current_Profit_Target'] = params['profit_target']
        df['Current_Stop_Loss'] = params['stop_loss']
        df['Current_Volume_Threshold'] = params['volume_threshold']
        df['Volatility_Regime'] = vol_regime
        df['Volume_Regime'] = volume_regime
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        df['Entry_Price'] = 0.0
        last_signal = 0
        entry_price = 0.0
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            
            # Get current parameters
            profit_target = df['Current_Profit_Target'].iloc[i]
            stop_loss = df['Current_Stop_Loss'].iloc[i]
            volume_threshold = df['Current_Volume_Threshold'].iloc[i]
            
            # Position management
            if last_signal != 0:
                price_change = (current_price - entry_price) / entry_price
                
                # Dynamic exit based on market conditions
                if df['Volume_Regime'].iloc[i] == 'high':
                    # Be more patient with exits in high volume
                    profit_target *= 1.2
                    stop_loss *= 1.1
                
                # Check exits
                if (last_signal == 1 and (price_change <= -stop_loss or price_change >= profit_target)) or \
                   (last_signal == -1 and (-price_change <= -stop_loss or -price_change >= profit_target)):
                    df.loc[df.index[i], 'Signal'] = -last_signal
                    last_signal = 0
                    continue
            
            # Enhanced trend conditions
            trend_up = (df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and 
                       df['MACD_Hist'].iloc[i] > 0 and
                       df['MACD_Hist'].iloc[i] > df['MACD_Hist'].iloc[i-1])
            
            trend_down = (df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and 
                         df['MACD_Hist'].iloc[i] < 0 and
                         df['MACD_Hist'].iloc[i] < df['MACD_Hist'].iloc[i-1])
            
            # Volume confirmation
            volume_confirmed = df['Volume_Ratio'].iloc[i] > volume_threshold
            
            # Trend strength with market cap adjustment
            trend_strength = abs(df['EMA_diff'].iloc[i]) > (0.4 if self.market_cap == 'mid' else 0.6)
            
            # Signal generation with market condition adjustments
            if last_signal == 0:  # Not in a position
                # More aggressive in high volume regimes
                if df['Volume_Regime'].iloc[i] == 'high':
                    trend_strength_threshold = 0.8 * (0.4 if self.market_cap == 'mid' else 0.6)
                else:
                    trend_strength_threshold = (0.4 if self.market_cap == 'mid' else 0.6)
                
                if trend_up and volume_confirmed and abs(df['EMA_diff'].iloc[i]) > trend_strength_threshold:
                    df.loc[df.index[i], 'Signal'] = 1
                    last_signal = 1
                    entry_price = current_price
                elif trend_down and volume_confirmed and abs(df['EMA_diff'].iloc[i]) > trend_strength_threshold:
                    df.loc[df.index[i], 'Signal'] = -1
                    last_signal = -1
                    entry_price = current_price
            
            # Track entry prices
            if df['Signal'].iloc[i] != 0:
                df.loc[df.index[i], 'Entry_Price'] = current_price
        
        return df

# 2.0
# more optimized
class TrendFollow2(BaseStrategy):
    def __init__(
        self,
        volatility_window: int = 20,
        high_vol_threshold: float = 0.30,  # annualized volatility threshold
        low_vol_threshold: float = 0.15,
        # High volatility parameters
        high_vol_fast_ema: int = 8,
        high_vol_slow_ema: int = 25,
        high_vol_volume_threshold: float = 1.8,
        high_vol_profit_target: float = 0.20,
        high_vol_stop_loss: float = 0.07,
        # Medium volatility parameters
        med_vol_fast_ema: int = 10,
        med_vol_slow_ema: int = 30,
        med_vol_volume_threshold: float = 1.5,
        med_vol_profit_target: float = 0.15,
        med_vol_stop_loss: float = 0.05,
        # Low volatility parameters
        low_vol_fast_ema: int = 12,
        low_vol_slow_ema: int = 35,
        low_vol_volume_threshold: float = 1.2,
        low_vol_profit_target: float = 0.10,
        low_vol_stop_loss: float = 0.03,
        # MACD parameters (kept constant)
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        self.volatility_window = volatility_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        
        # Store volatility-based parameters
        self.vol_params = {
            'high': {
                'fast_ema': high_vol_fast_ema,
                'slow_ema': high_vol_slow_ema,
                'volume_threshold': high_vol_volume_threshold,
                'profit_target': high_vol_profit_target,
                'stop_loss': high_vol_stop_loss
            },
            'medium': {
                'fast_ema': med_vol_fast_ema,
                'slow_ema': med_vol_slow_ema,
                'volume_threshold': med_vol_volume_threshold,
                'profit_target': med_vol_profit_target,
                'stop_loss': med_vol_stop_loss
            },
            'low': {
                'fast_ema': low_vol_fast_ema,
                'slow_ema': low_vol_slow_ema,
                'volume_threshold': low_vol_volume_threshold,
                'profit_target': low_vol_profit_target,
                'stop_loss': low_vol_stop_loss
            }
        }
        
        # MACD parameters
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine the volatility regime based on recent price action"""
        current_volatility = df['Returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        latest_volatility = current_volatility.iloc[-1]
        
        if latest_volatility >= self.high_vol_threshold:
            return 'high'
        elif latest_volatility <= self.low_vol_threshold:
            return 'low'
        else:
            return 'medium'

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with dynamic parameter selection"""
        df = df.copy()
        
        # Calculate volatility and determine regime
        df['Volatility'] = df['Returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        volatility_regime = self._calculate_volatility_regime(df)
        params = self.vol_params[volatility_regime]
        
        # Calculate EMAs using regime-specific parameters
        df['EMA_fast'] = df['Close'].ewm(span=params['fast_ema'], adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=params['slow_ema'], adjust=False).mean()
        df['EMA_diff'] = (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow'] * 100
        
        # MACD (constant parameters)
        df['MACD'] = (df['Close'].ewm(span=self.macd_fast, adjust=False).mean() - 
                      df['Close'].ewm(span=self.macd_slow, adjust=False).mean())
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Volume indicators with dynamic threshold
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Store current parameters for signal generation
        df['Current_Profit_Target'] = params['profit_target']
        df['Current_Stop_Loss'] = params['stop_loss']
        df['Current_Volume_Threshold'] = params['volume_threshold']
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with dynamic parameters"""
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        df['Entry_Price'] = 0.0
        last_signal = 0
        entry_price = 0.0
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            
            # Dynamic parameters based on stored values
            profit_target = df['Current_Profit_Target'].iloc[i]
            stop_loss = df['Current_Stop_Loss'].iloc[i]
            volume_threshold = df['Current_Volume_Threshold'].iloc[i]
            
            # Position management
            if last_signal != 0:
                price_change = (current_price - entry_price) / entry_price
                
                # Check stop loss and profit target
                if (last_signal == 1 and (price_change <= -stop_loss or price_change >= profit_target)) or \
                   (last_signal == -1 and (-price_change <= -stop_loss or -price_change >= profit_target)):
                    df.loc[df.index[i], 'Signal'] = -last_signal
                    last_signal = 0
                    continue
            
            # Enhanced trend conditions
            trend_up = (df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and 
                       df['MACD_Hist'].iloc[i] > 0 and
                       df['MACD_Hist'].iloc[i] > df['MACD_Hist'].iloc[i-1])
            
            trend_down = (df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and 
                         df['MACD_Hist'].iloc[i] < 0 and
                         df['MACD_Hist'].iloc[i] < df['MACD_Hist'].iloc[i-1])
            
            # Volume confirmation with dynamic threshold
            volume_confirmed = df['Volume_Ratio'].iloc[i] > volume_threshold
            
            # Trend strength filter
            trend_strength = abs(df['EMA_diff'].iloc[i]) > 0.5
            
            # Generate signals
            if last_signal == 0:  # Not in a position
                if trend_up and volume_confirmed and trend_strength:
                    df.loc[df.index[i], 'Signal'] = 1
                    last_signal = 1
                    entry_price = current_price
                elif trend_down and volume_confirmed and trend_strength:
                    df.loc[df.index[i], 'Signal'] = -1
                    last_signal = -1
                    entry_price = current_price
            
            # Update entry prices
            if df['Signal'].iloc[i] != 0:
                df.loc[df.index[i], 'Entry_Price'] = current_price
        
        return df

# Trend Follow Strategy 1.0
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

# Dev strategy (optimized 3.0)
class TrendFollowDev(BaseStrategy):
    def __init__(
        self,
        volatility_window: int = 15,
        volume_window: int = 15,
        high_vol_threshold: float = 0.28,
        low_vol_threshold: float = 0.17,
        market_cap: str = 'mid',  # 'mid' or 'large'
        # High volatility parameters
        high_vol_fast_ema: int = 6,  # Faster for mid-caps
        high_vol_slow_ema: int = 20,  # Shorter trend window
        high_vol_volume_threshold: float = 1.8,  # More aggressive volume filter
        high_vol_profit_target: float = 0.23,  # Higher profit target
        high_vol_stop_loss: float = 0.07,  # Wider stop for volatility
        # Medium volatility parameters
        med_vol_fast_ema: int = 9,
        med_vol_slow_ema: int = 28,
        med_vol_volume_threshold: float = 1.4,
        med_vol_profit_target: float = 0.13,
        med_vol_stop_loss: float = 0.04,
        # Low volatility parameters
        low_vol_fast_ema: int = 11,
        low_vol_slow_ema: int = 35,
        low_vol_volume_threshold: float = 1.1,
        low_vol_profit_target: float = 0.11,
        low_vol_stop_loss: float = 0.02,
        # MACD parameters
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        super().__init__()
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.market_cap = market_cap
        
        # Adjust parameters based on market cap
        if market_cap == 'mid':
            # More aggressive parameters for mid-caps
            high_vol_fast_ema = 6
            high_vol_slow_ema = 20
            high_vol_profit_target = 0.23
            high_vol_volume_threshold = 1.8
        
        # Store volatility-based parameters
        self.vol_params = {
            'high': {
                'fast_ema': high_vol_fast_ema,
                'slow_ema': high_vol_slow_ema,
                'volume_threshold': high_vol_volume_threshold,
                'profit_target': high_vol_profit_target,
                'stop_loss': high_vol_stop_loss
            },
            'medium': {
                'fast_ema': med_vol_fast_ema,
                'slow_ema': med_vol_slow_ema,
                'volume_threshold': med_vol_volume_threshold,
                'profit_target': med_vol_profit_target,
                'stop_loss': med_vol_stop_loss
            },
            'low': {
                'fast_ema': low_vol_fast_ema,
                'slow_ema': low_vol_slow_ema,
                'volume_threshold': low_vol_volume_threshold,
                'profit_target': low_vol_profit_target,
                'stop_loss': low_vol_stop_loss
            }
        }
        
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def _calculate_market_conditions(self, df: pd.DataFrame) -> tuple:
        """Calculate market conditions including volatility and volume regimes"""
        # Calculate annualized volatility
        volatility = df['Returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Calculate volume regime
        volume_ma = df['Volume'].rolling(window=self.volume_window).mean()
        volume_ratio = df['Volume'] / volume_ma
        
        # Determine volume regime (prefer extremes, avoid medium)
        if volume_ratio.iloc[-1] > 1.8:
            volume_regime = 'high'
        elif volume_ratio.iloc[-1] < 0.7:
            volume_regime = 'low'
        else:
            volume_regime = 'medium'
            
        # Determine volatility regime
        if volatility.iloc[-1] >= self.high_vol_threshold:
            vol_regime = 'high'
        elif volatility.iloc[-1] <= self.low_vol_threshold:
            vol_regime = 'low'
        else:
            vol_regime = 'medium'
            
        return vol_regime, volume_regime, volatility.iloc[-1]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate market conditions
        vol_regime, volume_regime, current_vol = self._calculate_market_conditions(df)
        params = self.vol_params[vol_regime]
        
        # Adjust parameters based on volume regime
        if volume_regime != 'medium':  # Prefer trading in high/low volume
            params['volume_threshold'] *= 0.9  # More lenient volume threshold
        else:
            params['volume_threshold'] *= 1.2  # Stricter threshold in medium volume
        
        # Calculate core indicators
        df['EMA_fast'] = df['Close'].ewm(span=params['fast_ema'], adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=params['slow_ema'], adjust=False).mean()
        df['EMA_diff'] = (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow'] * 100
        
        # Enhanced MACD with market cap adjustment
        if self.market_cap == 'mid':
            # More responsive MACD for mid-caps
            df['MACD'] = (df['Close'].ewm(span=10, adjust=False).mean() - 
                         df['Close'].ewm(span=20, adjust=False).mean())
        else:
            df['MACD'] = (df['Close'].ewm(span=self.macd_fast, adjust=False).mean() - 
                         df['Close'].ewm(span=self.macd_slow, adjust=False).mean())
        
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_window).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Store current parameters and conditions
        df['Current_Profit_Target'] = params['profit_target']
        df['Current_Stop_Loss'] = params['stop_loss']
        df['Current_Volume_Threshold'] = params['volume_threshold']
        df['Volatility_Regime'] = vol_regime
        df['Volume_Regime'] = volume_regime
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        df['Entry_Price'] = 0.0
        last_signal = 0
        entry_price = 0.0
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            
            # Get current parameters
            profit_target = df['Current_Profit_Target'].iloc[i]
            stop_loss = df['Current_Stop_Loss'].iloc[i]
            volume_threshold = df['Current_Volume_Threshold'].iloc[i]
            
            # Position management
            if last_signal != 0:
                price_change = (current_price - entry_price) / entry_price
                
                # Dynamic exit based on market conditions
                if df['Volume_Regime'].iloc[i] == 'high':
                    # Be more patient with exits in high volume
                    profit_target *= 1.2
                    stop_loss *= 1.1
                
                # Check exits
                if (last_signal == 1 and (price_change <= -stop_loss or price_change >= profit_target)) or \
                   (last_signal == -1 and (-price_change <= -stop_loss or -price_change >= profit_target)):
                    df.loc[df.index[i], 'Signal'] = -last_signal
                    last_signal = 0
                    continue
            
            # Enhanced trend conditions
            trend_up = (df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and 
                       df['MACD_Hist'].iloc[i] > 0 and
                       df['MACD_Hist'].iloc[i] > df['MACD_Hist'].iloc[i-1])
            
            trend_down = (df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and 
                         df['MACD_Hist'].iloc[i] < 0 and
                         df['MACD_Hist'].iloc[i] < df['MACD_Hist'].iloc[i-1])
            
            # Volume confirmation
            volume_confirmed = df['Volume_Ratio'].iloc[i] > volume_threshold
            
            # Trend strength with market cap adjustment
            trend_strength = abs(df['EMA_diff'].iloc[i]) > (0.4 if self.market_cap == 'mid' else 0.6)
            
            # Signal generation with market condition adjustments
            if last_signal == 0:  # Not in a position
                # More aggressive in high volume regimes
                if df['Volume_Regime'].iloc[i] == 'high':
                    trend_strength_threshold = 0.8 * (0.4 if self.market_cap == 'mid' else 0.6)
                else:
                    trend_strength_threshold = (0.4 if self.market_cap == 'mid' else 0.6)
                
                if trend_up and volume_confirmed and abs(df['EMA_diff'].iloc[i]) > trend_strength_threshold:
                    df.loc[df.index[i], 'Signal'] = 1
                    last_signal = 1
                    entry_price = current_price
                elif trend_down and volume_confirmed and abs(df['EMA_diff'].iloc[i]) > trend_strength_threshold:
                    df.loc[df.index[i], 'Signal'] = -1
                    last_signal = -1
                    entry_price = current_price
            
            # Track entry prices
            if df['Signal'].iloc[i] != 0:
                df.loc[df.index[i], 'Entry_Price'] = current_price
        
        return df
# Robust Trend Strategy - Simplified and reliable
class RobustTrend(BaseStrategy):
    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        atr_period: int = 14,
        volume_period: int = 20,
        volume_threshold: float = 1.5,
        atr_stop_multiplier: float = 2.0,
        atr_target_multiplier: float = 3.0,
        min_trend_strength: float = 0.02,
        max_position_risk: float = 0.02
    ):
        super().__init__()
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier
        self.min_trend_strength = min_trend_strength
        self.max_position_risk = max_position_risk

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Core trend indicators - simple and reliable
        df['EMA_fast'] = df['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['EMA_diff'] = (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow']
        
        # ATR for dynamic position sizing and stops
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=self.atr_period).mean()
        
        # Volume confirmation - simple average
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Trend momentum - price above/below recent highs/lows
        df['Recent_High'] = df['High'].rolling(window=5).max().shift(1)
        df['Recent_Low'] = df['Low'].rolling(window=5).min().shift(1)
        
        # Trend strength filter
        df['Trend_Strength'] = abs(df['EMA_diff'])
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        df['Entry_Price'] = 0.0
        df['Stop_Loss'] = 0.0
        df['Profit_Target'] = 0.0
        
        last_signal = 0
        entry_price = 0.0
        stop_loss = 0.0
        profit_target = 0.0
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            # Skip if ATR is NaN or zero
            if pd.isna(atr) or atr == 0:
                continue
            
            # Position management - check exits first
            if last_signal != 0:
                # Check stop loss and profit target
                if last_signal == 1:  # Long position
                    if current_price <= stop_loss or current_price >= profit_target:
                        df.loc[df.index[i], 'Signal'] = -1  # Exit long
                        last_signal = 0
                        continue
                elif last_signal == -1:  # Short position
                    if current_price >= stop_loss or current_price <= profit_target:
                        df.loc[df.index[i], 'Signal'] = 1  # Exit short
                        last_signal = 0
                        continue
            
            # Entry conditions - only when not in position
            if last_signal == 0:
                # Basic trend conditions
                trend_up = (df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and 
                           df['EMA_fast'].iloc[i] > df['EMA_fast'].iloc[i-1])
                
                trend_down = (df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and 
                             df['EMA_fast'].iloc[i] < df['EMA_fast'].iloc[i-1])
                
                # Trend strength filter
                strong_trend = df['Trend_Strength'].iloc[i] > self.min_trend_strength
                
                # Volume confirmation
                volume_confirmed = df['Volume_Ratio'].iloc[i] > self.volume_threshold
                
                # Breakout confirmation
                breakout_up = current_price > df['Recent_High'].iloc[i]
                breakout_down = current_price < df['Recent_Low'].iloc[i]
                
                # Generate entry signals
                if trend_up and strong_trend and volume_confirmed and breakout_up:
                    df.loc[df.index[i], 'Signal'] = 1
                    last_signal = 1
                    entry_price = current_price
                    stop_loss = entry_price - (atr * self.atr_stop_multiplier)
                    profit_target = entry_price + (atr * self.atr_target_multiplier)
                    
                elif trend_down and strong_trend and volume_confirmed and breakout_down:
                    df.loc[df.index[i], 'Signal'] = -1
                    last_signal = -1
                    entry_price = current_price
                    stop_loss = entry_price + (atr * self.atr_stop_multiplier)
                    profit_target = entry_price - (atr * self.atr_target_multiplier)
            
            # Record entry details
            if df['Signal'].iloc[i] != 0:
                df.loc[df.index[i], 'Entry_Price'] = entry_price
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Profit_Target'] = profit_target
        
        return df

# Robust Trend 2.0 - Enhanced with momentum filters and risk management
class RobustTrend2(BaseStrategy):
    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        trend_ema: int = 50,  # Long-term trend filter
        atr_period: int = 14,
        volume_period: int = 20,
        volume_threshold: float = 1.3,  # Slightly more lenient
        atr_stop_multiplier: float = 2.5,  # Wider stops
        atr_target_multiplier: float = 4.0,  # Better R:R ratio
        min_trend_strength: float = 0.015,  # Slightly more lenient
        momentum_period: int = 10,  # Price momentum lookback
        min_momentum: float = 0.02,  # Minimum momentum filter
        max_position_risk: float = 0.02,
        trend_alignment_required: bool = True  # Require alignment with long-term trend
    ):
        super().__init__()
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.trend_ema = trend_ema
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier
        self.min_trend_strength = min_trend_strength
        self.momentum_period = momentum_period
        self.min_momentum = min_momentum
        self.max_position_risk = max_position_risk
        self.trend_alignment_required = trend_alignment_required

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Core trend indicators
        df['EMA_fast'] = df['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['EMA_trend'] = df['Close'].ewm(span=self.trend_ema, adjust=False).mean()
        df['EMA_diff'] = (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow']
        
        # ATR for dynamic position sizing and stops
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=self.atr_period).mean()
        
        # Volume confirmation with smoothing
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Smooth'] = df['Volume_Ratio'].rolling(window=3).mean()  # Smooth volume spikes
        
        # Enhanced momentum indicators
        df['Price_Momentum'] = (df['Close'] - df['Close'].shift(self.momentum_period)) / df['Close'].shift(self.momentum_period)
        df['EMA_Momentum'] = (df['EMA_fast'] - df['EMA_fast'].shift(3)) / df['EMA_fast'].shift(3)
        
        # Trend strength and direction
        df['Trend_Strength'] = abs(df['EMA_diff'])
        df['Long_Trend_Bull'] = df['Close'] > df['EMA_trend']
        df['Long_Trend_Bear'] = df['Close'] < df['EMA_trend']
        
        # Support/Resistance levels
        df['Recent_High'] = df['High'].rolling(window=8).max().shift(1)  # Longer lookback
        df['Recent_Low'] = df['Low'].rolling(window=8).min().shift(1)
        
        # Volatility normalization
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Volatility_Rank'] = df['Price_Range'].rolling(window=20).rank(pct=True)
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        df['Entry_Price'] = 0.0
        df['Stop_Loss'] = 0.0
        df['Profit_Target'] = 0.0
        df['Trade_Reason'] = ''
        
        last_signal = 0
        entry_price = 0.0
        stop_loss = 0.0
        profit_target = 0.0
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            # Skip if insufficient data
            if pd.isna(atr) or atr == 0 or i < self.trend_ema:
                continue
            
            # Position management - check exits first
            if last_signal != 0:
                # Dynamic stop adjustment based on volatility
                volatility_adj = 1.0
                if df['Volatility_Rank'].iloc[i] > 0.8:  # High volatility
                    volatility_adj = 1.2
                elif df['Volatility_Rank'].iloc[i] < 0.2:  # Low volatility
                    volatility_adj = 0.8
                
                adjusted_stop = stop_loss * volatility_adj if last_signal == 1 else stop_loss / volatility_adj
                
                # Check exits with dynamic stops
                if last_signal == 1:  # Long position
                    if current_price <= adjusted_stop or current_price >= profit_target:
                        df.loc[df.index[i], 'Signal'] = -1
                        df.loc[df.index[i], 'Trade_Reason'] = 'Exit Long'
                        last_signal = 0
                        continue
                elif last_signal == -1:  # Short position
                    if current_price >= adjusted_stop or current_price <= profit_target:
                        df.loc[df.index[i], 'Signal'] = 1
                        df.loc[df.index[i], 'Trade_Reason'] = 'Exit Short'
                        last_signal = 0
                        continue
            
            # Entry conditions - only when not in position
            if last_signal == 0:
                # Basic trend conditions
                trend_up = (df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and 
                           df['EMA_fast'].iloc[i] > df['EMA_fast'].iloc[i-1])
                
                trend_down = (df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and 
                             df['EMA_fast'].iloc[i] < df['EMA_fast'].iloc[i-1])
                
                # Long-term trend alignment
                if self.trend_alignment_required:
                    trend_aligned_up = df['Long_Trend_Bull'].iloc[i]
                    trend_aligned_down = df['Long_Trend_Bear'].iloc[i]
                else:
                    trend_aligned_up = trend_aligned_down = True
                
                # Enhanced filters
                strong_trend = df['Trend_Strength'].iloc[i] > self.min_trend_strength
                volume_confirmed = df['Volume_Smooth'].iloc[i] > self.volume_threshold
                momentum_confirmed = abs(df['Price_Momentum'].iloc[i]) > self.min_momentum
                ema_momentum_confirmed = abs(df['EMA_Momentum'].iloc[i]) > 0.005
                
                # Breakout confirmation with better levels
                breakout_up = current_price > df['Recent_High'].iloc[i]
                breakout_down = current_price < df['Recent_Low'].iloc[i]
                
                # Volatility filter - avoid extreme volatility
                volatility_ok = 0.2 <= df['Volatility_Rank'].iloc[i] <= 0.8
                
                # Generate entry signals with comprehensive filters
                long_conditions = (trend_up and trend_aligned_up and strong_trend and 
                                 volume_confirmed and momentum_confirmed and 
                                 ema_momentum_confirmed and breakout_up and volatility_ok)
                
                short_conditions = (trend_down and trend_aligned_down and strong_trend and 
                                  volume_confirmed and momentum_confirmed and 
                                  ema_momentum_confirmed and breakout_down and volatility_ok)
                
                if long_conditions:
                    df.loc[df.index[i], 'Signal'] = 1
                    df.loc[df.index[i], 'Trade_Reason'] = 'Long Entry'
                    last_signal = 1
                    entry_price = current_price
                    stop_loss = entry_price - (atr * self.atr_stop_multiplier)
                    profit_target = entry_price + (atr * self.atr_target_multiplier)
                    
                elif short_conditions:
                    df.loc[df.index[i], 'Signal'] = -1
                    df.loc[df.index[i], 'Trade_Reason'] = 'Short Entry'
                    last_signal = -1
                    entry_price = current_price
                    stop_loss = entry_price + (atr * self.atr_stop_multiplier)
                    profit_target = entry_price - (atr * self.atr_target_multiplier)
            
            # Record entry details
            if df['Signal'].iloc[i] != 0:
                df.loc[df.index[i], 'Entry_Price'] = entry_price
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Profit_Target'] = profit_target
        
        return df

# Robust Trend 3.0 - Optimized for real trading with smart position sizing
class RobustTrend3(BaseStrategy):
    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        atr_period: int = 14,
        volume_period: int = 20,
        volume_threshold: float = 1.5,
        atr_stop_multiplier: float = 2.0,
        atr_target_multiplier: float = 3.0,
        min_trend_strength: float = 0.02,
        max_position_risk: float = 0.02,  # Max 2% risk per trade
        base_position_size: float = 0.1,  # Base 10% of portfolio
        volatility_lookback: int = 20,
        min_hold_period: int = 3,  # Minimum days to hold position
        transaction_cost: float = 0.001,  # 0.1% transaction cost
    ):
        super().__init__()
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier
        self.min_trend_strength = min_trend_strength
        self.max_position_risk = max_position_risk
        self.base_position_size = base_position_size
        self.volatility_lookback = volatility_lookback
        self.min_hold_period = min_hold_period
        self.transaction_cost = transaction_cost

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Core trend indicators (keep it simple)
        df['EMA_fast'] = df['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['EMA_diff'] = (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow']
        
        # ATR for position sizing
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=self.atr_period).mean()
        
        # Volatility for position sizing
        df['Volatility'] = df['Returns'].rolling(window=self.volatility_lookback).std() * np.sqrt(252)
        
        # Volume confirmation
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Support/Resistance levels
        df['Recent_High'] = df['High'].rolling(window=5).max().shift(1)
        df['Recent_Low'] = df['Low'].rolling(window=5).min().shift(1)
        
        # Trend strength
        df['Trend_Strength'] = abs(df['EMA_diff'])
        
        # Position sizing calculation
        df['Risk_Adjusted_Size'] = self._calculate_position_size(df)
        
        return df
    
    def _calculate_position_size(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility-adjusted position size"""
        # Base position size adjusted for volatility
        median_vol = df['Volatility'].rolling(window=60).median()
        current_vol = df['Volatility']
        
        # Inverse volatility scaling
        current_vol_safe = current_vol.fillna(median_vol)
        current_vol_safe = current_vol_safe.mask(current_vol_safe == 0, median_vol)
        vol_adjustment = median_vol / current_vol_safe
        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)  # Cap between 50% and 200%
        
        # Risk-based sizing using ATR
        atr_pct = df['ATR'] / df['Close']
        risk_adjustment = self.max_position_risk / (atr_pct * self.atr_stop_multiplier)
        risk_adjustment = np.clip(risk_adjustment, 0.2, 3.0)  # Cap between 20% and 300%
        
        # Combined position size
        position_size = self.base_position_size * vol_adjustment * risk_adjustment
        position_size = np.clip(position_size, 0.02, 0.25)  # Between 2% and 25% of portfolio
        
        return position_size

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['Signal'] = 0
        df['Entry_Price'] = 0.0
        df['Stop_Loss'] = 0.0
        df['Profit_Target'] = 0.0
        df['Position_Size'] = 0.0
        df['Hold_Days'] = 0
        df['Trade_Quality'] = 0.0
        
        last_signal = 0
        entry_price = 0.0
        stop_loss = 0.0
        profit_target = 0.0
        position_size = 0.0
        entry_day = 0
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            # Skip if insufficient data
            if pd.isna(atr) or atr == 0:
                continue
            
            # Update hold days
            if last_signal != 0:
                df.loc[df.index[i], 'Hold_Days'] = i - entry_day
            
            # Position management - check exits first
            if last_signal != 0:
                hold_days = i - entry_day
                
                # Apply transaction costs to exit conditions
                exit_threshold_long = stop_loss * (1 + self.transaction_cost)
                exit_threshold_short = stop_loss * (1 - self.transaction_cost)
                target_threshold_long = profit_target * (1 - self.transaction_cost)
                target_threshold_short = profit_target * (1 + self.transaction_cost)
                
                # Check exits with minimum hold period
                if hold_days >= self.min_hold_period:
                    if last_signal == 1:  # Long position
                        if current_price <= exit_threshold_long or current_price >= target_threshold_long:
                            df.loc[df.index[i], 'Signal'] = -1
                            last_signal = 0
                            continue
                    elif last_signal == -1:  # Short position
                        if current_price >= exit_threshold_short or current_price <= target_threshold_short:
                            df.loc[df.index[i], 'Signal'] = 1
                            last_signal = 0
                            continue
            
            # Entry conditions - only when not in position
            if last_signal == 0:
                # Core trend conditions (keep simple)
                trend_up = (df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and 
                           df['EMA_fast'].iloc[i] > df['EMA_fast'].iloc[i-1])
                
                trend_down = (df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and 
                             df['EMA_fast'].iloc[i] < df['EMA_fast'].iloc[i-1])
                
                # Quality filters
                strong_trend = df['Trend_Strength'].iloc[i] > self.min_trend_strength
                volume_confirmed = df['Volume_Ratio'].iloc[i] > self.volume_threshold
                breakout_up = current_price > df['Recent_High'].iloc[i]
                breakout_down = current_price < df['Recent_Low'].iloc[i]
                
                # Calculate trade quality score (0-1)
                trend_quality = min(df['Trend_Strength'].iloc[i] / 0.05, 1.0)
                volume_quality = min(df['Volume_Ratio'].iloc[i] / 3.0, 1.0)
                volatility_quality = 1.0 - min(df['Volatility'].iloc[i] / 0.5, 1.0)
                
                trade_quality = (trend_quality + volume_quality + volatility_quality) / 3.0
                
                # Only take high-quality trades (score > 0.6)
                quality_threshold = 0.6
                
                # Generate entry signals
                if (trend_up and strong_trend and volume_confirmed and 
                    breakout_up and trade_quality > quality_threshold):
                    
                    df.loc[df.index[i], 'Signal'] = 1
                    last_signal = 1
                    entry_price = current_price
                    entry_day = i
                    stop_loss = entry_price - (atr * self.atr_stop_multiplier)
                    profit_target = entry_price + (atr * self.atr_target_multiplier)
                    position_size = df['Risk_Adjusted_Size'].iloc[i]
                    
                elif (trend_down and strong_trend and volume_confirmed and 
                      breakout_down and trade_quality > quality_threshold):
                    
                    df.loc[df.index[i], 'Signal'] = -1
                    last_signal = -1
                    entry_price = current_price
                    entry_day = i
                    stop_loss = entry_price + (atr * self.atr_stop_multiplier)
                    profit_target = entry_price - (atr * self.atr_target_multiplier)
                    position_size = df['Risk_Adjusted_Size'].iloc[i]
                
                # Record trade quality for analysis
                df.loc[df.index[i], 'Trade_Quality'] = trade_quality
            
            # Record entry/position details
            if df['Signal'].iloc[i] != 0:
                df.loc[df.index[i], 'Entry_Price'] = entry_price
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Profit_Target'] = profit_target
                df.loc[df.index[i], 'Position_Size'] = position_size
        
        return df
