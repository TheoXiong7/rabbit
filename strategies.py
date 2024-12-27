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