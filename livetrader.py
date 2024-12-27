# livetrader.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List
from strategies import TrendFollowDev

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AlpacaTrader')

class AlpacaTrader:
    def __init__(self, strategy, api_key: str, api_secret: str, paper: bool = True):
        self.strategy = strategy
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        
        # Universe setup
        self.stock_universe = {
            'Tech Large-Cap': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'CSCO'],
            'Tech Mid-Cap': ['AMD', 'CRWD', 'SNOW', 'NET', 'FTNT', 'PANW', 'DDOG'],
            'Finance Large-Cap': ['JPM', 'BAC', 'GS', 'MS', 'BLK', 'SCHW'],
            'Finance Mid-Cap': ['COIN', 'RJF', 'LPLA', 'FDS'],
            'Healthcare Large-Cap': ['JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'TMO'],
            'Healthcare Mid-Cap': ['HOLX', 'CRL', 'WST', 'MTD'],
            'Consumer Staples': ['WMT', 'PG', 'KO', 'MCD', 'COST', 'PEP'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'NKE', 'SBUX', 'TJX'],
            'Industrial': ['CAT', 'DE', 'BA', 'HON', 'URI', 'PWR'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Materials': ['LIN', 'APD', 'ECL', 'FCX', 'NUE']
        }
        
        self.position_size = 0.1  # 10% of portfolio per position
        self.max_positions = 8    # Maximum number of concurrent positions
        
    def get_historical_data(self, symbol: str, lookback_days: int = 100) -> pd.DataFrame:
        """Fetch historical data from Alpaca"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=lookback_days),
                end=datetime.now()
            )
            bars = self.data_client.get_stock_bars(request)
            
            # Convert to DataFrame and calculate returns
            df = bars.df.reset_index()
            df = df.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df['Returns'] = df['Close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def check_market_hours(self) -> bool:
        """Check if market is open"""
        clock = self.trading_client.get_clock()
        return clock.is_open
    
    def get_available_positions(self) -> int:
        """Calculate how many new positions we can open"""
        positions = self.trading_client.get_all_positions()
        return self.max_positions - len(positions)

    def calculate_position_size(self, account_value: float) -> float:
        """Calculate position size based on account value"""
        return account_value * self.position_size

    def manage_existing_positions(self):
        """Manage existing positions based on strategy signals"""
        positions = self.trading_client.get_all_positions()
        
        for position in positions:
            try:
                # Get current data
                data = self.get_historical_data(position.symbol)
                if data is None:
                    continue
                
                # Generate signals
                signals = self.strategy.generate_signals(data)
                last_signal = signals['Signal'].iloc[-1]
                
                # Check for exit signal
                if last_signal == -1 * float(position.side):  # Exit signal
                    self.trading_client.submit_order(
                        MarketOrderRequest(
                            symbol=position.symbol,
                            qty=abs(float(position.qty)),
                            side=OrderSide.SELL if position.side == 'long' else OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                    )
                    logger.info(f"Closed position in {position.symbol}")
                    
            except Exception as e:
                logger.error(f"Error managing position in {position.symbol}: {str(e)}")

    def scan_for_entries(self):
        """Scan universe for new entry opportunities"""
        available_positions = self.get_available_positions()
        if available_positions <= 0:
            return
            
        account = self.trading_client.get_account()
        position_value = self.calculate_position_size(float(account.equity))
        
        for sector, symbols in self.stock_universe.items():
            for symbol in symbols:
                try:
                    # Skip if we already have a position
                    existing_positions = [p.symbol for p in self.trading_client.get_all_positions()]
                    if symbol in existing_positions:
                        continue
                    
                    # Get data and generate signals
                    data = self.get_historical_data(symbol)
                    if data is None:
                        continue
                        
                    signals = self.strategy.generate_signals(data)
                    last_signal = signals['Signal'].iloc[-1]
                    
                    # Check for entry signal
                    if last_signal != 0:
                        # Calculate number of shares
                        current_price = signals['Close'].iloc[-1]
                        shares = int(position_value / current_price)
                        
                        if shares > 0:
                            # Submit order
                            self.trading_client.submit_order(
                                MarketOrderRequest(
                                    symbol=symbol,
                                    qty=shares,
                                    side=OrderSide.BUY if last_signal == 1 else OrderSide.SELL,
                                    time_in_force=TimeInForce.DAY
                                )
                            )
                            logger.info(f"Opened {'long' if last_signal == 1 else 'short'} position in {symbol}")
                            available_positions -= 1
                            
                            if available_positions <= 0:
                                return
                                
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {str(e)}")

    def run(self, interval: int = 300):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        
        while True:
            try:
                if self.check_market_hours():
                    logger.info("Market is open, running strategy...")
                    
                    # Manage existing positions
                    self.manage_existing_positions()
                    
                    # Look for new entries
                    self.scan_for_entries()
                    
                    # Display current portfolio status
                    self.display_portfolio_status()
                else:
                    logger.info("Market is closed.")
                
                time.sleep(interval)  # Wait before next iteration
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait before retry
    
    def display_portfolio_status(self):
        """Display current portfolio status"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            logger.info("\n=== Portfolio Status ===")
            logger.info(f"Equity: ${float(account.equity):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Number of Positions: {len(positions)}")
            
            for position in positions:
                logger.info(f"\n{position.symbol}:")
                logger.info(f"Side: {position.side}")
                logger.info(f"Quantity: {position.qty}")
                logger.info(f"Market Value: ${float(position.market_value):,.2f}")
                logger.info(f"Unrealized P/L: ${float(position.unrealized_pl):,.2f}")
                
        except Exception as e:
            logger.error(f"Error displaying portfolio status: {str(e)}")

# Example usage
if __name__ == "__main__":
    def load_keys(path):
        with open(path, 'r') as f:
            key, secret = f.read().strip().split('\n')
        return key, secret
    
    # Initialize strategy with optimized parameters
    strategy = TrendFollowDev(
        volatility_window=15,
        volume_window=15,
        high_vol_threshold=0.28,
        low_vol_threshold=0.17,
        market_cap='mid',
        high_vol_fast_ema=6,
        high_vol_slow_ema=20,
        high_vol_volume_threshold=1.8,
        high_vol_profit_target=0.23,
        high_vol_stop_loss=0.07,
        med_vol_fast_ema=9,
        med_vol_slow_ema=28,
        med_vol_volume_threshold=1.4,
        med_vol_profit_target=0.13,
        med_vol_stop_loss=0.04,
        low_vol_fast_ema=11,
        low_vol_slow_ema=35,
        low_vol_volume_threshold=1.1,
        low_vol_profit_target=0.11,
        low_vol_stop_loss=0.02
    )
    
    # Load API keys
    api_key, api_secret = load_keys('key.txt')
    
    # Initialize and run trader
    trader = AlpacaTrader(strategy, api_key, api_secret, paper=True)
    trader.run()