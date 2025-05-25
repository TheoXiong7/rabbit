# livetrader.py
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import PositionSide
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
from strategies import RobustTrend
from performance_tracker import PerformanceTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AlpacaTrader')

class AlpacaTrader:
    def __init__(self, strategy, api_key: str, api_secret: str, paper: bool = True):
        self.strategy = strategy
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self.performance_tracker = PerformanceTracker()
        self.strategy_name = strategy.__class__.__name__
        
        # Universe setup
        self.stock_universe = {
            'Tech Large-Cap': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'CSCO', 'ORCL',
                               'PLTR', 'INTC', 'CRM', 'ADBE', 'ACN', 'SMCI', 'MU', 'AMD', 'ON',
                               'QCOM', 'UMC', 'MRVL', 'SNOW', 'AMAT', 'TSN'],
            'Tech Mid-Cap': ['CRWD', 'NET', 'FTNT', 'PANW', 'DDOG', 'ZS', 'SNPS', 
                             'CDNS', 'QUBT', 'RGTI', 'QS', 'IONQ', 'RIVN', 'LUNR', 'LCID', 'CFLT',
                             'GTLB', 'TTD', 'SOUN', 'QBTS', 'AI', 'S', 'QRVO', 'GTLB'],
            'Finance Large-Cap': ['JPM', 'BAC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'WFC', 'V', 'MA',
                                  'AXP'],
            'Finance Mid-Cap': ['COIN', 'HOOD', 'RJF', 'SEIC', 'LPLA', 'FDS', 'SOFI', 'AFRM'],
            'Healthcare Large-Cap': ['JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'TMO', 'DHR', 'BMY', 'MRNA',
                                     'MDT', 'BAX'],
            'Healthcare Mid-Cap': ['HOLX', 'VTRS', 'CRL', 'ICLR', 'WST', 'MTD', 'ENVX', 'TDOC',
                                   'VEEV'],
            'Consumer Staples': ['WMT', 'PG', 'KO', 'MCD', 'COST', 'PEP', 'TGT', 'DG'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'NKE', 'SBUX', 'TJX', 'BKNG', 'MAR'],
            'Industrial Large-Cap': ['CAT', 'DE', 'BA', 'HON', 'UNP', 'RTX', 'GE', 'MMM'],
            'Industrial Mid-Cap': ['URI', 'PWR', 'FAST', 'GGG', 'RBC', 'EME', 'IRBT', 'BE'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX'],
            'Materials': ['LIN', 'APD', 'ECL', 'NEM', 'FCX', 'DOW', 'NUE'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'SRE', 'AEP', 'XEL', 'PCG'],
            'Real Estate': ['PLD', 'AMT', 'EQIX', 'PSA', 'O', 'WELL', 'AVB'],
            'Communication': ['VZ', 'T', 'CMCSA', 'NFLX', 'DIS', 'TMUS', 'CHTR']
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
                if ((last_signal == -1 and position.side == PositionSide.LONG) or 
    (last_signal == 1 and position.side == PositionSide.SHORT)):  # Exit signal
                    
                    order = self.trading_client.submit_order(
                        MarketOrderRequest(
                            symbol=position.symbol,
                            qty=abs(float(position.qty)),
                            side=OrderSide.SELL if position.side == 'long' else OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                    )
                    
                    # Log exit trade
                    exit_price = signals['Close'].iloc[-1]  # Current price as proxy
                    realized_pnl = self._calculate_pnl(position, exit_price)
                    
                    self.performance_tracker.log_trade({
                        'symbol': position.symbol,
                        'side': 'LONG' if position.side == PositionSide.LONG else 'SHORT',
                        'action': 'EXIT',
                        'quantity': abs(float(position.qty)),
                        'price': exit_price,
                        'strategy': self.strategy_name,
                        'realized_pnl': realized_pnl,
                        'commission': 0  # Alpaca paper trading has no commission
                    })
                    
                    logger.info(f"Closed position in {position.symbol} at ${exit_price:.2f} | P&L: ${realized_pnl:.2f}")
                    
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
                            order = self.trading_client.submit_order(
                                MarketOrderRequest(
                                    symbol=symbol,
                                    qty=shares,
                                    side=OrderSide.BUY if last_signal == 1 else OrderSide.SELL,
                                    time_in_force=TimeInForce.DAY
                                )
                            )
                            
                            # Extract additional signal data for tracking
                            entry_data = {
                                'symbol': symbol,
                                'side': 'LONG' if last_signal == 1 else 'SHORT',
                                'action': 'ENTRY',
                                'quantity': shares,
                                'price': current_price,
                                'strategy': self.strategy_name,
                                'commission': 0
                            }
                            
                            # Add strategy-specific data if available
                            if 'Entry_Price' in signals.columns:
                                entry_data['entry_price'] = signals['Entry_Price'].iloc[-1]
                            if 'Stop_Loss' in signals.columns:
                                entry_data['stop_loss'] = signals['Stop_Loss'].iloc[-1]
                            if 'Profit_Target' in signals.columns:
                                entry_data['profit_target'] = signals['Profit_Target'].iloc[-1]
                            if 'Position_Size' in signals.columns:
                                entry_data['position_size'] = signals['Position_Size'].iloc[-1]
                            if 'Trade_Quality' in signals.columns:
                                entry_data['trade_quality'] = signals['Trade_Quality'].iloc[-1]
                            
                            self.performance_tracker.log_trade(entry_data)
                            
                            logger.info(f"Opened {'long' if last_signal == 1 else 'short'} position in {symbol} "
                                      f"| {shares} shares @ ${current_price:.2f}")
                            available_positions -= 1
                            
                            if available_positions <= 0:
                                return
                                
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {str(e)}")

    def run(self, interval: int = 60):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        
        while True:
            try:
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                if self.check_market_hours():
                    logger.info("Market is open, running strategy...")
                    
                    self.manage_existing_positions()
                    self.scan_for_entries()
                    self.log_portfolio_snapshot()
                    self.display_portfolio_status()
                    
                    # Log strategy performance every hour
                    if datetime.now().minute == 0:
                        self.performance_tracker.log_strategy_performance(self.strategy_name)

                else:
                    logger.info("Market is closed.")
                
                time.sleep(interval)  # Wait before next iteration
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait before retry

    def display_metrics(self, days: int = 30):
        """
        Display trading metrics and signals for all stocks in universe
        
        Args:
            days (int): Number of days to look back
        """
        try:
            logger.info("\n" + "="*80)
            logger.info(f"TRADING METRICS SUMMARY - {days} Day Analysis")
            logger.info("="*80)

            # Store metrics for all stocks
            all_metrics = []
            
            for sector, symbols in self.stock_universe.items():
                logger.info(f"\n{sector} Sector Analysis:")
                logger.info("-"*80)
                
                sector_metrics = []
                for symbol in symbols:
                    # Get historical data and signals
                    data = self.get_historical_data(symbol, days)
                    if data is None:
                        continue
                        
                    signals = self.strategy.generate_signals(data)
                    
                    # Calculate key metrics
                    metrics = {
                        'Symbol': symbol,
                        'Current_Price': signals['Close'].iloc[-1],
                        'Latest_Signal': signals['Signal'].iloc[-1],
                        'Buy_Signals': len(signals[signals['Signal'] == 1]),
                        'Sell_Signals': len(signals[signals['Signal'] == -1]),
                        'Avg_Return': signals[signals['Signal'] != 0]['Returns'].mean() * 100,
                        'Volatility': signals['Returns'].std() * np.sqrt(252) * 100,
                        'Volume': signals['Volume'].mean(),
                        'High': signals['High'].max(),
                        'Low': signals['Low'].min()
                    }
                    sector_metrics.append(metrics)
                    
                    # Format signal display
                    signal_text = {1: '🟢 BUY', -1: '🔴 SELL', 0: '⚪ HOLD'}[metrics['Latest_Signal']]
                    
                    # Display individual stock metrics
                    logger.info(f"\n{symbol:<6} | Price: ${metrics['Current_Price']:,.2f} | Signal: {signal_text}")
                    logger.info(f"       | Buy Signals: {metrics['Buy_Signals']} | Sell Signals: {metrics['Sell_Signals']}")
                    logger.info(f"       | Avg Return: {metrics['Avg_Return']:,.2f}% | Volatility: {metrics['Volatility']:,.2f}%")
                    
                # Calculate and display sector summary
                if sector_metrics:
                    sector_df = pd.DataFrame(sector_metrics)
                    logger.info(f"\n{sector} Summary:")
                    logger.info(f"Total Buy Signals: {sector_df['Buy_Signals'].sum()}")
                    logger.info(f"Total Sell Signals: {sector_df['Sell_Signals'].sum()}")
                    logger.info(f"Average Sector Return: {sector_df['Avg_Return'].mean():,.2f}%")
                    logger.info(f"Average Sector Volatility: {sector_df['Volatility'].mean():,.2f}%")
                    
                    # Store for overall summary
                    all_metrics.extend(sector_metrics)
            
            # Overall market summary
            if all_metrics:
                market_df = pd.DataFrame(all_metrics)
                logger.info("\n" + "="*80)
                logger.info("OVERALL MARKET SUMMARY")
                logger.info("="*80)
                logger.info(f"Total Stocks Analyzed: {len(market_df)}")
                logger.info(f"Current Buy Signals: {len(market_df[market_df['Latest_Signal'] == 1])}")
                logger.info(f"Current Sell Signals: {len(market_df[market_df['Latest_Signal'] == -1])}")
                logger.info(f"Average Market Return: {market_df['Avg_Return'].mean():,.2f}%")
                logger.info(f"Average Market Volatility: {market_df['Volatility'].mean():,.2f}%")
                
                # Display top opportunities
                logger.info("\nTop 5 Buy Opportunities:")
                buy_opps = market_df[market_df['Latest_Signal'] == 1].nlargest(5, 'Avg_Return')
                for _, row in buy_opps.iterrows():
                    logger.info(f"{row['Symbol']:<6} | Return: {row['Avg_Return']:,.2f}% | Volatility: {row['Volatility']:,.2f}%")
                
                logger.info("\nTop 5 Sell Signals:")
                sell_opps = market_df[market_df['Latest_Signal'] == -1].nsmallest(5, 'Avg_Return')
                for _, row in sell_opps.iterrows():
                    logger.info(f"{row['Symbol']:<6} | Return: {row['Avg_Return']:,.2f}% | Volatility: {row['Volatility']:,.2f}%")
                    
        except Exception as e:
            logger.error(f"Error displaying metrics: {str(e)}")

    def display_portfolio_status(self):
        """Display current portfolio status"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                logger.info(f"\n{position.symbol}:")
                logger.info(f"Side: {position.side}")
                logger.info(f"Quantity: {position.qty}")
                logger.info(f"Market Value: ${float(position.market_value):,.2f}")
                logger.info(f"Unrealized P/L: ${float(position.unrealized_pl):,.2f}")

            logger.info("\n=== Portfolio Status ===")
            logger.info(f"Equity: ${float(account.equity):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Total Account Value: {float(account.equity) + float(account.buying_power):,.2f}")
            logger.info(f"Number of Positions: {len(positions)}")
                
        except Exception as e:
            logger.error(f"Error displaying portfolio status: {str(e)}")
    
    def _calculate_pnl(self, position, exit_price: float) -> float:
        """Calculate realized P&L for a position"""
        try:
            entry_price = float(position.avg_entry_price)
            quantity = abs(float(position.qty))
            
            if position.side == PositionSide.LONG:
                return (exit_price - entry_price) * quantity
            else:
                return (entry_price - exit_price) * quantity
        except Exception:
            return 0.0
    
    def log_portfolio_snapshot(self):
        """Log current portfolio state for performance tracking"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            # Calculate portfolio metrics
            total_equity = float(account.equity)
            buying_power = float(account.buying_power)
            total_positions = len(positions)
            
            # Calculate daily P&L
            daily_pnl = float(account.equity) - float(account.last_equity) if hasattr(account, 'last_equity') else 0
            
            # Calculate unrealized P&L
            unrealized_pnl = sum(float(p.unrealized_pl) for p in positions)
            
            portfolio_data = {
                'total_equity': total_equity,
                'buying_power': buying_power,
                'total_positions': total_positions,
                'daily_pnl': daily_pnl,
                'total_pnl': unrealized_pnl
            }
            
            self.performance_tracker.log_portfolio_snapshot(portfolio_data)
            
        except Exception as e:
            logger.error(f"Error logging portfolio snapshot: {str(e)}")
    
    def generate_performance_report(self):
        """Generate and display performance report"""
        try:
            report = self.performance_tracker.generate_daily_report()
            logger.info("Daily Performance Report:")
            logger.info(report)
            return report
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    def load_keys(path):
        with open(path, 'r') as f:
            key, secret = f.read().strip().split('\n')
        return key, secret
    
    # Initialize strategy with proven parameters
    strategy = RobustTrend(
        fast_ema=10,
        slow_ema=30,
        atr_period=14,
        volume_period=20,
        volume_threshold=1.5,
        atr_stop_multiplier=2.0,
        atr_target_multiplier=3.0
    )
    
    # Load API keys
    api_key, api_secret = load_keys('key.txt')
    
    # Initialize and run trader
    trader = AlpacaTrader(strategy, api_key, api_secret, paper=True)
    
    # Run trading loop
    try:
        print("Starting paper trading with RobustTrend strategy...")
        trader.run_trading_loop()
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        print(f"Trading error: {e}")
        logger.error(f"Trading error: {e}")