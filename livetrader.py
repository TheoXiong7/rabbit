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
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AlpacaTrader')

class AlpacaTrader:
    def __init__(self, strategy, api_key: str, api_secret: str, paper: bool = True):
        self.strategy = strategy
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self.performance_tracker = PerformanceTracker()
        self.strategy_name = strategy.__class__.__name__
        
        # 30-40 bluechip stocks
        self.stock_universe = {
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'ORCL', 'CRM', 'ADBE'],
            'Finance': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'AXP'],
            'Healthcare': ['UNH', 'JNJ', 'ABBV', 'LLY', 'TMO', 'DHR'],
            'Consumer': ['AMZN', 'TSLA', 'WMT', 'COST', 'MCD', 'NKE'],
            'Industrial': ['CAT', 'DE', 'BA', 'HON', 'UNP', 'RTX', 'ENVX', 'LUNR'],
            'Energy': ['XOM', 'CVX', 'COP'],
            'Communication': ['NFLX', 'DIS', 'CMCSA']
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
                    
                    pnl_color = Fore.GREEN if realized_pnl >= 0 else Fore.RED
                    pnl_symbol = "📈" if realized_pnl >= 0 else "📉"
                    print(f"{Fore.YELLOW}🔚 CLOSED{Style.RESET_ALL} {position.symbol} @ ${exit_price:.2f} | {pnl_color}{pnl_symbol} P&L: ${realized_pnl:.2f}{Style.RESET_ALL}")
                    
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
                            
                            position_type = "LONG" if last_signal == 1 else "SHORT"
                            position_emoji = "🟢" if last_signal == 1 else "🔴"
                            print(f"{Fore.CYAN}🚀 OPENED{Style.RESET_ALL} {position_emoji} {position_type} {symbol} | {shares} shares @ ${current_price:.2f}")
                            available_positions -= 1
                            
                            if available_positions <= 0:
                                return
                                
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {str(e)}")

    def run(self, interval: int = 60):
        """Main trading loop"""
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🤖 LIVE TRADER STARTED{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Strategy: {self.strategy_name}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}\n")
        
        while True:
            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{Fore.BLUE}⏰ {current_time}{Style.RESET_ALL}")
                
                if self.check_market_hours():
                    print(f"{Fore.GREEN}🟢 Market OPEN - Running strategy...{Style.RESET_ALL}")
                    
                    self.manage_existing_positions()
                    self.scan_for_entries()
                    self.log_portfolio_snapshot()
                    # self.display_metrics()
                    self.display_portfolio_status()
                    # self.display_strategy_info()

                    # Log strategy performance every hour
                    if datetime.now().minute == 0:
                        self.performance_tracker.log_strategy_performance(self.strategy_name)

                else:
                    print(f"{Fore.RED}🔴 Market CLOSED - Waiting...{Style.RESET_ALL}")
                
                time.sleep(interval)  # Wait before next iteration
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait before retry

    def display_metrics(self, days: int = 30):
        """Display concise trading metrics and signals"""
        try:
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 MARKET SIGNALS ({days}D){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")

            all_metrics = []
            active_signals = {'BUY': [], 'SELL': []}
            
            for sector, symbols in self.stock_universe.items():
                for symbol in symbols:
                    try:
                        data = self.get_historical_data(symbol, days)
                        if data is None:
                            continue
                            
                        signals = self.strategy.generate_signals(data)
                        latest_signal = signals['Signal'].iloc[-1]
                        current_price = signals['Close'].iloc[-1]
                        
                        # Only track stocks with active signals
                        if latest_signal == 1:
                            active_signals['BUY'].append((symbol, current_price, sector))
                        elif latest_signal == -1:
                            active_signals['SELL'].append((symbol, current_price, sector))
                        
                        # Store for summary stats
                        signal_returns = signals[signals['Signal'] != 0]['Returns']
                        avg_return = signal_returns.mean() * 100 if len(signal_returns) > 0 else 0.0
                        volatility = signals['Returns'].std() * np.sqrt(252) * 100
                        
                        all_metrics.append({
                            'Symbol': symbol, 'Sector': sector, 'Signal': latest_signal,
                            'Price': current_price, 'Avg_Return': avg_return, 'Volatility': volatility
                        })
                        
                    except Exception as e:
                        logger.debug(f"Error processing {symbol}: {str(e)}")
                        continue
            
            # Display active signals concisely
            if active_signals['BUY']:
                print(f"\n{Fore.GREEN}🟢 BUY SIGNALS ({len(active_signals['BUY'])}){Style.RESET_ALL}")
                for symbol, price, sector in active_signals['BUY'][:5]:  # Top 5
                    print(f"{symbol:<6} ${price:>7.2f} ({sector})")
                if len(active_signals['BUY']) > 5:
                    print(f"   ... +{len(active_signals['BUY']) - 5} more")
            
            if active_signals['SELL']:
                print(f"\n{Fore.RED}🔴 SELL SIGNALS ({len(active_signals['SELL'])}){Style.RESET_ALL}")
                for symbol, price, sector in active_signals['SELL'][:5]:  # Top 5
                    print(f"{symbol:<6} ${price:>7.2f} ({sector})")
                if len(active_signals['SELL']) > 5:
                    print(f"   ... +{len(active_signals['SELL']) - 5} more")
            
            # Quick market summary
            if all_metrics:
                market_df = pd.DataFrame(all_metrics)
                total_signals = len(active_signals['BUY']) + len(active_signals['SELL'])
                avg_return = market_df['Avg_Return'].mean()
                avg_vol = market_df['Volatility'].mean()
                
                print(f"\n{Fore.WHITE}📈 MARKET SUMMARY{Style.RESET_ALL}")
                print(f"Stocks Scanned: {len(market_df)} | Active Signals: {total_signals}")
                print(f"Avg Return: {avg_return:.1f}% | Avg Volatility: {avg_vol:.1f}%")
            
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
                    
        except Exception as e:
            logger.error(f"Error displaying metrics: {str(e)}")

    def display_portfolio_status(self):
        """Display current portfolio status"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}💼 PORTFOLIO STATUS{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
            # Account summary
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            daily_pnl = float(account.equity) - float(account.last_equity) if hasattr(account, 'last_equity') else 0
            
            print(f"{Fore.WHITE}💰 Equity:{Style.RESET_ALL} ${equity:,.2f}")
            print(f"{Fore.WHITE}💳 Buying Power:{Style.RESET_ALL} ${buying_power:,.2f}")
            
            daily_pnl_color = Fore.GREEN if daily_pnl >= 0 else Fore.RED
            daily_pnl_symbol = "📈" if daily_pnl >= 0 else "📉"
            print(f"{Fore.WHITE}📊 Daily P&L:{Style.RESET_ALL} {daily_pnl_color}{daily_pnl_symbol} ${daily_pnl:,.2f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}📍 Positions:{Style.RESET_ALL} {len(positions)}/{self.max_positions}")
            
            # Individual positions
            if positions:
                print(f"\n{Fore.YELLOW}📋 ACTIVE POSITIONS:{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{'-'*50}{Style.RESET_ALL}")
                
                total_unrealized = 0
                for position in positions:
                    unrealized_pnl = float(position.unrealized_pl)
                    total_unrealized += unrealized_pnl
                    
                    pnl_color = Fore.GREEN if unrealized_pnl >= 0 else Fore.RED
                    pnl_symbol = "📈" if unrealized_pnl >= 0 else "📉"
                    side_emoji = "🟢" if position.side == 'long' else "🔴"
                    
                    print(f"{side_emoji} {position.symbol} | {position.side.upper()} | {position.qty} shares")
                    print(f"   💵 Value: ${float(position.market_value):,.2f} | {pnl_color}{pnl_symbol} P&L: ${unrealized_pnl:,.2f}{Style.RESET_ALL}")
                
                total_pnl_color = Fore.GREEN if total_unrealized >= 0 else Fore.RED
                total_pnl_symbol = "📈" if total_unrealized >= 0 else "📉"
                print(f"\n{Fore.WHITE}💯 Total Unrealized P&L:{Style.RESET_ALL} {total_pnl_color}{total_pnl_symbol} ${total_unrealized:,.2f}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}📭 No active positions{Style.RESET_ALL}")
            
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
                
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
            print("Daily Performance Report:")
            print(report)
            return report
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return None

    def display_strategy_info(self):
        """Display comprehensive strategy configuration and settings"""
        try:
            print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}🎯 STRATEGY CONFIGURATION{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
            
            # Strategy basics
            print(f"{Fore.WHITE}📈 Strategy:{Style.RESET_ALL} {self.strategy_name}")

            # Strategy parameters
            print(f"\n{Fore.YELLOW}⚙️  STRATEGY PARAMETERS{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'-'*40}{Style.RESET_ALL}")
            if hasattr(self.strategy, 'fast_ema'):
                print(f"Fast EMA: {self.strategy.fast_ema} periods")
                print(f"Slow EMA: {self.strategy.slow_ema} periods")
                print(f"ATR Period: {self.strategy.atr_period} periods")
                print(f"Volume Period: {self.strategy.volume_period} periods")
                print(f"Volume Threshold: {self.strategy.volume_threshold}x")
                print(f"Stop Loss: {self.strategy.atr_stop_multiplier}x ATR")
                print(f"Profit Target: {self.strategy.atr_target_multiplier}x ATR")
            
            # Trading configuration
            print(f"\n{Fore.CYAN}🎛️  TRADING CONFIGURATION{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
            print(f"Position Size: {self.position_size*100:.0f}% per trade")
            print(f"Max Positions: {self.max_positions}")
            print(f"Data Lookback: 100 days")
            print(f"Trading Mode: {'Paper' if True else 'Live'}")
            
            # Universe breakdown
            print(f"\n{Fore.GREEN}🌍 STOCK UNIVERSE{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'-'*40}{Style.RESET_ALL}")
            total_stocks = sum(len(symbols) for symbols in self.stock_universe.values())
            print(f"Total Stocks: {total_stocks}")
            for sector, symbols in self.stock_universe.items():
                print(f"{sector}: {len(symbols)} stocks ({', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''})")
            
            # Risk management
            print(f"\n{Fore.RED}🛡️  RISK MANAGEMENT{Style.RESET_ALL}")
            print(f"{Fore.RED}{'-'*40}{Style.RESET_ALL}")
            print(f"Dynamic stop losses via ATR")
            print(f"Volume confirmation required")
            print(f"Breakout confirmation required")
            print(f"Position sizing limits exposure")
            print(f"Diversification across {len(self.stock_universe)} sectors")
            
            # Current market context
            try:
                account = self.trading_client.get_account()
                positions = self.trading_client.get_all_positions()
                available_slots = self.max_positions - len(positions)
                
                print(f"\n{Fore.BLUE}📊 CURRENT STATUS{Style.RESET_ALL}")
                print(f"{Fore.BLUE}{'-'*40}{Style.RESET_ALL}")
                print(f"Account Equity: ${float(account.equity):,.2f}")
                print(f"Active Positions: {len(positions)}/{self.max_positions}")
                print(f"Available Slots: {available_slots}")
                print(f"Position Value: ${float(account.equity) * self.position_size:,.2f} per trade")
                
            except Exception as e:
                logger.debug(f"Could not get current status: {str(e)}")
            
            print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"Error displaying strategy info: {str(e)}")

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
        print(f"{Fore.MAGENTA}🎯 Starting paper trading with RobustTrend strategy...{Style.RESET_ALL}")
        trader.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Trading stopped by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Trading error: {e}{Style.RESET_ALL}")
        logger.error(f"Trading error: {e}")