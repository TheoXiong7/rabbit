# tester.py
import pandas as pd
import numpy as np
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from data import Data
from strategies import TrendFollow, TrendFollow2, TrendFollow3, TrendFollowDev

class MultiStrategyTester:
    def __init__(self):
        self.data_retriever = Data()
        self.strategies = {
            'Developmental Strategy': TrendFollowDev(),
            'TrendFollow': TrendFollow(),
            'TrendFollow 2.0': TrendFollow2(),
            'TrendFollow 3.0': TrendFollow3()
        }
        
        self.test_periods = {
            '1Y': "1y",
            '2Y': "2y",
            '5Y': "5y"
        }
        
        # Stock universe remains the same...
        self.stock_universe = {
            'Tech Large-Cap': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'CSCO', 'ORCL'],
            'Tech Mid-Cap': ['AMD', 'CRWD', 'SNOW', 'NET', 'FTNT', 'PANW', 'DDOG', 'ZS', 'SNPS', 'CDNS'],
            'Finance Large-Cap': ['JPM', 'BAC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'WFC'],
            'Finance Mid-Cap': ['COIN', 'HOOD', 'RJF', 'SEIC', 'LPLA', 'FDS'],
            'Healthcare Large-Cap': ['JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'TMO', 'DHR', 'BMY'],
            'Healthcare Mid-Cap': ['HOLX', 'VTRS', 'CRL', 'ICLR', 'WST', 'MTD'],
            'Consumer Staples': ['WMT', 'PG', 'KO', 'MCD', 'COST', 'PEP', 'TGT', 'DG'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'NKE', 'SBUX', 'TJX', 'BKNG', 'MAR'],
            'Industrial Large-Cap': ['CAT', 'DE', 'BA', 'HON', 'UNP', 'RTX', 'GE', 'MMM'],
            'Industrial Mid-Cap': ['URI', 'PWR', 'FAST', 'GGG', 'RBC', 'EME'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX'],
            'Materials': ['LIN', 'APD', 'ECL', 'NEM', 'FCX', 'DOW', 'NUE'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'SRE', 'AEP', 'XEL', 'PCG'],
            'Real Estate': ['PLD', 'AMT', 'EQIX', 'PSA', 'O', 'WELL', 'AVB'],
            'Communication': ['VZ', 'T', 'CMCSA', 'NFLX', 'DIS', 'TMUS', 'CHTR']
        }

    def calculate_trade_metrics(self, signals_df: pd.DataFrame) -> Dict:
        """Calculate detailed trade metrics"""
        try:
            # Count entries (non-zero signals that are different from previous signal)
            entries = signals_df[signals_df['Signal'] != 0]['Signal'].count()
            
            # Count exits (transitions from non-zero to zero)
            exits = len(signals_df[
                (signals_df['Signal'].shift(1) != 0) & 
                (signals_df['Signal'] == 0)
            ])
            
            # Total actions (entries + exits)
            total_actions = entries + exits
            
            return {
                'total_entries': entries,
                'total_exits': exits,
                'total_actions': total_actions
            }
        except Exception as e:
            print(f"Error calculating trade metrics: {str(e)}")
            return {
                'total_entries': 0,
                'total_exits': 0,
                'total_actions': 0
            }

    def test_strategy_on_stock(self, strategy_name: str, symbol: str, 
                             period: str, initial_capital: float = 5000) -> Dict:
        """Test a single strategy on a single stock"""
        try:
            # Get data
            data = self.data_retriever.get_historical_data(symbol, period=period, interval="1d")
            if data is None or data.empty:
                return None
            
            # Generate signals and calculate metrics
            strategy = self.strategies[strategy_name]
            signals_df = strategy.generate_signals(data)
            
            # Calculate portfolio metrics
            portfolio_metrics = strategy.calculate_portfolio_metrics(signals_df, initial_capital)
            trade_metrics = self.calculate_trade_metrics(signals_df)
            
            # Calculate dollar return
            dollar_return = initial_capital * portfolio_metrics['Total Return']
            
            # Combine all metrics
            combined_metrics = {
                'Symbol': symbol,
                'Strategy': strategy_name,
                'Period': period,
                'Dollar Return': dollar_return,
                'Percent Return': portfolio_metrics['Total Return'],
                **portfolio_metrics,
                **trade_metrics
            }
            
            return combined_metrics
            
        except Exception as e:
            print(f"Error testing {strategy_name} on {symbol} for {period}: {str(e)}")
            return None

    def test_all_combinations(self, initial_capital: float = 5000) -> Dict[str, pd.DataFrame]:
        """Test all strategies on all stocks for different periods"""
        results_by_period = {}
        
        for period_name, period in self.test_periods.items():
            print(f"\nTesting {period_name} period...")
            all_stocks = [stock for stocks in self.stock_universe.values() for stock in stocks]
            all_combinations = [(strategy_name, symbol, period) 
                              for strategy_name in self.strategies.keys() 
                              for symbol in all_stocks]
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(
                    lambda x: self.test_strategy_on_stock(x[0], x[1], x[2], initial_capital),
                    all_combinations
                ))
            
            # Filter out None results and convert to DataFrame
            results = [r for r in results if r is not None]
            
            if results:  # Only create DataFrame if we have results
                results_by_period[period_name] = pd.DataFrame(results)
            else:
                print(f"No valid results for {period_name}")
        
        return results_by_period

    def analyze_results(self, results_by_period: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze results across different periods"""
        analysis = {}
        
        for period, results_df in results_by_period.items():
            if results_df is None or results_df.empty:
                print(f"No data to analyze for {period}")
                continue
                
            try:
                strategy_groups = results_df.groupby('Strategy')
                
                period_analysis = {
                    strategy: {
                        'mean_dollar_return': group['Dollar Return'].mean(),
                        'mean_percent_return': group['Percent Return'].mean(),
                        'total_entries': group['total_entries'].sum(),
                        'total_exits': group['total_exits'].sum(),
                        'total_actions': group['total_actions'].sum(),
                        'mean_sharpe': group['Sharpe Ratio'].mean(),
                        'win_rate': len(group[group['Total Return'] > 0]) / len(group),
                        'return_std': group['Total Return'].std()
                    }
                    for strategy, group in strategy_groups
                }
                
                analysis[period] = period_analysis
                
            except Exception as e:
                print(f"Error analyzing {period} period: {str(e)}")
                continue
        
        return analysis

if __name__ == "__main__":
    # Initialize and run tests
    initial_capital = 5000
    tester = MultiStrategyTester()
    print(f"Testing all strategies across multiple stocks with ${initial_capital:,} initial capital...")
    
    # Run tests for all periods
    results_by_period = tester.test_all_combinations(initial_capital)
    analysis = tester.analyze_results(results_by_period)
    
    # Print detailed results for each period
    for period, period_analysis in analysis.items():
        print(f"\n{'='*20} {period} Period {'='*20}")
        for strategy, metrics in period_analysis.items():
            print(f"\n{strategy}:")
            print(f"Mean Dollar Return: ${metrics['mean_dollar_return']:,.2f}")
            print(f"Mean Percent Return: {metrics['mean_percent_return']:.2%}")
            print(f"Total Entries: {metrics['total_entries']:,}")
            print(f"Total Exits: {metrics['total_exits']:,}")
            print(f"Total Actions: {metrics['total_actions']:,}")
            print(f"Mean Sharpe Ratio: {metrics['mean_sharpe']:.2f}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Return Std Dev: {metrics['return_std']:.2%}")
            print('-' * 50)