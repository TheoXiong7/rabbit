import pandas as pd
import numpy as np
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from data import Data
from strategies import TrendFollow

class TrendParameterTester:
    def __init__(self):
        self.data_retriever = Data()
        
        # Define parameter ranges to test
        self.parameter_ranges = {
            'fast_ema': [8, 12],             # Reduced to 2 values
            'slow_ema': [25, 35],            # Reduced to 2 values
            'macd_fast': [12],               # Fixed value
            'macd_slow': [26],               # Fixed value
            'macd_signal': [9],              # Fixed value
            'volume_threshold': [1.2, 1.5],   # Keep 2 values
            'profit_target': [0.15],          # Fixed value
            'stop_loss': [0.05]              # Fixed value
        }
        
        # Define initial test stocks (reduced set)
        self.test_stocks = {
            'Tech_Mid': ['SNOW', 'CRWD'],      # Reduced to 2
            'Tech_Large': ['NVDA'],            # Reduced to 1
            'Consumer_Large': ['WMT'],         # Reduced to 1
            'Utilities': ['NEE']               # Reduced to 1
        }

    def generate_parameter_combinations(self) -> List[Dict]:
        """Generate all valid parameter combinations"""
        # Ensure fast_ema < slow_ema and macd_fast < macd_slow
        valid_combinations = []
        
        for params in product(*self.parameter_ranges.values()):
            param_dict = dict(zip(self.parameter_ranges.keys(), params))
            
            if (param_dict['fast_ema'] < param_dict['slow_ema'] and 
                param_dict['macd_fast'] < param_dict['macd_slow']):
                valid_combinations.append(param_dict)
        
        return valid_combinations

    def test_parameters(self, params: Dict, stock_data: pd.DataFrame) -> Dict:
        """Test a single parameter combination on one stock"""
        strategy = TrendFollow(**params)
        
        try:
            signals_df = strategy.generate_signals(stock_data)
            metrics = strategy.calculate_portfolio_metrics(signals_df, initial_capital=100000)
            
            # Add trade metrics
            trades = len(signals_df[signals_df['Signal'] != 0])
            winning_trades = len(signals_df[
                (signals_df['Signal'] != 0) & 
                (signals_df['Returns'].shift(-1) > 0)  # Look at next day's returns
            ])
            
            # Add parameter values directly to metrics
            metrics.update({
                'total_trades': trades,
                'win_rate': winning_trades / trades if trades > 0 else 0,
                'fast_ema': params['fast_ema'],
                'slow_ema': params['slow_ema'],
                'volume_threshold': params['volume_threshold'],
                'macd_fast': params['macd_fast'],
                'macd_slow': params['macd_slow'],
                'macd_signal': params['macd_signal'],
                'profit_target': params['profit_target'],
                'stop_loss': params['stop_loss']
            })
            
            return metrics
            
        except Exception as e:
            print(f"Error testing parameters: {str(e)}")
            return None

    def test_all_combinations(self) -> pd.DataFrame:
        """Test all parameter combinations on all stocks"""
        all_results = []
        param_combinations = self.generate_parameter_combinations()
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for stock_group, symbols in self.test_stocks.items():
            for symbol in symbols:
                try:
                    # Get stock data
                    data = self.data_retriever.get_historical_data(symbol, period="1y", interval="1d")
                    
                    # Test each parameter combination
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = [
                            executor.submit(self.test_parameters, params, data.copy())
                            for params in param_combinations
                        ]
                        
                        for future, params in zip(futures, param_combinations):
                            result = future.result()
                            if result is not None:
                                result.update({
                                    'Symbol': symbol,
                                    'Group': stock_group
                                })
                                all_results.append(result)
                                
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
        
        return pd.DataFrame(all_results)

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Analyze parameter testing results"""
        analysis = {
            'best_overall': {},
            'best_by_group': {},
            'parameter_impact': {},
            'robust_parameters': {}
        }
        
        # Best overall parameters
        best_sharpe = results_df.nlargest(10, 'Sharpe Ratio')
        best_return = results_df.nlargest(10, 'Total Return')
        
        analysis['best_overall'] = {
            'by_sharpe': best_sharpe[['Symbol', 'Group', 'Total Return', 'Sharpe Ratio', 'params']].to_dict('records'),
            'by_return': best_return[['Symbol', 'Group', 'Total Return', 'Sharpe Ratio', 'params']].to_dict('records')
        }
        
        # Best parameters by stock group
        for group in results_df['Group'].unique():
            group_df = results_df[results_df['Group'] == group]
            analysis['best_by_group'][group] = {
                'best_params': group_df.nlargest(5, 'Sharpe Ratio')[
                    ['Symbol', 'Total Return', 'Sharpe Ratio', 'params']
                ].to_dict('records')
            }
        
        # Parameter impact analysis
        for param in self.parameter_ranges.keys():
            param_performance = results_df.groupby(results_df['params'].apply(lambda x: x[param])).agg({
                'Sharpe Ratio': ['mean', 'std'],
                'Total Return': ['mean', 'std'],
                'win_rate': 'mean'
            }).round(4)
            
            analysis['parameter_impact'][param] = param_performance.to_dict()
        
        # Find robust parameters (good performance across different stocks)
        param_stability = results_df.groupby(results_df['params'].apply(str)).agg({
            'Sharpe Ratio': ['mean', 'std'],
            'Total Return': ['mean', 'std'],
            'win_rate': 'mean'
        })
        
        # Get top 10 most robust parameter sets (high mean, low std)
        param_stability['robustness_score'] = (
            param_stability[('Sharpe Ratio', 'mean')] / 
            param_stability[('Sharpe Ratio', 'std')]
        )
        
        analysis['robust_parameters'] = param_stability.nlargest(
            10, ('robustness_score')
        ).to_dict()
        
        return analysis

if __name__ == "__main__":
    # Initialize and run parameter tests
    tester = TrendParameterTester()
    results_df = tester.test_all_combinations()
    analysis = tester.analyze_results(results_df)
    
    # Print best overall parameters
    print("\nTop 10 Parameter Sets by Sharpe Ratio:")
    for i, result in enumerate(analysis['best_overall']['by_sharpe'], 1):
        print(f"\n{i}. {result['Symbol']} ({result['Group']}):")
        print(f"Return: {result['Total Return']:.2%}")
        print(f"Sharpe: {result['Sharpe Ratio']:.2f}")
        print("Parameters:")
        for param, value in result['params'].items():
            print(f"  {param}: {value}")
    
    # Print best parameters by group
    print("\nBest Parameters by Stock Group:")
    for group, results in analysis['best_by_group'].items():
        print(f"\n{group}:")
        for result in results['best_params'][:3]:  # Top 3 for each group
            print(f"\n{result['Symbol']}:")
            print(f"Return: {result['Total Return']:.2%}")
            print(f"Sharpe: {result['Sharpe Ratio']:.2f}")
            print("Parameters:")
            for param, value in result['params'].items():
                print(f"  {param}: {value}")
    
    # Print parameter impact analysis
    print("\nParameter Impact Analysis:")
    for param, impact in analysis['parameter_impact'].items():
        print(f"\n{param}:")
        print("Value  |  Mean Sharpe  |  Mean Return  |  Win Rate")
        print("-" * 50)
        for value, metrics in impact['Sharpe Ratio'].items():
            print(f"{value:5}  |  {metrics:.2f}  |  {impact['Total Return'][value]:.2%}  |  {impact['win_rate'][value]:.2%}")
    
    # Print most robust parameters
    print("\nMost Robust Parameter Sets:")
    for i, (params, metrics) in enumerate(analysis['robust_parameters'].items(), 1):
        print(f"\n{i}. Parameters: {params}")
        print(f"Mean Sharpe: {metrics['Sharpe Ratio']['mean']:.2f}")
        print(f"Sharpe Std: {metrics['Sharpe Ratio']['std']:.2f}")
        print(f"Mean Return: {metrics['Total Return']['mean']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")