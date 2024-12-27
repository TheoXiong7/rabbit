# optimizer.pt
import numpy as np
from strategies import TrendFollow3

class TrendFollowOptimizer:
    def __init__(self, data_retriever, initial_capital=5000):
        self.data_retriever = data_retriever
        self.initial_capital = initial_capital
        self.best_params = None
        self.best_metrics = None
        
        # Test universe - using a subset for faster optimization
        self.test_universe = {
            'Tech Large-Cap': ['AAPL', 'MSFT'],
            'Tech Mid-Cap': ['AMD', 'CRWD'],
            'Finance': ['JPM', 'GS'],
            'Healthcare': ['JNJ', 'UNH'],
        }

    def _generate_param_combinations(self):
        """Generate parameter combinations to test"""
        param_ranges = {
            'volatility_window': [15, 20, 25],
            'volume_window': [15, 20, 25],
            'high_vol_threshold': [0.28, 0.30, 0.32],
            'low_vol_threshold': [0.13, 0.15, 0.17],
            'high_vol_fast_ema': [6, 7, 8],
            'high_vol_slow_ema': [20, 22, 24],
            'high_vol_volume_threshold': [1.8, 2.0, 2.2],
            'high_vol_profit_target': [0.23, 0.25, 0.27],
            'high_vol_stop_loss': [0.07, 0.08, 0.09],
            'med_vol_fast_ema': [9, 10, 11],
            'med_vol_slow_ema': [28, 30, 32],
            'med_vol_volume_threshold': [1.4, 1.5, 1.6],
            'med_vol_profit_target': [0.13, 0.15, 0.17],
            'med_vol_stop_loss': [0.04, 0.05, 0.06],
            'low_vol_fast_ema': [11, 12, 13],
            'low_vol_slow_ema': [33, 35, 37],
            'low_vol_volume_threshold': [1.1, 1.2, 1.3],
            'low_vol_profit_target': [0.09, 0.10, 0.11],
            'low_vol_stop_loss': [0.02, 0.03, 0.04]
        }
        
        # Generate base combinations
        base_params = [
            {
                'volatility_window': p1,
                'volume_window': p2,
                'high_vol_threshold': p3,
                'low_vol_threshold': p4
            }
            for p1 in param_ranges['volatility_window']
            for p2 in param_ranges['volume_window']
            for p3 in param_ranges['high_vol_threshold']
            for p4 in param_ranges['low_vol_threshold']
        ]
        
        # Generate volatility regime specific combinations
        vol_regime_params = [
            {
                'high_vol_fast_ema': h1,
                'high_vol_slow_ema': h2,
                'high_vol_volume_threshold': h3,
                'high_vol_profit_target': h4,
                'high_vol_stop_loss': h5,
                'med_vol_fast_ema': m1,
                'med_vol_slow_ema': m2,
                'med_vol_volume_threshold': m3,
                'med_vol_profit_target': m4,
                'med_vol_stop_loss': m5,
                'low_vol_fast_ema': l1,
                'low_vol_slow_ema': l2,
                'low_vol_volume_threshold': l3,
                'low_vol_profit_target': l4,
                'low_vol_stop_loss': l5
            }
            for h1 in param_ranges['high_vol_fast_ema']
            for h2 in param_ranges['high_vol_slow_ema']
            for h3 in param_ranges['high_vol_volume_threshold']
            for h4 in param_ranges['high_vol_profit_target']
            for h5 in param_ranges['high_vol_stop_loss']
            for m1 in param_ranges['med_vol_fast_ema']
            for m2 in param_ranges['med_vol_slow_ema']
            for m3 in param_ranges['med_vol_volume_threshold']
            for m4 in param_ranges['med_vol_profit_target']
            for m5 in param_ranges['med_vol_stop_loss']
            for l1 in param_ranges['low_vol_fast_ema']
            for l2 in param_ranges['low_vol_slow_ema']
            for l3 in param_ranges['low_vol_volume_threshold']
            for l4 in param_ranges['low_vol_profit_target']
            for l5 in param_ranges['low_vol_stop_loss']
        ][:50]  # Limit combinations for computational efficiency
        
        # Combine parameters
        all_params = []
        for base in base_params[:10]:  # Limit base combinations
            for vol_params in vol_regime_params:
                combined_params = {**base, **vol_params}
                all_params.append(combined_params)
        
        return all_params

    def _test_parameters(self, params):
        """Test a single parameter combination"""
        results = []
        
        # Create strategy instance with current parameters
        strategy = TrendFollow3(**params)
        
        # Test on each stock
        for sector_stocks in self.test_universe.values():
            for symbol in sector_stocks:
                try:
                    # Get data
                    data = self.data_retriever.get_historical_data(symbol, period="1y", interval="1d")
                    
                    # Generate signals and calculate metrics
                    signals_df = strategy.generate_signals(data)
                    metrics = strategy.calculate_portfolio_metrics(signals_df, self.initial_capital)
                    
                    metrics['Symbol'] = symbol
                    results.append(metrics)
                    
                except Exception as e:
                    print(f"Error testing parameters on {symbol}: {str(e)}")
                    continue
        
        if not results:
            return None
        
        # Calculate average metrics
        avg_metrics = {
            'Total Return': np.mean([r['Total Return'] for r in results]),
            'Sharpe Ratio': np.mean([r['Sharpe Ratio'] for r in results]),
            'Max Drawdown': np.mean([r['Max Drawdown'] for r in results]),
            'Win Rate': len([r for r in results if r['Total Return'] > 0]) / len(results)
        }
        
        return avg_metrics

    def optimize(self, metric='Sharpe Ratio'):
        """Run optimization process"""
        print("Starting parameter optimization...")
        param_combinations = self._generate_param_combinations()
        
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        for i, params in enumerate(param_combinations):
            print(f"Testing parameter combination {i+1}/{len(param_combinations)}")
            metrics = self._test_parameters(params)
            
            if metrics is None:
                continue
                
            current_score = metrics[metric]
            if current_score > best_score:
                best_score = current_score
                best_params = params
                best_metrics = metrics
                print(f"New best {metric}: {best_score:.4f}")
                print("Parameters:", best_params)
                print("Metrics:", best_metrics)
        
        self.best_params = best_params
        self.best_metrics = best_metrics
        return best_params, best_metrics

    def get_optimized_strategy(self):
        """Return a TrendFollow3 instance with optimized parameters"""
        if self.best_params is None:
            raise ValueError("Run optimize() first to find best parameters")
        return TrendFollow3(**self.best_params)

# Example usage:
if __name__ == "__main__":
    from data import Data
    
    # Initialize optimizer
    data_retriever = Data()
    optimizer = TrendFollowOptimizer(data_retriever)
    
    # Run optimization
    best_params, best_metrics = optimizer.optimize()
    
    print("\nOptimization complete!")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
        
    print("\nBest metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")