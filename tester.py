import pandas as pd
import numpy as np
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from data import Data
from strategies import Breakout, MeanReversion, TrendFollow, Adaptive, TrendFollowOptimized, TrendFollowEnhanced

class MultiStrategyTester:
    def __init__(self):
        self.data_retriever = Data()
        self.strategies = {
            'TrendFollow': TrendFollow(),
            'TrendFollow Optimized': TrendFollowOptimized(),
            'TrendFollow Enhanced' : TrendFollowEnhanced()
        }
        
        # Define stock universe
        self.stock_universe = {
            'Tech Large-Cap': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Tech Mid-Cap': ['AMD', 'CRWD', 'SNOW', 'NET'],
            'Finance': ['JPM', 'BAC', 'GS', 'MS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV'],
            'Consumer': ['WMT', 'PG', 'KO', 'MCD'],
            'Industrial': ['CAT', 'DE', 'BA', 'HON'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D']
        }

    def test_strategy_on_stock(self, strategy_name: str, symbol: str, 
                             initial_capital: float = 100000) -> Dict:
        """Test a single strategy on a single stock"""
        try:
            # Get data
            data = self.data_retriever.get_historical_data(symbol, period="1y", interval="1d")
            
            # Generate signals and calculate metrics
            strategy = self.strategies[strategy_name]
            signals_df = strategy.generate_signals(data)
            metrics = strategy.calculate_portfolio_metrics(signals_df, initial_capital)
            
            # Add identifiers to metrics
            metrics['Symbol'] = symbol
            metrics['Strategy'] = strategy_name
            return metrics
            
        except Exception as e:
            print(f"Error testing {strategy_name} on {symbol}: {str(e)}")
            return None

    def test_all_combinations(self, initial_capital: float = 100000) -> pd.DataFrame:
        """Test all strategies on all stocks"""
        all_stocks = [stock for stocks in self.stock_universe.values() for stock in stocks]
        all_combinations = [(strategy_name, symbol) 
                          for strategy_name in self.strategies.keys() 
                          for symbol in all_stocks]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(
                lambda x: self.test_strategy_on_stock(x[0], x[1], initial_capital),
                all_combinations
            ))
        
        # Filter out None results and convert to DataFrame
        results = [r for r in results if r is not None]
        return pd.DataFrame(results)

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Enhanced analysis including stock characteristics"""
        # Get characteristics for each stock and merge with results
        stock_characteristics = self._calculate_stock_characteristics()
        results_df = pd.merge(results_df, stock_characteristics, on='Symbol', how='left')
        
        analysis = {
            'overall_strategy_performance': {},
            'sector_performance': {},
            'market_cap_performance': {},
            'volatility_performance': {},
            'volume_performance': {},
            'best_combinations': {},
            'strategy_characteristics': {}
        }
        
        # Overall strategy performance
        strategy_groups = results_df.groupby('Strategy')
        analysis['overall_strategy_performance'] = {
            strategy: {
                'mean_return': group['Total Return'].mean(),
                'return_std': group['Total Return'].std(),
                'mean_sharpe': group['Sharpe Ratio'].mean(),
                'win_rate': len(group[group['Total Return'] > 0]) / len(group)
            }
            for strategy, group in strategy_groups
        }
        
        # Performance by market cap and other characteristics
        for strategy in self.strategies.keys():
            strategy_df = results_df[results_df['Strategy'] == strategy].copy()
            
            # Market cap analysis
            if 'market_cap_performance' not in analysis:
                analysis['market_cap_performance'] = {}
            analysis['market_cap_performance'][strategy] = {}
            
            for cap_category in strategy_df['Market_Cap_Category'].unique():
                cap_df = strategy_df[strategy_df['Market_Cap_Category'] == cap_category]
                if not cap_df.empty:
                    analysis['market_cap_performance'][strategy][cap_category] = {
                        'mean_return': cap_df['Total Return'].mean(),
                        'sharpe_ratio': cap_df['Sharpe Ratio'].mean(),
                        'win_rate': len(cap_df[cap_df['Total Return'] > 0]) / len(cap_df)
                    }
            
            # Create volatility buckets
            strategy_df.loc[:, 'Volatility_Bucket'] = pd.qcut(
                strategy_df['Volatility'].rank(method='first'), 
                3, 
                labels=['Low', 'Medium', 'High']
            )
            
            # Create volume buckets
            strategy_df.loc[:, 'Volume_Bucket'] = pd.qcut(
                strategy_df['Avg_Volume'].rank(method='first'), 
                3, 
                labels=['Low', 'Medium', 'High']
            )
            
            # Volatility analysis
            if strategy not in analysis['volatility_performance']:
                analysis['volatility_performance'][strategy] = {}
                
            vol_stats = strategy_df.groupby('Volatility_Bucket', observed=True).agg({
                'Total Return': 'mean',
                'Sharpe Ratio': 'mean'
            })
            analysis['volatility_performance'][strategy] = vol_stats.to_dict()
            
            # Volume analysis
            if strategy not in analysis['volume_performance']:
                analysis['volume_performance'][strategy] = {}
                
            vol_stats = strategy_df.groupby('Volume_Bucket', observed=True).agg({
                'Total Return': 'mean',
                'Sharpe Ratio': 'mean'
            })
            analysis['volume_performance'][strategy] = vol_stats.to_dict()
            
            # Strategy characteristics preference
            top_performers = strategy_df.nlargest(10, 'Sharpe Ratio')
            analysis['strategy_characteristics'][strategy] = {
                'preferred_sectors': top_performers['Sector'].value_counts().nlargest(3).index.tolist(),
                'preferred_market_cap': top_performers['Market_Cap_Category'].value_counts().nlargest(2).index.tolist(),
                'preferred_volatility': top_performers['Volatility_Bucket'].value_counts().nlargest(2).index.tolist(),
                'preferred_volume': top_performers['Volume_Bucket'].value_counts().nlargest(2).index.tolist()
            }
        
        # Best combinations with additional characteristics
        analysis['best_combinations'] = {
            'by_return': results_df.nlargest(10, 'Total Return')[
                ['Strategy', 'Symbol', 'Total Return', 'Sharpe Ratio', 'Market_Cap_Category', 'Sector', 'Volatility']
            ].to_dict('records'),
            'by_sharpe': results_df.nlargest(10, 'Sharpe Ratio')[
                ['Strategy', 'Symbol', 'Total Return', 'Sharpe Ratio', 'Market_Cap_Category', 'Sector', 'Volatility']
            ].to_dict('records')
        }
        
        return analysis

    def _calculate_stock_characteristics(self) -> pd.DataFrame:
        """Calculate various characteristics for each stock"""
        characteristics = []
        
        for sector, symbols in self.stock_universe.items():
            for symbol in symbols:
                try:
                    data = self.data_retriever.get_historical_data(symbol, period="1y", interval="1d")
                    
                    # Calculate volatility
                    volatility = data['Close'].pct_change().std() * np.sqrt(252)
                    
                    # Calculate average volume
                    avg_volume = data['Volume'].mean()
                    
                    # Determine market cap category from sector name
                    if 'Large-Cap' in sector:
                        market_cap = 'Large-Cap'
                    elif 'Mid-Cap' in sector:
                        market_cap = 'Mid-Cap'
                    elif 'Small-Cap' in sector:
                        market_cap = 'Small-Cap'
                    else:
                        market_cap = 'Large-Cap'  # Default for sectors without explicit size
                    
                    # Get base sector without cap size
                    base_sector = sector.split()[0]
                    
                    characteristics.append({
                        'Symbol': symbol,
                        'Sector': base_sector,
                        'Market_Cap_Category': market_cap,
                        'Volatility': volatility,
                        'Avg_Volume': avg_volume
                    })
                    
                except Exception as e:
                    print(f"Error calculating characteristics for {symbol}: {str(e)}")
        
        return pd.DataFrame(characteristics)

if __name__ == "__main__":
    # Initialize and run tests
    tester = MultiStrategyTester()
    print("Testing all strategies across multiple stocks...")
    
    # Run tests
    results_df = tester.test_all_combinations()
    analysis = tester.analyze_results(results_df)
    
    # Print overall strategy performance
    print("\nOverall Strategy Performance:")
    for strategy, metrics in analysis['overall_strategy_performance'].items():
        print(f"\n{strategy}:")
        print(f"Mean Return: {metrics['mean_return']:.2%}")
        print(f"Return Std Dev: {metrics['return_std']:.2%}")
        print(f"Mean Sharpe Ratio: {metrics['mean_sharpe']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    # Print best combinations
    print("\nTop 10 Strategy-Stock Combinations by Return:")
    for combo in analysis['best_combinations']['by_return']:
        print(f"Strategy: {combo['Strategy']}, Symbol: {combo['Symbol']}, "
              f"Return: {combo['Total Return']:.2%}, Sharpe: {combo['Sharpe Ratio']:.2f}, "
              f"Category: {combo['Market_Cap_Category']}, Sector: {combo['Sector']}")
    
    print("\nTop 10 Strategy-Stock Combinations by Sharpe Ratio:")
    for combo in analysis['best_combinations']['by_sharpe']:
        print(f"Strategy: {combo['Strategy']}, Symbol: {combo['Symbol']}, "
              f"Return: {combo['Total Return']:.2%}, Sharpe: {combo['Sharpe Ratio']:.2f}, "
              f"Category: {combo['Market_Cap_Category']}, Sector: {combo['Sector']}")
    
    # Print strategy characteristics
    print("\nStrategy Characteristics:")
    for strategy, characteristics in analysis['strategy_characteristics'].items():
        print(f"\n{strategy}:")
        print(f"Preferred Sectors: {', '.join(characteristics['preferred_sectors'])}")
        print(f"Preferred Market Cap: {', '.join(characteristics['preferred_market_cap'])}")
        print(f"Preferred Volatility: {', '.join(characteristics['preferred_volatility'])}")
        print(f"Preferred Volume: {', '.join(characteristics['preferred_volume'])}")
    
    # Print market cap performance
    print("\nMarket Cap Performance by Strategy:")
    for strategy, cap_performance in analysis['market_cap_performance'].items():
        print(f"\n{strategy}:")
        for cap_category, metrics in cap_performance.items():
            print(f"{cap_category}:")
            print(f"  Mean Return: {metrics['mean_return']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")

    # Print volatility performance
    print("\nVolatility Performance by Strategy:")
    for strategy, vol_performance in analysis['volatility_performance'].items():
        print(f"\n{strategy}:")
        for metric, values in vol_performance.items():
            print(f"{metric}:")
            for bucket, value in values.items():
                print(f"  {bucket}: {value:.2%}")

    # Print volume performance
    print("\nVolume Performance by Strategy:")
    for strategy, vol_performance in analysis['volume_performance'].items():
        print(f"\n{strategy}:")
        for metric, values in vol_performance.items():
            print(f"{metric}:")
            for bucket, value in values.items():
                print(f"  {bucket}: {value:.2%}")