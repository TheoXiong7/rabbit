import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import logging
from data import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StockScreener')

class StockScreener:
    def __init__(self):
        self.data_retriever = Data()
        
        # Base universe - S&P 500
        self.sp500_tickers = self._get_sp500_tickers()
        
        # Screening criteria
        self.criteria = {
            'min_market_cap': 1e9,        # $1B minimum
            'min_avg_volume': 500000,     # 500K shares minimum
            'min_price': 10,              # $10 minimum
            'max_price': 500,             # $500 maximum
            'min_volatility': 0.15,       # 15% minimum annualized volatility
            'max_volatility': 0.50,       # 50% maximum annualized volatility
            'min_sharpe': 0.5,            # Minimum Sharpe ratio
            'lookback_period': "1y"       # Analysis period
        }

    def _get_sp500_tickers(self) -> List[str]:
        """Get initial stock universe"""
        try:
            # Using Yahoo Finance Major Indices
            indices = ['^DJI', '^GSPC', '^IXIC']  # Dow, S&P 500, NASDAQ
            stocks = set()
            
            for index in indices:
                try:
                    index_data = yf.Ticker(index)
                    if hasattr(index_data, 'components'):
                        stocks.update(index_data.components)
                except:
                    continue

            # If indices fetching fails, add major stocks manually
            if not stocks:
                additional_stocks = [
                # Tech
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'AMD', 'TSLA',
                # Finance
                'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',
                # Healthcare
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
                # Industrial
                'CAT', 'BA', 'GE', 'HON', 'MMM',
                # Energy
                'XOM', 'CVX', 'COP', 'SLB',
                # Consumer
                'WMT', 'PG', 'KO', 'PEP', 'MCD'
            ]
            
            stocks.extend([s for s in additional_stocks if s not in stocks])
            return list(set(stocks))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error fetching stock tickers: {e}")
            # Return base universe if fetch fails
            return [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'AMD', 'TSLA',
                'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
                'CAT', 'BA', 'GE', 'HON', 'MMM',
                'XOM', 'CVX', 'COP', 'SLB',
                'WMT', 'PG', 'KO', 'PEP', 'MCD'
            ]

    def _analyze_stock(self, ticker: str) -> Dict:
        """Analyze a single stock"""
        try:
            # Get historical data
            data = self.data_retriever.get_historical_data(
                ticker, 
                period=self.criteria['lookback_period'],
                interval="1d"
            )
            
            if data is None or data.empty:
                return None
                
            # Get current stock info
            stock = yf.Ticker(ticker)
            try:
                info = stock.info if hasattr(stock, 'info') else {}
            except:
                info = {}
            
            # Calculate key metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            avg_volume = data['Volume'].mean()
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            # Get market cap category
            market_cap = info.get('marketCap', 0)
            if market_cap >= 200e9:
                cap_category = 'Mega-Cap'
            elif market_cap >= 10e9:
                cap_category = 'Large-Cap'
            elif market_cap >= 2e9:
                cap_category = 'Mid-Cap'
            else:
                cap_category = 'Small-Cap'
            
            # Get sector
            sector = info.get('sector', 'Unknown')
            
            metrics = {
                'symbol': ticker,
                'company_name': info.get('longName', ticker),
                'sector': sector,
                'industry': info.get('industry', 'Unknown'),
                'market_cap': market_cap,
                'market_cap_category': cap_category,
                'current_price': data['Close'].iloc[-1],
                'avg_volume': avg_volume,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'beta': info.get('beta', 0),
                'avg_dollar_volume': avg_volume * data['Close'].iloc[-1]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    def screen_stocks(self) -> pd.DataFrame:
        """Screen stocks based on criteria"""
        logger.info("Starting stock screening process...")
        
        # Analyze stocks in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self._analyze_stock, self.sp500_tickers))
        
        # Filter out None results and convert to DataFrame
        results = [r for r in results if r is not None]
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.warning("No stocks passed the screening criteria")
            return pd.DataFrame()
        
        # Apply screening criteria
        screened = df[
            (df['market_cap'] >= self.criteria['min_market_cap']) &
            (df['avg_volume'] >= self.criteria['min_avg_volume']) &
            (df['current_price'] >= self.criteria['min_price']) &
            (df['current_price'] <= self.criteria['max_price']) &
            (df['volatility'] >= self.criteria['min_volatility']) &
            (df['volatility'] <= self.criteria['max_volatility']) &
            (df['sharpe_ratio'] >= self.criteria['min_sharpe'])
        ]
        
        return screened.sort_values('market_cap', ascending=False)

    def generate_universe(self) -> Dict[str, List[str]]:
        """Generate stock universe from screened stocks"""
        screened = self.screen_stocks()
        if screened.empty:
            return {}
            
        # Group by market cap and sector
        universe = {}
        
        for cap in ['Mega-Cap', 'Large-Cap', 'Mid-Cap']:
            cap_stocks = screened[screened['market_cap_category'] == cap]
            
            for sector in cap_stocks['sector'].unique():
                sector_stocks = cap_stocks[cap_stocks['sector'] == sector]
                
                if len(sector_stocks) > 0:
                    key = f"{sector} {cap}"
                    universe[key] = sector_stocks['symbol'].tolist()
        
        return universe

    def get_universe_analysis(self) -> pd.DataFrame:
        """Get detailed analysis of the universe"""
        screened = self.screen_stocks()
        if screened.empty:
            return pd.DataFrame()
            
        # Calculate universe metrics
        analysis = screened.groupby(['market_cap_category', 'sector']).agg({
            'symbol': 'count',
            'market_cap': 'mean',
            'avg_volume': 'mean',
            'volatility': 'mean',
            'sharpe_ratio': 'mean',
            'beta': 'mean'
        }).round(2)
        
        return analysis

    def update_screening_criteria(self, new_criteria: Dict):
        """Update screening criteria"""
        self.criteria.update(new_criteria)
        logger.info("Screening criteria updated")

if __name__ == "__main__":
    # Example usage
    screener = StockScreener()
    
    # Generate universe
    universe = screener.generate_universe()
    print("\nGenerated Stock Universe:")
    for category, stocks in universe.items():
        print(f"\n{category}: {len(stocks)} stocks")
        print(f"Symbols: {', '.join(stocks)}")
    
    # Get detailed analysis
    analysis = screener.get_universe_analysis()
    print("\nUniverse Analysis:")
    print(analysis)