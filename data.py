import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Data')

class Data:
    def __init__(self):
        self.cache = {}  # Simple cache to store recent queries
        
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical data for a given stock symbol.
        
        Args:
            symbol (str): Stock ticker symbol
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache first
            if cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                # Cache valid for 1 hour for intraday data, 1 day for daily data
                cache_validity = timedelta(hours=1) if 'm' in interval else timedelta(days=1)
                if datetime.now() - cache_time < cache_validity:
                    logger.info(f"Returning cached data for {symbol}")
                    return data
            
            # Fetch new data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            
            # Basic data validation
            if df.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Clean the data
            df = self._clean_data(df)
            
            # Update cache
            self.cache[cache_key] = (datetime.now(), df)
            
            logger.info(f"Successfully retrieved data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the data.
        
        Args:
            df (pd.DataFrame): Raw data from yfinance
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Ensure all numeric columns are float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add useful derived columns
        df['Returns'] = df['Close'].pct_change()
        
        return df

    def get_multiple_stocks(self, symbols: list, period: str = "1y", interval: str = "1d") -> dict:
        """
        Fetch historical data for multiple stock symbols.
        
        Args:
            symbols (list): List of stock ticker symbols
            period (str): Data period
            interval (str): Data interval
            
        Returns:
            dict: Dictionary of DataFrames with symbols as keys
        """
        return {symbol: self.get_historical_data(symbol, period, interval) 
                for symbol in symbols}

# Example usage
if __name__ == "__main__":
    retriever = Data()
    
    # Example: Get daily data for the last year
    try:
        aapl_data = retriever.get_historical_data("AAPL", period="1y", interval="1d")
        print(f"Retrieved {len(aapl_data)} days of data for AAPL")
        print("\nFirst few rows of data:")
        print(aapl_data.head())
        
    except Exception as e:
        print(f"Error in example: {str(e)}")