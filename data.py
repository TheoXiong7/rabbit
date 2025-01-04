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
class DataMinutes:
    def __init__(self):
        self.cache = {}
        
    def get_intraday_data(self, symbol: str, period: str, interval: str = "1h") -> pd.DataFrame:
        """
        Fetch intraday data by chunking requests for minute data.
        For hourly data, fetch directly.
        """
        try:
            if interval == "1h":
                # Hourly data can be fetched directly
                stock = yf.Ticker(symbol)
                df = stock.history(period=period, interval=interval)
                if df.empty:
                    logger.error(f"No data retrieved for {symbol}")
                    return None
                
                df['Returns'] = df['Close'].pct_change()
                return df
                
            elif interval == "1m":
                # For minute data, we need to fetch in 7-day chunks
                end_date = datetime.now()
                if period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif period == "6mo":
                    start_date = end_date - timedelta(days=180)
                else:
                    start_date = end_date - timedelta(days=30)  # default
                
                # Initialize empty list to store chunks
                chunks = []
                current_end = end_date
                current_start = max(current_end - timedelta(days=7), start_date)
                
                while current_start >= start_date:
                    try:
                        stock = yf.Ticker(symbol)
                        chunk = stock.history(
                            start=current_start,
                            end=current_end,
                            interval="1m"
                        )
                        if not chunk.empty:
                            chunks.append(chunk)
                        
                        # Move window back
                        current_end = current_start
                        current_start = max(current_end - timedelta(days=7), start_date)
                        
                        # Add delay to avoid rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error fetching chunk for {symbol}: {str(e)}")
                        continue
                
                if not chunks:
                    logger.error(f"No data chunks retrieved for {symbol}")
                    return None
                
                # Combine chunks and sort by index
                df = pd.concat(chunks).sort_index()
                df['Returns'] = df['Close'].pct_change()
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            return None

    def get_hourly_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Convenience method for hourly data"""
        return self.get_intraday_data(symbol, period, interval="1h")
        
    def get_minute_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Convenience method for minute data"""
        return self.get_intraday_data(symbol, period, interval="1m")
    
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