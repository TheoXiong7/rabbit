import logging
from datetime import datetime
import os
import json
from typing import Dict, Any

class TradingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging directory and files"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            os.makedirs(os.path.join(self.log_dir, "daily"))
        
        # Setup error logger
        error_logger = logging.getLogger('error_logger')
        error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(os.path.join(self.log_dir, 'error.log'))
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        error_logger.addHandler(error_handler)
        self.error_logger = error_logger
        
        # Setup trade logger
        trade_logger = logging.getLogger('trade_logger')
        trade_logger.setLevel(logging.INFO)
        trade_handler = logging.FileHandler(os.path.join(self.log_dir, 'trades.log'))
        trade_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        trade_logger.addHandler(trade_handler)
        self.trade_logger = trade_logger
        
    def log_error(self, error: str, additional_info: Dict = None):
        """Log error with timestamp"""
        try:
            error_msg = error
            if additional_info:
                error_msg += f" - Additional Info: {json.dumps(additional_info)}"
            self.error_logger.error(error_msg)
        except Exception as e:
            print(f"Error logging error: {str(e)}")
            
    def log_trade(self, trade_info: Dict):
        """Log trade with timestamp"""
        try:
            trade_msg = (
                f"Symbol: {trade_info.get('symbol')} | "
                f"Side: {trade_info.get('side')} | "
                f"Qty: {trade_info.get('quantity')} | "
                f"Price: ${trade_info.get('price', 0):.2f} | "
                f"Type: {trade_info.get('type', 'unknown')}"
            )
            if trade_info.get('status'):
                trade_msg += f" | Status: {trade_info['status']}"
            if trade_info.get('notes'):
                trade_msg += f" | Notes: {trade_info['notes']}"
                
            self.trade_logger.info(trade_msg)
        except Exception as e:
            self.log_error(f"Error logging trade: {str(e)}", trade_info)
            
    def create_daily_summary(self, summary_data: Dict):
        """Create daily trading summary"""
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            summary_file = os.path.join(self.log_dir, "daily", f"summary_{date_str}.json")
            
            # Prepare summary
            summary = {
                "date": date_str,
                "portfolio_value": summary_data.get('portfolio_value', 0),
                "daily_pnl": summary_data.get('daily_pnl', 0),
                "total_positions": summary_data.get('total_positions', 0),
                "positions": summary_data.get('positions', []),
                "trades_made": summary_data.get('trades_made', 0),
                "win_rate": summary_data.get('win_rate', 0),
                "best_performer": summary_data.get('best_performer', ''),
                "worst_performer": summary_data.get('worst_performer', ''),
                "market_conditions": summary_data.get('market_conditions', {}),
                "notes": summary_data.get('notes', '')
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
                
        except Exception as e:
            self.log_error(f"Error creating daily summary: {str(e)}", summary_data)
            
    def get_recent_errors(self, num_errors: int = 10) -> list:
        """Get most recent errors"""
        try:
            error_file = os.path.join(self.log_dir, 'error.log')
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    errors = f.readlines()[-num_errors:]
                return errors
        except Exception as e:
            print(f"Error reading error log: {str(e)}")
        return []
        
    def get_daily_trades(self, date_str: str = None) -> list:
        """Get trades for a specific date"""
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y-%m-%d')
                
            trades = []
            trade_file = os.path.join(self.log_dir, 'trades.log')
            
            if os.path.exists(trade_file):
                with open(trade_file, 'r') as f:
                    for line in f:
                        if date_str in line:
                            trades.append(line.strip())
            return trades
        except Exception as e:
            self.log_error(f"Error reading trades: {str(e)}")
            return []
            
    def get_daily_summary(self, date_str: str = None) -> Dict:
        """Get summary for a specific date"""
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y-%m-%d')
                
            summary_file = os.path.join(self.log_dir, "daily", f"summary_{date_str}.json")
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.log_error(f"Error reading daily summary: {str(e)}")
        return {}