# performance_tracker.py
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from pathlib import Path

class PerformanceTracker:
    """
    Comprehensive performance tracking system for live trading
    Tracks trades, portfolio metrics, strategy performance, and generates reports
    """
    
    def __init__(self, db_path: str = "trading_performance.db"):
        self.db_path = db_path
        self.setup_database()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup structured logging for performance tracking"""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Setup performance logger
        self.perf_logger = logging.getLogger('performance')
        self.perf_logger.setLevel(logging.INFO)
        
        # File handler for performance logs
        perf_handler = logging.FileHandler(f'logs/performance_{datetime.now().strftime("%Y%m%d")}.log')
        perf_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        perf_handler.setFormatter(perf_formatter)
        self.perf_logger.addHandler(perf_handler)
        
        # Setup trade logger
        self.trade_logger = logging.getLogger('trades')
        self.trade_logger.setLevel(logging.INFO)
        
        trade_handler = logging.FileHandler(f'logs/trades_{datetime.now().strftime("%Y%m%d")}.log')
        trade_formatter = logging.Formatter('%(asctime)s | %(message)s')
        trade_handler.setFormatter(trade_formatter)
        self.trade_logger.addHandler(trade_handler)
        
    def setup_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                strategy TEXT NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                profit_target REAL,
                position_size REAL,
                trade_quality REAL,
                hold_days INTEGER,
                realized_pnl REAL,
                commission REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Portfolio snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_equity REAL NOT NULL,
                buying_power REAL NOT NULL,
                total_positions INTEGER NOT NULL,
                daily_pnl REAL,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Strategy performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                total_return REAL NOT NULL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_trade_return REAL,
                avg_hold_period REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_trade(self, trade_data: Dict):
        """Log a trade to database and file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                timestamp, symbol, side, action, quantity, price, strategy,
                entry_price, stop_loss, profit_target, position_size, 
                trade_quality, hold_days, realized_pnl, commission
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('timestamp', datetime.now().isoformat()),
            trade_data['symbol'],
            trade_data['side'],
            trade_data['action'],
            trade_data['quantity'],
            trade_data['price'],
            trade_data['strategy'],
            trade_data.get('entry_price'),
            trade_data.get('stop_loss'),
            trade_data.get('profit_target'),
            trade_data.get('position_size'),
            trade_data.get('trade_quality'),
            trade_data.get('hold_days'),
            trade_data.get('realized_pnl'),
            trade_data.get('commission')
        ))
        
        conn.commit()
        conn.close()
        
        # Log to file
        self.trade_logger.info(f"TRADE | {trade_data['action']} | {trade_data['symbol']} | "
                              f"{trade_data['quantity']}@${trade_data['price']:.2f} | "
                              f"Strategy: {trade_data['strategy']}")
        
    def log_portfolio_snapshot(self, portfolio_data: Dict):
        """Log portfolio snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolio_snapshots (
                timestamp, total_equity, buying_power, total_positions,
                daily_pnl, total_pnl, max_drawdown, sharpe_ratio,
                win_rate, avg_win, avg_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            portfolio_data.get('timestamp', datetime.now().isoformat()),
            portfolio_data['total_equity'],
            portfolio_data['buying_power'],
            portfolio_data['total_positions'],
            portfolio_data.get('daily_pnl'),
            portfolio_data.get('total_pnl'),
            portfolio_data.get('max_drawdown'),
            portfolio_data.get('sharpe_ratio'),
            portfolio_data.get('win_rate'),
            portfolio_data.get('avg_win'),
            portfolio_data.get('avg_loss')
        ))
        
        conn.commit()
        conn.close()
        
        self.perf_logger.info(f"PORTFOLIO | Equity: ${portfolio_data['total_equity']:,.2f} | "
                             f"Positions: {portfolio_data['total_positions']} | "
                             f"Daily P&L: ${portfolio_data.get('daily_pnl', 0):,.2f}")
        
    def calculate_strategy_metrics(self, strategy_name: str, days: int = 30) -> Dict:
        """Calculate comprehensive strategy performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get trades for the strategy in the last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT * FROM trades 
            WHERE strategy = ? AND timestamp >= ? 
            ORDER BY timestamp
        '''
        
        trades_df = pd.read_sql_query(query, conn, params=(strategy_name, cutoff_date))
        conn.close()
        
        if trades_df.empty:
            return {}
        
        # Calculate metrics
        entry_trades = trades_df[trades_df['action'] == 'ENTRY']
        exit_trades = trades_df[trades_df['action'] == 'EXIT']
        
        total_trades = len(entry_trades)
        completed_trades = trades_df[trades_df['realized_pnl'].notna()]
        
        if completed_trades.empty:
            return {
                'strategy_name': strategy_name,
                'total_trades': total_trades,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_hold_period': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        winning_trades = len(completed_trades[completed_trades['realized_pnl'] > 0])
        losing_trades = len(completed_trades[completed_trades['realized_pnl'] <= 0])
        
        total_return = completed_trades['realized_pnl'].sum()
        win_rate = winning_trades / len(completed_trades) if len(completed_trades) > 0 else 0
        
        wins = completed_trades[completed_trades['realized_pnl'] > 0]['realized_pnl']
        losses = completed_trades[completed_trades['realized_pnl'] <= 0]['realized_pnl']
        
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0
        avg_hold_period = completed_trades['hold_days'].mean() if 'hold_days' in completed_trades.columns else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = completed_trades['realized_pnl'] / 10000  # Assume $10k base
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        return {
            'strategy_name': strategy_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_period': avg_hold_period,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    def log_strategy_performance(self, strategy_name: str, days: int = 30):
        """Log strategy performance metrics"""
        metrics = self.calculate_strategy_metrics(strategy_name, days)
        
        if not metrics:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO strategy_performance (
                timestamp, strategy_name, total_trades, winning_trades,
                losing_trades, total_return, sharpe_ratio, max_drawdown,
                avg_trade_return, avg_hold_period
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            metrics['strategy_name'],
            metrics['total_trades'],
            metrics['winning_trades'],
            metrics['losing_trades'],
            metrics['total_return'],
            metrics['sharpe_ratio'],
            metrics['max_drawdown'],
            metrics['total_return'] / max(metrics['total_trades'], 1),
            metrics['avg_hold_period']
        ))
        
        conn.commit()
        conn.close()
        
        self.perf_logger.info(f"STRATEGY | {strategy_name} | "
                             f"Trades: {metrics['total_trades']} | "
                             f"Win Rate: {metrics['win_rate']:.1%} | "
                             f"Total Return: ${metrics['total_return']:,.2f} | "
                             f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        
        # Get today's trades
        trades_query = '''
            SELECT * FROM trades 
            WHERE date(timestamp) = date('now', 'localtime')
            ORDER BY timestamp
        '''
        trades_df = pd.read_sql_query(trades_query, conn)
        
        # Get latest portfolio snapshot
        portfolio_query = '''
            SELECT * FROM portfolio_snapshots 
            ORDER BY timestamp DESC LIMIT 1
        '''
        portfolio_df = pd.read_sql_query(portfolio_query, conn)
        
        conn.close()
        
        report = f"""
=== DAILY TRADING REPORT - {today} ===

PORTFOLIO SUMMARY:
"""
        
        if not portfolio_df.empty:
            p = portfolio_df.iloc[0]
            report += f"""
Total Equity: ${p['total_equity']:,.2f}
Daily P&L: ${p.get('daily_pnl', 0):,.2f}
Active Positions: {p['total_positions']}
Win Rate: {p.get('win_rate', 0):.1%}
"""
        
        report += f"""
TRADING ACTIVITY:
Total Trades Today: {len(trades_df)}
"""
        
        if not trades_df.empty:
            entries = trades_df[trades_df['action'] == 'ENTRY']
            exits = trades_df[trades_df['action'] == 'EXIT']
            
            report += f"""
Entries: {len(entries)}
Exits: {len(exits)}

TRADE DETAILS:
"""
            
            for _, trade in trades_df.iterrows():
                report += f"""
{trade['timestamp'][:19]} | {trade['action']} | {trade['symbol']} | 
{trade['side']} | {trade['quantity']}@${trade['price']:.2f}
"""
        
        # Get strategy performance
        strategies = trades_df['strategy'].unique() if not trades_df.empty else []
        
        if strategies:
            report += "\nSTRATEGY PERFORMANCE (30 days):\n"
            for strategy in strategies:
                metrics = self.calculate_strategy_metrics(strategy, 30)
                if metrics:
                    report += f"""
{strategy}:
  Total Trades: {metrics['total_trades']}
  Win Rate: {metrics['win_rate']:.1%}
  Total Return: ${metrics['total_return']:,.2f}
  Avg Hold: {metrics['avg_hold_period']:.1f} days
  Sharpe: {metrics['sharpe_ratio']:.2f}
"""
        
        return report
        
    def export_data(self, start_date: str, end_date: str, output_path: str):
        """Export trading data to CSV for analysis"""
        conn = sqlite3.connect(self.db_path)
        
        # Export trades
        trades_query = '''
            SELECT * FROM trades 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        trades_df = pd.read_sql_query(trades_query, conn, params=(start_date, end_date))
        trades_df.to_csv(f"{output_path}_trades.csv", index=False)
        
        # Export portfolio snapshots
        portfolio_query = '''
            SELECT * FROM portfolio_snapshots 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        portfolio_df = pd.read_sql_query(portfolio_query, conn, params=(start_date, end_date))
        portfolio_df.to_csv(f"{output_path}_portfolio.csv", index=False)
        
        conn.close()
        
        self.perf_logger.info(f"Data exported to {output_path}_*.csv")