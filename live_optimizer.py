"""
Live Performance Optimizer
Analyzes paper trading results to improve strategy parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from performance_tracker import PerformanceTracker
import sqlite3
from datetime import datetime, timedelta

class LiveOptimizer:
    def __init__(self, performance_tracker: PerformanceTracker):
        self.tracker = performance_tracker
        
    def analyze_execution_quality(self) -> Dict:
        """Compare live vs backtest execution"""
        conn = sqlite3.connect(self.tracker.db_path)
        
        # Get all trades
        trades_df = pd.read_sql_query("""
            SELECT * FROM trades 
            WHERE timestamp >= date('now', '-90 days')
        """, conn)
        
        if trades_df.empty:
            return {"error": "Insufficient data - need at least 90 days"}
        
        analysis = {
            "total_trades": len(trades_df),
            "avg_slippage": 0,  # Calculate vs expected entry prices
            "avg_hold_period": trades_df['hold_days'].mean(),
            "execution_lag": 0,  # Time between signal and execution
            "market_impact": 0   # Price movement during execution
        }
        
        return analysis
    
    def identify_underperforming_conditions(self) -> Dict:
        """Find market conditions where strategy underperforms"""
        conn = sqlite3.connect(self.tracker.db_path)
        
        trades_df = pd.read_sql_query("""
            SELECT t.*, p.daily_pnl, p.total_equity
            FROM trades t
            LEFT JOIN portfolio_snapshots p ON date(t.timestamp) = date(p.timestamp)
            WHERE t.timestamp >= date('now', '-90 days')
        """, conn)
        
        if trades_df.empty:
            return {"error": "Insufficient data"}
        
        # Analyze by trade quality, volatility, etc.
        poor_trades = trades_df[trades_df['realized_pnl'] < 0]
        good_trades = trades_df[trades_df['realized_pnl'] > 0]
        
        analysis = {
            "worst_symbols": poor_trades['symbol'].value_counts().head(5).to_dict(),
            "best_symbols": good_trades['symbol'].value_counts().head(5).to_dict(),
            "avg_quality_winners": good_trades['trade_quality'].mean() if not good_trades.empty else 0,
            "avg_quality_losers": poor_trades['trade_quality'].mean() if not poor_trades.empty else 0,
            "win_rate_by_quality": self._analyze_by_quality_buckets(trades_df)
        }
        
        return analysis
    
    def _analyze_by_quality_buckets(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by trade quality buckets"""
        if df.empty:
            return {}
        
        # Filter out rows with None trade_quality
        quality_df = df[df['trade_quality'].notna()].copy()
        
        if quality_df.empty:
            return {"error": "No trades with quality scores found"}
        
        quality_df['quality_bucket'] = pd.cut(quality_df['trade_quality'], 
                                    bins=[0, 0.6, 0.7, 0.8, 1.0], 
                                    labels=['Low', 'Medium', 'High', 'Very High'])
        
        return quality_df.groupby('quality_bucket', observed=False).agg({
            'realized_pnl': ['count', 'mean', lambda x: (x > 0).mean()]
        }).to_dict()
    
    def suggest_parameter_adjustments(self) -> Dict:
        """Suggest parameter changes based on live performance"""
        
        execution_analysis = self.analyze_execution_quality()
        condition_analysis = self.identify_underperforming_conditions()
        
        suggestions = {
            "immediate_actions": [],
            "parameter_tweaks": {},
            "risk_management": [],
            "confidence_level": "low"  # low/medium/high based on data quality
        }
        
        # Only suggest changes if we have sufficient data
        if execution_analysis.get("total_trades", 0) < 50:
            suggestions["immediate_actions"].append(
                "Continue collecting data - need minimum 50 trades for reliable analysis"
            )
            return suggestions
        
        suggestions["confidence_level"] = "medium"
        
        # Trade quality analysis
        if condition_analysis.get("avg_quality_winners", 0) > condition_analysis.get("avg_quality_losers", 0) + 0.1:
            suggestions["parameter_tweaks"]["min_trade_quality"] = 0.7
            suggestions["immediate_actions"].append("Raise trade quality threshold to 0.7")
        
        # Symbol performance
        worst_symbols = condition_analysis.get("worst_symbols", {})
        if worst_symbols:
            worst_symbol = list(worst_symbols.keys())[0]
            if worst_symbols[worst_symbol] >= 5:  # At least 5 losing trades
                suggestions["immediate_actions"].append(f"Consider removing {worst_symbol} from universe")
        
        return suggestions
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        
        execution = self.analyze_execution_quality()
        conditions = self.identify_underperforming_conditions()
        suggestions = self.suggest_parameter_adjustments()
        
        report = f"""
LIVE TRADING OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

=== EXECUTION QUALITY ===
Total Trades: {execution.get('total_trades', 0)}
Avg Hold Period: {execution.get('avg_hold_period', 0):.1f} days
Data Quality: {suggestions['confidence_level'].upper()}

=== PERFORMANCE ANALYSIS ===
Win Rate by Quality:
{conditions.get('win_rate_by_quality', 'Insufficient data')}

Worst Performing Symbols:
{conditions.get('worst_symbols', {})}

Best Performing Symbols:
{conditions.get('best_symbols', {})}

=== RECOMMENDATIONS ===
Immediate Actions:
"""
        for action in suggestions.get('immediate_actions', []):
            report += f"• {action}\n"
        
        report += f"""
Parameter Adjustments:
"""
        for param, value in suggestions.get('parameter_tweaks', {}).items():
            report += f"• {param}: {value}\n"
        
        return report

# Usage example for paper trading optimization
def optimize_from_paper_trading():
    """Example workflow for optimizing after paper trading"""
    
    tracker = PerformanceTracker()
    optimizer = LiveOptimizer(tracker)
    
    # Generate report
    report = optimizer.generate_optimization_report()
    print(report)
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('optreps', exist_ok=True)
    
    # Save report
    with open(f'optreps/optimization_report_{datetime.now().strftime("%Y%m%d")}.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    optimize_from_paper_trading()