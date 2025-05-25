"""
Monthly Paper Trading Review Process
"""

from live_optimizer import LiveOptimizer
from performance_tracker import PerformanceTracker
from strategies import RobustTrend
import pandas as pd

def monthly_optimization_workflow():
    """
    Step-by-step monthly review process
    """
    
    print("=== MONTHLY PAPER TRADING REVIEW ===\n")
    
    # 1. Data Quality Check
    tracker = PerformanceTracker()
    optimizer = LiveOptimizer(tracker)
    
    execution_data = optimizer.analyze_execution_quality()
    
    if execution_data.get("total_trades", 0) < 30:
        print("⚠️  INSUFFICIENT DATA - Continue paper trading")
        print(f"Current trades: {execution_data.get('total_trades', 0)}")
        print("Need minimum 30 trades for meaningful analysis\n")
        return
    
    # 2. Performance vs Backtest Comparison
    print("📊 PERFORMANCE COMPARISON")
    
    # Get live metrics
    live_metrics = tracker.calculate_strategy_metrics('RobustTrend')
    
    # Compare to backtest expectations
    expected_win_rate = 0.63  # From your backtest results
    expected_returns = 0.26   # Annual from backtest
    
    live_win_rate = live_metrics.get('win_rate', 0)
    live_returns = live_metrics.get('total_return', 0)
    
    print(f"Win Rate - Live: {live_win_rate:.1%} | Expected: {expected_win_rate:.1%}")
    print(f"Returns - Live: {live_returns:.1%} | Expected: {expected_returns:.1%}")
    
    # 3. Identify Issues
    if live_win_rate < expected_win_rate * 0.8:  # 20% worse than expected
        print("🚨 WIN RATE BELOW EXPECTATIONS")
        suggestions = optimizer.suggest_parameter_adjustments()
        
        for action in suggestions.get('immediate_actions', []):
            print(f"  → {action}")
    
    # 4. Make Incremental Adjustments (NOT optimization)
    print("\n🔧 RECOMMENDED ADJUSTMENTS")
    print("(Make ONE change at a time, test for 2 weeks)")
    
    condition_analysis = optimizer.identify_underperforming_conditions()
    
    # Quality threshold adjustment
    avg_winner_quality = condition_analysis.get('avg_quality_winners', 0)
    avg_loser_quality = condition_analysis.get('avg_quality_losers', 0)
    
    if avg_winner_quality > avg_loser_quality + 0.1:
        current_threshold = 0.6  # RobustTrend default
        new_threshold = min(0.75, current_threshold + 0.05)
        print(f"  → Raise trade quality threshold: {current_threshold} → {new_threshold}")
    
    # Universe refinement
    worst_symbols = condition_analysis.get('worst_symbols', {})
    if worst_symbols:
        worst_symbol = list(worst_symbols.keys())[0]
        loss_count = worst_symbols[worst_symbol]
        if loss_count >= 5:
            print(f"  → Consider removing {worst_symbol} (5+ consecutive losses)")
    
    # 5. Document Changes
    print(f"\n📝 LOG THIS REVIEW")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    print(f"Trades analyzed: {execution_data.get('total_trades', 0)}")
    print(f"Key finding: {'Performance in line with expectations' if live_win_rate >= expected_win_rate * 0.9 else 'Performance below expectations'}")

if __name__ == "__main__":
    monthly_optimization_workflow()