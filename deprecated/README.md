# Deprecated Code Archive

This folder contains deprecated code that performed poorly in testing and is kept for reference only.

## Files

### Strategies
- `strategies_original.py` - Full backup of original strategies.py file
- `strategies_full_backup.py` - Another backup copy
- `deprecated_strategies.py` - Template for deprecated strategy classes

### Failed Strategies (DO NOT USE)
- **TrendFollow, TrendFollow2, TrendFollow3** - Over-complex trend following strategies
- **TrendFollowDev** - Original problematic strategy (5.13% vs 25.91% returns)
- **RobustTrend2** - Failed complexity addition to RobustTrend
- **RobustTrend3** - Failed "optimization" with position sizing (22% vs 154% returns)

### Tools
- `optimizer.py` - Historical parameter optimizer (causes overfitting)
- `screener.py` - Stock screener (not part of main system)
- `app.py` - Debug/exploration file
- `notes.txt` - Personal development notes

## Key Lessons Learned

1. **Complexity Kills Performance** - Every optimization attempt reduced returns
2. **Over-optimization Fails Live** - RobustTrend3 had 7x worse performance
3. **Simple Beats Sophisticated** - RobustTrend (7 parameters) >> TrendFollowDev (20+ parameters)
4. **Historical Optimization ≠ Live Performance** - Curve-fitting to backtests fails

## Performance Comparison (5Y)
- **RobustTrend (WINNER)**: 154% returns, 71.9% win rate
- **RobustTrend3 (FAILED)**: 22.45% returns, 39.87% win rate
- **TrendFollowDev (FAILED)**: 5.13% returns, poor win rate

## Current Production Code
Use only the files in the main directory:
- `strategies.py` - Clean version with only RobustTrend
- `livetrader.py` - Production live trading
- `performance_tracker.py` - Performance monitoring
- `live_optimizer.py` - Post-trading analysis

**Never use code from this deprecated folder for live trading.**