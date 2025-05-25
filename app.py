#!/usr/bin/env python3
"""
A Flask webapp version of getsignal.py, serving a one-page interface styled with Tailwind CSS.
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from strategies import TrendFollowDev

# Flask app setup
app = Flask(__name__)

# Load Alpaca credentials
def load_keys(path):
        with open(path, 'r') as f:
            key, secret = f.read().strip().split('\n')
        return key, secret
API_KEY, API_SECRET = load_keys('key.txt')

if not API_KEY or not API_SECRET:
    raise RuntimeError("Set environment vars APCA_API_KEY_ID and APCA_API_SECRET_KEY")

# Initialize data client and strategy
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
strategy = TrendFollowDev(
        volatility_window=15,
        volume_window=15,
        high_vol_threshold=0.3,
        low_vol_threshold=0.17,
        market_cap='mid',
        high_vol_fast_ema=6,
        high_vol_slow_ema=20,
        high_vol_volume_threshold=1.8,
        high_vol_profit_target=0.23,
        high_vol_stop_loss=0.07,
        med_vol_fast_ema=9,
        med_vol_slow_ema=28,
        med_vol_volume_threshold=1.4,
        med_vol_profit_target=0.13,
        med_vol_stop_loss=0.04,
        low_vol_fast_ema=11,
        low_vol_slow_ema=33,
        low_vol_volume_threshold=1.1,
        low_vol_profit_target=0.09,
        low_vol_stop_loss=0.02
    )

# Helper functions

def get_historical_data(symbol: str, lookback_days: int) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=lookback_days),
        end=datetime.now()
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    df['Returns'] = df['Close'].pct_change()
    return df


def analyze_symbol(symbol: str, lookback_days: int):
    df = get_historical_data(symbol, lookback_days)
    signals = strategy.generate_signals(df)

    # Compute metrics
    current_price = signals['Close'].iloc[-1]
    latest_signal = signals['Signal'].iloc[-1]
    buy_signals = int((signals['Signal'] == 1).sum())
    sell_signals = int((signals['Signal'] == -1).sum())
    avg_return = signals.loc[signals['Signal'] != 0, 'Returns'].mean() * 100
    volatility = signals['Returns'].std() * np.sqrt(252) * 100
    avg_volume = signals['Volume'].mean()
    high_52 = signals['High'].max()
    low_52 = signals['Low'].min()

    return {
        'symbol': symbol,
        'current_price': f"${current_price:.2f}",
        'signal_text': {1: '🟢 BUY', -1: '🔴 SELL', 0: '⚪ HOLD'}.get(latest_signal, '⚪ HOLD'),
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'avg_return': f"{avg_return:.2f}%",
        'volatility': f"{volatility:.2f}%",
        'avg_volume': f"{avg_volume:,.0f}",
        'high_52': f"${high_52:.2f}",
        'low_52': f"${low_52:.2f}"        
    }

# One-page template with Tailwind CDN
TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <title>Quick Analysis</title>
</head>
<body class="bg-gray-100 text-gray-800">
  <div class="max-w-2xl mx-auto p-6">
    <h1 class="text-3xl font-bold mb-4 text-center">Quick Analysis</h1>
    <form method="post" class="mb-6">
      <div class="flex space-x-2">
        <input name="symbol" placeholder="Ticker (e.g., AAPL)" required
               class="flex-1 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"/>
        <input name="days" type="number" value="100" min="1"
               class="w-20 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"/>
        <button type="submit"
                class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Analyze</button>
      </div>
    </form>

    {% if result %}
    <div class="bg-white shadow rounded p-4">
      <h2 class="text-2xl font-semibold mb-2">Results for {{ result.symbol }}</h2>
      <div class="grid grid-cols-2 gap-4">
        <div><strong>Current Price:</strong> {{ result.current_price }}</div>
        <div><strong>Signal:</strong> {{ result.signal_text }}</div>
        <div><strong>Buy Signals:</strong> {{ result.buy_signals }}</div>
        <div><strong>Sell Signals:</strong> {{ result.sell_signals }}</div>
        <div><strong>Avg Return:</strong> {{ result.avg_return }}</div>
        <div><strong>Volatility:</strong> {{ result.volatility }}</div>
        <div><strong>Avg Volume:</strong> {{ result.avg_volume }}</div>
        <div><strong>52-Day High:</strong> {{ result.high_52 }}</div>
        <div><strong>52-Day Low:</strong> {{ result.low_52 }}</div>
      </div>
    </div>
    {% endif %}
  </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        symbol = request.form['symbol'].upper().strip()
        days = int(request.form.get('days', 100))
        try:
            result = analyze_symbol(symbol, days)
        except Exception as e:
            result = {'symbol': symbol, 'error': str(e)}
    return render_template_string(TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=18602)
