"""
Multi-strategy F&O engine: NIFTY, BANKNIFTY, FINNIFTY, SENSEX and every NSE stock that currently has options.

    python -m src.fno                 live Upstox data, paper trading, dashboard on http://127.0.0.1:8520
    python -m src.fno --mock          synthetic market (labelled MOCK), no token
    python -m src.fno backtest ...    per-strategy backtest, out-of-sample split, walk-forward

Paper trading only. This package contains no order-placement code.
"""
