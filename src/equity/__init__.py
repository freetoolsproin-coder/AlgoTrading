"""
NIFTY 50 cash-equity trading: two independent modules on one service.

    intraday   same-day trades on 5-minute bars (1m / 15m / 30m / 1h for context), squared off before the close
    swing      2-5 session holds on daily bars (weekly / 4h / 1h for context)

    python -m src.equity                 service on http://127.0.0.1:8530 (the desk on :8500 shows it as two tabs)
    python -m src.equity.backtest        backtest + out-of-sample / walk-forward validation of every strategy

Settings: config/equity.ini. Default mode is PAPER; nothing here places a real order unless live trading is
armed for the day and each order is confirmed (src/equity/execution.py).
"""
