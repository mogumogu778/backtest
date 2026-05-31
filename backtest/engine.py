import pandas as pd
import numpy as np
from .strategies.base import Strategy


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 1_000_000,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
    """
    Simulate trades based on strategy signals.
    Signals generated at day T close are executed at day T+1 open
    to eliminate lookahead bias.
    """
    signals = strategy.generate_signals(df)

    position = 0
    cash = initial_capital
    portfolio_values = []
    trades = []
    pending_signal = 0
    entry_cost = 0.0

    for date, row in signals.iterrows():
        open_price = row["Open"]
        close_price = row["Close"]

        # Execute the previous bar's signal at today's open (no lookahead bias)
        if pending_signal == 1 and position == 0 and cash > 0:
            exec_price = open_price * (1 + slippage)
            shares = int(cash / (exec_price * (1 + commission)))
            cost = shares * exec_price * (1 + commission)
            if shares > 0:
                cash -= cost
                position = shares
                entry_cost = cost
                trades.append({"date": date, "type": "買い", "price": exec_price, "shares": shares})

        elif pending_signal == -1 and position > 0:
            exec_price = open_price * (1 - slippage)
            proceeds = position * exec_price * (1 - commission)
            pnl = proceeds - entry_cost
            return_pct = (proceeds / entry_cost - 1) * 100
            cash += proceeds
            trades.append({
                "date": date,
                "type": "売り",
                "price": exec_price,
                "shares": position,
                "pnl": pnl,
                "return_pct": return_pct,
            })
            position = 0
            entry_cost = 0.0

        portfolio_values.append(cash + position * close_price)
        pending_signal = row["signal"]

    # Close open position at last close price (end of period)
    if position > 0:
        last_price = signals["Close"].iloc[-1]
        proceeds = position * last_price * (1 - commission)
        pnl = proceeds - entry_cost
        return_pct = (proceeds / entry_cost - 1) * 100
        cash += proceeds
        trades.append({
            "date": signals.index[-1],
            "type": "売り(強制)",
            "price": last_price,
            "shares": position,
            "pnl": pnl,
            "return_pct": return_pct,
        })
        portfolio_values[-1] = cash

    equity_curve = pd.Series(portfolio_values, index=signals.index, name="portfolio")
    trades_df = pd.DataFrame(trades)

    return {
        "signals": signals,
        "equity_curve": equity_curve,
        "trades": trades_df,
        "final_value": equity_curve.iloc[-1],
        "initial_capital": initial_capital,
    }
