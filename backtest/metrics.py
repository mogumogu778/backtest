import pandas as pd
import numpy as np


def calc_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    bh_return_pct: float | None = None,
) -> dict:
    equity_curve = equity_curve.dropna()
    if equity_curve.empty:
        raise ValueError("資産推移データが空です。データ期間や銘柄コードを確認してください。")

    final = equity_curve.iloc[-1]
    total_return = (final / initial_capital - 1) * 100

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    annual_return = ((final / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    daily_returns = equity_curve.pct_change().dropna()
    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    sell_trades = trades[trades["type"].str.contains("売り")] if not trades.empty else pd.DataFrame()
    num_trades = len(sell_trades)
    win_rate = (sell_trades["pnl"] > 0).mean() * 100 if num_trades > 0 else 0.0
    avg_return = sell_trades["return_pct"].mean() if num_trades > 0 else 0.0

    if num_trades > 0:
        gross_profit = sell_trades.loc[sell_trades["pnl"] > 0, "pnl"].sum()
        gross_loss = sell_trades.loc[sell_trades["pnl"] < 0, "pnl"].abs().sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        profit_factor = 0.0

    result = {
        "総リターン (%)": round(total_return, 2),
        "年率リターン (%)": round(annual_return, 2),
        "シャープレシオ": round(sharpe, 3),
        "カルマー比率": round(calmar, 3),
        "最大ドローダウン (%)": round(max_drawdown, 2),
        "トレード回数": num_trades,
        "勝率 (%)": round(win_rate, 1),
        "Profit Factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "平均リターン/トレード (%)": round(avg_return, 2),
        "最終資産 (円)": int(round(final)),
    }
    if bh_return_pct is not None:
        result["Buy & Hold リターン (%)"] = round(bh_return_pct, 2)

    return result
