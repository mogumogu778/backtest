import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from data.fetcher import fetch_ohlcv, get_ticker_info
from backtest import run_backtest, calc_metrics
from backtest.strategies import MACrossStrategy, RSIStrategy, BollingerStrategy, MACDStrategy

st.set_page_config(page_title="日本株バックテスト", page_icon="📈", layout="wide")


def _build_price_chart(signals, strategy_name, buy_signals, sell_signals):
    has_subpanel = strategy_name in ["RSI", "MACD"]

    if has_subpanel:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.05)
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=signals.index,
        open=signals["Open"], high=signals["High"],
        low=signals["Low"], close=signals["Close"],
        name="株価", increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71",
    ), row=1, col=1)

    if strategy_name == "移動平均クロス":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["short_ma"], name="短期MA", line=dict(color="orange", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["long_ma"], name="長期MA", line=dict(color="blue", width=1.2)), row=1, col=1)
    elif strategy_name == "ボリンジャーバンド":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["bb_upper"], name="上限", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["bb_mid"], name="中央", line=dict(color="gray", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["bb_lower"], name="下限", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
    elif strategy_name == "RSI":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["rsi"], name="RSI", line=dict(color="purple", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    elif strategy_name == "MACD":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["macd"], name="MACD", line=dict(color="blue", width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["macd_signal"], name="シグナル", line=dict(color="orange", width=1.2)), row=2, col=1)
        fig.add_trace(go.Bar(x=signals.index, y=signals["macd_hist"], name="ヒストグラム", marker_color="lightblue"), row=2, col=1)

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals["date"], y=buy_signals["price"],
            mode="markers", name="買いシグナル",
            marker=dict(symbol="triangle-up", size=12, color="red"),
        ), row=1, col=1)
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals["date"], y=sell_signals["price"],
            mode="markers", name="売りシグナル",
            marker=dict(symbol="triangle-down", size=12, color="blue"),
        ), row=1, col=1)

    fig.update_layout(
        title="株価チャート＋売買シグナル",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=550 if has_subpanel else 450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _show_metrics(m: dict):
    """指標を3行に分けて表示する。"""
    # 行1: リターン系
    bh = m.get("Buy & Hold リターン (%)")
    bh_label = f"{bh:+.2f}%" if bh is not None else "—"
    bh_delta = f"{m['総リターン (%)'] - bh:+.2f}% vs B&H" if bh is not None else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総リターン", f"{m['総リターン (%)']:+.2f}%", delta=bh_delta)
    c2.metric("年率リターン", f"{m['年率リターン (%)']:+.2f}%")
    c3.metric("Buy & Hold リターン", bh_label)
    c4.metric("シャープレシオ", f"{m['シャープレシオ']:.3f}")

    # 行2: リスク系
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("最大ドローダウン", f"{m['最大ドローダウン (%)']:.2f}%")
    c6.metric("カルマー比率", f"{m['カルマー比率']:.3f}")
    pf = m["Profit Factor"]
    c7.metric("Profit Factor", str(pf) if pf == "∞" else f"{pf:.2f}")
    c8.metric("最終資産", f"¥{m['最終資産 (円)']:,}")

    # 行3: トレード系
    c9, c10, c11, _ = st.columns(4)
    c9.metric("トレード回数", m["トレード回数"])
    c10.metric("勝率", f"{m['勝率 (%)']:.1f}%")
    c11.metric("平均リターン/トレード", f"{m['平均リターン/トレード (%)']:+.2f}%")


st.title("📈 日本株バックテストツール")

# ── サイドバー設定 ──────────────────────────────────────────
st.sidebar.header("設定")

ticker_input = st.sidebar.text_input(
    "銘柄コード（例: 7203.T, 1321.T）",
    value="7203.T",
    help="東証銘柄は末尾に .T を付けてください",
)

col1, col2 = st.sidebar.columns(2)
default_end = date.today()
default_start = default_end - timedelta(days=3 * 365)
start_date = col1.date_input("開始日", value=default_start)
end_date = col2.date_input("終了日", value=default_end)

strategy_name = st.sidebar.selectbox(
    "戦略",
    ["移動平均クロス", "RSI", "ボリンジャーバンド", "MACD"],
)

st.sidebar.subheader("戦略パラメータ")
if strategy_name == "移動平均クロス":
    short_w = st.sidebar.slider("短期MA (日)", 5, 50, 25)
    long_w = st.sidebar.slider("長期MA (日)", 20, 200, 75)
    strategy = MACrossStrategy(short_window=short_w, long_window=long_w)
elif strategy_name == "RSI":
    period = st.sidebar.slider("RSI期間 (日)", 5, 30, 14)
    oversold = st.sidebar.slider("売られすぎ閾値", 10, 45, 30)
    overbought = st.sidebar.slider("買われすぎ閾値", 55, 90, 70)
    strategy = RSIStrategy(period=period, oversold=oversold, overbought=overbought)
elif strategy_name == "ボリンジャーバンド":
    window = st.sidebar.slider("期間 (日)", 5, 50, 20)
    num_std = st.sidebar.slider("標準偏差の倍数", 1.0, 3.0, 2.0, step=0.1)
    strategy = BollingerStrategy(window=window, num_std=num_std)
else:
    fast = st.sidebar.slider("短期EMA", 5, 20, 12)
    slow = st.sidebar.slider("長期EMA", 15, 50, 26)
    signal_p = st.sidebar.slider("シグナル期間", 5, 20, 9)
    strategy = MACDStrategy(fast=fast, slow=slow, signal=signal_p)

st.sidebar.subheader("資金・コスト設定")
initial_capital = st.sidebar.number_input(
    "初期資金 (円)", min_value=100_000, max_value=100_000_000, value=1_000_000, step=100_000
)
commission = st.sidebar.number_input(
    "手数料率 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f"
) / 100
slippage = st.sidebar.number_input(
    "スリッページ (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f",
    help="注文がズレて約定するコスト。買いは高く、売りは安く約定する想定。",
) / 100

st.sidebar.subheader("アウト・オブ・サンプル検証")
use_oos = st.sidebar.checkbox(
    "OOS検証を有効化",
    value=False,
    help="期間を前半（パラメータ調整用）と後半（未知データ検証用）に分割し、過剰最適化を検出します。",
)
oos_ratio = 0.3
if use_oos:
    oos_ratio = st.sidebar.slider("テスト期間の割合 (%)", 10, 50, 30) / 100

run_btn = st.sidebar.button("バックテスト実行", type="primary", use_container_width=True)

# ── バックテスト実行 ────────────────────────────────────────
if run_btn:
    if start_date >= end_date:
        st.error("開始日は終了日より前に設定してください。")
        st.stop()

    with st.spinner("データを取得中..."):
        try:
            df = fetch_ohlcv(ticker_input, str(start_date), str(end_date))
            info = get_ticker_info(ticker_input)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            st.stop()

    with st.spinner("バックテスト実行中..."):
        result = run_backtest(df, strategy, initial_capital, commission, slippage)
        bh_return_pct = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        metrics = calc_metrics(result["equity_curve"], result["trades"], initial_capital, bh_return_pct)

    signals = result["signals"]
    equity = result["equity_curve"]
    trades = result["trades"]

    st.subheader(f"{info['name']}  ({ticker_input})  ／  戦略: {strategy_name}")

    # ── パフォーマンス指標 ────────────────────────────────
    _show_metrics(metrics)

    # ── チャート ─────────────────────────────────────────
    buy_signals = trades[trades["type"] == "買い"] if not trades.empty else pd.DataFrame()
    sell_signals = trades[trades["type"].str.contains("売り")] if not trades.empty else pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["株価チャート", "資産推移", "トレード履歴"])

    with tab1:
        fig = _build_price_chart(signals, strategy_name, buy_signals, sell_signals)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        bh_equity = (df["Close"] / df["Close"].iloc[0]) * initial_capital
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            name="戦略", line=dict(color="#1f77b4", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=bh_equity.index, y=bh_equity.values,
            name="Buy&Hold", line=dict(color="#aaa", width=1.5, dash="dash"),
        ))
        fig2.update_layout(
            title="資産推移",
            xaxis_title="日付",
            yaxis_title="資産 (円)",
            hovermode="x unified",
            height=450,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if trades.empty:
            st.info("トレードが発生しませんでした。")
        else:
            display_cols = [c for c in ["date", "type", "price", "shares", "pnl", "return_pct"] if c in trades.columns]
            rename = {"date": "日付", "type": "種別", "price": "価格", "shares": "株数", "pnl": "損益 (円)", "return_pct": "リターン (%)"}
            display_df = trades[display_cols].rename(columns=rename).reset_index(drop=True)
            if "損益 (円)" in display_df.columns:
                display_df["損益 (円)"] = display_df["損益 (円)"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            if "リターン (%)" in display_df.columns:
                display_df["リターン (%)"] = display_df["リターン (%)"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
            st.dataframe(display_df, use_container_width=True)

    # ── アウト・オブ・サンプル検証 ────────────────────────
    if use_oos:
        st.divider()
        st.subheader("インサンプル / アウト・オブ・サンプル 比較")
        st.caption("同じパラメータで期間を分割検証します。IS（最適化期間）より OOS（未知データ）の成績が極端に落ちる場合、過剰最適化の疑いがあります。")

        split_idx = int(len(df) * (1 - oos_ratio))
        if split_idx < 30 or (len(df) - split_idx) < 30:
            st.warning("データ期間が短すぎてOOS分割できません。期間を広げてください。")
        else:
            df_is = df.iloc[:split_idx]
            df_oos = df.iloc[split_idx:]
            split_date = df.index[split_idx].date()

            with st.spinner("OOS検証実行中..."):
                result_is = run_backtest(df_is, strategy, initial_capital, commission, slippage)
                bh_is = (df_is["Close"].iloc[-1] / df_is["Close"].iloc[0] - 1) * 100
                metrics_is = calc_metrics(result_is["equity_curve"], result_is["trades"], initial_capital, bh_is)

                result_oos = run_backtest(df_oos, strategy, initial_capital, commission, slippage)
                bh_oos = (df_oos["Close"].iloc[-1] / df_oos["Close"].iloc[0] - 1) * 100
                metrics_oos = calc_metrics(result_oos["equity_curve"], result_oos["trades"], initial_capital, bh_oos)

            col_is, col_oos = st.columns(2)

            key_metrics = [
                "総リターン (%)", "年率リターン (%)", "Buy & Hold リターン (%)",
                "シャープレシオ", "カルマー比率", "最大ドローダウン (%)",
                "Profit Factor", "勝率 (%)", "トレード回数",
            ]

            with col_is:
                st.markdown(f"**インサンプル**  \n{df.index[0].date()} ～ {split_date}")
                rows = []
                for k in key_metrics:
                    v = metrics_is.get(k)
                    rows.append({"指標": k, "値": v if v is not None else "—"})
                st.dataframe(pd.DataFrame(rows).set_index("指標"), use_container_width=True)

            with col_oos:
                st.markdown(f"**アウト・オブ・サンプル**  \n{split_date} ～ {df.index[-1].date()}")
                rows = []
                for k in key_metrics:
                    v = metrics_oos.get(k)
                    rows.append({"指標": k, "値": v if v is not None else "—"})
                st.dataframe(pd.DataFrame(rows).set_index("指標"), use_container_width=True)
