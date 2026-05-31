import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from data.fetcher import fetch_ohlcv, get_ticker_info
from backtest import run_backtest, calc_metrics
from backtest.strategies import MACrossStrategy, RSIStrategy, BollingerStrategy, MACDStrategy

st.set_page_config(page_title="日本株バックテスト", page_icon="📈", layout="wide")

STRATEGY_COLORS = {
    "移動平均クロス": "#1f77b4",
    "RSI":           "#d62728",
    "ボリンジャーバンド": "#2ca02c",
    "MACD":          "#ff7f0e",
}

METRIC_DISPLAY_ORDER = [
    "総リターン (%)", "年率リターン (%)", "Buy & Hold リターン (%)",
    "シャープレシオ", "カルマー比率", "最大ドローダウン (%)",
    "Profit Factor", "勝率 (%)", "トレード回数",
    "平均リターン/トレード (%)", "最終資産 (円)",
]


def _fmt_metric(key: str, val) -> str:
    if val is None:
        return "—"
    if key in ("総リターン (%)", "年率リターン (%)", "Buy & Hold リターン (%)",
               "平均リターン/トレード (%)"):
        return f"{val:+.2f}%" if isinstance(val, (int, float)) else str(val)
    if key in ("勝率 (%)",):
        return f"{val:.1f}%"
    if key in ("最大ドローダウン (%)",):
        return f"{val:.2f}%"
    if key in ("シャープレシオ", "カルマー比率"):
        return f"{val:.3f}"
    if key == "Profit Factor":
        return str(val) if val == "∞" else f"{val:.2f}"
    if key == "最終資産 (円)":
        return f"¥{val:,}"
    return str(val)


def _build_price_chart(signals, strategy_name, buy_sigs, sell_sigs, color):
    has_subpanel = strategy_name in ["RSI", "MACD"]
    fig = make_subplots(
        rows=2 if has_subpanel else 1, cols=1,
        row_heights=[0.7, 0.3] if has_subpanel else [1],
        shared_xaxes=True, vertical_spacing=0.05,
    )

    fig.add_trace(go.Candlestick(
        x=signals.index,
        open=signals["Open"], high=signals["High"],
        low=signals["Low"], close=signals["Close"],
        name="株価", increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71",
    ), row=1, col=1)

    if strategy_name == "移動平均クロス":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["short_ma"], name="短期MA", line=dict(color="orange", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["long_ma"],  name="長期MA", line=dict(color="blue",   width=1.2)), row=1, col=1)
    elif strategy_name == "ボリンジャーバンド":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["bb_upper"], name="上限", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["bb_mid"],   name="中央", line=dict(color="gray", width=1)),            row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["bb_lower"], name="下限", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
    elif strategy_name == "RSI":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["rsi"], name="RSI", line=dict(color="purple", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    elif strategy_name == "MACD":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["macd"],        name="MACD",    line=dict(color="blue",   width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["macd_signal"], name="シグナル", line=dict(color="orange", width=1.2)), row=2, col=1)
        fig.add_trace(go.Bar(x=signals.index, y=signals["macd_hist"], name="ヒストグラム", marker_color="lightblue"), row=2, col=1)

    if not buy_sigs.empty:
        fig.add_trace(go.Scatter(
            x=buy_sigs["date"], y=buy_sigs["price"], mode="markers", name="買い",
            marker=dict(symbol="triangle-up", size=12, color=color),
        ), row=1, col=1)
    if not sell_sigs.empty:
        fig.add_trace(go.Scatter(
            x=sell_sigs["date"], y=sell_sigs["price"], mode="markers", name="売り",
            marker=dict(symbol="triangle-down", size=12, color=color),
        ), row=1, col=1)

    fig.update_layout(
        title=f"株価チャート — {strategy_name}",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=550 if has_subpanel else 450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _show_single_metrics(m: dict):
    bh = m.get("Buy & Hold リターン (%)")
    bh_label = f"{bh:+.2f}%" if bh is not None else "—"
    bh_delta = f"{m['総リターン (%)'] - bh:+.2f}% vs B&H" if bh is not None else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総リターン",         f"{m['総リターン (%)']:+.2f}%", delta=bh_delta)
    c2.metric("年率リターン",       f"{m['年率リターン (%)']:+.2f}%")
    c3.metric("Buy & Hold リターン", bh_label)
    c4.metric("シャープレシオ",      f"{m['シャープレシオ']:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("最大ドローダウン", f"{m['最大ドローダウン (%)']:.2f}%")
    c6.metric("カルマー比率",    f"{m['カルマー比率']:.3f}")
    pf = m["Profit Factor"]
    c7.metric("Profit Factor", str(pf) if pf == "∞" else f"{pf:.2f}")
    c8.metric("最終資産",       f"¥{m['最終資産 (円)']:,}")

    c9, c10, c11, _ = st.columns(4)
    c9.metric("トレード回数",           m["トレード回数"])
    c10.metric("勝率",                 f"{m['勝率 (%)']:.1f}%")
    c11.metric("平均リターン/トレード", f"{m['平均リターン/トレード (%)']:+.2f}%")


def _show_compare_metrics(all_metrics: dict):
    rows = {}
    for key in METRIC_DISPLAY_ORDER:
        rows[key] = {sname: _fmt_metric(key, m.get(key)) for sname, m in all_metrics.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "指標"
    st.dataframe(df, use_container_width=True)


def _trades_table(trades: pd.DataFrame):
    if trades.empty:
        st.info("トレードが発生しませんでした。")
        return
    display_cols = [c for c in ["date", "type", "price", "shares", "pnl", "return_pct"] if c in trades.columns]
    rename = {"date": "日付", "type": "種別", "price": "価格", "shares": "株数",
              "pnl": "損益 (円)", "return_pct": "リターン (%)"}
    df = trades[display_cols].rename(columns=rename).reset_index(drop=True)
    if "損益 (円)" in df.columns:
        df["損益 (円)"] = df["損益 (円)"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    if "リターン (%)" in df.columns:
        df["リターン (%)"] = df["リターン (%)"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    st.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────────────────────────
st.title("📈 日本株バックテストツール")
st.sidebar.header("設定")

ticker_input = st.sidebar.text_input(
    "銘柄コード（例: 7203.T, 1321.T）", value="7203.T",
    help="東証銘柄は末尾に .T を付けてください",
)

col1, col2 = st.sidebar.columns(2)
default_end   = date.today()
default_start = default_end - timedelta(days=3 * 365)
start_date = col1.date_input("開始日", value=default_start)
end_date   = col2.date_input("終了日", value=default_end)

st.sidebar.subheader("戦略選択（複数可）")
selected_names = st.sidebar.multiselect(
    "使用する戦略",
    ["移動平均クロス", "RSI", "ボリンジャーバンド", "MACD"],
    default=["移動平均クロス"],
    help="複数選択するとパフォーマンスを並べて比較できます",
)

if not selected_names:
    st.sidebar.warning("戦略を1つ以上選択してください。")

# 戦略ごとのパラメータ設定
strategies: dict = {}
for sname in selected_names:
    with st.sidebar.expander(f"{sname} のパラメータ", expanded=(len(selected_names) == 1)):
        if sname == "移動平均クロス":
            sw = st.slider("短期MA (日)", 5,  50,  25,  key=f"{sname}_short")
            lw = st.slider("長期MA (日)", 20, 200, 75,  key=f"{sname}_long")
            strategies[sname] = MACrossStrategy(short_window=sw, long_window=lw)
        elif sname == "RSI":
            p  = st.slider("RSI期間 (日)",   5,  30, 14, key=f"{sname}_period")
            os = st.slider("売られすぎ閾値", 10,  45, 30, key=f"{sname}_oversold")
            ob = st.slider("買われすぎ閾値", 55,  90, 70, key=f"{sname}_overbought")
            strategies[sname] = RSIStrategy(period=p, oversold=os, overbought=ob)
        elif sname == "ボリンジャーバンド":
            w   = st.slider("期間 (日)",        5,   50, 20,  key=f"{sname}_window")
            std = st.slider("標準偏差の倍数", 1.0,  3.0, 2.0, step=0.1, key=f"{sname}_std")
            strategies[sname] = BollingerStrategy(window=w, num_std=std)
        elif sname == "MACD":
            fast = st.slider("短期EMA",    5,  20, 12, key=f"{sname}_fast")
            slow = st.slider("長期EMA",   15,  50, 26, key=f"{sname}_slow")
            sig  = st.slider("シグナル期間", 5, 20,  9, key=f"{sname}_signal")
            strategies[sname] = MACDStrategy(fast=fast, slow=slow, signal=sig)

st.sidebar.subheader("資金・コスト設定")
initial_capital = st.sidebar.number_input(
    "初期資金 (円)", min_value=100_000, max_value=100_000_000, value=1_000_000, step=100_000
)
commission = st.sidebar.number_input(
    "手数料率 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f"
) / 100
slippage = st.sidebar.number_input(
    "スリッページ (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f",
    help="注文がズレて約定するコスト。",
) / 100

st.sidebar.subheader("アウト・オブ・サンプル検証")
use_oos = st.sidebar.checkbox(
    "OOS検証を有効化", value=False,
    help="期間を前半（最適化用）と後半（未知データ）に分割し、過剰最適化を検出します。",
)
oos_ratio = 0.3
if use_oos:
    oos_ratio = st.sidebar.slider("テスト期間の割合 (%)", 10, 50, 30) / 100

run_btn = st.sidebar.button(
    "バックテスト実行", type="primary", use_container_width=True,
    disabled=(not selected_names),
)

# ─────────────────────────────────────────────────────────────
# バックテスト実行
# ─────────────────────────────────────────────────────────────
if run_btn and strategies:
    if start_date >= end_date:
        st.error("開始日は終了日より前に設定してください。")
        st.stop()

    with st.spinner("データを取得中..."):
        try:
            df   = fetch_ohlcv(ticker_input, str(start_date), str(end_date))
            info = get_ticker_info(ticker_input)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            st.stop()

    bh_return_pct = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

    with st.spinner(f"{len(strategies)}戦略のバックテスト実行中..."):
        all_results: dict = {}
        all_metrics: dict = {}
        for sname, strategy in strategies.items():
            result  = run_backtest(df, strategy, initial_capital, commission, slippage)
            metrics = calc_metrics(result["equity_curve"], result["trades"], initial_capital, bh_return_pct)
            all_results[sname] = result
            all_metrics[sname] = metrics

    st.subheader(f"{info['name']}  ({ticker_input})")

    # ── パフォーマンス指標 ──────────────────────────────────
    if len(strategies) == 1:
        sname = list(strategies.keys())[0]
        st.caption(f"戦略: {sname}")
        _show_single_metrics(all_metrics[sname])
    else:
        st.caption("戦略比較（指標一覧）")
        _show_compare_metrics(all_metrics)

    # ── タブ ───────────────────────────────────────────────
    tab_equity, tab_chart, tab_trades = st.tabs(["資産推移", "株価チャート", "トレード履歴"])

    # 資産推移：全戦略 + B&H を1枚に重ねて表示
    with tab_equity:
        bh_equity = (df["Close"] / df["Close"].iloc[0]) * initial_capital
        fig_eq = go.Figure()
        for sname, result in all_results.items():
            fig_eq.add_trace(go.Scatter(
                x=result["equity_curve"].index,
                y=result["equity_curve"].values,
                name=sname,
                line=dict(color=STRATEGY_COLORS.get(sname, "#333"), width=2),
            ))
        fig_eq.add_trace(go.Scatter(
            x=bh_equity.index, y=bh_equity.values,
            name="Buy & Hold", line=dict(color="#aaa", width=1.5, dash="dash"),
        ))
        fig_eq.update_layout(
            title="資産推移比較",
            xaxis_title="日付", yaxis_title="資産 (円)",
            hovermode="x unified", height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # 株価チャート：戦略ごとのサブタブ
    with tab_chart:
        if len(strategies) == 1:
            sname  = list(strategies.keys())[0]
            result = all_results[sname]
            trades = result["trades"]
            buy_s  = trades[trades["type"] == "買い"] if not trades.empty else pd.DataFrame()
            sell_s = trades[trades["type"].str.contains("売り")] if not trades.empty else pd.DataFrame()
            st.plotly_chart(
                _build_price_chart(result["signals"], sname, buy_s, sell_s, STRATEGY_COLORS.get(sname, "#333")),
                use_container_width=True,
            )
        else:
            subtabs = st.tabs(list(all_results.keys()))
            for subtab, (sname, result) in zip(subtabs, all_results.items()):
                with subtab:
                    trades = result["trades"]
                    buy_s  = trades[trades["type"] == "買い"] if not trades.empty else pd.DataFrame()
                    sell_s = trades[trades["type"].str.contains("売り")] if not trades.empty else pd.DataFrame()
                    st.plotly_chart(
                        _build_price_chart(result["signals"], sname, buy_s, sell_s, STRATEGY_COLORS.get(sname, "#333")),
                        use_container_width=True,
                    )

    # トレード履歴：戦略ごとのサブタブ
    with tab_trades:
        if len(strategies) == 1:
            _trades_table(list(all_results.values())[0]["trades"])
        else:
            subtabs = st.tabs(list(all_results.keys()))
            for subtab, (sname, result) in zip(subtabs, all_results.items()):
                with subtab:
                    _trades_table(result["trades"])

    # ── OOS検証 ────────────────────────────────────────────
    if use_oos:
        st.divider()
        st.subheader("インサンプル / アウト・オブ・サンプル 比較")
        st.caption("IS（最適化期間）より OOS（未知データ）の成績が極端に落ちる場合は過剰最適化の疑いがあります。")

        split_idx = int(len(df) * (1 - oos_ratio))
        if split_idx < 30 or (len(df) - split_idx) < 30:
            st.warning("データ期間が短すぎてOOS分割できません。期間を広げてください。")
        else:
            df_is  = df.iloc[:split_idx]
            df_oos = df.iloc[split_idx:]
            split_date = df.index[split_idx].date()

            oos_key_metrics = [
                "総リターン (%)", "年率リターン (%)", "Buy & Hold リターン (%)",
                "シャープレシオ", "カルマー比率", "最大ドローダウン (%)",
                "Profit Factor", "勝率 (%)", "トレード回数",
            ]

            for sname, strategy in strategies.items():
                with st.expander(f"{sname}  のOOS検証", expanded=(len(strategies) == 1)):
                    with st.spinner(f"{sname} OOS実行中..."):
                        r_is  = run_backtest(df_is,  strategy, initial_capital, commission, slippage)
                        r_oos = run_backtest(df_oos, strategy, initial_capital, commission, slippage)
                        bh_is  = (df_is["Close"].iloc[-1]  / df_is["Close"].iloc[0]  - 1) * 100
                        bh_oos = (df_oos["Close"].iloc[-1] / df_oos["Close"].iloc[0] - 1) * 100
                        m_is  = calc_metrics(r_is["equity_curve"],  r_is["trades"],  initial_capital, bh_is)
                        m_oos = calc_metrics(r_oos["equity_curve"], r_oos["trades"], initial_capital, bh_oos)

                    col_is, col_oos = st.columns(2)
                    with col_is:
                        st.markdown(f"**インサンプル**  \n{df.index[0].date()} ～ {split_date}")
                        rows = [{"指標": k, "値": _fmt_metric(k, m_is.get(k))} for k in oos_key_metrics]
                        st.dataframe(pd.DataFrame(rows).set_index("指標"), use_container_width=True)
                    with col_oos:
                        st.markdown(f"**アウト・オブ・サンプル**  \n{split_date} ～ {df.index[-1].date()}")
                        rows = [{"指標": k, "値": _fmt_metric(k, m_oos.get(k))} for k in oos_key_metrics]
                        st.dataframe(pd.DataFrame(rows).set_index("指標"), use_container_width=True)
