import yfinance as yf
import pandas as pd


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance. Japanese stocks use .T suffix (e.g. 7203.T)."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"データが取得できませんでした: {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError(f"有効なデータがありません: {ticker}")
    return df


def get_ticker_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": info.get("currency", "JPY"),
    }
