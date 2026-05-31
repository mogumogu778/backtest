import json
from pathlib import Path
from datetime import date

import yfinance as yf
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"


def _meta_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}.json"


def _load_meta(ticker: str) -> dict | None:
    p = _meta_path(ticker)
    return json.loads(p.read_text()) if p.exists() else None


def _save_meta(ticker: str, meta: dict):
    _meta_path(ticker).write_text(json.dumps(meta, ensure_ascii=False))


def _has_splits_since(ticker: str, since: str) -> bool:
    """最終取得日以降に株式分割があったか確認する。"""
    try:
        splits = yf.Ticker(ticker).splits
        if splits.empty:
            return False
        idx = splits.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return bool((idx > pd.Timestamp(since)).any())
    except Exception:
        return False


def _download_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"データが取得できませんでした: {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError(f"有効なデータがありません: {ticker}")
    return df


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    meta = _load_meta(ticker)
    cache_file = _cache_path(ticker)

    today = date.today().isoformat()

    if meta and cache_file.exists():
        # 分割チェックは1日1回だけ実施（毎リクエストの通信を避ける）
        already_checked_today = meta.get("splits_checked_at") == today
        if not already_checked_today and _has_splits_since(ticker, meta["fetched_at"]):
            # 株式分割を検出 → キャッシュ全削除して再取得
            cache_file.unlink(missing_ok=True)
            _meta_path(ticker).unlink(missing_ok=True)
        else:
            if not already_checked_today:
                meta["splits_checked_at"] = today
                _save_meta(ticker, meta)
            if start >= meta["start"] and end <= meta["end"]:
                # キャッシュが要求範囲を完全にカバー → キャッシュを返す
                df = pd.read_parquet(cache_file)
                return df.loc[start:end]

    # 新規取得してキャッシュ保存
    df = _download_raw(ticker, start, end)
    df.to_parquet(cache_file)
    _save_meta(ticker, {
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "fetched_at": today,
        "splits_checked_at": today,
    })
    return df


def get_ticker_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": info.get("currency", "JPY"),
    }
