# -*- coding: utf-8 -*-
"""Фичи Phase 0 (раздел 3.4) и метки (раздел 3.5).

Реализовано БЕЗ сторонних TA-библиотек (чистый pandas/numpy) — меньше зависимостей,
полный контроль над look-ahead. Используются только закрытые свечи (принцип №4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ───────────────────────── базовые индикаторы ─────────────────────────

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_s = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def vwap_deviation(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Отклонение цены от VWAP скользящего окна (сутки для 1h)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(window).sum()
    vv = df["volume"].rolling(window).sum().replace(0, np.nan)
    return (df["close"] - pv / vv) / (pv / vv)


# ───────────────────────── сборка фичей ─────────────────────────

FEATURE_COLUMNS = [
    # трендовые (5)
    "ema_20", "ema_50", "macd", "macd_signal", "adx",
    # осцилляторы (4)
    "rsi_14", "stoch_k", "stoch_d", "cci_20",
    # объемные (3)
    "volume_ratio_20", "obv_slope_10", "vwap_deviation",
    # контекст (2)
    "volatility_24h", "trend_strength_4h",
]


def compute_features_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Базовый набор фичей на 1h (13 из 14–15; hour_of_day опционален — раздел 3.4)."""
    out = df.copy()
    m_line, m_sig = macd(out["close"])
    k, d = stochastic(out)
    ob = obv(out)

    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["macd"] = m_line
    out["macd_signal"] = m_sig
    out["adx"] = adx(out)
    out["rsi_14"] = rsi(out["close"])
    out["stoch_k"] = k
    out["stoch_d"] = d
    out["cci_20"] = cci(out)
    out["volume_ratio_20"] = out["volume"] / out["volume"].rolling(20).mean()
    out["obv_slope_10"] = ob.diff(10) / 1e6  # масштабирование
    out["vwap_deviation"] = vwap_deviation(out)
    ret = out["close"].pct_change()
    out["volatility_24h"] = ret.rolling(24).std()
    return out


def merge_multitimeframe_correctly(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """Раздел 4.2: merge_asof direction='backward' по ВРЕМЕНИ ЗАКРЫТИЯ 4h свечи.

    Свеча 4h с ts=00:00 закрывается в 04:00 → доступна 1h-свечам с timestamp >= 04:00.
    Это исключает look-ahead.
    """
    right = df_4h.copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    right["close_time"] = right["timestamp"] + pd.Timedelta(hours=4)
    right = right.sort_values("close_time")
    # сила тренда на 4h: расстояние EMA20 к EMA50 в единицах ATR-4h
    right["ema_20_4h"] = ema(right["close"], 20)
    right["ema_50_4h"] = ema(right["close"], 50)
    right["atr_4h"] = atr(right)
    right["trend_strength_4h"] = (right["ema_20_4h"] - right["ema_50_4h"]) / right["atr_4h"]

    left = df_1h.sort_values("timestamp").copy()
    left_key = (left["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
                .astype("datetime64[ns]").astype("int64").rename("close_time"))
    right_key = (right["close_time"].dt.tz_convert("UTC").dt.tz_localize(None)
                 .astype("datetime64[ns]").astype("int64"))
    left_out = left.drop(columns=["timestamp"])
    left_out["close_time"] = left_key.reset_index(drop=True)
    right_out = right[["close_time", "trend_strength_4h"]]
    right_out["close_time"] = right_key.reset_index(drop=True)

    merged = pd.merge_asof(
        left_out,
        right_out,
        on="close_time",
        direction="backward",
        allow_exact_matches=False,  # закрытие 4h ровно в момент 1h-свечи = свеча ещё не закрыта
    )
    merged["timestamp"] = pd.to_datetime(left["timestamp"].reset_index(drop=True))
    return merged


def build_dataset(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    feats = compute_features_1h(df_1h)
    feats = merge_multitimeframe_correctly(feats, df_4h)
    return feats.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


# ───────────────────────── метки (раздел 3.5) ─────────────────────────

def generate_labels(df: pd.DataFrame, forward_period: int = 24,
                    atr_period: int = 14, atr_multiplier: float = 2.0):
    """1 (long): future close > close + 2×ATR; -1 (short): < close − 2×ATR; иначе 0."""
    a = atr(df, atr_period)
    future_price = df["close"].shift(-forward_period)
    threshold = atr_multiplier * a
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_price > df["close"] + threshold] = 1
    labels[future_price < df["close"] - threshold] = -1
    dist = labels.value_counts(normalize=True).to_dict()
    return labels, dist


if __name__ == "__main__":
    import sys
    d1 = pd.read_parquet(sys.argv[1])
    d4 = pd.read_parquet(sys.argv[2])
    ds = build_dataset(d1, d4)
    lab, dist = generate_labels(ds)
    ds["label"] = lab
    print(f"Строк: {len(ds)}, диапазон: {ds['timestamp'].min()} .. {ds['timestamp'].max()}")
    print("Распределение классов:", {k: round(v, 3) for k, v in dist.items()})
    ds.to_parquet("data/features/dataset.parquet", index=False)
    print("Сохранено: data/features/dataset.parquet")
