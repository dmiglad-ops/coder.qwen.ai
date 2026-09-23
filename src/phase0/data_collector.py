# -*- coding: utf-8 -*-
"""Загрузка исторических данных с Bybit MAINNET (публичный REST API v5).

Раздел 0.5 документа: исторические данные и funding — только MAINNET, ключи не нужны.
Реализует пагинацию по end_time (Bybit отдаёт максимум 1000 свечей за запрос).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://api.bybit.com"
MAX_LIMIT = 1000  # v5/market/kline max


def _get(path: str, params: dict, retries: int = 5) -> dict:
    url = BASE_URL + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phase0-collector/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            if payload.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {payload.get('retMsg')}")
            return payload["result"]
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {exc}; sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Не удалось получить {path} после {retries} попыток")


def fetch_klines(symbol: str, interval: str = "60",
                 start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    """Пагинированная загрузка клайнов (интервал в минутах: '60', '240')."""
    frames = []
    step_ms = MAX_LIMIT * int(interval) * 60_000  # окно на ~1000 свечей
    cursor = start_ms or (end_ms or int(time.time() * 1000)) - 90 * 86400_000
    stop = end_ms or int(time.time() * 1000)
    n_pages = 0
    while cursor < stop:
        result = _get("/v5/market/kline", {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": cursor,
            "end": min(cursor + step_ms - 1, stop),
            "limit": MAX_LIMIT,
        })
        rows = result.get("list") or []
        if not rows:
            cursor += step_ms  # пропустить пустое окно (нет листинга / разрыв)
            continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype({c: float for c in ["open", "high", "low", "close", "volume", "turnover"]})
        df["ts"] = df["ts"].astype("int64")
        frames.append(df)
        newest = df["ts"].max()
        cursor = max(newest, cursor) + int(interval) * 60_000  # жёсткий прогресс: без зацикливания
        n_pages += 1
        if n_pages % 25 == 0:
            print(f"  ...{n_pages} страниц, до {pd.to_datetime(newest, unit='ms')}", flush=True)
        time.sleep(0.1)  # вежливая пауза к rate limit
    if not frames:
        raise RuntimeError(f"Нет данных для {symbol}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    out = out[out["ts"] <= stop].reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    return out[["timestamp", "ts", "open", "high", "low", "close", "volume", "turnover"]]


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Исторические funding rates (каждые 8 часов), пагинация по 200 записей."""
    frames = []
    cursor = start_ms
    while cursor < end_ms:
        result = _get("/v5/market/funding/history", {
            "category": "linear",
            "symbol": symbol,
            "startTime": cursor,
            "endTime": min(cursor + 30 * 86400_000, end_ms),
            "limit": 200,
        })
        rows = result.get("list") or []
        if not rows:
            break
        df = pd.DataFrame(rows)[["fundingRate", "fundingRateTimestamp"]]
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["ts"] = df["fundingRateTimestamp"].astype("int64")
        frames.append(df[["ts", "fundingRate"]])
        newest = df["ts"].max()
        if newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(0.15)
    if not frames:
        raise RuntimeError(f"Нет funding данных для {symbol}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    out["timestamp"] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    return out.reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Сохранено: {path} ({len(df)} строк)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Сбор данных Bybit MAINNET (Phase 0, чек-лист п.1-2)")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--interval", default="60")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(args.end, tz="UTC") if args.end
                  else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)

    root = Path(args.outdir)
    sym = args.symbol

    print(f"[1/2] Клайны {sym} interval={args.interval} с {args.start}")
    kl = fetch_klines(sym, args.interval, start_ms, end_ms)
    tf_dir = {"60": "1h", "240": "4h", "D": "1d"}.get(args.interval, args.interval)
    save_parquet(kl, root / "historical/bybit" / f"{sym}_USDT" / tf_dir / "klines.parquet")

    print(f"[2/2] Funding {sym}")
    fr = fetch_funding(sym, start_ms, end_ms)
    save_parquet(fr, root / "funding_rates" / f"{sym}_USDT" / "funding_history.parquet")
    print("Готово.")
