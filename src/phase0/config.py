# -*- coding: utf-8 -*-
"""Единый источник конфигурации (раздел 2.1, config/system.yaml).

Ревизия: конфиг раньше нигде не читался — параметры дублировались в
run_phase0.py и oos_guard.py. Теперь значения берутся отсюда с фолбэком
на встроенные дефолты (YAML можно заполнять не полностью).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("TS_CONFIG", PROJECT_ROOT / "config" / "system.yaml"))

DEFAULTS: dict = {
    "project": {"name": "crypto-ml-trading", "phase": 0, "doc_version": "4.2"},
    "data": {
        "symbol": "SOL_USDT",
        "ccxt_symbol": "SOL/USDT",
        "bybit_symbol": "SOLUSDT",
        "timeframe": "1h",
        "source": "mainnet",
    },
    "splits": {
        "train": ["2022-01-01", "2024-06-30"],
        "validation": ["2024-07-01", "2024-12-31"],
        "oos_iterative": ["2025-01-01", "2026-02-28"],
        "oos_final": ["2026-03-01", "2026-09-01"],
    },
    "labels": {"forward_period": 24, "atr_period": 14, "atr_multiplier": 2.0},
    "costs": {"taker_fee": 0.001, "slippage": 0.0005, "funding_enabled": True},
    "backtest": {
        "embargo_bars": 72,
        "holding_bars": 24,
        "atr_exit": True,
        "atr_multiplier": 2.0,
        "non_overlapping": True,
    },
    "model": {
        "algorithm": "xgboost",
        "base_params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
        # sample_weight — НЕ scale_pos_weight (мультикласс!)
        "class_weighting": "sample_weight",
    },
    "optuna": {"n_trials": 100, "metric": "val_sharpe", "early_stopping_rounds": 50},
    "walkforward": {"n_splits": 5},
    "benchmarks": {"n_runs": 1000, "percentile_threshold": 0.95},
    "pairs": {
        "SOL_USDT": {"ccxt": "SOL/USDT", "bybit": "SOLUSDT", "fee": 0.001, "slippage": 0.0005},
        "BTC_USDT": {"ccxt": "BTC/USDT", "bybit": "BTCUSDT", "fee": 0.001, "slippage": 0.0003},
        "ETH_USDT": {"ccxt": "ETH/USDT", "bybit": "ETHUSDT", "fee": 0.001, "slippage": 0.0003},
    },
}


def _deep_merge(base: dict, upd: dict) -> dict:
    """Рекурсивное слияние: YAML перекрывает дефолты, отсутствующие ключи сохраняются."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Запуск: load_config() — читает config/system.yaml поверх DEFAULTS."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if isinstance(user, dict):
            cfg = _deep_merge(cfg, user)
        else:
            raise RuntimeError(f"Конфиг {p} не является YAML-словарём")
    return cfg


def split_bounds(cfg: dict, name: str) -> tuple:
    """Границы сплита как (start, end) datetime с tz=UTC (end — включительно, до конца суток)."""
    lo, hi = cfg["splits"][name]
    return pd_start(lo), pd_end(hi)


def pd_start(iso: str) -> pd.Timestamp:
    import pandas as pd
    return pd.Timestamp(iso, tz="UTC")


def pd_end(iso: str) -> pd.Timestamp:
    import pandas as pd
    return pd.Timestamp(iso, tz="UTC") + pd.Timedelta(hours=23)


if __name__ == "__main__":
    import json
    cfg = load_config()
    print(json.dumps(cfg, indent=2, ensure_ascii=False))