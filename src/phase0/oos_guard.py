# -*- coding: utf-8 -*-
"""Защита OOS_FINAL (раздел 3.3): физическая, хеш, флаг --allow-final-oos.

Все скрипты фазы 0/1 ОБЯЗАНЫ вызывать guard_split() перед любым доступом к данным.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

OOS_FINAL_START = pd.Timestamp("2026-03-01", tz="UTC")
OOS_FINAL_END = pd.Timestamp("2026-09-01", tz="UTC")
HASH_FILE = Path("data/oos_final/OOS_HASHES.txt")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_oos_final(dataset: pd.DataFrame, out_path: Path) -> None:
    """Вырежьте и заморозьте OOS_FINAL ОДИН РАЗ; запись chmod 400 + хеш в файл."""
    part = dataset[(dataset["timestamp"] >= OOS_FINAL_START) & (dataset["timestamp"] < OOS_FINAL_END)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part.to_parquet(out_path, index=False)
    os.chmod(out_path, 0o400)
    entry = f"{out_path.name}: {_sha256(out_path)}\n"
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = HASH_FILE.read_text() if HASH_FILE.exists() else ""
    if out_path.name not in existing:
        HASH_FILE.open("a").write(entry)
    print(f"OOS_FINAL заморожен: {out_path} ({len(part)} строк), права 400, хеш записан")


def verify_hashes() -> bool:
    if not HASH_FILE.exists():
        return False
    ok = True
    for line in HASH_FILE.read_text().splitlines():
        name, _, digest = line.partition(": ")
        p = Path("data/oos_final") / name
        if not p.exists() or _sha256(p) != digest:
            ok = False
    return ok


def guard_split(name: str, allow_final_oos: bool = False) -> None:
    """Запрет доступа к OOS_FINAL без явного флага (защита от случайного подглядывания)."""
    if name.upper() in ("OOS_FINAL", "OOS-FINAL", "FINAL") and not allow_final_oos:
        raise PermissionError(
            "Доступ к OOS_FINAL запрещён. Нужен явный флаг --allow-final-oos "
            "(раздел 3.3: один запуск в самом конце фазы 2)."
        )


def assert_no_final_overlap(index: pd.DatetimeIndex, split: str,
                            allow_final_oos: bool = False) -> None:
    """Проверка, что запрос данных не пересекается с замороженным периодом."""
    if allow_final_oos:
        return
    if len(index) == 0:
        return
    lo = pd.Timestamp(index.min())
    hi = pd.Timestamp(index.max())
    if lo.tzinfo is None:
        lo, hi = lo.tz_localize("UTC"), hi.tz_localize("UTC")
    else:
        lo, hi = lo.tz_convert("UTC"), hi.tz_convert("UTC")
    overlaps = not (hi < OOS_FINAL_START or lo >= OOS_FINAL_END)
    if overlaps and split.upper() != "OOS_FINAL":
        # данные перекрывают финальный OOS — обрезаем, чтобы скрипт случайно не обучился на нём
        raise ValueError(
            f"Сплит '{split}' пересекается с замороженным OOS_FINAL "
            f"({lo} .. {hi}). Обрежьте данные до {OOS_FINAL_START}."
        )
