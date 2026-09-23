# PROJECT_STATE — coder.qwen.ai (C:\42)

## 1) Назначение и схема потока
ML-крипто-торговая система SOL/USDT (Bybit). ТЗ: «структура торговой системы.txt» v4.2.
Принцип — **«Сигнал СНАЧАЛА»**: сначала доказать альфу на XGBoost/LightGBM против бенчмарков, только потом инфраструктура.

Поток Phase 0 (src/phase0/run_phase0.py):
```
data_collector (Bybit mainnet OHLCV+funding) → features (+labels) → dataset.parquet
→ train/val/oos сплиты (oos_guard: OOS_FINAL заморожен)
→ базовый XGBoost → Optuna (val_sharpe) → walk-forward 5 фолдов + embargo 72
→ non-overlapping бэктест OOS_ITERATIVE (ATR-выход, funding_paid) → Newey-West / block bootstrap
→ бенчмарки 1–4 (один движок симуляции) → KILL CRITERIA (IR>0.5, gain-importance) → GO/NO-GO
→ report.py: HTML-отчёт (backtests/reports/phase0_*.json → *.html)
```

## 2) Запуск / остановка
```
# данные уже собраны (Журнал: 41421 строка) — при чистом клоне сначала data_collector
python src/phase0/run_phase0.py [--trials 40] [--bench-runs 150] [--quick]
# --quick — smoke-тест без Optuna/WF; данные ожидаются в data/features/dataset.parquet
python src/phase0/report.py              # последний JSON в backtests/reports/ → HTML
git -C C:\42 pull   # обновление (рабочая папка = репозиторий)
```
Остановка: Ctrl+C (долгие прогоны — Optuna; не требует сервера/демона).

## 3) Внешние зависимости
- Python 3.10+ (локально в C:\42 — 3.14.5, часть пакетов отсутствует: xgboost/statsmodels/matplotlib/optuna → ставить через requirements.txt). Пакеты: numpy, pandas, xgboost, scikit-learn, statsmodels, pyarrow, optuna, matplotlib, PyYAML (см. requirements.txt).
- Bybit MAINNET: открытые эндпоинты OHLCV/funding (ключи НЕ требуются для данных; исполнение — Demo/Testnet в Phase 2+).
- Данные: `data/` в .gitignore (в репо нет — данные локальные!).

## 4) Критичные примечания и фиксы
- **OOS_FINAL (2026-03-01..2026-09-01) заморожен**: chmod 400 + SHA-256 в git + флаг `--allow-final-oos`. Ни один скрипт фазы 0/1 не смеет его трогать.
- **Фикс шкалы Sharpe**: все решения на per-bar базисах `sharpe_per_bar(rets, 8760)` (период = 8ч цикл фандинга); B&H тоже на часовых барах → шкалы сопоставимы (раздел 3.2).
- **Фикс статистики**: newey_west_significance(maxlags=24) и block_bootstrap_significance(block_size=24) — выровнены на forward_period (было 5/10).
- **ATR-выход в Phase 0**: ранний выход при пробое 2×ATR внутри holding=24 (atr_exit+atr_multiplier в конфиге) — совпадает с метками 3.5.
- **funding_paid** добавляется на сделку (сторона × mean_fr × n_settle/3, 8-часовые сеттлменты); анализ по сторонам — `funding_by_side(trades)`, в HTML-отчёте.
- **kill_criteria**: добавлен IR > 0.5 (info_ratio модели к B&H на per-bar ретёрнах) и no_dominant_feature (gain ≤ 0.90).
- **Воспроизводимость**: make_model setdefault random_state=42; train_and_predict early_stopping_rounds=50.
- **Бенчмарки B2/B3** симулируются тем же движком (holding, atr_target, funding_per_8h) — честное сравнение.
- Конфиг читается из YAML: детали из `config/system.yaml` переопределяют DEFAULTS в `config.py` (deep-merge). `data_collector` пишет `data/funding_rates/SOL_USDT/funding_history.parquet` (символ нормализован). Путь к конфигу можно переопределить env `TS_CONFIG`.

## 5) Чеклист здоровья
- [ ] git status чистый (текущая ветка main, коммит f6e9965)
- [ ] `python -m py_compile src/phase0/*.py` без ошибок
- [ ] config/system.yaml валидный YAML (проверено: блок backtest `holding_bars: 24, atr_exit: true`)
- [ ] data/features/dataset.parquet существует (при наличии локальных данных)
- [ ] Журнал разработки актуален (вердикт GO/NO-GO после прогона)
- [ ] Прогнать `run_phase0.py --quick` → сгенерировать `backtests/reports/phase0_*.json` → `report.py` → HTML

## 6) Ключевые папки
- `src/phase0/` — config, data_collector, features, backtest, benchmarks, oos_guard, run_phase0, report
- `config/system.yaml` — все параметры фаз (сплиты, этикетки, издержки, модель, Optuna, бенчмарки, backtest)
- `структура торговой системы.txt` — ТЗ v4.2 (источник истины)
- `Журнал разработки` — дневник решений/ограничений (обновлять!)