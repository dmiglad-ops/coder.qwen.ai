# -*- coding: utf-8 -*-
"""HTML-отчёт Phase 0 (раздел 3.12, чек-лист п.13).

Читает results JSON (backtests/reports/phase0_*.json или --json) и строит
одностраничный HTML: метрики, бенчмарки, walk-forward, KILL CRITERIA,
фандинг по сторонам, важность фичей. matplotlib используется опционально
(если доступен — equity-кривые и гистограммы встраиваются base64).

Использование:
  python3 report.py [--json backtests/reports/phase0_2026-09-24.json] [-o report.html]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("backtests/reports")


def _meta(report: dict) -> str:
    c = report.get("config", {})
    best = report.get("best_params", {})
    return (
        f"<li>Символ: <b>{report.get('symbol')}</b>, ТФ: <b>{report.get('timeframe')}</b>, "
        f"фаза {report.get('phase')}</li>"
        f"<li>Сплиты: {c.get('splits', {}).get('oos_iterative')} → OOS_ITERATIVE</li>"
        f"<li>backtest: holding={c.get('backtest', {}).get('holding_bars')}, "
        f"embargo={c.get('backtest', {}).get('embargo_bars')}, "
        f"atr_exit={c.get('backtest', {}).get('atr_exit')}</li>"
        f"<li>Optuna trials: {c.get('n_trials')}; bench runs: {c.get('n_bench_runs')}</li>"
        f"<li>best params: {json.dumps(best, ensure_ascii=False)[:200]}</li>"
    )


def _table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "<p><i>нет данных</i></p>"
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def _metrics_table(m: dict, bbm: dict | None = None) -> str:
    rows = [
        ["Total return", f"{m.get('total_return', 0):+.2%}"],
        ["Sharpe (per-trade, √365)", f"{m.get('sharpe', 0):.2f}"],
        ["Sharpe (per-bar, √8760)", f"{m.get('sharpe_pb', 0):.2f}"],
        ["Max drawdown", f"{m.get('max_drawdown', 0):.1%}"],
        ["Win rate", f"{m.get('win_rate', 0):.1%}"],
        ["Сделок (OOS)", m.get("n_trades", 0)],
        ["ATR-выходы", f"{m.get('n_exit_atr', 0)}/{m.get('n_trades', 0)}"],
        ["t-stat (Newey-West)", f"{m.get('t_statistic', 0):.2f}"],
        ["p-value", f"{m.get('p_value', 1):.4f}"],
    ]
    if bbm and "p_value" in bbm:
        rows.append(["block-bootstrap p", f"{bbm['p_value']:.4f} (ci {bbm.get('ci_95')})"])
    else:
        rows.append(["block-bootstrap p", "—"])
    return _table(["Метрика", "Значение"], rows)


def _benchmarks_table(bm: dict) -> str:
    b1, b2, b3, b4 = bm["buyhold"], bm["benchmark_2_timing"], bm["benchmark_3_direction"], bm["benchmark_4_sma"]
    rows = [
        ["1. Buy & Hold", f"{b1.get('total_return', 0):+.2%}", f"{b1.get('sharpe_pb', 0):.2f}", f"{b1.get('max_drawdown', 0):.1%}", "—"],
        ["2. Случайные таймстампы", f"{b2.get('mean', 0):+.2%}", "—", "—", f"{b2.get('model_percentile', 0):.0f}%"],
        ["3. Случайные направления", f"{b3.get('mean', 0):+.2%}", "—", "—", f"{b3.get('model_percentile', 0):.0f}%"],
        ["4. SMA cross 20/50", f"{b4.get('total_return', 0):+.2%}", f"{b4.get('sharpe_pb', 0):.2f}", f"{b4.get('max_drawdown', 0):.1%}", "—"],
    ]
    return _table(["Бенчмарк", "Return", "Sharpe_pb", "MaxDD", "Перцентиль модели"], rows) + \
        f"<p>Интерпретация: <b>{bm.get('interpretation', '—')}</b></p>"


def _wf_table(wf: list[dict]) -> str:
    rows = [[f['fold'], f.get('period', ''), f.get('n_trades', 0),
             f"{f.get('total_return', 0):+.2%}", f"{f.get('sharpe_pb', 0):.2f}",
             f"{f.get('win_rate', 0):.1%}"] for f in wf]
    return _table(["Fold", "Период", "Сделок", "Return", "Sharpe_pb", "WinRate"], rows)


def _kill_table(k: dict) -> str:
    rows = [[name, "✅" if v else "❌"] for name, v in k.get("checks", {}).items()]
    head = _table(["Критерий (3.11)", "Статус"], rows)
    verdict = k.get("verdict", "—")
    color = "#1a7f37" if verdict == "GO" else "#cf222e"
    return (f"<p style='font-size:20px'>ВЕРДИКТ: <b style='color:{color}'>{verdict}</b></p>"
            f"<p>IR={k.get('info_ratio', 0):.2f} | Sharpe_pb: модель "
            f"{k.get('sharpe_pb_model', 0):.2f} / B&H {k.get('sharpe_pb_buyhold', 0):.2f} / "
            f"SMA {k.get('sharpe_pb_sma', 0):.2f} | MaxDD: модель "
            f"{k.get('maxdd_model', 0):.1%} / B&H {k.get('maxdd_buyhold', 0):.1%}</p>"
            f"{head}")


def _funding_table(fbx: dict) -> str:
    if not fbx:
        return "<p><i>фандинг-анализ отсутствует в данных отчёта</i></p>"
    rows = [[side.title(), d.get("n", 0), f"{d.get('total_paid', 0):+.4f}",
             f"{d.get('mean_paid', 0):+.5f}"] for side, d in fbx.items()]
    return _table(["Сторона", "Сделок", "Funding суммарно", "Funding в среднем"], rows)


def _charts(report: dict) -> str:
    """Построение графиков через matplotlib, если он доступен. Иначе — заглушки."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return "<p><i>matplotlib недоступен — графики пропущены</i></p>"

    oos = report.get("oos_iterative_metrics", {})
    bm = report.get("benchmarks", {})
    imgs = []

    col = (int(oos.get("n_exit_atr", 0) or 0), oos.get("n_trades", 0) or 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["ATR-выход", "Time-выход"], col, color=["#1a7f37", "#8250df"])
    axes[0].set_title("Типы выходов (OOS)")
    axes[1].axis("off")
    axes[1].text(0.02, 0.6, f"Sharpe_pb модели={oos.get('sharpe_pb', 0):.2f}\n"
                            f"IR={report.get('kill_criteria', {}).get('info_ratio', 0):.2f}\n"
                            f"MaxDD={oos.get('max_drawdown', 0):.1%}",
                 family="monospace", fontsize=11)
    axes[1].set_title("Итоговые метрики")
    imgs.append(_fig_to_img(fig))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dist, title, model_line in (
        (axes[0], report.get("bench2_dist_sample") or [], "Bench 2: случайные таймстампы",
         oos.get("total_return")),
        (axes[1], report.get("bench3_dist_sample") or [], "Bench 3: случайные направления",
         oos.get("total_return")),
    ):
        if dist:
            ax.hist(dist, bins=30, alpha=0.6)
            ax.axvline(model_line, color="red", ls="--", label="модель")
            ax.legend()
        else:
            ax.text(0.4, 0.5, "нет распределения в JSON", transform=ax.transAxes)
        ax.set_title(title)
    imgs.append(_fig_to_img(fig))
    plt.close(fig)

    return "".join(f'<img src="data:image/png;base64,{b64}"/>' for b64 in imgs)


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def build_html(report: dict, output_path: Path) -> None:
    kc = report.get("kill_criteria", {})
    funding = report.get("funding_by_side", {})
    body = f"""
<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Phase 0 v{report.get('phase', 0)} — {report.get('symbol', '?')} {report.get('generated_at', '')[:10]}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; background: #0d1117; color: #e6edf3; }}
 h1 {{ color: #58a6ff; }} h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
 th, td {{ border: 1px solid #30363d; padding: 5px 8px; font-size: 14px; text-align: left; }}
 th {{ background: #161b22; color: #58a6ff; }}
 img {{ max-width: 95%; border: 1px solid #30363d; margin: 6px 0; }}
 ul {{ line-height: 1.6; }}
</style></head><body>
<h1>Phase {report.get('phase', 0)}: валидация сигнала — {report.get('symbol', '?')}</h1>
<p>Сгенерировано: {report.get('generated_at', '—')}</p>
<h2>Конфигурация</h2><ul>{_meta(report)}</ul>
<h2>Метрики OOS_ITERATIVE</h2>{_metrics_table(report.get('oos_iterative_metrics', {}), report.get('block_bootstrap'))}
<h2>Бенчмарки 1–4</h2>{_benchmarks_table(report.get('benchmarks', {}))}
<h2>Walk-forward</h2>{_wf_table(report.get('walkforward', []))}
<h2>KILL CRITERIA</h2>{_kill_table(kc)}
<h2>Фандинг по сторонам</h2>{_funding_table(funding)}
<h2>Feature importance (gain)</h2>{_table(['Фича', 'Gain'],
     [[k, f"{v:.4f}"] for k, v in report.get('feature_importance_gain', {}).items()])}
<h2>Графики</h2>{_charts(report)}
</body></html>"""
    output_path.write_text(body, encoding="utf-8")
    print(f"Отчёт сохранён: {output_path.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="путь к results JSON (иначе последний phase0_*.json)")
    ap.add_argument("-o", "--out", default=None, help="путь к HTML (по умолчанию рядом с JSON)")
    args = ap.parse_args()

    if args.json:
        jp = Path(args.json)
    else:
        files = sorted(REPORTS_DIR.glob("phase0_*.json"))
        if not files:
            print(f"Нет отчётов в {REPORTS_DIR}: сначала запустите run_phase0.py", file=sys.stderr)
            sys.exit(1)
        jp = files[-1]
    report = json.loads(jp.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else jp.with_suffix(".html")
    build_html(report, out)


if __name__ == "__main__":
    main()