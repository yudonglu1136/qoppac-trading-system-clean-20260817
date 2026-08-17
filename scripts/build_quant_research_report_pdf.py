#!/usr/bin/env python3
"""Build the final Chinese PDF research report for the Rob-style research project."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
REPORT = OUT / "rob_style_quant_research_report_cn.pdf"
EMAIL_REPORT = OUT / "rob_style_quant_research_report_cn_email.pdf"

FONT_PATHS = [
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]
FONT_NAME = "ArialUnicode"
BUSINESS_DAYS = 256.0


def register_font() -> None:
    for path in FONT_PATHS:
        if path.exists():
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
            return
    raise FileNotFoundError("No Chinese-capable TTF font found. Tried Arial Unicode paths.")


def pct(value: float | str, digits: int = 1) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"{value:.{digits}%}"


def num(value: float | str, digits: int = 2) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def money(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"${value:,.0f}"


def load_cta_stream(path: Path, column: str = "buffered_integer") -> pd.Series:
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return frame[(column, "daily_return")].rename(path.parent.name).sort_index()


def load_spy_returns() -> pd.Series:
    spy = pd.read_csv(ROOT / "data" / "sp500_yfinance" / "spy_adj_close.csv", parse_dates=["Date"])
    close = spy.set_index("Date")["SPY"].sort_index()
    return close.pct_change().rename("SPY")


def nav_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def stream_metrics(returns: pd.Series, spy: pd.Series) -> dict[str, float | str]:
    aligned = pd.concat([returns, spy], axis=1).dropna()
    if aligned.empty:
        return {}
    r = aligned.iloc[:, 0]
    s = aligned["SPY"]
    nav = nav_from_returns(r)
    days = (r.index.max() - r.index.min()).days
    years = days / 365.25
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan
    ann_return = r.mean() * BUSINESS_DAYS
    ann_vol = r.std() * math.sqrt(BUSINESS_DAYS)
    drawdown = nav / nav.cummax() - 1.0
    beta = r.cov(s) / s.var() if s.var() else np.nan
    return {
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "years": years,
        "cagr": cagr,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else np.nan,
        "max_drawdown": float(drawdown.min()),
        "corr_to_spy": r.corr(s),
        "beta_to_spy": beta,
    }


def read_simple_return_csv(path: Path, date_col: str, return_col: str, name: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=[date_col])
    return frame.set_index(date_col)[return_col].rename(name).sort_index()


def build_spy_correlation_table() -> pd.DataFrame:
    spy = load_spy_returns()
    streams: list[tuple[str, pd.Series, str]] = [
        (
            "Rob-style futures, all available markets",
            load_cta_stream(ROOT / "backtests" / "rob_style_multirule_long" / "portfolio_daily.csv"),
            "Futures CTA",
        ),
        (
            "40 no-equity CTA",
            load_cta_stream(ROOT / "backtests" / "rob_style_no_equity_40_long" / "portfolio_daily.csv"),
            "Futures CTA",
        ),
        (
            "30 CTA / 70 SPY annual rebalance",
            read_simple_return_csv(
                ROOT / "backtests" / "cta_30_spy_70_annual_rebalance" / "daily_returns_nav.csv",
                "date",
                "portfolio_return",
                "30/70",
            ),
            "Portfolio mix",
        ),
        (
            "17 selected no-vol futures",
            read_simple_return_csv(
                ROOT / "backtests" / "selected_no_vol_vs_40_vs_spy" / "daily_comparison.csv",
                "date",
                "17 selected_return",
                "17 selected",
            ),
            "Futures CTA",
        ),
        (
            "KMLM-like rolling 15V",
            read_simple_return_csv(
                ROOT / "backtests" / "kmlm_like_2000" / "kmlmlike_rolling_15v_daily.csv",
                "date",
                "daily_return",
                "KMLM-like rolling",
            ),
            "Futures CTA",
        ),
        (
            "Actual KMLM ETF",
            read_simple_return_csv(
                ROOT / "backtests" / "kmlm_like_2000" / "actual_kmlm_fit_daily_returns.csv",
                "date",
                "KMLM actual",
                "KMLM actual",
            ),
            "ETF / benchmark",
        ),
        (
            "SPY stock top40 annual PIT",
            read_simple_return_csv(
                ROOT / "backtests" / "point_in_time_annual_ranked_long_only" / "sp500" / "portfolio_daily_top40.csv",
                "Date",
                "net_return",
                "SPY stock top40",
            ),
            "Stock selection",
        ),
        (
            "EM stock top40 annual PIT",
            read_simple_return_csv(
                ROOT / "backtests" / "point_in_time_annual_ranked_long_only" / "eem" / "portfolio_daily_top40.csv",
                "Date",
                "net_return",
                "EM stock top40",
            ),
            "Stock selection",
        ),
        (
            "Developed ex-US stock top40 annual PIT",
            read_simple_return_csv(
                ROOT / "backtests" / "point_in_time_annual_ranked_long_only" / "efa" / "portfolio_daily_top40.csv",
                "Date",
                "net_return",
                "DM stock top40",
            ),
            "Stock selection",
        ),
        (
            "SPY benchmark-aware stock momentum",
            read_simple_return_csv(
                ROOT / "backtests" / "benchmark_aware_stock_momentum" / "sp500" / "core80_signal20_daily.csv",
                "Date",
                "net_return",
                "Benchmark aware stock",
            ),
            "Stock selection",
        ),
        (
            "SPY sector-relative stock top10",
            read_simple_return_csv(
                ROOT
                / "backtests"
                / "stock_forecast_method_matrix_vol10_sector_fixed"
                / "sp500"
                / "sector_rel_mom_12_1__sector_top10_long_daily.csv",
                "Date",
                "daily_return",
                "Sector relative top10",
            ),
            "Stock selection",
        ),
    ]
    rows = []
    for name, returns, group in streams:
        metrics = stream_metrics(returns, spy)
        metrics.update({"strategy": name, "group": group})
        rows.append(metrics)
    table = pd.DataFrame(rows)[
        [
            "group",
            "strategy",
            "start",
            "end",
            "years",
            "cagr",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "corr_to_spy",
            "beta_to_spy",
        ]
    ]
    out_dir = ROOT / "backtests" / "strategy_spy_correlation_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "correlation_to_spy.csv", index=False)
    return table


def build_styles(font_name: str = FONT_NAME):
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=font_name,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "H1": ParagraphStyle(
            "H1CN",
            parent=base["Heading1"],
            fontName=font_name,
            fontSize=15,
            leading=20,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1f2937"),
        ),
        "H2": ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=12,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#374151"),
        ),
        "Body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9.2,
            leading=14,
            spaceAfter=6,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "Small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=7.5,
            leading=10,
            spaceAfter=4,
            wordWrap="CJK",
        ),
    }
    return styles


def paragraph(text: str, styles, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"- {text}", styles["Body"])


def table_flowable(
    rows: list[list[str]],
    styles,
    col_widths: list[float] | None = None,
    font_size: float = 7.2,
    header_color: str = "#e5e7eb",
) -> Table:
    data = [[Paragraph(str(cell), styles["Small"]) for cell in row] for row in rows]
    table_font = styles["Small"].fontName
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, -1), table_font),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def image_flowable(path: Path, width_cm: float = 17.0) -> Image:
    image = Image(str(path))
    max_width = width_cm * cm
    scale = max_width / image.imageWidth
    image.drawWidth = max_width
    image.drawHeight = image.imageHeight * scale
    return image


def metrics_table_from_csv(path: Path, rows: Iterable[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if rows is not None:
        frame = frame[frame["series"].isin(rows)]
    return frame


def add_title(story, styles) -> None:
    story.append(paragraph("Robert Carver 风格系统化交易研究报告", styles, "Title"))
    story.append(paragraph("期货趋势、股票横截面、SPY 相关性与 30/70 CTA-SPY 组合测试", styles, "H2"))
    story.append(paragraph("生成日期：2026-08-12。所有结果来自本地回测输出；本报告不是投资建议。", styles, "Small"))
    story.append(Spacer(1, 0.4 * cm))


def add_executive_summary(story, styles) -> None:
    story.append(paragraph("1. 执行摘要", styles, "H1"))
    for item in [
        "最有效的部分是多资产期货趋势系统，尤其是去掉 equity 后的 40 品种 CTA。它与 SPY 的日收益相关性约 -0.17，在 dot-com、GFC、COVID、2022 inflation bear 中都提供了对冲收益。",
        "Rob-style 风险系统的核心价值不在预测，而在把 forecast 转换成按波动率、权重、IDM、FDM、整数合约和成本约束后的可交易仓位。",
        "股票横截面版本没有稳定击败 SPY。SPY/发达市场股票趋势大多只是高 beta 或弱 alpha；EM 的 point-in-time long-only Top40 相对更有价值，但仍有数据完整性和交易成本限制。",
        "神经网络没有在当前数据条件下证明稳定增量。ML 的目标应该是增强已有弱 alpha 的非线性交互，而不是从噪音中创造 alpha。",
        "30% 40 no-equity CTA + 70% SPY、每 12 个月再平衡，在 2000-01-04 到 2024-03-28 的复利口径下，CAGR 16.5%、Vol 14.1%、Sharpe 1.17、MaxDD -21.0%，明显改善 SPY 的 -55.2% 最大回撤。",
    ]:
        story.append(bullet(item, styles))


def add_architecture(story, styles) -> None:
    story.append(paragraph("2. 系统架构和算法设计", styles, "H1"))
    story.append(paragraph("整体系统分成六层：数据层、forecast 层、forecast 合成层、风险和仓位层、执行成本层、报告和诊断层。核心原则是只修改 forecast，不让 alpha 模型直接决定杠杆、资金分配或风险预算。", styles))
    rows = [
        ["层", "职责", "实现要点"],
        ["Data", "读取期货 adjusted price、multiple/carry、FX、成本和元数据", "使用 cloned pysystemtrade CSV；股票用 yfinance 和 point-in-time 成分股快照"],
        ["Forecast", "生成 EWMAC、breakout、carry、relative carry、relative momentum、skew、accel 等信号", "Forecast cap 为 +/-20；股票实验另测 momentum、reversal、residual momentum、low vol、sector-relative momentum"],
        ["Forecast Combine", "按 Rob config 的 forecast weights 与 FDM 合成单品种 forecast", "保留 Rob 的 forecast scaling/weighting 框架；失败信号不进入生产配置"],
        ["Risk/Position", "按波动率目标、instrument weights、IDM 和合约风险计算目标仓位", "使用 buffered integer position，降低无意义换手和交易成本"],
        ["Costs", "按本地 spread/commission 估算交易成本", "报告 gross/net 和 annual cost as pct NAV"],
        ["Diagnostics", "年度收益、危机窗口、相关性、回撤、成本、active instruments", "所有关键输出落到 backtests/ 下"],
    ]
    story.append(table_flowable(rows, styles, [2.4 * cm, 5.0 * cm, 9.0 * cm]))
    story.append(paragraph("关键修正：长历史回测时，vol/FX 不再从未来向前 bfill；仓位有效性必须要求合约当天已有 price。这避免了早期历史中未上市/未有数据合约被错误分配权重。", styles))


def add_futures_results(story, styles) -> None:
    story.append(paragraph("3. 期货 CTA 结果", styles, "H1"))
    long_metrics = pd.read_csv(ROOT / "backtests" / "rob_long_history_comparison" / "full_sample_metrics.csv")
    rows = [["策略", "起点", "CAGR/年化均值", "Vol", "Sharpe", "MaxDD", "成本/年"]]
    for _, row in long_metrics.iterrows():
        rows.append(
            [
                row["stream"],
                row["start"],
                pct(row["ann_return"]),
                pct(row["ann_vol"]),
                num(row["sharpe"]),
                pct(row["max_drawdown"]),
                pct(row["costs_per_year_pct_capital"]),
            ]
        )
    story.append(table_flowable(rows, styles, [4.5 * cm, 2.1 * cm, 2.4 * cm, 2.0 * cm, 1.6 * cm, 2.0 * cm, 2.0 * cm]))
    story.append(paragraph("长历史显示，1970s 对收益贡献极大，但当时 active instruments 很少。这个结果更适合作为长期压力测试和趋势因子历史证据，不应被理解为今天可直接复制的真实容量。", styles))
    story.append(image_flowable(ROOT / "backtests" / "rob_long_history_comparison" / "long_history_equity_active_annual.png"))


def add_rob_live_comparison(story, styles) -> None:
    story.append(paragraph("4. 与 Rob 公布实盘收益对比", styles, "H1"))
    overlap = pd.read_csv(ROOT / "backtests" / "rob_published_comparison" / "overlap_fit_metrics.csv")
    rows = [["本地流", "重叠年份", "本地均值", "Rob 实盘均值", "平均差", "相关性", "同方向率"]]
    for _, row in overlap.iterrows():
        rows.append(
            [
                row["stream"],
                str(int(row["years"])),
                pct(row["mean_return"]),
                pct(row["published_mean_return"]),
                pct(row["mean_difference"]),
                num(row["correlation"]),
                pct(row["same_sign_rate"]),
            ]
        )
    story.append(table_flowable(rows, styles, [4.2 * cm, 1.8 * cm, 2.0 * cm, 2.1 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm]))
    story.append(paragraph("Rob 公布的是 live futures 表现，而本地是 compact replica，不是他的生产数据库、品种池和执行栈。2015-2024 重叠期，本地 Rob-style buffered 与 Rob 实盘相关性约 0.56；40 no-equity 与 Rob 实盘相关性约 0.71，但它不是 Rob 复制，而是更偏商品/利率/FX 的替代组合。", styles))
    story.append(image_flowable(ROOT / "backtests" / "rob_published_comparison" / "published_vs_local_annual_returns.png"))


def add_stock_and_ml_results(story, styles) -> None:
    story.append(paragraph("5. 股票横截面与 ML 实验结论", styles, "H1"))
    rows = [
        ["实验", "结论", "原因"],
        ["SPY 股票趋势/排名", "多数没有跑赢 SPY", "美股指数本身动量和 mega-cap 权重强，分散选股容易稀释赢家；交易成本和调仓噪音吃掉 alpha"],
        ["Top 10% long-only", "比 long-short 稳，但 SPY 仍更强", "股票横截面负 forecast 不等于可做空 alpha，short side 噪音和挤压大"],
        ["EM point-in-time Top40", "相对更有效", "EM 个股离散度更高，指数效率较低，Top40 CAGR/Sharpe 好于 EEM"],
        ["Developed ex-US", "不稳定", "市场结构分散但数据质量、货币/国家风险和成本更复杂"],
        ["Neural Network", "没有证明稳健增量", "样本 2016-2026 偏短，label 噪音高；MLP/Ridge/LightGBM 没有稳定 after-cost OOS 改善"],
        ["Benchmark-aware/weight-aware", "可改善风险形态，但不是决定性 alpha", "把指数权重纳入下注能降低偏离，但仍受 equity beta 主导"],
    ]
    story.append(table_flowable(rows, styles, [3.6 * cm, 4.4 * cm, 8.4 * cm]))
    story.append(paragraph("股票实验的核心教训：Rob 框架可以套到股票，但 forecast 本身必须适合股票。期货趋势是时间序列趋势和跨资产危机 beta；股票横截面更像相对收益排序，受 sector、size、index weight、liquidity 和 survivorship bias 影响更大。", styles))
    chart = ROOT / "backtests" / "stock_forecast_sector_idm_vol10_cross_universe" / "cross_universe_intermediate_mom_vs_benchmarks.png"
    if chart.exists():
        story.append(image_flowable(chart))


def add_spy_correlation(story, styles, corr_table: pd.DataFrame) -> None:
    story.append(paragraph("6. 各策略与 SPY 的相关性", styles, "H1"))
    display = corr_table.copy().sort_values("corr_to_spy")
    rows = [["类别", "策略", "区间", "CAGR", "Vol", "Sharpe", "MaxDD", "Corr SPY", "Beta SPY"]]
    for _, row in display.iterrows():
        rows.append(
            [
                row["group"],
                row["strategy"],
                f"{row['start']} - {row['end']}",
                pct(row["cagr"]),
                pct(row["ann_vol"]),
                num(row["sharpe"]),
                pct(row["max_drawdown"]),
                num(row["corr_to_spy"]),
                num(row["beta_to_spy"]),
            ]
        )
    story.append(table_flowable(rows, styles, [2.4 * cm, 4.4 * cm, 3.0 * cm, 1.6 * cm, 1.5 * cm, 1.3 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm], font_size=6.4))
    story.append(paragraph("解释：期货 CTA 的价值来自低相关或负相关，不是来自比 SPY 更像 SPY。股票策略即使 alpha 有一些信号，也通常与 SPY 高相关，因此改善组合回撤的能力弱于 CTA。30/70 组合仍与 SPY 相关 0.73，因为 70% 权重是股票，但 CTA 把最大回撤从 SPY 的 -55.2% 降到 -21.0%。", styles))


def add_3070_results(story, styles) -> None:
    story.append(paragraph("7. 30% CTA + 70% SPY，12 个月再平衡", styles, "H1"))
    metrics_df = pd.read_csv(ROOT / "backtests" / "cta_30_spy_70_annual_rebalance" / "metrics.csv")
    rows = [["组合", "CAGR", "Ann Ret", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar"]]
    for _, row in metrics_df.iterrows():
        rows.append(
            [
                row["series"],
                pct(row["cagr"]),
                pct(row["annual_return_arithmetic"]),
                pct(row["annual_vol"]),
                num(row["sharpe_0rf"]),
                num(row["sortino_0rf"]),
                pct(row["max_drawdown"]),
                num(row["calmar"]),
            ]
        )
    story.append(table_flowable(rows, styles, [4.8 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 1.5 * cm]))
    crisis = pd.read_csv(ROOT / "backtests" / "cta_30_spy_70_annual_rebalance" / "crisis_metrics.csv")
    rows = [["危机", "CTA", "SPY", "30/70", "组合 MaxDD"]]
    for window in crisis["window"].unique():
        sub = crisis[crisis["window"] == window].set_index("series")
        rows.append(
            [
                window,
                pct(sub.loc["CTA_no_equity_40", "total_return"]),
                pct(sub.loc["SPY", "total_return"]),
                pct(sub.loc["CTA30_SPY70_annual", "total_return"]),
                pct(sub.loc["CTA30_SPY70_annual", "max_drawdown"]),
            ]
        )
    story.append(table_flowable(rows, styles, [4.2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm]))
    story.append(paragraph("行为分析：组合不是简单提高收益，而是改变左尾。dot-com、GFC、COVID 和 2022 中，CTA sleeve 都在 SPY 下跌期取得正收益；30/70 组合仍可能短期下跌，因为 SPY 权重高，但回撤幅度显著收敛。年度再平衡让 CTA 在趋势大年自然漂移变大，随后下一年回到 30%。", styles))
    story.append(image_flowable(ROOT / "backtests" / "cta_30_spy_70_annual_rebalance" / "cta30_spy70_annual_rebalance.png"))


def add_what_worked(story, styles) -> None:
    story.append(paragraph("8. 什么有效，什么没有效果", styles, "H1"))
    rows = [
        ["模块", "有效性", "判断"],
        ["Futures trend following", "有效", "跨资产 trend/carry/breakout 在危机与通胀趋势中有可重复收益"],
        ["Buffered integer execution", "有效", "降低换手和成本，保持 Rob risk system 的仓位纪律"],
        ["No-equity 40 universe", "有效但激进", "去掉 equity 后对 SPY 更有对冲价值，但更集中于商品/利率/FX 宏观趋势"],
        ["KMLM/DBMF 对比", "有解释力", "2022 这类 inflation shock 中商品权重和规则速度决定爆发力"],
        ["QE/QT 分类", "弱证据", "不能简单说 QE 一定差、QT 一定好；趋势方向和资产路径比政策标签更重要"],
        ["SPY stock cross-section", "弱/无效", "多数版本跑不赢 SPY，alpha 被 mega-cap beta 和成本稀释"],
        ["EM stock cross-section", "局部有效", "Top40 point-in-time 表现好于 EEM，但数据质量和可交易性仍需审计"],
        ["Neural Network forecast", "当前无效", "未证明稳定正 Rank IC 或 after-cost 组合改善"],
    ]
    story.append(table_flowable(rows, styles, [4.0 * cm, 2.6 * cm, 9.5 * cm]))


def add_implementation(story, styles) -> None:
    story.append(paragraph("9. 可复现文件和下一步", styles, "H1"))
    rows = [
        ["用途", "文件"],
        ["Rob-style futures 回测", "scripts/run_rob_style_backtest.py"],
        ["No-equity 40 回测", "scripts/run_rob_style_no_equity_40_backtest.py"],
        ["Rob 公布实盘对比", "scripts/compare_rob_published_returns.py"],
        ["长历史报告", "scripts/compare_long_history_results.py"],
        ["30/70 CTA-SPY 回测", "scripts/backtest_cta_spy_70_30.py"],
        ["本 PDF 报告", "scripts/build_quant_research_report_pdf.py"],
        ["30/70 输出", "backtests/cta_30_spy_70_annual_rebalance/"],
        ["相关性输出", "backtests/strategy_spy_correlation_report/correlation_to_spy.csv"],
    ]
    story.append(table_flowable(rows, styles, [5.0 * cm, 10.5 * cm]))
    story.append(paragraph("下一步如果要进入实盘研究，重点不是继续调参，而是做三件事：第一，接入更稳定的 futures 数据源并核对 continuous contract/carry；第二，把 no-equity 40 做容量和保证金压力测试；第三，用 IB paper trading 验证 execution、slippage 和动态优化输出。", styles))


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(1.5 * cm, 1.0 * cm, "Rob-style systematic trading research report")
    canvas.drawRightString(19.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    register_font()
    styles = build_styles()
    corr_table = build_spy_correlation_table()

    doc = SimpleDocTemplate(
        str(REPORT),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.5 * cm,
        title="Robert Carver 风格系统化交易研究报告",
        author="Codex",
    )
    story = []
    add_title(story, styles)
    add_executive_summary(story, styles)
    add_architecture(story, styles)
    story.append(PageBreak())
    add_futures_results(story, styles)
    story.append(PageBreak())
    add_rob_live_comparison(story, styles)
    story.append(PageBreak())
    add_stock_and_ml_results(story, styles)
    story.append(PageBreak())
    add_spy_correlation(story, styles, corr_table)
    story.append(PageBreak())
    add_3070_results(story, styles)
    story.append(PageBreak())
    add_what_worked(story, styles)
    add_implementation(story, styles)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    email_styles = styles
    email_doc = SimpleDocTemplate(
        str(EMAIL_REPORT),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.5 * cm,
        title="Robert Carver 风格系统化交易研究报告 - 邮件版",
        author="Codex",
    )
    email_story = []
    add_title(email_story, email_styles)
    email_story.append(paragraph("邮件附件轻量版：不嵌入大图，保留算法设计、系统搭建、有效/无效结论、SPY 相关性和 30/70 CTA-SPY 回测指标。完整版带图文件在同一目录：output/pdf/rob_style_quant_research_report_cn.pdf。", email_styles, "Small"))
    add_executive_summary(email_story, email_styles)
    add_architecture(email_story, email_styles)
    email_story.append(PageBreak())
    long_metrics = pd.read_csv(ROOT / "backtests" / "rob_long_history_comparison" / "full_sample_metrics.csv")
    email_story.append(paragraph("3. 期货 CTA 结果", email_styles, "H1"))
    rows = [["策略", "起点", "CAGR/年化均值", "Vol", "Sharpe", "MaxDD", "成本/年"]]
    for _, row in long_metrics.iterrows():
        rows.append([row["stream"], row["start"], pct(row["ann_return"]), pct(row["ann_vol"]), num(row["sharpe"]), pct(row["max_drawdown"]), pct(row["costs_per_year_pct_capital"])])
    email_story.append(table_flowable(rows, email_styles, [4.5 * cm, 2.1 * cm, 2.4 * cm, 2.0 * cm, 1.6 * cm, 2.0 * cm, 2.0 * cm]))
    add_spy_correlation(email_story, email_styles, corr_table)
    email_story.append(PageBreak())
    add_3070_results_without_image(email_story, email_styles)
    add_what_worked(email_story, email_styles)
    add_implementation(email_story, email_styles)
    email_doc.build(email_story, onFirstPage=footer, onLaterPages=footer)


def add_3070_results_without_image(story, styles) -> None:
    story.append(paragraph("7. 30% CTA + 70% SPY，12 个月再平衡", styles, "H1"))
    metrics_df = pd.read_csv(ROOT / "backtests" / "cta_30_spy_70_annual_rebalance" / "metrics.csv")
    rows = [["组合", "CAGR", "Ann Ret", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar"]]
    for _, row in metrics_df.iterrows():
        rows.append(
            [
                row["series"],
                pct(row["cagr"]),
                pct(row["annual_return_arithmetic"]),
                pct(row["annual_vol"]),
                num(row["sharpe_0rf"]),
                num(row["sortino_0rf"]),
                pct(row["max_drawdown"]),
                num(row["calmar"]),
            ]
        )
    story.append(table_flowable(rows, styles, [4.8 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 1.5 * cm]))
    crisis = pd.read_csv(ROOT / "backtests" / "cta_30_spy_70_annual_rebalance" / "crisis_metrics.csv")
    rows = [["危机", "CTA", "SPY", "30/70", "组合 MaxDD"]]
    for window in crisis["window"].unique():
        sub = crisis[crisis["window"] == window].set_index("series")
        rows.append(
            [
                window,
                pct(sub.loc["CTA_no_equity_40", "total_return"]),
                pct(sub.loc["SPY", "total_return"]),
                pct(sub.loc["CTA30_SPY70_annual", "total_return"]),
                pct(sub.loc["CTA30_SPY70_annual", "max_drawdown"]),
            ]
        )
    story.append(table_flowable(rows, styles, [4.2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm]))
    story.append(paragraph("行为分析：30/70 组合的主要价值是降低左尾风险。它保留 SPY 的长期 beta，同时利用 no-equity CTA 在股市危机和通胀趋势中的低相关/负相关收益。", styles))


def main() -> None:
    build_pdf()
    print(f"Wrote {REPORT}")
    print(f"Wrote {EMAIL_REPORT}")


if __name__ == "__main__":
    main()
