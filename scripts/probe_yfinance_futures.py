#!/usr/bin/env python3
"""Probe Yahoo/yfinance coverage for candidate futures symbols.

The probe is intentionally small and non-destructive. It checks whether Yahoo
has recent daily history for continuous and specific-contract futures symbols,
then writes a coverage report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "yfinance_futures_probe.csv"


SYMBOLS = [
    # Continuous nearby tickers.
    "ES=F",
    "NQ=F",
    "YM=F",
    "RTY=F",
    "GC=F",
    "MGC=F",
    "CL=F",
    "NG=F",
    "HO=F",
    "RB=F",
    "ZC=F",
    "ZW=F",
    "ZS=F",
    "ZM=F",
    "HE=F",
    "LE=F",
    "ZT=F",
    "ZF=F",
    "ZN=F",
    "6E=F",
    "6A=F",
    "6B=F",
    "6C=F",
    "VX=F",
    "FDAX.EX",
    "FVS=F",
    # Specific contracts observed on Yahoo futures-chain pages.
    "ESU26.CME",
    "ESZ26.CME",
    "ESH27.CME",
    "GCZ26.CMX",
    "CLU26.NYM",
    "CLV26.NYM",
    "ZCZ26.CBT",
    "ZCH27.CBT",
    "6EU26.CME",
]


def probe(symbol: str) -> dict[str, object]:
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1mo", interval="1d", auto_adjust=False)
        close = history["Close"].dropna() if "Close" in history else pd.Series(dtype=float)
        fast_info = getattr(ticker, "fast_info", {}) or {}
        return {
            "symbol": symbol,
            "rows": len(history),
            "first": history.index.min().date().isoformat() if len(history) else "",
            "last": history.index.max().date().isoformat() if len(history) else "",
            "last_close": float(close.iloc[-1]) if len(close) else None,
            "currency": fast_info.get("currency", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "rows": 0,
            "first": "",
            "last": "",
            "last_close": None,
            "currency": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame([probe(symbol) for symbol in SYMBOLS])
    report.to_csv(OUT, index=False)
    print(report.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
