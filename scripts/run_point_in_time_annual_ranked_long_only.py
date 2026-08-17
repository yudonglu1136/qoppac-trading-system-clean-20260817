#!/usr/bin/env python3
"""Point-in-time annual ranked long-only stock trend tests.

This replaces the current-constituent stock tests with annual, point-in-time
constituent snapshots:

- Wikipedia universes use the latest page revision available before Jan 1.
- iShares ETF universes use archived holdings files whose holdings-as-of date
  and Wayback capture date are both before Jan 1.
- The portfolio only ranks tickers in that year's snapshot.
- If a historical page or holdings file does not contain a real constituent
  table, that year is skipped instead of filling it with today's membership.
- Forecast scaling uses fixed Rob/pysystemtrade scalars, not full-sample
  calibration.

This still depends on free yfinance prices. Delisted or renamed historical
tickers often have missing Yahoo data, so this is a no-future-function research
run, not institutional-grade point-in-time equity data.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
import urllib.parse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

import run_sp500_stock_trend_backtest as stock_base


warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "point_in_time_index_members"
OUT = ROOT / "backtests" / "point_in_time_annual_ranked_long_only"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WAYBACK_CDX = "https://web.archive.org/cdx"
START = "2000-01-01"
BUSINESS_DAYS = 252.0
COST_PER_DOLLAR_TRADED = 0.0005
MIN_HISTORY_DAYS = 260
TOP_COUNTS = [20, 40]
FORECAST_CAP = 20.0

# Fixed scalars from pysystemtrade's provided Rob system config. These avoid the
# full-sample scalar fitting that the first stock-only sketch used.
FIXED_EWMAC_SCALARS = {
    16: 4.104172020369661,
    32: 2.786994330124792,
    64: 1.9093945630747895,
}
FIXED_BREAKOUT_SCALARS = {
    80: 0.726260784624834,
    160: 0.7388310187414805,
    320: 0.7366197028421859,
}


@dataclass(frozen=True)
class UniverseSpec:
    key: str
    title: str
    wiki_title: str
    wiki_url: str
    benchmark_ticker: str
    benchmark_label: str
    expected_rows: int
    min_active_floor: int
    symbol_suffix: str = ""
    source_kind: str = "wiki"


UNIVERSES = [
    UniverseSpec(
        key="sp500",
        title="S&P 500 / SPY",
        wiki_title="List of S&P 500 companies",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        benchmark_ticker="SPY",
        benchmark_label="SPY",
        expected_rows=500,
        min_active_floor=100,
    ),
    UniverseSpec(
        key="ftse100",
        title="FTSE 100",
        wiki_title="FTSE 100 Index",
        wiki_url="https://en.wikipedia.org/wiki/FTSE_100_Index",
        benchmark_ticker="^FTSE",
        benchmark_label="FTSE 100 (^FTSE)",
        expected_rows=100,
        min_active_floor=50,
        symbol_suffix=".L",
    ),
    UniverseSpec(
        key="nasdaq100",
        title="Nasdaq-100 / QQQ",
        wiki_title="Nasdaq-100",
        wiki_url="https://en.wikipedia.org/wiki/Nasdaq-100",
        benchmark_ticker="QQQ",
        benchmark_label="QQQ",
        expected_rows=100,
        min_active_floor=50,
    ),
    UniverseSpec(
        key="eem",
        title="Emerging Markets / EEM",
        wiki_title="iShares MSCI Emerging Markets ETF",
        wiki_url="https://www.ishares.com/us/products/239637/ishares-msci-emerging-markets-etf",
        benchmark_ticker="EEM",
        benchmark_label="EEM",
        expected_rows=1100,
        min_active_floor=100,
        source_kind="ishares_holdings",
    ),
    UniverseSpec(
        key="efa",
        title="Developed Markets ex-US / EFA",
        wiki_title="iShares MSCI EAFE ETF",
        wiki_url="https://www.ishares.com/us/products/239623/ishares-msci-eafe-etf",
        benchmark_ticker="EFA",
        benchmark_label="EFA",
        expected_rows=700,
        min_active_floor=100,
        source_kind="ishares_holdings",
    ),
]

ISHARES_HOLDINGS_SOURCES = {
    "eem": {
        "file_name": "EEM_holdings",
        "cdx_prefixes": [
            "www.ishares.com/us/products/239637/ishares-msci-emerging-markets-etf/1395165510754.ajax",
            "www.ishares.com/us/products/239637/ishares-msci-emerging-markets-etf/1449138789749.ajax",
            "www.ishares.com/us/products/239637/ishares-msci-emerging-markets-etf/1467271812596.ajax",
        ],
    },
    "efa": {
        "file_name": "EFA_holdings",
        "cdx_prefixes": [
            "www.ishares.com/us/products/239623/ishares-msci-eafe-etf/1395165510754.ajax",
            "www.ishares.com/us/products/239623/ishares-msci-eafe-etf/1449138789749.ajax",
            "www.ishares.com/us/products/239623/ishares-msci-eafe-etf/1467271812596.ajax",
        ],
    },
}

EM_EXCHANGE_SUFFIXES = {
    "tokyo stock exchange": ".T",
    "osaka securities exchange": ".T",
    "london stock exchange": ".L",
    "nyse euronext - euronext paris": ".PA",
    "euronext paris": ".PA",
    "euronext amsterdam": ".AS",
    "nyse euronext - euronext amsterdam": ".AS",
    "six swiss exchange": ".SW",
    "swiss exchange": ".SW",
    "xetra": ".DE",
    "deutsche boerse": ".DE",
    "asx": ".AX",
    "australian securities exchange": ".AX",
    "bolsa de madrid": ".MC",
    "borsa italiana": ".MI",
    "omx nordic exchange stockholm": ".ST",
    "stockholm": ".ST",
    "omx nordic exchange copenhagen": ".CO",
    "copenhagen": ".CO",
    "nasdaq omx helsinki": ".HE",
    "helsinki": ".HE",
    "nyse euronext - euronext brussels": ".BR",
    "euronext brussels": ".BR",
    "oslo bors": ".OL",
    "oslo": ".OL",
    "tel aviv": ".TA",
    "irish stock exchange": ".IR",
    "nyse euronext - euronext lisbon": ".LS",
    "euronext lisbon": ".LS",
    "wiener boerse": ".VI",
    "vienna": ".VI",
    "new zealand exchange": ".NZ",
    "hong kong exchanges": ".HK",
    "taiwan stock exchange": ".TW",
    "taipei exchange": ".TWO",
    "korea exchange (stock market)": ".KS",
    "kosdaq": ".KQ",
    "national stock exchange of india": ".NS",
    "bombay stock exchange": ".BO",
    "xbsp": ".SA",
    "sao paulo": ".SA",
    "bolsa mexicana": ".MX",
    "johannesburg": ".JO",
    "saudi stock exchange": ".SR",
    "warsaw": ".WA",
    "budapest": ".BD",
    "stock exchange of thailand": ".BK",
    "indonesia stock exchange": ".JK",
    "bursa malaysia": ".KL",
    "borsa istanbul": ".IS",
    "philippine": ".PS",
    "singapore exchange": ".SI",
    "shanghai stock exchange": ".SS",
    "shenzhen stock exchange": ".SZ",
    "qatar": ".QA",
    "abu dhabi": ".AD",
    "dubai": ".DU",
    "kuwait": ".KW",
    "santiago": ".SN",
    "egyptian": ".CA",
    "prague": ".PR",
    "colombia": ".CL",
    "lima": ".LM",
    "athens": ".AT",
}

FX_TICKERS = {
    "AED": ("AED=X", "divide"),
    "AUD": ("AUDUSD=X", "multiply"),
    "BRL": ("BRL=X", "divide"),
    "CHF": ("CHF=X", "divide"),
    "CLP": ("CLP=X", "divide"),
    "CNY": ("CNY=X", "divide"),
    "COP": ("COP=X", "divide"),
    "CZK": ("CZK=X", "divide"),
    "DKK": ("DKK=X", "divide"),
    "EGP": ("EGP=X", "divide"),
    "EUR": ("EURUSD=X", "multiply"),
    "GBP": ("GBPUSD=X", "multiply"),
    "HKD": ("HKD=X", "divide"),
    "HUF": ("HUF=X", "divide"),
    "IDR": ("IDR=X", "divide"),
    "INR": ("INR=X", "divide"),
    "ILS": ("ILS=X", "divide"),
    "JPY": ("JPY=X", "divide"),
    "KRW": ("KRW=X", "divide"),
    "KWD": ("KWD=X", "divide"),
    "MXN": ("MXN=X", "divide"),
    "MYR": ("MYR=X", "divide"),
    "NOK": ("NOK=X", "divide"),
    "NZD": ("NZDUSD=X", "multiply"),
    "PEN": ("PEN=X", "divide"),
    "PHP": ("PHP=X", "divide"),
    "PLN": ("PLN=X", "divide"),
    "QAR": ("QAR=X", "divide"),
    "SAR": ("SAR=X", "divide"),
    "SEK": ("SEK=X", "divide"),
    "SGD": ("SGD=X", "divide"),
    "THB": ("THB=X", "divide"),
    "TRY": ("TRY=X", "divide"),
    "TWD": ("TWD=X", "divide"),
    "USD": ("", "identity"),
    "ZAR": ("ZAR=X", "divide"),
}

YAHOO_SYMBOL_OVERRIDES = {
    (".CO", "AMBUB"): "AMBU-B",
    (".CO", "CARLA"): "CARL-A",
    (".CO", "CARLB"): "CARL-B",
    (".CO", "COLOB"): "COLO-B",
    (".CO", "MAERSKA"): "MAERSK-A",
    (".CO", "MAERSKB"): "MAERSK-B",
    (".CO", "NOVOB"): "NOVO-B",
    (".CO", "NSISB"): "NSIS-B",
    (".CO", "ROCKB"): "ROCK-B",
    (".HE", "NOK1V"): "NOKIA",
    (".HE", "SAMAS"): "SAMPO",
    (".HE", "UPM1V"): "UPM",
    (".ST", "NDA"): "NDA-SE",
    (".ST", "NDASE"): "NDA-SE",
    (".ST", "NDASEK"): "NDA-SE",
    (".SW", "ROG"): "RO",
    (".SW", "SIK"): "SIKA",
}

PRICE_ALIASES = {
    # Same listed company/security after ticker changes. The downloaded alias
    # series is stored under the original point-in-time ticker.
    "ABC": "COR",
    "ADS": "BFH",
    "ANTM": "ELV",
    "BLL": "BALL",
    "CBG": "CBRE",
    "COH": "TPR",
    "CTL": "LUMN",
    "DLPH": "APTV",
    "FB": "META",
    "FLT": "CPAY",
    "HFC": "DINO",
    "JEC": "J",
    "KORS": "CPRI",
    "MMC": "MRSH",
    "NLOK": "GEN",
    "PCLN": "BKNG",
    "PKI": "RVTY",
    "RE": "EG",
    "SYMC": "GEN",
    "TMK": "GL",
    "WLP": "ELV",
    "WLTW": "WTW",
}


def headers() -> dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 qoppac point-in-time annual backtest"}


def get_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    max_attempts: int = 6,
    timeout: int = 20,
) -> requests.Response:
    for attempt in range(max_attempts):
        response = session.get(url, params=params, headers=headers(), timeout=timeout)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            sleep_for = min(float(retry_after), 30.0)
        else:
            sleep_for = min(2.0 * (attempt + 1), 20.0)
        print(f"HTTP {response.status_code}; sleeping {sleep_for:.0f}s before retry", flush=True)
        time.sleep(sleep_for)

    response.raise_for_status()
    return response


def flatten_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for column in df.columns:
        if isinstance(column, tuple):
            parts = [str(part) for part in column if not str(part).startswith("Unnamed")]
            out.append(" ".join(parts).strip())
        else:
            out.append(str(column).strip())
    return out


def find_column(columns: list[str], needles: list[str]) -> str | None:
    for column in columns:
        lower = column.lower()
        if any(needle in lower for needle in needles):
            return column
    return None


def clean_symbol(symbol: object, suffix: str) -> str | None:
    if pd.isna(symbol):
        return None
    cleaned = str(symbol).strip()
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    cleaned = cleaned.replace("\xa0", " ").strip()
    cleaned = cleaned.split()[0].strip()
    if not cleaned or cleaned.lower() in {"nan", "ticker", "symbol", "epic"}:
        return None
    if suffix == ".L":
        if cleaned.endswith("."):
            cleaned = cleaned[:-1]
        else:
            cleaned = cleaned.replace(".", "-")
    else:
        cleaned = cleaned.replace(".", "-")
    if suffix and not cleaned.endswith(suffix):
        cleaned = f"{cleaned}{suffix}"
    return cleaned


def revision_before(session: requests.Session, wiki_title: str, date: str) -> dict | None:
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": wiki_title,
        "rvlimit": 1,
        "rvstart": f"{date}T00:00:00Z",
        "rvdir": "older",
        "format": "json",
        "redirects": 1,
    }
    response = get_with_retries(session, WIKI_API, params=params, timeout=20)
    data = response.json()
    for page in data.get("query", {}).get("pages", {}).values():
        revisions = page.get("revisions", [])
        if revisions:
            return {"title": page.get("title", wiki_title), **revisions[0]}
    return None


def old_revision_url(title: str, revid: int) -> str:
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    return f"https://en.wikipedia.org/w/index.php?title={encoded_title}&oldid={revid}"


def parse_constituent_table(html: str, spec: UniverseSpec) -> tuple[pd.DataFrame | None, str]:
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return None, "no_html_tables"

    candidates: list[tuple[int, int, pd.DataFrame, str, str, str | None]] = []
    for table_index, table in enumerate(tables):
        table = table.copy()
        columns = flatten_columns(table)
        table.columns = columns
        joined = "|".join(column.lower() for column in columns)
        is_changes_table = "reason" in joined or ("added" in joined and "removed" in joined)
        if is_changes_table:
            continue
        if any(column.lower().startswith("vte") for column in columns):
            continue

        symbol_col = find_column(columns, ["symbol", "ticker", "epic"])
        name_col = find_column(columns, ["security", "company", "name"])
        sector_col = find_column(columns, ["gics sector", "sector", "industry"])
        if not symbol_col or not name_col:
            continue
        if len(table) < max(20, int(spec.expected_rows * 0.35)):
            continue

        row_score = abs(len(table) - spec.expected_rows)
        candidates.append((row_score, table_index, table, symbol_col, name_col, sector_col))

    if not candidates:
        return None, "no_constituent_table"

    _, table_index, table, symbol_col, name_col, sector_col = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    selected_columns = [symbol_col, name_col] + ([sector_col] if sector_col else [])
    out = table[selected_columns].copy()
    out.columns = ["symbol_raw", "name"] + (["sector"] if sector_col else [])
    if "sector" not in out.columns:
        out["sector"] = pd.NA
    out["symbol"] = out["symbol_raw"].map(lambda value: clean_symbol(value, spec.symbol_suffix))
    out = out.dropna(subset=["symbol"]).drop_duplicates("symbol")
    out = out[~out["symbol"].str.contains("—|–|No", regex=True, na=False)]
    if len(out) < max(20, int(spec.expected_rows * 0.35)):
        return None, f"too_few_symbols_table_{table_index}"
    out["table_index"] = table_index
    return out[["symbol", "name", "symbol_raw", "sector", "table_index"]], "ok"


def load_annual_constituents(spec: UniverseSpec, start_year: int, end_year: int, refresh: bool) -> pd.DataFrame:
    if spec.source_kind.startswith("ishares"):
        return load_ishares_annual_constituents(spec, start_year, end_year, refresh)

    data_dir = DATA_ROOT / spec.key
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "annual_constituents.csv"
    audit_cache = data_dir / "annual_snapshot_audit.csv"
    if cache.exists() and audit_cache.exists() and not refresh:
        annual = pd.read_csv(cache)
        return annual[(annual["year"] >= start_year) & (annual["year"] <= end_year)].copy()

    rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    session = requests.Session()
    for year in range(start_year, end_year + 1):
        snapshot_date = f"{year}-01-01"
        status = "unknown"
        count = 0
        revision = revision_before(session, spec.wiki_title, snapshot_date)
        if revision is None:
            audit_rows.append({"year": year, "snapshot_date": snapshot_date, "status": "no_revision"})
            print(f"{spec.key} {year}: no revision", flush=True)
            continue

        url = old_revision_url(str(revision["title"]), int(revision["revid"]))
        try:
            response = get_with_retries(session, url, timeout=20)
            parsed, status = parse_constituent_table(response.text, spec)
        except Exception as exc:  # pragma: no cover - network edge
            parsed = None
            status = f"error: {exc}"

        if parsed is not None:
            parsed["year"] = year
            parsed["snapshot_date"] = snapshot_date
            parsed["revision_timestamp"] = revision.get("timestamp")
            parsed["revision_id"] = revision.get("revid")
            parsed["source_url"] = url
            rows.append(parsed)
            count = len(parsed)

        audit_rows.append(
            {
                "year": year,
                "snapshot_date": snapshot_date,
                "revision_timestamp": revision.get("timestamp"),
                "revision_id": revision.get("revid"),
                "source_url": url,
                "status": status,
                "constituent_count": count,
            }
        )
        print(f"{spec.key} {year}: {status} ({count})", flush=True)
        time.sleep(1.0)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_cache, index=False)
    if not rows:
        raise RuntimeError(f"No annual constituent snapshots parsed for {spec.key}")
    annual = pd.concat(rows, ignore_index=True)
    annual.to_csv(cache, index=False)
    return annual


def parse_date_text(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().strip('"')
    if not text or text == "-":
        return None
    for fmt in ["%b %d, %Y", "%d-%b-%Y", "%Y%m%d", "%Y-%m-%d"]:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt).date())
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed.date())


def parse_wayback_timestamp(timestamp: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(timestamp[:8], "%Y%m%d").date())


def wayback_capture_url(timestamp: str, original: str) -> str:
    return f"https://web.archive.org/web/{timestamp}if_/{original}"


def value_raw(value: object) -> float | str | None:
    if isinstance(value, dict):
        return value.get("raw", value.get("display"))
    return value


def parse_float_value(value: object) -> float | None:
    raw = value_raw(value)
    if raw is None or pd.isna(raw):
        return None
    try:
        return float(str(raw).replace("$", "").replace(",", "").replace("%", ""))
    except ValueError:
        return None


def clean_market_symbol(ticker: object) -> str | None:
    if pd.isna(ticker):
        return None
    symbol = str(ticker).strip().strip('"')
    if not symbol or symbol == "-" or symbol.lower() in {"nan", "cash"}:
        return None
    return re.sub(r"\s+", " ", symbol)


def exchange_suffix(exchange_text: str, currency: str, location_text: str) -> str | None:
    if "nasdaq omx nordic" in exchange_text:
        if currency == "DKK" or "denmark" in location_text:
            return ".CO"
        if currency == "EUR" or "finland" in location_text:
            return ".HE"
        return ".ST"

    for marker, candidate_suffix in EM_EXCHANGE_SUFFIXES.items():
        if marker in exchange_text:
            return candidate_suffix
    return None


def format_yahoo_symbol(symbol: str, suffix: str) -> str:
    yahoo = symbol.strip()
    normalized = re.sub(r"[\s/.\-]+", "", yahoo).upper()
    if (suffix, normalized) in YAHOO_SYMBOL_OVERRIDES:
        yahoo = YAHOO_SYMBOL_OVERRIDES[(suffix, normalized)]
    if suffix == ".HK" and yahoo.isdigit():
        yahoo = yahoo.zfill(4)
    elif suffix == ".L" and yahoo.endswith((".", "/")):
        yahoo = yahoo[:-1]
    yahoo = yahoo.replace("*", "")
    yahoo = re.sub(r"[\s/.]+", "-", yahoo).strip("-")
    return f"{yahoo}{suffix}"


def em_yahoo_symbol(
    ticker: object,
    exchange: object,
    market_currency: object | None = None,
    location: object | None = None,
) -> str | None:
    symbol = clean_market_symbol(ticker)
    if symbol is None:
        return None
    exchange_text = "" if pd.isna(exchange) else str(exchange).lower()
    currency = "" if market_currency is None or pd.isna(market_currency) else str(market_currency).upper()
    location_text = "" if location is None or pd.isna(location) else str(location).lower()

    is_us_exchange = "new york stock exchange" in exchange_text or "nyse" == exchange_text.strip()
    is_us_exchange = is_us_exchange or (
        "nasdaq" in exchange_text and "omx" not in exchange_text and "nordic" not in exchange_text
    )
    if is_us_exchange:
        return re.sub(r"[\s/.]+", "-", symbol).strip("-")

    suffix = exchange_suffix(exchange_text, currency, location_text)
    if suffix is None:
        return None
    if "india" in exchange_text and symbol.isdigit():
        suffix = ".BO"

    return format_yahoo_symbol(symbol, suffix)


def parse_ishares_csv(text: str) -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    lines = text.replace("\ufeff", "").splitlines()
    asof = None
    for line in lines[:12]:
        if "Fund Holdings as of" in line:
            parts = line.split(",", 1)
            if len(parts) == 2:
                asof = parse_date_text(parts[1])
            break

    header_idx = next((index for index, line in enumerate(lines) if line.startswith("Ticker,")), None)
    if header_idx is None:
        return None, asof
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    if "Ticker" not in frame.columns or "Name" not in frame.columns:
        return None, asof

    out = pd.DataFrame(
        {
            "symbol_raw": frame["Ticker"],
            "name": frame["Name"],
            "asset_class": frame.get("Asset Class", frame.get("Type", "")),
            "sector": frame.get("Sector", ""),
            "location": frame.get("Location", frame.get("Country", "")),
            "exchange": frame.get("Exchange", ""),
            "market_currency": frame.get("Market Currency", frame.get("Currency", "")),
            "weight": frame.get("Weight (%)", np.nan),
        }
    )
    out["weight"] = out["weight"].map(parse_float_value)
    return out, asof


def parse_ishares_json(text: str, asof_hint: pd.Timestamp | None) -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    try:
        data = json.loads(text)
    except ValueError:
        return None, asof_hint
    rows = data.get("aaData") if isinstance(data, dict) else None
    if not rows:
        return None, asof_hint

    parsed_rows = []
    for row in rows:
        if len(row) < 16:
            continue
        if str(row[2]).lower() == "equity":
            parsed_rows.append(
                {
                    "symbol_raw": row[0],
                    "name": row[1],
                    "asset_class": row[2],
                    "weight": parse_float_value(row[3]),
                    "sector": row[8],
                    "exchange": row[11],
                    "location": row[12],
                    "market_currency": row[14],
                }
            )
        else:
            parsed_rows.append(
                {
                    "symbol_raw": row[0],
                    "name": row[1],
                    "sector": row[2],
                    "asset_class": row[3],
                    "weight": parse_float_value(row[5]),
                    "location": row[12],
                    "exchange": row[13],
                    "market_currency": row[14],
                }
            )
    if not parsed_rows:
        return None, asof_hint
    return pd.DataFrame(parsed_rows), asof_hint


def ishares_source(spec: UniverseSpec) -> dict[str, object]:
    source = ISHARES_HOLDINGS_SOURCES.get(spec.key)
    if source is None:
        raise KeyError(f"No iShares holdings source configured for {spec.key}")
    return source


def ishares_candidate_rows(session: requests.Session, spec: UniverseSpec, start_year: int, end_year: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    source = ishares_source(spec)
    file_name = str(source["file_name"])
    for prefix in source["cdx_prefixes"]:
        params = {
            "url": prefix,
            "from": max(2014, start_year - 2),
            "to": end_year,
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype,digest",
            "filter": "statuscode:200",
            "collapse": "timestamp:8",
            "matchType": "prefix",
        }
        response = get_with_retries(session, WAYBACK_CDX, params=params, timeout=60)
        try:
            data = response.json()
        except ValueError:
            continue
        for item in data[1:]:
            timestamp, original, _, mimetype, digest = item
            is_csv = "fileType=csv" in original and file_name in original
            is_json = "fileType=json" in original and "tab=all" in original
            if not (is_csv or is_json):
                continue
            key = (timestamp, original)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"timestamp": timestamp, "original": original, "mimetype": mimetype, "digest": digest})
    return sorted(rows, key=lambda row: row["timestamp"])


def load_ishares_holding_snapshots(spec: UniverseSpec, start_year: int, end_year: int, refresh: bool) -> pd.DataFrame:
    data_dir = DATA_ROOT / spec.key
    snapshot_cache = data_dir / "holding_snapshots.csv"
    if snapshot_cache.exists() and not refresh:
        return pd.read_csv(snapshot_cache, parse_dates=["holding_asof", "archive_date"])

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    candidate_rows = ishares_candidate_rows(session, spec, start_year, end_year)
    for candidate in candidate_rows:
        timestamp = candidate["timestamp"]
        original = candidate["original"]
        archive_date = parse_wayback_timestamp(timestamp)
        asof_match = re.search(r"asOfDate=(\d{8})", original)
        asof_hint = parse_date_text(asof_match.group(1)) if asof_match else None
        url = wayback_capture_url(timestamp, original)
        try:
            response = get_with_retries(session, url, timeout=45)
        except Exception as exc:  # pragma: no cover - network edge
            print(f"{spec.key} {timestamp}: download error {exc}", flush=True)
            continue
        text = response.text
        if "aaData" in text[:200]:
            parsed, holding_asof = parse_ishares_json(text, asof_hint or archive_date)
        else:
            parsed, holding_asof = parse_ishares_csv(text)
            holding_asof = holding_asof or asof_hint or archive_date
        if parsed is None or holding_asof is None:
            print(f"{spec.key} {timestamp}: no holdings parsed", flush=True)
            continue

        parsed = parsed.copy()
        parsed["asset_class"] = parsed["asset_class"].astype(str)
        parsed = parsed[parsed["asset_class"].str.lower().eq("equity")]
        market_currency = parsed["market_currency"] if "market_currency" in parsed.columns else [None] * len(parsed)
        location = parsed["location"] if "location" in parsed.columns else [None] * len(parsed)
        parsed["symbol"] = [
            em_yahoo_symbol(ticker, exchange, currency, place)
            for ticker, exchange, currency, place in zip(
                parsed["symbol_raw"],
                parsed["exchange"],
                market_currency,
                location,
                strict=False,
            )
        ]
        parsed = parsed.dropna(subset=["symbol"]).drop_duplicates("symbol")
        parsed["holding_asof"] = holding_asof
        parsed["archive_timestamp"] = timestamp
        parsed["archive_date"] = archive_date
        parsed["source_url"] = url
        frames.append(parsed)
        print(
            f"{spec.key} {timestamp}: holdings_asof={holding_asof.date()} mapped={len(parsed)}",
            flush=True,
        )
        time.sleep(0.5)

    if not frames:
        raise RuntimeError(f"No {spec.key} holding snapshots parsed from Wayback")
    snapshots = pd.concat(frames, ignore_index=True)
    snapshots.to_csv(snapshot_cache, index=False)
    return snapshots


def load_ishares_annual_constituents(spec: UniverseSpec, start_year: int, end_year: int, refresh: bool) -> pd.DataFrame:
    data_dir = DATA_ROOT / spec.key
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "annual_constituents.csv"
    audit_cache = data_dir / "annual_snapshot_audit.csv"
    if cache.exists() and audit_cache.exists() and not refresh:
        annual = pd.read_csv(cache)
        return annual[(annual["year"] >= start_year) & (annual["year"] <= end_year)].copy()

    snapshots = load_ishares_holding_snapshots(spec, start_year, end_year, refresh)
    snapshots["holding_asof"] = pd.to_datetime(snapshots["holding_asof"])
    snapshots["archive_date"] = pd.to_datetime(snapshots["archive_date"])
    snapshot_keys = (
        snapshots[["holding_asof", "archive_timestamp", "archive_date", "source_url"]]
        .drop_duplicates()
        .sort_values(["holding_asof", "archive_timestamp"])
    )

    rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        snapshot_date = pd.Timestamp(f"{year}-01-01")
        eligible = snapshot_keys[
            (snapshot_keys["holding_asof"] <= snapshot_date)
            & (snapshot_keys["archive_date"] <= snapshot_date)
        ]
        if eligible.empty:
            audit_rows.append(
                {
                    "year": year,
                    "snapshot_date": snapshot_date.date(),
                    "status": "no_holding_snapshot",
                    "constituent_count": 0,
                }
            )
            print(f"{spec.key} {year}: no holding snapshot", flush=True)
            continue
        selected = eligible.iloc[-1]
        selected_frame = snapshots[
            (snapshots["holding_asof"] == selected["holding_asof"])
            & (snapshots["archive_timestamp"] == selected["archive_timestamp"])
        ].copy()
        selected_frame["year"] = year
        selected_frame["snapshot_date"] = snapshot_date.date()
        selected_frame["revision_timestamp"] = selected["archive_timestamp"]
        selected_frame["revision_id"] = selected["archive_timestamp"]
        selected_frame["source_asof"] = selected["holding_asof"]
        selected_frame["source_url"] = selected["source_url"]
        rows.append(selected_frame)
        age_days = int((snapshot_date - selected["holding_asof"]).days)
        audit_rows.append(
            {
                "year": year,
                "snapshot_date": snapshot_date.date(),
                "status": "ok",
                "constituent_count": len(selected_frame),
                "holding_asof": selected["holding_asof"].date(),
                "archive_timestamp": selected["archive_timestamp"],
                "archive_date": selected["archive_date"].date(),
                "snapshot_age_days": age_days,
                "source_url": selected["source_url"],
            }
        )
        print(
            f"{spec.key} {year}: ok ({len(selected_frame)}) asof={selected['holding_asof'].date()} age={age_days}d",
            flush=True,
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_cache, index=False)
    if not rows:
        raise RuntimeError(f"No annual {spec.key} snapshots selected")
    annual = pd.concat(rows, ignore_index=True)
    annual.to_csv(cache, index=False)
    return annual


def close_from_download(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        if len(tickers) == 1 and "Close" in data.columns:
            return data[["Close"]].rename(columns={"Close": tickers[0]})
        return pd.DataFrame()

    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        if (ticker, "Close") in data.columns:
            closes[ticker] = data[(ticker, "Close")]
        elif ("Close", ticker) in data.columns:
            closes[ticker] = data[("Close", ticker)]
    return pd.DataFrame(closes)


def download_prices(tickers: list[str], data_dir: Path, refresh: bool, chunk_size: int) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "adj_close.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date").sort_index()

    chunks: list[pd.DataFrame] = []
    failures: list[str] = []
    alias_rows: list[dict[str, str]] = []
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        request_map = {ticker: PRICE_ALIASES.get(ticker, ticker) for ticker in chunk}
        request_tickers = sorted(set(request_map.values()))
        print(f"{data_dir.name}: downloading {start + 1}-{start + len(chunk)} / {len(tickers)}", flush=True)
        try:
            data = yf.download(
                request_tickers,
                start=START,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=30,
            )
            downloaded = close_from_download(data, request_tickers)
            close = pd.DataFrame(index=downloaded.index)
            missing = []
            for ticker, request_ticker in request_map.items():
                if request_ticker not in downloaded:
                    missing.append(ticker)
                    continue
                close[ticker] = downloaded[request_ticker]
                if request_ticker != ticker:
                    alias_rows.append({"symbol": ticker, "download_symbol": request_ticker})
            if close.empty:
                failures.append(",".join(chunk))
            else:
                chunks.append(close)
                if missing:
                    failures.append(",".join(missing))
        except Exception as exc:  # pragma: no cover - network edge
            failures.append(f"{','.join(chunk)} :: {exc}")
        time.sleep(0.3)

    if not chunks:
        raise RuntimeError(f"No prices downloaded for {data_dir}")
    prices = pd.concat(chunks, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(how="all")
    prices.index.name = "Date"
    prices.to_csv(cache)
    if failures:
        (data_dir / "failed_download_chunks.txt").write_text("\n".join(failures), encoding="utf-8")
    if alias_rows:
        pd.DataFrame(alias_rows).drop_duplicates().to_csv(data_dir / "price_aliases_used.csv", index=False)
    return prices


def download_benchmark(spec: UniverseSpec, data_dir: Path, refresh: bool) -> pd.Series:
    cache = data_dir / "benchmark_adj_close.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date")[spec.benchmark_ticker].sort_index()

    data = yf.download(spec.benchmark_ticker, start=START, auto_adjust=True, progress=False, timeout=30)
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", spec.benchmark_ticker)]
    else:
        close = data["Close"]
    benchmark = close.rename(spec.benchmark_ticker).dropna()
    benchmark.index.name = "Date"
    benchmark.to_csv(cache)
    return benchmark


def symbol_currency_map(annual: pd.DataFrame) -> dict[str, str]:
    if "market_currency" not in annual.columns:
        return {}
    mapping: dict[str, str] = {}
    usable = annual.dropna(subset=["symbol", "market_currency"])
    for symbol, frame in usable.groupby("symbol"):
        currencies = frame["market_currency"].astype(str).str.upper()
        currencies = currencies[currencies.ne("-") & currencies.ne("NAN")]
        if currencies.empty:
            continue
        mapping[str(symbol)] = currencies.mode().iloc[0]
    return mapping


def download_fx_rates(currencies: set[str], data_dir: Path, refresh: bool) -> pd.DataFrame:
    cache = data_dir / "fx_rates.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date").sort_index()

    needed = sorted(currency for currency in currencies if currency in FX_TICKERS and currency != "USD")
    if not needed:
        return pd.DataFrame()

    ticker_to_currency = {FX_TICKERS[currency][0]: currency for currency in needed}
    data = yf.download(
        sorted(ticker_to_currency),
        start=START,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
        timeout=30,
    )
    close = close_from_download(data, sorted(ticker_to_currency))
    fx = pd.DataFrame(index=close.index)
    for ticker, currency in ticker_to_currency.items():
        if ticker in close:
            fx[currency] = close[ticker]
    fx.index.name = "Date"
    fx.to_csv(cache)
    return fx


def convert_em_prices_to_usd(price: pd.DataFrame, annual: pd.DataFrame, data_dir: Path, refresh: bool) -> pd.DataFrame:
    mapping = symbol_currency_map(annual)
    currencies = set(mapping.values())
    fx = download_fx_rates(currencies, data_dir, refresh).reindex(price.index).ffill()
    columns: dict[str, pd.Series] = {}
    missing_fx: set[str] = set()

    for symbol in price.columns:
        currency = mapping.get(symbol, "USD")
        if currency == "USD":
            columns[symbol] = price[symbol]
            continue
        fx_ticker = FX_TICKERS.get(currency)
        if fx_ticker is None or currency not in fx:
            missing_fx.add(currency)
            continue
        _, operation = fx_ticker
        if operation == "multiply":
            columns[symbol] = price[symbol] * fx[currency]
        elif operation == "divide":
            columns[symbol] = price[symbol] / fx[currency]
        else:
            columns[symbol] = price[symbol]

    if missing_fx:
        (data_dir / "missing_fx_currencies.txt").write_text(
            "\n".join(sorted(missing_fx)) + "\n",
            encoding="utf-8",
        )
    out = pd.DataFrame(columns, index=price.index)
    out.index.name = price.index.name
    return out.dropna(how="all")


def ewmac_forecast(price: pd.DataFrame, pct_vol: pd.DataFrame, fast: int) -> pd.DataFrame:
    slow = fast * 4
    fast_ewma = price.ewm(span=fast, min_periods=max(2, fast // 2)).mean()
    slow_ewma = price.ewm(span=slow, min_periods=max(2, slow // 2)).mean()
    raw = ((fast_ewma - slow_ewma) / price.abs().replace(0.0, np.nan)) / pct_vol
    return (raw * FIXED_EWMAC_SCALARS[fast]).clip(-FORECAST_CAP, FORECAST_CAP)


def breakout_forecast(price: pd.DataFrame, lookback: int) -> pd.DataFrame:
    smooth = max(int(lookback / 4.0), 1)
    roll_max = price.rolling(lookback, min_periods=max(20, lookback // 2)).max()
    roll_min = price.rolling(lookback, min_periods=max(20, lookback // 2)).min()
    roll_range = (roll_max - roll_min).replace(0.0, np.nan)
    raw = 40.0 * ((price - (roll_max + roll_min) / 2.0) / roll_range)
    forecast = raw.ewm(span=smooth, min_periods=max(2, smooth // 2)).mean()
    return (forecast * FIXED_BREAKOUT_SCALARS[lookback]).clip(-FORECAST_CAP, FORECAST_CAP)


def build_forecasts_no_lookahead(price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = price.pct_change()
    pct_vol = stock_base.mixed_vol(returns)
    valid_history = price.notna().rolling(MIN_HISTORY_DAYS, min_periods=MIN_HISTORY_DAYS).sum() >= MIN_HISTORY_DAYS

    rule_forecasts: list[pd.DataFrame] = []
    rows: list[dict[str, float | str]] = []
    for fast in [16, 32, 64]:
        rule_forecasts.append(ewmac_forecast(price, pct_vol, fast))
        rows.append({"rule": f"ewmac{fast}_{fast * 4}", "scalar": FIXED_EWMAC_SCALARS[fast], "source": "Rob config"})
    for lookback in [80, 160, 320]:
        rule_forecasts.append(breakout_forecast(price, lookback))
        rows.append({"rule": f"breakout{lookback}", "scalar": FIXED_BREAKOUT_SCALARS[lookback], "source": "Rob config"})

    stacked = pd.concat(rule_forecasts, axis=1, keys=range(len(rule_forecasts)))
    combined = stacked.T.groupby(level=1).mean().T
    return combined.where(valid_history).clip(-FORECAST_CAP, FORECAST_CAP), pd.DataFrame(rows)


def membership_by_year(annual: pd.DataFrame) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for year, frame in annual.groupby("year"):
        out[int(year)] = set(frame["symbol"].dropna().astype(str))
    return out


def build_long_only_weights(
    forecast: pd.DataFrame,
    annual: pd.DataFrame,
    total_names: int,
    min_active_floor: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_active = max(total_names, min_active_floor)
    year_members = membership_by_year(annual)
    rebalance_dates = stock_base.last_rebalance_dates(forecast.index)
    target_on_rebalance = pd.DataFrame(0.0, index=rebalance_dates, columns=forecast.columns)
    rows: list[dict[str, float | int | str]] = []

    for date in rebalance_dates:
        allowed = list(year_members.get(int(date.year), set()))
        if not allowed:
            continue
        available_allowed = [ticker for ticker in allowed if ticker in forecast.columns]
        scores = forecast.loc[date, available_allowed].replace([np.inf, -np.inf], np.nan).dropna()
        if len(scores) < min_active:
            continue
        names = scores.nlargest(total_names).index
        weight = 1.0 / total_names
        target_on_rebalance.loc[date, names] = weight
        for rank, ticker in enumerate(names, start=1):
            rows.append(
                {
                    "date": date,
                    "year": int(date.year),
                    "portfolio_size": total_names,
                    "rank": rank,
                    "ticker": ticker,
                    "forecast": float(scores[ticker]),
                    "weight": weight,
                    "allowed_members": len(allowed),
                    "scored_members": len(scores),
                    "min_active": min_active,
                }
            )

    weights = target_on_rebalance.reindex(forecast.index).ffill().fillna(0.0)
    return weights, pd.DataFrame(rows)


def run_portfolio(
    price: pd.DataFrame,
    forecast: pd.DataFrame,
    annual: pd.DataFrame,
    total_names: int,
    min_active_floor: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights, selections = build_long_only_weights(forecast, annual, total_names, min_active_floor)
    returns = price.pct_change().fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross_return = (held * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * COST_PER_DOLLAR_TRADED
    net_return = gross_return - costs
    equity = (1.0 + net_return.fillna(0.0)).cumprod()
    daily = pd.DataFrame(
        {
            "gross_return": gross_return,
            "costs": costs,
            "net_return": net_return,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
            "turnover": turnover,
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
            "long_count": (weights > 0.0).sum(axis=1),
        },
        index=price.index,
    )
    return daily, weights, selections


def trim_to_first_position(returns: pd.Series, weights: pd.DataFrame) -> pd.Series:
    gross = weights.abs().sum(axis=1)
    active = gross[gross > 0.0]
    if active.empty:
        return returns.dropna()
    return returns.loc[active.index[0] :].dropna()


def performance_stats(returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    ann_return = returns.mean() * BUSINESS_DAYS
    vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": years,
        "total_return": equity.iloc[-1] - 1.0,
        "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
        "ann_return": ann_return,
        "vol": vol,
        "sharpe": ann_return / vol if vol else np.nan,
        "mdd": (equity / equity.cummax() - 1.0).min(),
    }


def make_stats_table(streams: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in streams.items():
        row = {"strategy": name}
        row.update(performance_stats(returns))
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_returns(streams: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(streams).dropna(how="all").groupby(pd.Grouper(freq="YE")).apply(
        lambda frame: (1.0 + frame).prod() - 1.0
    )


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 126) -> pd.Series:
    aligned = pd.concat([a, b], axis=1).dropna()
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def crisis_table(streams: dict[str, pd.Series]) -> pd.DataFrame:
    windows = {
        "gfc_2007_2009": ("2007-10-09", "2009-03-09"),
        "covid_2020": ("2020-02-19", "2020-03-23"),
        "inflation_2022": ("2022-01-03", "2022-10-12"),
    }
    rows = []
    benchmark_name = next(name for name in streams if name.startswith("Benchmark"))
    benchmark = streams[benchmark_name]
    for window, (start, end) in windows.items():
        for name, series in streams.items():
            period = series.loc[start:end].dropna()
            if period.empty:
                continue
            equity = (1.0 + period).cumprod()
            row = {
                "window": window,
                "strategy": name,
                "return": equity.iloc[-1] - 1.0,
                "vol": period.std() * math.sqrt(BUSINESS_DAYS),
                "mdd": (equity / equity.cummax() - 1.0).min(),
            }
            if name != benchmark_name:
                row["corr_to_benchmark"] = series.loc[start:end].corr(benchmark.loc[start:end])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_results(
    spec: UniverseSpec,
    streams: dict[str, pd.Series],
    annual_returns: pd.DataFrame,
    corr20: pd.Series,
    corr40: pd.Series,
    out_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    top20 = "Top 20 long-only"
    top40 = "Top 40 long-only"
    benchmark = f"Benchmark: {spec.benchmark_label}"
    colors = {top20: "#1f77b4", top40: "#ff7f0e", benchmark: "#4c4c4c"}

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(4, 1, height_ratios=[3.0, 1.1, 1.1, 1.45], hspace=0.18)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(grid[3, 0])

    aligned = pd.DataFrame(streams).dropna(how="all")
    for name in streams:
        series = aligned[name].dropna()
        equity = (1.0 + series).cumprod()
        ax0.plot(equity.index, equity, label=name, color=colors.get(name), linewidth=1.8)
        drawdown = equity / equity.cummax() - 1.0
        ax1.plot(drawdown.index, drawdown, label=name, color=colors.get(name), linewidth=1.1)

    ax2.plot(corr20.index, corr20, color=colors[top20], label=f"{top20} corr", linewidth=1.2)
    ax2.plot(corr40.index, corr40, color=colors[top40], label=f"{top40} corr", linewidth=1.2)
    ax2.axhline(0.0, color="#777777", linewidth=0.8)
    ax2.legend(loc="upper left", ncol=2)

    annual_returns = annual_returns[[top20, top40, benchmark]].dropna(how="all")
    annual_returns.index = annual_returns.index.year
    x = np.arange(len(annual_returns.index))
    width = 0.25
    for offset, name in zip([-width, 0, width], annual_returns.columns):
        ax3.bar(x + offset, annual_returns[name], width=width, label=name, color=colors.get(name), alpha=0.9)
    ax3.axhline(0.0, color="#555555", linewidth=0.8)
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([str(year) for year in annual_returns.index[::2]], rotation=45, ha="right")

    ax0.set_title(f"{spec.title} Point-In-Time Annual Ranked Long-Only vs Benchmark")
    ax0.set_ylabel("Growth of $1")
    ax0.set_yscale("log")
    ax0.legend(loc="upper left", ncol=3)
    ax1.set_ylabel("Drawdown")
    ax2.set_ylabel("126d corr")
    ax3.set_ylabel("Year return")
    ax1.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax3.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{spec.key}_point_in_time_annual_long_only_vs_benchmark.png", dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def write_summary(
    spec: UniverseSpec,
    stats: pd.DataFrame,
    crisis: pd.DataFrame,
    annual: pd.DataFrame,
    price: pd.DataFrame,
    out_dir: Path,
) -> None:
    snapshot_counts = annual.groupby("year")["symbol"].nunique()
    price_cols = set(price.columns)
    priced_counts = annual.assign(has_price=annual["symbol"].isin(price_cols)).groupby("year")["has_price"].sum()
    if spec.source_kind.startswith("ishares"):
        source_lines = [
            f"- Constituent snapshots: each year's latest {spec.benchmark_label} holdings file with both holdings-as-of date and Internet Archive capture date on or before Jan 1.",
            f"- Source: {spec.wiki_url}.",
            "- Local-market prices are converted to USD using yfinance FX series before ranking and performance calculation.",
        ]
    else:
        source_lines = [
            f"- Constituent snapshots: each year's latest Wikipedia revision before Jan 1 from {spec.wiki_url}.",
        ]
    lines = [
        f"# {spec.title} Point-In-Time Annual Ranked Long-Only Backtest",
        "",
        *source_lines,
        f"- Snapshot years used: {int(snapshot_counts.index.min())}-{int(snapshot_counts.index.max())}; {len(snapshot_counts)} annual snapshots.",
        f"- Snapshot member count median: {snapshot_counts.median():.0f}; yfinance-priced member count median: {priced_counts.median():.0f}.",
        "- Signal: EWMAC 16/64, 32/128, 64/256 plus breakout 80, 160, 320.",
        "- Forecast scalars: fixed Rob/pysystemtrade scalars; no full-sample forecast scalar calibration.",
        "- Portfolio: weekly rebalance; top 20 or top 40 forecast names from that year's snapshot; equal-weight; 100% long-only.",
        f"- Trading cost assumption: {COST_PER_DOLLAR_TRADED:.2%} of notional traded.",
        "- Data caveat: yfinance often lacks delisted/renamed historical tickers; missing prices are not backfilled with current tickers.",
    ]
    if spec.key == "ftse100":
        lines.append("- Benchmark caveat: ^FTSE is a price index, not a total-return benchmark.")
    lines += [
        "",
        "## Performance",
        "",
        "| Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )
    lines += [
        "",
        "## Crisis Windows",
        "",
        "| Window | Strategy | Return | Vol | MDD | Corr To Benchmark |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in crisis.iterrows():
        corr = (
            ""
            if "corr_to_benchmark" not in row or pd.isna(row["corr_to_benchmark"])
            else f"{row['corr_to_benchmark']:.2f}"
        )
        lines.append(
            f"| {row['window']} | {row['strategy']} | {pct(row['return'])} | {pct(row['vol'])} | {pct(row['mdd'])} | {corr} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_universe(spec: UniverseSpec, start_year: int, end_year: int, refresh: bool, chunk_size: int) -> pd.DataFrame:
    data_dir = DATA_ROOT / spec.key
    out_dir = OUT / spec.key
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    annual = load_annual_constituents(spec, start_year, end_year, refresh)
    tickers = sorted(annual["symbol"].dropna().astype(str).unique())
    price = download_prices(tickers, data_dir, refresh, chunk_size)
    benchmark_price = download_benchmark(spec, data_dir, refresh)
    if spec.source_kind.startswith("ishares"):
        price = convert_em_prices_to_usd(price, annual, data_dir, refresh)

    usable = [column for column in price.columns if price[column].notna().sum() >= MIN_HISTORY_DAYS]
    price = price[usable].sort_index().ffill(limit=5)
    forecast, rule_table = build_forecasts_no_lookahead(price)

    streams: dict[str, pd.Series] = {}
    selections_all: list[pd.DataFrame] = []
    benchmark_name = f"Benchmark: {spec.benchmark_label}"
    for count in TOP_COUNTS:
        daily, weights, selections = run_portfolio(price, forecast, annual, count, spec.min_active_floor)
        label = f"Top {count} long-only"
        streams[label] = trim_to_first_position(daily["net_return"].rename(label), weights)
        daily.to_csv(out_dir / f"portfolio_daily_top{count}.csv")
        weights.iloc[::5].to_csv(out_dir / f"weekly_weights_top{count}.csv")
        selections_all.append(selections)

    benchmark_returns = benchmark_price.pct_change().rename(benchmark_name).dropna()
    common_start = max([series.index.min() for series in streams.values()] + [benchmark_returns.index.min()])
    common_end = min([series.index.max() for series in streams.values()] + [benchmark_returns.index.max()])
    streams = {name: series.loc[common_start:common_end] for name, series in streams.items()}
    streams[benchmark_name] = benchmark_returns.loc[common_start:common_end]
    aligned_streams = pd.DataFrame(streams).dropna(how="any")
    streams = {name: aligned_streams[name] for name in aligned_streams.columns}

    stats = make_stats_table(streams)
    annual_returns = yearly_returns(streams)
    crisis = crisis_table(streams)
    corr20 = rolling_corr(streams["Top 20 long-only"], streams[benchmark_name])
    corr40 = rolling_corr(streams["Top 40 long-only"], streams[benchmark_name])

    annual.to_csv(out_dir / "annual_constituents_used.csv", index=False)
    pd.concat(selections_all, ignore_index=True).to_csv(out_dir / "rebalance_selections.csv", index=False)
    rule_table.to_csv(out_dir / "rule_scalars.csv", index=False)
    stats.to_csv(out_dir / "stats.csv", index=False)
    annual_returns.to_csv(out_dir / "yearly_returns.csv")
    crisis.to_csv(out_dir / "crisis_windows.csv", index=False)
    pd.DataFrame({"Top 20 to benchmark": corr20, "Top 40 to benchmark": corr40}).to_csv(
        out_dir / "rolling_corr_to_benchmark.csv"
    )
    plot_results(spec, streams, annual_returns, corr20, corr40, out_dir)
    write_summary(spec, stats, crisis, annual, price, out_dir)

    stats.insert(0, "universe", spec.title)
    print(f"\n{spec.title}")
    print(stats.to_string(index=False))
    return stats


def write_combined_summary(all_stats: pd.DataFrame) -> None:
    lines = [
        "# Point-In-Time Annual Ranked Long-Only Backtests",
        "",
        "| Universe | Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_stats.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )
    lines += [
        "",
        "- Constituents are annual point-in-time snapshots from old Wikipedia revisions, not today's member lists.",
        "- EEM/EFA constituents use archived iShares holdings files, selected by holdings-as-of date rather than today's ETF holdings.",
        "- Years without a parsable real constituent table are skipped rather than filled with future members.",
        "- Forecast scalars are fixed Rob/pysystemtrade values, not fitted on the full backtest sample.",
        "- Free yfinance price availability for delisted/renamed stocks remains a limitation.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh Wikipedia snapshots and yfinance caches.")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument(
        "--universes",
        nargs="+",
        default=[spec.key for spec in UNIVERSES],
        choices=[spec.key for spec in UNIVERSES],
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in UNIVERSES if spec.key in args.universes]
    all_stats = pd.concat(
        [run_universe(spec, args.start_year, args.end_year, args.refresh, args.chunk_size) for spec in specs],
        ignore_index=True,
    )
    all_stats.to_csv(OUT / "stats_all.csv", index=False)
    write_combined_summary(all_stats)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
