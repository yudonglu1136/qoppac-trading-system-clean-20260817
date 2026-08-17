# Qoppac / Rob-Style Systematic Trading Research

This repository is a clean research and paper-trading harness around a
Robert Carver style systematic futures workflow. It intentionally separates:

- research and backtests
- forecast/model experiments
- IBKR paper/live adapters
- read-only monitoring UI
- local-only data, secrets, logs, and broker state

Start with:

- [Architecture](docs/ARCHITECTURE.md)
- [IBKR Paper/Live Boundary](docs/IBKR_PAPER_LIVE_BOUNDARY.md)
- [GitHub Hygiene Audit](docs/GITHUB_HYGIENE_AUDIT.md)

## Repository Boundaries

Tracked source code and docs live in:

- `scripts/` - research/backtest runners and IBKR adapter scripts
- `cross_sectional_nn/` - cross-sectional forecast research modules
- `ibkr-monitor-prototype/` - read-only React monitor
- `config/` - non-secret strategy guardrails and sanitized templates
- `docs/` - architecture, operating boundaries, and audit notes
- `tests/` - leakage, safety, and persistence tests

Local-only artifacts are ignored and must not be uploaded:

- `env/` and `.env*`
- `data/`
- `output/`
- `reports/`
- `research/`
- `notes/`
- `blog/`
- `github/`
- SQLite databases, logs, screenshots, node modules, and caches

## Backtest Path

Backtests must not connect to IBKR and must not read broker state. Typical
entry points:

```bash
python3 scripts/run_minimal_ewmac_backtest.py
python3 scripts/run_rob_style_no_equity_40_backtest.py
python3 scripts/run_ibkr_capacity_stress.py
```

## IBKR Paper Path

IBKR scripts are adapter and execution-boundary code. They require an explicit
paper account through CLI or environment:

```bash
export IBKR_EXPECTED_ACCOUNT="YOUR_PAPER_ACCOUNT"
python3 scripts/ibkr_connection_smoke.py --port 4002
python3 scripts/ibkr_historical_bar_gate.py --all
python3 scripts/ibkr_strategy_order_dry_run.py
```

Order transmission additionally requires explicit confirmation flags. See
[IBKR Paper/Live Boundary](docs/IBKR_PAPER_LIVE_BOUNDARY.md).

## Upstream References

- Rob Carver blog: https://qoppac.blogspot.com/2021/12/my-trading-system.html
- `pysystemtrade`: https://github.com/pst-group/pysystemtrade
