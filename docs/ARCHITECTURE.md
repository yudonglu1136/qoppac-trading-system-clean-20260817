# Architecture

This project has three separate layers. Keep them separate when researching,
running backtests, or connecting IBKR.

## 1. Research And Backtest Path

Purpose: test ideas, generate diagnostics, and produce target-position files.

Main paths:

- `scripts/run_*backtest.py`
- `scripts/compare_*.py`
- `scripts/run_stock_*.py`
- `scripts/run_cross_sectional_nn_forecast.py`
- `cross_sectional_nn/`
- local-only `backtests/`, `research/`, `reports/`, `data/`

Rules:

- No IBKR socket connection.
- No order placement.
- No account, margin, open-order, or fill dependency.
- No live broker state as model input.
- Outputs are research artifacts, not execution authority.

Typical output:

- portfolio daily returns
- positions or target positions
- diagnostics and charts
- target manifest for downstream review

## 2. Strategy Engine Boundary

Purpose: preserve the Robert Carver style risk and sizing system.

The strategy engine boundary is:

```text
prices / forecasts
-> Rob-style volatility estimation
-> forecast scaling
-> instrument risk calculation
-> instrument weights / diversification
-> volatility targeting
-> integer position targets
```

Research may change forecasts or universes, but should not silently change:

- volatility estimator
- forecast scale/cap
- risk target
- diversification multiplier / IDM
- instrument weights
- transaction cost assumptions
- integer position logic
- portfolio aggregation

When the engine is intentionally changed, document it as an engine change, not
as a forecast experiment.

## 3. IBKR Paper/Live Adapter Path

Purpose: connect strategy targets to IBKR paper infrastructure with hard gates.

Main paths:

- `scripts/ibkr_connection_smoke.py`
- `scripts/ibkr_contract_qualification.py`
- `scripts/ibkr_market_data_gate.py`
- `scripts/ibkr_historical_bar_gate.py`
- `scripts/ibkr_strategy_order_dry_run.py`
- `scripts/ibkr_paper_strategy_runner.py`
- `scripts/ibkr_paper_daemon.py`
- `config/ibkr_paper_live_guardrails.yaml`
- `config/launchagents/*.plist.template`

Rules:

- IBKR scripts are adapters, not research engines.
- Paper execution requires explicit paper account identity.
- Order transmission requires explicit confirmation flags.
- Runtime state is local-only under ignored `output/` and `data/`.
- Concrete LaunchAgent plists are local machine files and are not committed.

Current implementation is paper-first. Live trading must be a separate,
reviewed deployment path.

## 4. Read-Only Monitor

Purpose: observe account, positions, gate status, data freshness, and local
price overlays.

Main path:

- `ibkr-monitor-prototype/`

Rules:

- No execution controls in the UI.
- The monitor may show broker/account values and local marks.
- The monitor must not be treated as the source of target positions.

## 5. Data Flow

```mermaid
flowchart TD
    Research["Research/backtest scripts"] --> Targets["Target positions + manifest"]
    Targets --> DryRun["IBKR dry run / reconciliation"]
    DryRun --> Gate["Guardrails and target integrity gates"]
    Gate --> Runner["Paper execution runner"]
    Runner --> Broker["IBKR paper account"]
    Broker --> StateDB["Local SQLite state"]
    MarketData["IBKR historical 15m bars"] --> PriceDB["Local market-data SQLite"]
    StateDB --> Monitor["Read-only monitor API/UI"]
    PriceDB --> Monitor
```

## 6. Trading Decision Flow

There are two decision systems. The first is the strategy engine that decides
target positions. The second is the IBKR adapter that decides whether the
targets may be sent to a paper account.

### 6.1 Strategy Engine Decisions

```mermaid
flowchart TD
    A["Price history and instrument metadata"]
    B{"Enough clean history?"}
    C["Generate raw rule forecasts"]
    D["Apply forecast scalar and cap"]
    E["Combine forecast rules"]
    F["Estimate volatility using prior data only"]
    G["Compute risk-adjusted position"]
    H["Apply instrument weights and IDM/FDM"]
    I["Apply buffer / reduce churn"]
    J["Round to integer contracts"]
    K["Write target file and manifest"]
    X["No target: instrument excluded or frozen"]

    A --> B
    B -- "no" --> X
    B -- "yes" --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
```

Important boundary: the engine can output positive, negative, or zero integer
targets, but it does not know about open IBKR orders, fills, current account
cash, or API connection state.

### 6.2 IBKR Adapter Decisions

```mermaid
flowchart TD
    Start["Paper daemon cycle"]
    Expected{"Expected account supplied?"}
    Api{"IBKR API connected?"}
    Paper{"Connected account is expected paper account?"}
    Target{"Target manifest current and verified?"}
    Universe{"Target universe matches production contract list?"}
    Contracts{"Contracts qualified?"}
    Data{"Market data gate passes?"}
    Positions{"Broker positions downloaded?"}
    Delta{"Target minus current position creates action?"}
    Margin{"Projected margin within guardrails?"}
    Flags{"Execution confirmation flags present?"}
    Submit["Submit paper market orders"]
    Save["Persist local state and monitor export"]
    Dry["Dry run / no transmission"]
    Block["Block and record gate reason"]

    Start --> Expected
    Expected -- "no" --> Block
    Expected -- "yes" --> Api
    Api -- "no" --> Block
    Api -- "yes" --> Paper
    Paper -- "no" --> Block
    Paper -- "yes" --> Target
    Target -- "no" --> Block
    Target -- "yes" --> Universe
    Universe -- "no" --> Block
    Universe -- "yes" --> Contracts
    Contracts -- "no" --> Block
    Contracts -- "yes" --> Data
    Data -- "no" --> Block
    Data -- "yes" --> Positions
    Positions -- "no" --> Block
    Positions -- "yes" --> Delta
    Delta -- "no" --> Save
    Delta -- "yes" --> Margin
    Margin -- "no" --> Block
    Margin -- "yes" --> Flags
    Flags -- "no" --> Dry
    Flags -- "yes" --> Submit
    Submit --> Save
    Dry --> Save
    Block --> Save
```

Adapter decisions are deliberately operational. They should not change the
forecast, the volatility target, or the portfolio construction logic.

### 6.3 Local State Outputs

```mermaid
flowchart LR
    IBKR["IBKR paper API"]
    Bars["15m historical bars"]
    Account["Account values / NAV"]
    Positions["Broker positions"]
    Orders["Orders and fills"]
    SQLite["Local SQLite"]
    JSON["Monitor JSON export"]
    UI["Read-only monitor"]

    IBKR --> Bars
    IBKR --> Account
    IBKR --> Positions
    IBKR --> Orders
    Bars --> SQLite
    Account --> SQLite
    Positions --> SQLite
    Orders --> SQLite
    SQLite --> JSON
    JSON --> UI
```

Runtime state is not a research input and is not committed to git.

## 7. Directory Policy

Committed:

- source code
- tests
- docs
- sanitized templates

Ignored/local-only:

- credentials
- raw data
- SQLite databases
- IBKR account/fill/order state
- generated reports
- generated backtests/research outputs
- screenshots
- cloned upstream repos
- node modules and build output

## 8. Operational Rule

If a file can reveal account identity, API keys, local machine paths, broker
orders, fills, NAV, or private runtime state, it should be local-only unless it
has been explicitly sanitized.
