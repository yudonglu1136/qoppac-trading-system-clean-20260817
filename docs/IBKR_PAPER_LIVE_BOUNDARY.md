# IBKR Paper/Live Boundary

The IBKR path is deliberately separate from the research/backtest path.

## Required Runtime Inputs

Set the expected paper account outside git:

```bash
export IBKR_EXPECTED_ACCOUNT="YOUR_PAPER_ACCOUNT"
```

The scripts also accept:

```bash
--expected-account YOUR_PAPER_ACCOUNT
```

Do not commit real account identifiers, tokens, API keys, `.env` files, local
SQLite databases, order logs, fill logs, or account snapshots.

## Paper Mode

Current paper assumptions:

- IB Gateway paper port: `4002`
- order runner refuses non-paper-looking account IDs
- order runner requires `--execute-orders`
- order runner requires `--confirm-paper-strategy-market-orders`
- smoke test requires `--confirm-paper-market-order`
- daemon requires `--confirm-paper-24h-daemon` before trading mode

The LaunchAgent template under `config/launchagents/` is intentionally safe by
default. It does not include trading flags. Add those only in the local installed
plist after manual review.

## Live Mode

Live trading is not just changing the port.

Before live mode exists, create a separate live adapter/config and review:

- account identity and permissions
- live market data subscriptions
- margin schedule
- contract mappings and roll rules
- close-out rules for physical delivery contracts
- kill switch and manual stop file
- maximum order size
- maximum margin/equity
- maximum daily loss
- reconciliation failure handling
- logging retention and backup

Do not reuse the paper LaunchAgent template for live execution.

## Execution Flow

```text
1. Generate or refresh research target positions.
2. Create/verify target manifest and hash.
3. Qualify IBKR contracts.
4. Fetch/update 15m historical bars for freshness.
5. Run order dry run and broker-position reconciliation.
6. Check guardrails.
7. Execute paper market orders only when every gate passes.
8. Persist orders, fills, account values, positions, NAV, and holdings locally.
9. Serve read-only monitor from local JSON/SQLite state.
```

## Important Separation

Backtest outputs answer:

```text
What would the strategy have done historically?
```

IBKR adapter outputs answer:

```text
Can these targets be safely reconciled and executed in this broker account now?
```

Never let IBKR account state feed back into model selection or historical
research results.

## Local One-Click Runtime

The local one-click runtime can live outside this repository. Its job is to:

- wait for a real IBKR API handshake
- clear stale client/process state
- start or restart services in the correct order
- refresh guardrails, market data, and monitor snapshots
- print data freshness and gate status

Keep that runtime machine-specific. Commit only sanitized templates and docs.
