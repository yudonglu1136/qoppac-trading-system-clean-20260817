# GitHub Hygiene Audit

Audit date: 2026-08-17

## Cleaned Before Upload

- Removed hard-coded paper account defaults from IBKR order scripts.
- Removed the tracked concrete LaunchAgent plist with local absolute paths.
- Replaced concrete LaunchAgent config with a sanitized `.plist.template`.
- Ignored concrete `config/*.plist` files.
- Expanded `.gitignore` for credentials, runtime DBs, broker state, reports,
  local research outputs, cloned upstream repos, screenshots, and caches.
- Removed local absolute screenshot paths from monitor design QA notes.
- Documented the separation between research/backtest, engine, IBKR adapter, and
  monitor paths.

## Files That Must Stay Local

- `env/`
- `.env*`
- `data/`
- `output/`
- `reports/`
- `research/`
- `notes/`
- `blog/`
- `github/`
- `*.sqlite`
- `*.db`
- `*.log`
- concrete LaunchAgent `.plist` files

## Sensitive Runtime State Not To Commit

- IBKR account identifiers
- broker positions
- open orders
- fills
- commissions
- NAV/account values
- target hashes generated from live target files
- API keys
- local machine absolute paths
- local screenshots of account state

## Publish Policy

For GitHub publication, prefer a private repository. If the repository has no
remote and contains previous local commits with sensitive runtime experiments,
publish a clean squashed branch rather than pushing the full local history.

## Remaining Review Notes

- Historical backtest, research, report, and PDF artifacts are treated as
  local-only generated evidence, not as engine code.
- IBKR paper execution code remains in the repo, but it is gated by explicit
  account identity and confirmation flags.
- Live execution is not enabled by any committed LaunchAgent template.
