# Polymarket BTC 5-Minute Bot

Dry-run-first Python bot for Polymarket BTC 5-minute Up/Down markets.

It finds the current BTC 5-minute market, streams live BTC prices, estimates the probability that the window resolves Up or Down, compares that with the Polymarket order book, and only trades when the edge clears every risk gate.

## Important

This is not financial advice and it does not guarantee profit. By default the bot runs in dry-run (paper) mode: it logs candidate trades, simulates fills against the live order book, and settles them against the official market resolution, but it never places orders. Stay in paper mode until you have reviewed the paper PnL in `trade_log.jsonl` and are comfortable with the risk.

## Setup

Requires Python 3.10+.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env
python btc_5m_bot.py
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.example.env .env
python btc_5m_bot.py
```

For dry-run you can leave the keys in `.env` blank. Invalid settings stop the bot at startup with a message naming the bad key.

Run the tests with:

```bash
python -m unittest discover -s tests
```

## Live Trading

Live trading needs two switches, so one typo cannot send real orders:

```env
POLYMARKET_DRY_RUN=false
LIVE_TRADING=true
POLYMARKET_PRIVATE_KEY=...
POLYMARKET_FUNDER=...
POLYMARKET_SIGNATURE_TYPE=1
```

The bot authenticates at startup, so bad credentials fail immediately rather than on the first trade. Orders are FOK market orders sent through `py-clob-client`, with a limit price: the order fills only if every share costs no more than the price that still clears `MIN_EDGE_AFTER_COSTS`. Each market gets at most one order attempt, even if the attempt errors. Your wallet must already be funded and have the required Polymarket allowances.

## Strategy

- **Market discovery**: looks up the current window directly by its deterministic slug (`btc-updown-5m-<window start>`), falling back to a Gamma scan.
- **Price feeds**: Binance BTC-USDT by default. Coinbase BTC-USD and OKX BTC-USDT are optional. When several feeds are fresh, the bot uses one source, preferring Coinbase, then Binance, then OKX, so the open and current prices always come from the same venue. Feeds that stop sending prices are reconnected. Stale feeds or a cross-exchange divergence disable new trades.
- **Window open**: taken from the chosen feed's price at the window start. If the bot joined mid-window and has no sample near the start, it skips that market instead of guessing.
- **Model**: driftless Brownian motion, `P(up) = Φ(ln(price/open) / (σ·√seconds_left))`, where σ is realized volatility of 1-second returns over the last 5 minutes. The result is clamped to [0.01, 0.99].
- **Pricing**: edge is checked against the best ask (`MIN_EDGE`) and against the average fill for the full stake plus `EXECUTION_COST` (`MIN_EDGE_AFTER_COSTS`). A book too thin for the stake means no trade.
- **Gates**: `MIN_SECONDS_LEFT` to `MAX_SECONDS_LEFT`, `MIN_MARKET_VOLUME`, `MIN_LIQUIDITY`, `MAX_SPREAD`, `MAX_BUY_PRICE`.
- **Risk limits**: `MAX_TRADES_PER_DAY`, `MAX_TRADES_PER_HOUR`, `MAX_OPEN_POSITIONS`, `MAX_DAILY_LOSS`, `MAX_CONSECUTIVE_LOSSES`, `KILL_SWITCH`. Daily counters reset at 00:00 UTC. They are held in memory, so a restart also resets them.
- **Settlement**: after each window closes, positions are settled from Gamma's official resolution. PnL feeds the loss limits and is logged.

## Files

- `btc_5m_bot.py`: entry point, price feeds, and the `TradingBot` loop.
- `polymarket_client.py`: Gamma discovery and resolution, CLOB order-book quotes, and order execution.
- `strategy.py`: probability model, signal generation, risk manager, paper ledger, and the JSONL trade log.
- `settings.py`: loads and validates configuration from the environment and `.env`.
- `config.example.env`: configuration template.
- `validate_edge.py`, `diagnose_network.py`, `diagnose_okx_websocket.py`: standalone diagnostics.

## Trade log

`TRADE_LOG_PATH` (default `trade_log.jsonl`) holds one JSON line per decision. It also records `order`, `order_error` and `resolution` events, including paper PnL.

## Known limitations

- Markets resolve on the Chainlink BTC/USD data stream, while the bot models exchange prices. Close to the open price, the two can disagree.
- Paper PnL does not model taker fees except through the `EXECUTION_COST` buffer used when deciding to trade.
- Risk counters and open positions are not persisted across restarts.
