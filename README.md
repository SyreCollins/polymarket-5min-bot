# Polymarket BTC 5-Minute Bot

Dry-run-first Python bot for Polymarket BTC 5-minute Up/Down markets.

It follows the shape from the linked article: discover active BTC 5-minute markets, stream live BTC ticks, estimate the probability that the window resolves Up or Down, compare that to Polymarket CLOB prices, and only trade when the edge clears configurable risk gates.

## Important

This is not financial advice and it does not guarantee profit. The default mode is `POLYMARKET_DRY_RUN=true`, which prints candidate trades without placing orders. Keep it that way until you have watched logs, checked fills manually, and are comfortable with the risk.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.example.env .env
```

Edit `.env`. For dry-run you can leave keys blank.

```powershell
python btc_5m_bot.py
```

## Live Trading

Set these only when you are ready:

```env
POLYMARKET_DRY_RUN=false
POLYMARKET_PRIVATE_KEY=...
POLYMARKET_FUNDER=...
POLYMARKET_SIGNATURE_TYPE=1
```

The bot uses `py-clob-client` for signing and posting FOK market orders. Your wallet must already be funded and have the required Polymarket allowances.

## Strategy

- Uses persistent Coinbase Advanced Trade BTC-USD and OKX BTC-USDT WebSocket feeds, with optional Binance BTC-USDT fallback.
- Rejects stale feeds, detects cross-exchange divergence, and disables new trades when no valid consolidated BTC price is available.
- Uses timestamped WebSocket price history for each market's opening reference; no REST polling is used for live BTC prices.
- Finds active Polymarket BTC 5-minute Up/Down markets from Gamma.
- Reads CLOB ask prices for each outcome token.
- Estimates final probability with a simple short-horizon Brownian model using realized tick volatility.
- Trades only inside `MIN_SECONDS_LEFT` to `MAX_SECONDS_LEFT`.
- Skips markets below `MIN_MARKET_VOLUME`.
- Caps buy price with `MAX_BUY_PRICE`.
- Caps frequency with `MAX_TRADES_PER_DAY`.

## Files

- `btc_5m_bot.py` - entry point and event loop.
- `polymarket_client.py` - Gamma discovery, CLOB pricing, optional execution.
- `strategy.py` - probability model, signal generation, risk settings.
- `config.example.env` - configuration template.

## Notes

The hardest part of these markets is not writing an order call; it is getting the correct active market, correct outcome token, correct time remaining, and avoiding stale assumptions. This bot errs on skipping trades when data is ambiguous.
