from __future__ import annotations

import asyncio
import json
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from polymarket_client import GammaClient, Market


NETWORK_CONTEXT = ssl._create_unverified_context()
GAMMA_URL = "https://gamma-api.polymarket.com"
DISPLAY_TZ = ZoneInfo("America/New_York")


def fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=15, context=NETWORK_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_spot_price() -> tuple[float | None, str | None]:
    sources = [
        ("Binance", "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"),
        ("Coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot"),
    ]
    for name, url in sources:
        try:
            data = fetch_json(url)
            if isinstance(data, dict):
                value = data["price"] if "price" in data else data["data"]["amount"]
                return float(value), name
        except Exception:
            continue
    return None, None


def fetch_binance_1m_open(ts_seconds: float) -> float | None:
    params = urllib.parse.urlencode(
        {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": str(int(ts_seconds * 1000)),
            "limit": "1",
        }
    )
    try:
        data = fetch_json(f"https://api.binance.com/api/v3/klines?{params}")
        if isinstance(data, list) and data:
            return float(data[0][1])
    except Exception:
        return None
    return None


def fetch_market(market_id: str) -> dict:
    data = fetch_json(f"{GAMMA_URL}/markets/{market_id}")
    return data if isinstance(data, dict) else {}


def official_winner(final_market: dict) -> str | None:
    for key in ("outcome", "resolvedOutcome", "winningOutcome"):
        value = final_market.get(key)
        if value in ("0", 0, "Up", "UP", "Yes", "YES"):
            return "UP"
        if value in ("1", 1, "Down", "DOWN", "No", "NO"):
            return "DOWN"

    outcomes = final_market.get("outcomes")
    prices = final_market.get("outcomePrices")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = None
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            prices = None
    if isinstance(outcomes, list) and isinstance(prices, list):
        numeric = [float(price) for price in prices]
        if numeric:
            return str(outcomes[numeric.index(max(numeric))]).upper()
    return None


def choose_current_market(markets: list[Market]) -> Market | None:
    now = datetime.now(timezone.utc).timestamp()
    future = [market for market in markets if market.end_ts > now]
    return min(future, key=lambda market: market.end_ts, default=None)


def print_candidates(markets: list[Market]) -> None:
    now = datetime.now(timezone.utc).timestamp()
    print("No strict active BTC 5m market found. Nearby BTC candidates:")
    for market in markets[:10]:
        seconds_left = market.end_ts - now
        end = datetime.fromtimestamp(market.end_ts, tz=DISPLAY_TZ).strftime("%H:%M:%S %Z")
        print(
            f"  end={end} left={seconds_left:6.1f}s "
            f"slug={market.slug or market.event_slug} q={market.question}"
        )


async def audit_edge() -> None:
    print("=== Polymarket BTC 5m Edge Validator ===")
    print("This validates market discovery and Binance proxy agreement. It does not fetch Chainlink Data Streams.")

    gamma = GammaClient(GAMMA_URL, ssl_context=NETWORK_CONTEXT)
    audited_markets: set[str] = set()

    while True:
        try:
            markets = gamma.find_btc_5m_markets(limit=500)
            active_market = choose_current_market(markets)

            if not active_market:
                print_candidates(gamma.find_btc_5m_candidates(limit=500))
                await asyncio.sleep(15)
                continue

            market_key = active_market.id or active_market.slug
            if market_key in audited_markets:
                await asyncio.sleep(5)
                continue

            now_ts = datetime.now(timezone.utc).timestamp()
            start_ts = active_market.start_ts or active_market.end_ts - 300
            wait_time = max(0, active_market.end_ts - now_ts + 15)

            print(f"\n--- MATCH FOUND: {active_market.question} ---")
            print(f"Slug: {active_market.slug or active_market.event_slug}")
            print(f"Start: {datetime.fromtimestamp(start_ts, tz=DISPLAY_TZ).isoformat()}")
            print(f"End:   {datetime.fromtimestamp(active_market.end_ts, tz=DISPLAY_TZ).isoformat()}")

            binance_open = fetch_binance_1m_open(start_ts)
            spot, spot_source = fetch_spot_price()
            print(f"Binance 1m open near start: {format_price(binance_open)}")
            print(f"Current spot proxy: {format_price(spot)} from {spot_source or 'none'}")
            print(f"Waiting {int(wait_time)}s for window to close...")
            await asyncio.sleep(wait_time)

            final_market = fetch_market(active_market.id)
            winner = official_winner(final_market)
            print(f"[RESULT] Gamma winner field: {winner or 'not found yet'}")

            binance_end = fetch_binance_1m_open(active_market.end_ts)
            if binance_open and binance_end:
                binance_winner = "UP" if binance_end >= binance_open else "DOWN"
                print(f"Binance end 1m open: {format_price(binance_end)}")
                print(f"Binance proxy winner: {binance_winner}")
                if winner:
                    print("MATCH" if winner == binance_winner else "DISCREPANCY / possible ghost edge")

            audited_markets.add(market_key)
            await asyncio.sleep(5)

        except Exception as exc:
            print(f"Diagnostic loop error: {exc}")
            await asyncio.sleep(15)


def format_price(value: float | None) -> str:
    return "not found" if value is None else f"${value:,.2f}"


if __name__ == "__main__":
    try:
        asyncio.run(audit_edge())
    except KeyboardInterrupt:
        print("\nStopped.")
