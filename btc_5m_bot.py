from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import median
from time import time
from zoneinfo import ZoneInfo

import websockets

from polymarket_client import ClobMarketData, Executor, GammaClient, Market
from strategy import FeatureEngine, PaperTradeLedger, RiskConfig, TradeLogger, build_signal


DISPLAY_TZ = ZoneInfo("America/New_York")
COINBASE_SUBSCRIPTION = {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"}


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class SourcePrice:
    price: float
    timestamp: float
    received_at: float


@dataclass(frozen=True)
class BTCStateSnapshot:
    status: str
    price: float | None
    timestamp: float | None
    sources: tuple[str, ...]
    source_prices: dict[str, float]
    source_ages: dict[str, float]
    divergence_bps: float | None = None
    reason: str = ""


@dataclass
class PriceState:
    max_age_seconds: float = 10.0
    max_divergence_bps: float = 50.0
    coinbase_stale_after_seconds: float | None = None
    window_open: float | None = None
    prices: deque[float] | None = None
    history: deque[tuple[float, float]] | None = None
    latest: dict[str, SourcePrice] | None = None
    connections: dict[str, bool] | None = None
    feed_status: dict[str, str] | None = None
    last_update_timestamp: dict[str, float] | None = None
    _last_state_signature: tuple[str, ...] | None = None
    _last_status: str | None = None

    def __post_init__(self) -> None:
        if self.prices is None:
            self.prices = deque(maxlen=120)
        if self.history is None:
            self.history = deque(maxlen=5000)
        if self.latest is None:
            self.latest = {}
        if self.connections is None:
            self.connections = {}
        if self.feed_status is None:
            self.feed_status = {}
        if self.last_update_timestamp is None:
            self.last_update_timestamp = {}

    @property
    def current(self) -> float | None:
        return self.get_current_btc_state().price

    def update_source(self, source: str, price: float, timestamp: float, received_at: float | None = None) -> None:
        if not isfinite(price) or price <= 0 or not isfinite(timestamp):
            return
        received_at = time() if received_at is None else received_at
        current = self.latest.get(source) if self.latest is not None else None
        if current is not None and timestamp < current.timestamp:
            return
        self.latest[source] = SourcePrice(price, timestamp, received_at)
        self.last_update_timestamp[source] = received_at
        if source == "coinbase":
            self.set_feed_status(source, "HEALTHY")
        else:
            self.feed_status[source] = "HEALTHY"
        snapshot = self.get_current_btc_state(now=received_at)
        if snapshot.status == "OK" and self.prices is not None:
            self.prices.append(snapshot.price)
            self.history.append((received_at, snapshot.price))
            if self.window_open is None:
                self.window_open = snapshot.price

    def market_open_price(self, start_timestamp: float, fallback: float) -> float:
        matches = [
            item for item in (self.history or [])
            if start_timestamp <= item[0] <= start_timestamp + self.max_age_seconds
        ]
        if matches:
            return min(matches, key=lambda item: item[0])[1]
        print("[price] no websocket sample near market start; using current price as open fallback")
        return fallback

    def set_connected(self, source: str, connected: bool, detail: str = "") -> None:
        was_connected = self.connections.get(source, False)
        self.connections[source] = connected
        state = "connected" if connected else "disconnected"
        suffix = f": {detail}" if detail else ""
        if was_connected != connected:
            print(f"[price][{source}] {state}{suffix}")

    def set_feed_status(self, source: str, status: str) -> None:
        if self.feed_status.get(source) != status:
            self.feed_status[source] = status
            print(f"[price][{source}] {status}")

    def get_current_btc_state(self, now: float | None = None) -> BTCStateSnapshot:
        now = time() if now is None else now
        fresh: dict[str, SourcePrice] = {}
        for source, tick in (self.latest or {}).items():
            receive_age = now - tick.received_at
            max_age = (self.coinbase_stale_after_seconds or self.max_age_seconds) if source == "coinbase" else self.max_age_seconds
            if 0.0 <= receive_age <= max_age:
                fresh[source] = tick
            elif source == "coinbase" and self.last_update_timestamp.get(source) is not None and now - self.last_update_timestamp[source] > max_age:
                self.set_feed_status(source, "STALE")

        if not fresh:
            snapshot = BTCStateSnapshot(
                status="DATA_UNAVAILABLE",
                price=None,
                timestamp=None,
                sources=(),
                source_prices={},
                source_ages={},
                reason="all BTC price feeds are stale or unavailable",
            )
        else:
            values = [tick.price for tick in fresh.values()]
            center = median(values)
            divergence_bps = (max(values) - min(values)) / center * 10_000 if len(values) > 1 and center else 0.0
            if divergence_bps > self.max_divergence_bps:
                snapshot = BTCStateSnapshot(
                    status="DATA_UNAVAILABLE",
                    price=None,
                    timestamp=max(tick.timestamp for tick in fresh.values()),
                    sources=tuple(sorted(fresh)),
                    source_prices={source: tick.price for source, tick in fresh.items()},
                    source_ages={source: max(0.0, now - tick.timestamp) for source, tick in fresh.items()},
                    divergence_bps=divergence_bps,
                    reason=f"exchange price divergence {divergence_bps:.1f} bps exceeds limit",
                )
            else:
                primary = fresh.get("coinbase")
                if primary is None:
                    primary = fresh.get("binance")
                selected_price = primary.price if primary is not None else sum(values) / len(values)
                snapshot = BTCStateSnapshot(
                    status="OK",
                    price=selected_price,
                    timestamp=max(tick.timestamp for tick in fresh.values()),
                    sources=tuple(sorted(fresh)),
                    source_prices={source: tick.price for source, tick in fresh.items()},
                    source_ages={source: max(0.0, now - tick.timestamp) for source, tick in fresh.items()},
                    divergence_bps=divergence_bps,
                )

        if snapshot.status != self._last_status or snapshot.sources != self._last_state_signature:
            if snapshot.status == "OK":
                print(f"[price] active source(s): {', '.join(snapshot.sources)} price={snapshot.price:.2f}")
            else:
                print(f"[price] {snapshot.status}: {snapshot.reason}")
            self._last_status = snapshot.status
            self._last_state_signature = snapshot.sources
        return snapshot

    def reset_window_open(self, price: float) -> None:
        self.window_open = price


def parse_exchange_timestamp(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)):
        stamp = float(value)
        return stamp / 1000.0 if stamp > 10_000_000_000 else stamp
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                stamp = float(value)
                return stamp / 1000.0 if stamp > 10_000_000_000 else stamp
            except ValueError:
                pass
    return fallback


def parse_coinbase_tickers(message: dict, received_at: float) -> list[tuple[float, float]]:
    if message.get("channel") != "ticker":
        return []
    ticks: list[tuple[float, float]] = []
    message_timestamp = message.get("timestamp")
    for event in message.get("events", []):
        for ticker in event.get("tickers", []):
            if ticker.get("product_id") != "BTC-USD":
                continue
            try:
                raw_price = ticker.get("price") or ticker.get("last_trade_price") or ticker.get("last")
                price = float(raw_price)
            except (KeyError, TypeError, ValueError):
                continue
            timestamp = parse_exchange_timestamp(ticker.get("time") or message_timestamp, received_at)
            ticks.append((price, timestamp))
    return ticks


def parse_okx_tickers(message: dict, received_at: float) -> list[tuple[float, float]]:
    ticks: list[tuple[float, float]] = []
    for item in message.get("data", []):
        if item.get("instId") != "BTC-USDT":
            continue
        try:
            price = float(item["last"])
        except (KeyError, TypeError, ValueError):
            continue
        ticks.append((price, parse_exchange_timestamp(item.get("ts"), received_at)))
    return ticks


def parse_binance_trades(message: dict, received_at: float) -> list[tuple[float, float]]:
    payload = message.get("data", message)
    try:
        price = float(payload["p"])
    except (KeyError, TypeError, ValueError):
        return []
    return [(price, parse_exchange_timestamp(payload.get("T", payload.get("E")), received_at))]


async def run_price_feed(
    state: PriceState,
    *,
    source: str,
    url: str,
    subscription: dict | None,
    parser,
) -> None:
    min_backoff = float(os.environ.get("PRICE_RECONNECT_MIN_SECONDS", "1"))
    max_backoff = float(os.environ.get("PRICE_RECONNECT_MAX_SECONDS", "30"))
    backoff = min_backoff
    connected_once = False
    first_message_logged = False
    first_price_logged = False
    while True:
        try:
            print(f"[price][{source}] connecting to {url}...")
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, open_timeout=10) as ws:
                state.set_connected(source, True)
                if connected_once:
                    print(f"[price][{source}] reconnected")
                connected_once = True
                if subscription is not None:
                    await ws.send(json.dumps(subscription))
                async for message in ws:
                    if message == "ping":
                        await ws.send("pong")
                        continue
                    if not first_message_logged:
                        print(f"[price][{source}] first message: {message}")
                        first_message_logged = True
                    received_at = time()
                    try:
                        payload = json.loads(message)
                    except (json.JSONDecodeError, TypeError) as exc:
                        print(f"[price][{source}] JSON parse error: {type(exc).__name__}: {exc}; raw={message!r}")
                        continue
                    try:
                        ticks = parser(payload, received_at)
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        print(f"[price][{source}] feed parse error: {type(exc).__name__}: {exc}; payload={payload!r}")
                        continue
                    if not ticks and not first_price_logged:
                        print(f"[price][{source}] received message but no valid BTC ticker/trade was extracted")
                    for price, timestamp in ticks:
                        state.update_source(source, price, timestamp, received_at)
                        if not first_price_logged:
                            print(f"[price][{source}] first valid BTC price: {price} timestamp={timestamp}")
                            first_price_logged = True
                        backoff = min_backoff
        except Exception as exc:
            state.set_connected(source, False, str(exc))
            print(f"[price][{source}] reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, max(min_backoff, backoff * 2))


async def stream_coinbase_prices(state: PriceState) -> None:
    url = os.environ.get("COINBASE_WS_URL", "wss://advanced-trade-ws.coinbase.com")
    subscription = COINBASE_SUBSCRIPTION
    stale_after = float(os.environ.get("COINBASE_STALE_AFTER_SECONDS", str(state.coinbase_stale_after_seconds or state.max_age_seconds)))
    min_backoff = float(os.environ.get("PRICE_RECONNECT_MIN_SECONDS", "1"))
    max_backoff = float(os.environ.get("PRICE_RECONNECT_MAX_SECONDS", "30"))
    backoff = min_backoff
    first_raw_logged = False
    first_price_logged = False

    while True:
        received_valid_data = False
        try:
            print(f"[price][coinbase] connecting to {url}...")
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, open_timeout=10) as ws:
                state.connections["coinbase"] = True
                state.set_feed_status("coinbase", "CONNECTED")
                print("[price][coinbase] websocket connected")
                subscription_json = json.dumps(subscription, separators=(",", ":"))
                print(f"[price][coinbase] sending subscription: {subscription_json}")
                await ws.send(subscription_json)
                state.set_feed_status("coinbase", "SUBSCRIBED")

                while True:
                    try:
                        raw_message = await asyncio.wait_for(ws.recv(), timeout=min(1.0, max(stale_after, 0.1)))
                    except asyncio.TimeoutError:
                        last_update = state.last_update_timestamp.get("coinbase")
                        if last_update is not None and time() - last_update > stale_after:
                            state.set_feed_status("coinbase", "STALE")
                        continue

                    if not first_raw_logged:
                        print(f"[price][coinbase] first message: {raw_message}")
                        first_raw_logged = True

                    try:
                        payload = json.loads(raw_message)
                    except (json.JSONDecodeError, TypeError) as exc:
                        print(f"[price][coinbase] message JSON parse error: {type(exc).__name__}: {exc}")
                        continue

                    if not isinstance(payload, dict):
                        print(f"[price][coinbase] unexpected message shape: {type(payload).__name__}")
                        continue

                    try:
                        ticks = parse_coinbase_tickers(payload, time())
                    except (AttributeError, TypeError, ValueError, KeyError) as exc:
                        print(f"[price][coinbase] ticker parse error: {type(exc).__name__}: {exc}")
                        continue

                    if not ticks:
                        if payload.get("channel") == "ticker":
                            print("[price][coinbase] ticker message contained no valid BTC-USD price")
                        continue

                    received_at = time()
                    for price, timestamp in ticks:
                        state.update_source("coinbase", price, timestamp, received_at)
                        if not first_price_logged:
                            print(f"[price][coinbase] first BTC price received: {price} timestamp={timestamp}")
                            first_price_logged = True
                    state.connections["coinbase"] = True
                    received_valid_data = True
                    backoff = min_backoff
        except Exception as exc:
            state.connections["coinbase"] = False
            state.set_feed_status("coinbase", "DISCONNECTED")
            close_code = getattr(exc, "code", None)
            close_reason = getattr(exc, "reason", None)
            print(f"[price][coinbase] exception type: {type(exc).__name__}")
            print(f"[price][coinbase] exception message: {exc}")
            print(f"[price][coinbase] WebSocket close code: {close_code}")
            print(f"[price][coinbase] WebSocket close reason: {close_reason}")
            print(f"[price][coinbase] reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            if received_valid_data:
                backoff = min_backoff
            else:
                backoff = min(max_backoff, max(min_backoff, backoff * 2))


async def stream_okx_prices(state: PriceState) -> None:
    await run_price_feed(
        state,
        source="okx",
        url=os.environ.get("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public"),
        subscription={"op": "subscribe", "args": [{"channel": "tickers", "instId": "BTC-USDT"}]},
        parser=parse_okx_tickers,
    )


async def stream_binance_prices(state: PriceState) -> None:
    base = os.environ.get("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws").rstrip("/")
    symbol = os.environ.get("BTC_SYMBOL", "btcusdt").lower()
    await run_price_feed(
        state,
        source="binance",
        url=f"{base}/{symbol}@trade",
        subscription=None,
        parser=parse_binance_trades,
    )


async def trade_loop(state: PriceState) -> None:
    gamma = GammaClient(os.environ.get("GAMMA_URL", "https://gamma-api.polymarket.com"))
    clob = ClobMarketData(os.environ.get("CLOB_URL", "https://clob.polymarket.com"))
    dry_run = os.environ.get("POLYMARKET_DRY_RUN", "true").lower() != "false"
    executor = Executor(os.environ.get("CLOB_URL", "https://clob.polymarket.com"), dry_run=dry_run)
    logger = TradeLogger(os.environ.get("TRADE_LOG_PATH", "trade_log.jsonl"))
    paper_trades = PaperTradeLedger()
    feature_engine = FeatureEngine()
    risk = RiskConfig(
        trade_usdc=float(os.environ.get("TRADE_USDC", "5")),
        max_trades_per_day=int(os.environ.get("MAX_TRADES_PER_DAY", "20")),
        min_edge=float(os.environ.get("MIN_EDGE", "0.07")),
        min_seconds_left=float(os.environ.get("MIN_SECONDS_LEFT", "8")),
        max_seconds_left=float(os.environ.get("MAX_SECONDS_LEFT", "45")),
        min_market_volume=float(os.environ.get("MIN_MARKET_VOLUME", "1000")),
        max_buy_price=float(os.environ.get("MAX_BUY_PRICE", "0.92")),
        min_price_history_seconds=float(os.environ.get("MIN_PRICE_HISTORY_SECONDS", "60")),
    )
    poll_seconds = float(os.environ.get("POLL_SECONDS", "2"))
    trades_today = 0
    traded_market_ids: set[str] = set()
    active_market_id: str | None = None

    print(f"[startup] running with dry_run={dry_run}, trade_usdc={risk.trade_usdc}")

    while True:
        try:
            markets = await asyncio.to_thread(gamma.find_btc_5m_markets)
            market = choose_current_market(markets)
            if not market:
                print("[scan] no active BTC 5-minute market found")
                await asyncio.sleep(max(poll_seconds, 5))
                continue

            btc_state = state.get_current_btc_state()
            if btc_state.status != "OK" or btc_state.price is None:
                print(f"[decision] {btc_state.status}: {btc_state.reason}; new trades disabled")
                await asyncio.sleep(poll_seconds)
                continue

            if market.id != active_market_id:
                active_market_id = market.id
                state.reset_window_open(state.market_open_price(market.start_ts or market.end_ts - 300, btc_state.price))
                print(
                    f"[market] {market.question} | volume={market.volume:.0f} "
                    f"liquidity={market.liquidity:.0f} "
                    f"window_open={state.window_open:.2f}"
                )

            seconds_left = market.end_ts - time()
            if seconds_left <= 0:
                await asyncio.sleep(poll_seconds)
                continue

            up_ask, down_ask = await asyncio.gather(
                asyncio.to_thread(clob.buy_price, market.up.token_id),
                asyncio.to_thread(clob.buy_price, market.down.token_id),
            )
            activity_measure = max(market.volume, market.liquidity)
            recent_samples = list(state.history or [])
            current_sample_timestamp = recent_samples[-1][0] if recent_samples else None

            signal = build_signal(
                open_price=state.window_open or btc_state.price,
                current_price=btc_state.price,
                seconds_left=seconds_left,
                recent_prices=list(state.prices or []),
                recent_samples=recent_samples,
                current_timestamp=current_sample_timestamp,
                up_token_id=market.up.token_id,
                down_token_id=market.down.token_id,
                up_ask=up_ask,
                down_ask=down_ask,
                market_volume=activity_measure,
                trades_today=trades_today,
                risk=risk,
            )
            features = feature_engine.compute_features(
                open_price=state.window_open or btc_state.price,
                current_price=btc_state.price,
                recent_prices=list(state.prices or []),
                recent_samples=recent_samples,
                current_timestamp=current_sample_timestamp,
                seconds_left=seconds_left,
                market_volume=activity_measure,
                up_ask=up_ask,
                down_ask=down_ask,
            )

            logger.record_decision(
                timestamp=datetime.now(DISPLAY_TZ).isoformat(),
                market_id=market.id,
                time_remaining=seconds_left,
                btc_price=btc_state.price,
                btc_open_price=state.window_open or 0.0,
                btc_return_30s=features["btc_30s_return"],
                btc_return_1m=features["btc_1m_return"],
                btc_return_5m=features["btc_5m_return"],
                volume_features={
                    "market_volume": market.volume,
                    "market_liquidity": market.liquidity,
                    "activity_measure": activity_measure,
                    "volume_feature": features["volume_feature"],
                },
                order_flow_features={
                    "order_flow": features["order_flow"],
                    "volume_acceleration": features["volume_acceleration"],
                },
                polymarket_bid=0.0,
                polymarket_ask=signal.ask_price,
                model_probability_up=signal.model_probability if signal.outcome == "UP" else 1.0 - signal.model_probability,
                model_probability_down=1.0 - signal.model_probability if signal.outcome == "UP" else signal.model_probability,
                estimated_edge=signal.edge,
                trade_decision="TRADE" if signal.should_trade else "NO_TRADE",
                decision_reason=signal.reason,
                position_size=risk.trade_usdc if signal.should_trade else 0.0,
            )

            stamp = datetime.now(DISPLAY_TZ).strftime("%H:%M:%S %Z")
            print(
                f"[{stamp}] left={seconds_left:5.1f}s btc={btc_state.price:.2f} "
                f"sources={','.join(btc_state.sources)} "
                f"up_ask={fmt(up_ask)} down_ask={fmt(down_ask)} "
                f"pick={signal.outcome} p={signal.model_probability:.3f} "
                f"edge={signal.edge:.3f} {signal.reason}"
            )

            if signal.should_trade and market.id not in traded_market_ids:
                if dry_run:
                    paper_trades.enter(
                        market_id=market.id,
                        outcome=signal.outcome,
                        entry_price=signal.ask_price,
                        stake=risk.trade_usdc,
                        model_probability=signal.model_probability,
                        edge=signal.edge,
                    )
                response = await asyncio.to_thread(executor.buy_market, signal.token_id, risk.trade_usdc)
                trades_today += 1
                traded_market_ids.add(market.id)
                print(f"[trade] {signal.outcome} {risk.trade_usdc} USDC response={response}")

        except Exception as exc:
            print(f"[loop] error: {exc}")

        await asyncio.sleep(poll_seconds)


def choose_current_market(markets: list[Market]) -> Market | None:
    now = time()
    running = [
        market
        for market in markets
        if (market.start_ts is None or market.start_ts <= now) and market.end_ts > now
    ]
    if running:
        return min(running, key=lambda market: market.end_ts)

    upcoming = [market for market in markets if market.end_ts > now]
    return min(upcoming, key=lambda market: market.start_ts or market.end_ts, default=None)


def fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


async def main() -> None:
    load_dotenv()
    state = PriceState(
        max_age_seconds=float(os.environ.get("PRICE_MAX_AGE_SECONDS", "10")),
        max_divergence_bps=float(os.environ.get("PRICE_MAX_DIVERGENCE_BPS", "50")),
        coinbase_stale_after_seconds=float(os.environ["COINBASE_STALE_AFTER_SECONDS"]) if "COINBASE_STALE_AFTER_SECONDS" in os.environ else None,
    )
    feeds = []
    if os.environ.get("ENABLE_COINBASE_FEED", "false").lower() == "true":
        feeds.append(stream_coinbase_prices(state))
    if os.environ.get("ENABLE_OKX_FALLBACK", "false").lower() == "true":
        feeds.append(stream_okx_prices(state))
    if os.environ.get("ENABLE_BINANCE_FALLBACK", "true").lower() == "true":
        feeds.append(stream_binance_prices(state))
    await asyncio.gather(*feeds, trade_loop(state))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[shutdown] stopped")
