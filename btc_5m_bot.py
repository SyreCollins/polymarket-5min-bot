from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite, log as ln, sqrt
from statistics import median
from time import monotonic, time
from typing import Callable
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import websockets
from dotenv import load_dotenv

from polymarket_client import WINDOW_SECONDS, BookQuote, ClobMarketData, Executor, GammaClient, Market
from settings import ConfigError, Settings
from strategy import UNFILLABLE, DecisionLogger, FeatureEngine, PaperTradeLedger, RiskManager, Signal, TradeLogger, build_signal, per_second_volatility, probability_up


log = logging.getLogger("btc_5m_bot")

DISPLAY_TZ = ZoneInfo("America/New_York")
COINBASE_SUBSCRIPTION = {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"}
# Coinbase closes idle connections; heartbeats keep it open between ticker updates.
COINBASE_HEARTBEAT_SUBSCRIPTION = {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "heartbeats"}
# Coinbase BTC-USD tracks the USD-denominated resolution source most closely; USDT venues carry a basis.
SOURCE_PREFERENCE = ("coinbase", "binance", "okx")
HISTORY_RETENTION_SECONDS = 900.0
RESOLUTION_GRACE_SECONDS = 15.0
RESOLUTION_RETRY_SECONDS = 10.0
RESOLUTION_SLOW_RETRY_SECONDS = 60.0
GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
GEOBLOCK_INTERVAL_SECONDS = 2 * 60 * 60
TELEGRAM_POLL_TIMEOUT_SECONDS = 2
TELEGRAM_REQUEST_TIMEOUT_SECONDS = 5
TELEGRAM_HEARTBEAT_SECONDS = 60 * 60


def check_geoblock() -> tuple[bool, str]:
    request = urllib.request.Request(GEOBLOCK_URL, headers={
        "accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    })
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("blocked"), bool):
        raise ValueError("invalid geoblock response")
    country = str(payload.get("country") or "unknown")
    return payload["blocked"], country


def send_telegram_alert(message: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.error("[telegram] alert not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            result = json.loads(response.read().decode("utf-8"))
        return isinstance(result, dict) and result.get("ok") is True
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        log.warning("[telegram] alert delivery failed: HTTP %s - %s", exc.code, error_body)
        return False
    except Exception as exc:
        log.warning("[telegram] alert delivery failed (%s): %s", type(exc).__name__, exc)
        return False


def fetch_telegram_updates(offset: int) -> list[dict] | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None
    query = urllib.parse.urlencode({
        "offset": offset,
        "timeout": TELEGRAM_POLL_TIMEOUT_SECONDS,
        "allowed_updates": json.dumps(["message"]),
    })
    url = f"https://api.telegram.org/bot{token}/getUpdates?{query}"
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TELEGRAM_REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload.get("result") if isinstance(payload, dict) and payload.get("ok") is True else None
        return result if isinstance(result, list) else None
    except Exception as exc:
        log.warning("[telegram] command polling failed (%s)", type(exc).__name__)
        return None


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
    primary_source: str | None = None


TWAP_WINDOW_SECONDS = 60.0


@dataclass
class PriceState:
    max_age_seconds: float = 10.0
    max_divergence_bps: float = 50.0
    coinbase_stale_after_seconds: float | None = None
    history_retention_seconds: float = HISTORY_RETENTION_SECONDS
    # (received_at, price) samples; time-bounded so a busy feed cannot evict the window open.
    history: deque[tuple[float, float]] = field(default_factory=deque)
    source_history: dict[str, deque[tuple[float, float]]] = field(default_factory=dict)
    latest: dict[str, SourcePrice] = field(default_factory=dict)
    connections: dict[str, bool] = field(default_factory=dict)
    feed_status: dict[str, str] = field(default_factory=dict)
    last_update_timestamp: dict[str, float] = field(default_factory=dict)
    _last_state_signature: tuple[str, ...] | None = field(default=None, repr=False)
    _last_status: str | None = field(default=None, repr=False)
    # 60-second TWAP tracking for primary source (used for calibration/probability model)
    twap_samples: deque[tuple[float, float]] = field(default_factory=deque)  # (timestamp, price)
    _twap_primary_source: str | None = field(default=None, repr=False)

    @property
    def current(self) -> float | None:
        return self.get_current_btc_state().price

    def update_source(self, source: str, price: float, timestamp: float, received_at: float | None = None) -> None:
        if not isfinite(price) or price <= 0 or not isfinite(timestamp):
            return
        received_at = time() if received_at is None else received_at
        current = self.latest.get(source)
        if current is not None and timestamp < current.timestamp:
            return
        self.latest[source] = SourcePrice(price, timestamp, received_at)
        self.last_update_timestamp[source] = received_at
        self.set_feed_status(source, "HEALTHY")
        self._append(self.source_history.setdefault(source, deque()), received_at, price)
        snapshot = self.get_current_btc_state(now=received_at)
        if snapshot.status == "OK" and snapshot.price is not None:
            self._append(self.history, received_at, snapshot.price)
            # Update TWAP samples if this is the primary source
            if snapshot.primary_source == source:
                self._update_twap(source, snapshot.price, received_at)

    def _append(self, history: deque[tuple[float, float]], timestamp: float, price: float) -> None:
        history.append((timestamp, price))
        cutoff = timestamp - self.history_retention_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def _update_twap(self, source: str, price: float, received_at: float) -> None:
        """Update 60-second TWAP samples for the primary source."""
        if self._twap_primary_source is None:
            self._twap_primary_source = source
        if source != self._twap_primary_source:
            return
        # Append new sample
        self.twap_samples.append((received_at, price))
        # Remove samples older than TWAP_WINDOW_SECONDS
        cutoff = received_at - TWAP_WINDOW_SECONDS
        while self.twap_samples and self.twap_samples[0][0] < cutoff:
            self.twap_samples.popleft()

    def get_twap(self) -> float | None:
        """Return current 60-second TWAP for primary source, or None if insufficient data."""
        if len(self.twap_samples) < 2:
            return None
        total_weight = 0.0
        weighted_sum = 0.0
        # Time-weighted average: sum(price_i * dt_i) / sum(dt_i)
        for i in range(1, len(self.twap_samples)):
            dt = self.twap_samples[i][0] - self.twap_samples[i-1][0]
            if dt > 0:
                total_weight += dt
                weighted_sum += self.twap_samples[i][1] * dt
        if total_weight <= 0:
            return None
        return weighted_sum / total_weight

    def get_twap_at(self, timestamp: float) -> float | None:
        """Return 60-second TWAP ending at the given timestamp, using historical samples.
        Returns None if insufficient data in [timestamp-60, timestamp]."""
        if not self.history:
            return None
        window_start = timestamp - TWAP_WINDOW_SECONDS
        window_end = timestamp
        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(1, len(self.history)):
            ts, price = self.history[i]
            if ts > window_end:
                break
            if ts < window_start:
                continue
            dt = ts - self.history[i-1][0]
            if dt > 0:
                total_weight += dt
                weighted_sum += price * dt
        if total_weight <= 0:
            return None
        return weighted_sum / total_weight

    def samples(self, source: str | None = None) -> list[tuple[float, float]]:
        return list(self.history if source is None else self.source_history.get(source, ()))

    def market_open_price(self, start_timestamp: float, source: str | None = None) -> float | None:
        """Price at the window start, or None if no sample lies within max_age_seconds of it."""
        before: tuple[float, float] | None = None
        after: tuple[float, float] | None = None
        for sample in self.samples(source):
            if sample[0] <= start_timestamp:
                before = sample
            else:
                after = sample
                break
        candidates = [
            sample for sample in (before, after)
            if sample is not None and abs(sample[0] - start_timestamp) <= self.max_age_seconds
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda sample: abs(sample[0] - start_timestamp))[1]

    def market_open_twap(self, start_timestamp: float) -> float | None:
        """60-second TWAP at the window start (ending at start_timestamp).
        Returns the TWAP for [start_timestamp-60, start_timestamp], or None if insufficient data."""
        return self.get_twap_at(start_timestamp)

    def set_connected(self, source: str, connected: bool, detail: str = "") -> None:
        was_connected = self.connections.get(source, False)
        self.connections[source] = connected
        if was_connected != connected:
            suffix = f": {detail}" if detail else ""
            log.info("[price][%s] %s%s", source, "connected" if connected else "disconnected", suffix)

    def set_feed_status(self, source: str, status: str) -> None:
        if self.feed_status.get(source) != status:
            self.feed_status[source] = status
            log.info("[price][%s] %s", source, status)

    def _max_age(self, source: str) -> float:
        if source == "coinbase" and self.coinbase_stale_after_seconds:
            return self.coinbase_stale_after_seconds
        return self.max_age_seconds

    def get_current_btc_state(self, now: float | None = None) -> BTCStateSnapshot:
        now = time() if now is None else now
        fresh: dict[str, SourcePrice] = {}
        for source, tick in self.latest.items():
            receive_age = now - tick.received_at
            if 0.0 <= receive_age <= self._max_age(source):
                fresh[source] = tick
            elif receive_age > self._max_age(source) and self.feed_status.get(source) == "HEALTHY":
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
            common = {
                "timestamp": max(tick.timestamp for tick in fresh.values()),
                "sources": tuple(sorted(fresh)),
                "source_prices": {source: tick.price for source, tick in fresh.items()},
                "source_ages": {source: max(0.0, now - tick.timestamp) for source, tick in fresh.items()},
                "divergence_bps": divergence_bps,
            }
            if divergence_bps > self.max_divergence_bps:
                snapshot = BTCStateSnapshot(
                    status="DATA_UNAVAILABLE",
                    price=None,
                    reason=f"exchange price divergence {divergence_bps:.1f} bps exceeds limit",
                    **common,
                )
            else:
                # Use one source rather than an average so open and current prices stay comparable.
                primary = next((source for source in SOURCE_PREFERENCE if source in fresh), sorted(fresh)[0])
                snapshot = BTCStateSnapshot(status="OK", price=fresh[primary].price, primary_source=primary, **common)

        if snapshot.status != self._last_status or snapshot.sources != self._last_state_signature:
            if snapshot.status == "OK":
                log.info("[price] active source(s): %s price=%.2f", ", ".join(snapshot.sources), snapshot.price)
            else:
                log.warning("[price] %s: %s", snapshot.status, snapshot.reason)
            self._last_status = snapshot.status
            self._last_state_signature = snapshot.sources
        return snapshot


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


@dataclass(frozen=True)
class FeedSpec:
    source: str
    url: str
    subscriptions: tuple[dict, ...]
    parser: Callable[[dict, float], list[tuple[float, float]]]


def build_feeds(settings: Settings) -> list[FeedSpec]:
    feeds: list[FeedSpec] = []
    if settings.enable_coinbase:
        feeds.append(FeedSpec(
            "coinbase",
            settings.coinbase_ws_url,
            (COINBASE_SUBSCRIPTION, COINBASE_HEARTBEAT_SUBSCRIPTION),
            parse_coinbase_tickers,
        ))
    if settings.enable_okx:
        feeds.append(FeedSpec(
            "okx",
            settings.okx_ws_url,
            ({"op": "subscribe", "args": [{"channel": "tickers", "instId": "BTC-USDT"}]},),
            parse_okx_tickers,
        ))
    if settings.enable_binance:
        feeds.append(FeedSpec(
            "binance",
            f"{settings.binance_ws_url.rstrip('/')}/{settings.btc_symbol}@trade",
            (),
            parse_binance_trades,
        ))
    return feeds


async def run_price_feed(
    state: PriceState,
    feed: FeedSpec,
    *,
    min_backoff: float = 1.0,
    max_backoff: float = 30.0,
    silence_timeout: float = 30.0,
) -> None:
    source = feed.source
    backoff = min_backoff
    first_price_logged = False
    while True:
        try:
            log.info("[price][%s] connecting to %s", source, feed.url)
            async with websockets.connect(feed.url, ping_interval=20, ping_timeout=20, open_timeout=10) as ws:
                state.set_connected(source, True)
                state.set_feed_status(source, "CONNECTED")
                for subscription in feed.subscriptions:
                    await ws.send(json.dumps(subscription, separators=(",", ":")))
                if feed.subscriptions:
                    state.set_feed_status(source, "SUBSCRIBED")

                last_tick_at = time()
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=min(5.0, silence_timeout))
                    except asyncio.TimeoutError:
                        message = None
                    # An open socket that stops delivering prices is as bad as a closed one.
                    if time() - last_tick_at > silence_timeout:
                        raise ConnectionError(f"no BTC prices for {silence_timeout:.0f}s")
                    if message is None:
                        continue
                    if message == "ping":
                        await ws.send("pong")
                        continue

                    received_at = time()
                    try:
                        payload = json.loads(message)
                    except (json.JSONDecodeError, TypeError) as exc:
                        log.debug("[price][%s] JSON parse error: %s; raw=%r", source, exc, message)
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("event") == "error" or payload.get("type") == "error":
                        log.error("[price][%s] feed error: %s", source, payload)
                        continue
                    try:
                        ticks = feed.parser(payload, received_at)
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        log.warning("[price][%s] parse error: %s: %s", source, type(exc).__name__, exc)
                        continue

                    for price, timestamp in ticks:
                        state.update_source(source, price, timestamp, received_at)
                    if ticks:
                        last_tick_at = received_at
                        backoff = min_backoff
                        if not first_price_logged:
                            log.info("[price][%s] first BTC price: %.2f", source, ticks[-1][0])
                            first_price_logged = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.set_connected(source, False)
            state.set_feed_status(source, "DISCONNECTED")
            log.warning("[price][%s] %s: %s; reconnecting in %.1fs", source, type(exc).__name__, exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, max(min_backoff, backoff * 2))


def choose_current_market(markets: list[Market], now: float | None = None) -> Market | None:
    now = time() if now is None else now
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


def fill_price(quote: BookQuote | None) -> float | None:
    if quote is None or quote.best_ask is None:
        return None
    return quote.avg_fill_price if quote.avg_fill_price is not None else UNFILLABLE


class TradingBot:
    def __init__(
        self,
        settings: Settings,
        state: PriceState,
        *,
        gamma: GammaClient | None = None,
        clob: ClobMarketData | None = None,
        executor: Executor | None = None,
        logger: TradeLogger | None = None,
        clock: Callable[[], float] = time,
    ) -> None:
        self.settings = settings
        self.state = state
        self.clock = clock
        self.gamma = gamma or GammaClient(settings.gamma_url)
        self.clob = clob or ClobMarketData(settings.clob_url)
        self.executor = executor or Executor(
            settings.clob_url,
            dry_run=settings.dry_run,
            private_key=settings.private_key,
            funder=settings.funder,
            signature_type=settings.signature_type,
        )
        self.logger = logger or TradeLogger(settings.trade_log_path)
        self.decision_logger = DecisionLogger(settings.trade_log_path.replace("trade_log", "decision_log"))
        self.risk_manager = RiskManager(settings.risk, clock=clock, state_path=settings.risk_state_path)
        self.ledger = PaperTradeLedger()
        self.feature_engine = FeatureEngine()
        self.market: Market | None = None
        self.next_market_refresh = 0.0
        # market id -> end_ts; a market is never retried once an order was attempted.
        self.attempted_markets = self.risk_manager.active_order_attempts()
        self.open_prices: dict[tuple[str, str], float] = {}
        self.open_price_warned: set[str] = set()
        self.next_resolution_check: dict[str, float] = {}
        self.geoblock_blocked: bool | None = None
        self.geoblock_country = "unknown"
        self.next_geoblock_check = 0.0
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.telegram_offset = 0
        self.telegram_stopped = False
        self.telegram_outbox: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._last_alerted_risk = ""
        self._last_error_alert_at = 0.0
        self._next_heartbeat = monotonic() + TELEGRAM_HEARTBEAT_SECONDS

    async def run(self) -> None:
        sender_task = asyncio.create_task(self.telegram_sender_loop(), name="telegram-sender")
        command_task = asyncio.create_task(self.telegram_command_loop(), name="telegram-commands")
        mode = "DRY RUN" if self.settings.dry_run else "LIVE"
        self.notify(f"Bot started ({mode}); daily trades={self.risk_manager.trades_today}.")
        if self.risk_manager.day_changed_on_load:
            self.notify("Daily UTC trade counter reset after the date changed.")
        startup_backoff = 1.0
        try:
            await self.check_geoblock_status(force=True)
            while True:
                try:
                    await asyncio.to_thread(self.executor.ensure_ready)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.error("[startup] exchange authentication failed (%s); retrying in %.1fs", type(exc).__name__, startup_backoff)
                    self.notify_error(f"Exchange authentication failed ({type(exc).__name__}); retrying.")
                    await asyncio.sleep(startup_backoff)
                    startup_backoff = min(60.0, startup_backoff * 2)
            log.info("[startup] mode=%s stake=%.2f USDC", mode, self.settings.stake_usdc)
            while True:
                try:
                    await self.step()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.exception("[loop] unexpected error")
                    self.notify_error(f"Trading cycle failed ({type(exc).__name__}).")
                if monotonic() >= self._next_heartbeat:
                    status = "paused" if self.telegram_stopped or self.geoblock_blocked is not False else "running"
                    self.notify(f"Heartbeat: {status}; daily trades={self.risk_manager.trades_today}.")
                    self._next_heartbeat = monotonic() + TELEGRAM_HEARTBEAT_SECONDS
                await asyncio.sleep(self.settings.poll_seconds)
        finally:
            for task in (command_task, sender_task):
                task.cancel()
            await asyncio.gather(command_task, sender_task, return_exceptions=True)
            shutdown_prefix = "[DRY RUN] " if self.settings.dry_run else ""
            await asyncio.to_thread(send_telegram_alert, shutdown_prefix + "Bot is shutting down.")

    def notify(self, message: str) -> None:
        prefix = "[DRY RUN] " if self.settings.dry_run else ""
        try:
            self.telegram_outbox.put_nowait(prefix + message)
        except asyncio.QueueFull:
            log.warning("[telegram] alert queue full; dropping notification")

    def notify_once(self, key: str, message: str) -> None:
        if key != self._last_alerted_risk:
            self._last_alerted_risk = key
            self.notify(message)

    def notify_error(self, message: str) -> None:
        now = monotonic()
        if now - self._last_error_alert_at >= 60.0:
            self._last_error_alert_at = now
            self.notify(message)

    async def telegram_sender_loop(self) -> None:
        """Send Telegram alerts with rate limiting (Telegram allows ~1 msg/sec per chat)."""
        last_send = 0.0
        min_interval = 1.1  # seconds between sends to same chat
        while True:
            message = await self.telegram_outbox.get()
            try:
                now = monotonic()
                wait = max(0.0, last_send + min_interval - now)
                if wait > 0:
                    await asyncio.sleep(wait)
                await asyncio.to_thread(send_telegram_alert, message)
                last_send = monotonic()
            finally:
                self.telegram_outbox.task_done()

    async def telegram_command_loop(self) -> None:
        while True:
            if not self.telegram_chat_id or not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
                await asyncio.sleep(30.0)
                continue
            updates = await asyncio.to_thread(fetch_telegram_updates, self.telegram_offset)
            if updates is None:
                await asyncio.sleep(5.0)
                continue
            self.process_telegram_updates(updates)

    def process_telegram_updates(self, updates: list[dict]) -> None:
        for update in updates:
            if not isinstance(update, dict):
                continue
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self.telegram_offset = max(self.telegram_offset, update_id + 1)
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat") or {}
            sender = message.get("from") or {}
            if str(chat.get("id", "")) != self.telegram_chat_id:
                continue
            if str(sender.get("id", "")) != self.telegram_chat_id:
                continue
            words = str(message.get("text", "")).strip().split()
            command = words[0].split("@", 1)[0].lower() if words else ""
            if command == "/stop":
                self.telegram_stopped = True
                self.notify("Telegram stop command accepted; new orders are paused.")
            elif command == "/resume":
                if self.settings.risk.kill_switch:
                    self.notify("Cannot resume: KILL_SWITCH is enabled in configuration.")
                else:
                    self.telegram_stopped = False
                    self.notify("Telegram resume command accepted; trading resumes when other safety checks allow.")
            elif command == "/status":
                status = "paused" if self.telegram_stopped or self.geoblock_blocked is not False else "running"
                self.notify(f"Bot status: {status}; daily trades={self.risk_manager.trades_today}.")

    async def step(self) -> None:
        now = self.clock()
        previous_day = self.risk_manager._day
        risk_block = self.risk_manager.block_reason()
        if previous_day is not None and self.risk_manager._day != previous_day:
            self._last_alerted_risk = ""
            self.notify("Daily UTC risk limits reset; trading may resume.")
        if risk_block:
            key = "daily-cap" if "daily trade limit" in risk_block else risk_block
            self.notify_once(key, f"Risk limit: {risk_block}.")
        else:
            self._last_alerted_risk = ""
        await self.check_geoblock_status()
        await self.resolve_positions(now)
        if self.telegram_stopped or self.geoblock_blocked is not False:
            return
        market = await self.current_market(now)
        if market is None:
            return

        start_ts = market.start_ts or market.end_ts - WINDOW_SECONDS
        if now < start_ts or market.end_ts <= now:
            return

        btc = self.state.get_current_btc_state(now=now)
        if btc.status != "OK" or btc.price is None or btc.primary_source is None:
            log.warning("[decision] %s: %s; new trades disabled", btc.status, btc.reason)
            self.notify_once(f"feed-{btc.status}", f"Price feed unavailable ({btc.status}); new trades are paused.")
            return
        self._last_alerted_risk = ""

        source = btc.primary_source
        open_price = self.open_price(market, source, start_ts)
        if open_price is None:
            if market.id not in self.open_price_warned:
                self.open_price_warned.add(market.id)
                log.warning(
                    "[market] no %s price near window start; skipping %s (bot joined mid-window or feed gap)",
                    source,
                    market.slug or market.id,
                )
            return

        # Get TWAP prices for probability model
        twap_open_price = self.state.market_open_twap(start_ts)
        twap_current_price = self.state.get_twap()
        
        # Use TWAP if available, fall back to spot
        model_open_price = twap_open_price if twap_open_price is not None else open_price
        model_current_price = twap_current_price if twap_current_price is not None else btc.price
        
        stake = self.settings.stake_usdc
        up_quote, down_quote = await asyncio.gather(
            asyncio.to_thread(self.clob.quote, market.up.token_id, stake),
            asyncio.to_thread(self.clob.quote, market.down.token_id, stake),
        )
        if up_quote is None or down_quote is None:
            self.notify_error("Polymarket order-book quote unavailable; affected trades are skipped.")
        seconds_left = market.end_ts - self.clock()
        up_ask = up_quote.best_ask if up_quote else None
        down_ask = down_quote.best_ask if down_quote else None
        activity_measure = max(market.volume, market.liquidity)
        samples = self.state.samples(source)
        current_sample_timestamp = samples[-1][0] if samples else None

        signal = build_signal(
            open_price=model_open_price,
            current_price=model_current_price,
            seconds_left=seconds_left,
            recent_prices=[price for _, price in samples],
            recent_samples=samples,
            current_timestamp=current_sample_timestamp,
            up_token_id=market.up.token_id,
            down_token_id=market.down.token_id,
            up_ask=up_ask,
            down_ask=down_ask,
            up_fill_price=fill_price(up_quote),
            down_fill_price=fill_price(down_quote),
            market_volume=activity_measure,
            trades_today=self.risk_manager.trades_today,
            risk=self.settings.risk,
        )

        # Log decision cycle for calibration (only within execution window)
        if (self.settings.risk.min_seconds_left <= seconds_left <= self.settings.risk.max_seconds_left
                and signal.probability_up is not None):
            sigma = per_second_volatility(
                sorted((float(ts), float(price)) for ts, price in samples if ts is not None and price > 0),
                end_timestamp=current_sample_timestamp,
            )
            z_score = 0.0
            if sigma > 0 and seconds_left > 0:
                z_score = ln(model_open_price / model_current_price) / (sigma * sqrt(seconds_left))
            trade_outcome = signal.outcome if signal.should_trade else None
            self.decision_logger.record_decision(
                timestamp=datetime.now(DISPLAY_TZ).isoformat(),
                market_id=market.id,
                market_slug=market.slug or market.event_slug,
                seconds_left=seconds_left,
                open_price=open_price,
                current_price=btc.price,
                model_open_price=model_open_price,
                model_current_price=model_current_price,
                twap_open_price=twap_open_price,
                twap_current_price=twap_current_price,
                sigma=sigma,
                z_score=z_score,
                model_probability_up=signal.probability_up,
                trade_decision="TRADE" if signal.should_trade else "NO_TRADE",
                trade_outcome=trade_outcome,
            )

        side_quote = up_quote if signal.outcome == "UP" else down_quote
        self.log_decision(market, signal, btc.price, open_price, samples, current_sample_timestamp,
                          seconds_left, activity_measure, up_ask, down_ask, side_quote)

        stamp = datetime.now(DISPLAY_TZ).strftime("%H:%M:%S %Z")
        log.info(
            "[%s] left=%5.1fs btc=%.2f src=%s open=%.2f up_ask=%s down_ask=%s pick=%s p=%.3f edge=%.3f %s",
            stamp, seconds_left, btc.price, source, open_price, fmt(up_ask), fmt(down_ask),
            signal.outcome, signal.model_probability, signal.edge, signal.reason,
        )

        if signal.should_trade and market.id not in self.attempted_markets:
            await self.execute(market, signal, side_quote, stake)
        elif signal.reason == "daily trade limit reached":
            self.notify_once("daily-cap", "Daily trade cap reached; trading is paused until the next UTC day.")

    async def check_geoblock_status(self, *, force: bool = False) -> None:
        now = monotonic()
        if not force and now < self.next_geoblock_check:
            return
        self.next_geoblock_check = now + GEOBLOCK_INTERVAL_SECONDS
        previous = self.geoblock_blocked
        try:
            blocked, country = await asyncio.to_thread(check_geoblock)
            self.geoblock_blocked = blocked
            self.geoblock_country = country
        except Exception as exc:
            self.geoblock_blocked = None
            log.error("[geoblock] check failed (%s); new trades paused", type(exc).__name__)
            self.notify_error("Polymarket geoblock check failed; trading is paused.")
            return

        if self.geoblock_blocked and previous is not True:
            self.notify(f"Polymarket geoblock blocked this connection (country: {self.geoblock_country}); new trades are paused.")
        elif previous is True and not self.geoblock_blocked:
            self.notify("Polymarket geoblock is clear; trading has resumed.")

    async def current_market(self, now: float) -> Market | None:
        previous = self.market
        refreshed = False
        if self.market is None or now >= self.market.end_ts or now >= self.next_market_refresh:
            refreshed = True
            try:
                markets = await asyncio.to_thread(self.gamma.find_btc_5m_markets)
            except Exception as exc:
                log.warning("[scan] market discovery failed: %s", exc)
                self.notify_error(f"Market discovery failed ({type(exc).__name__}).")
                markets = None
            if markets is not None:
                self.market = choose_current_market(markets, now)
            elif self.market is not None and now >= self.market.end_ts:
                self.market = None
            self.next_market_refresh = now + (self.settings.market_refresh_seconds if self.market else 5.0)

        if self.market is None:
            if refreshed:
                log.info("[scan] no active BTC 5-minute market found")
        elif previous is None or previous.id != self.market.id:
            self.open_prices = {key: value for key, value in self.open_prices.items() if key[0] == self.market.id}
            log.info(
                "[market] %s | volume=%.0f liquidity=%.0f",
                self.market.question, self.market.volume, self.market.liquidity,
            )
        return self.market

    def open_price(self, market: Market, source: str, start_ts: float) -> float | None:
        key = (market.id, source)
        if key not in self.open_prices:
            price = self.state.market_open_price(start_ts, source=source)
            if price is None:
                return None
            self.open_prices[key] = price
            log.info("[market] window open %.2f (%s)", price, source)
        return self.open_prices[key]

    async def execute(self, market: Market, signal: Signal, quote: BookQuote | None, stake: float) -> None:
        if quote and quote.best_ask is not None and quote.best_bid is None:
            spread = None
        elif quote and quote.spread is not None:
            spread = quote.spread
        else:
            spread = float("inf")
        blocked = self.risk_manager.block_reason(spread=spread, liquidity=market.liquidity)
        if blocked:
            log.info("[risk] skipping %s %s: %s", market.slug or market.id, signal.outcome, blocked)
            key = "daily-cap" if "daily trade limit" in blocked else blocked
            self.notify_once(key, f"Risk limit: {blocked}.")
            return

        if self.telegram_stopped or self.geoblock_blocked is not False:
            return

        # Persist the intent before sending so a restart cannot repeat an ambiguous order.
        if not self.risk_manager.reserve_order_attempt(market.id, market.end_ts):
            self.attempted_markets[market.id] = market.end_ts
            return
        self.attempted_markets[market.id] = market.end_ts
        try:
            result = await asyncio.to_thread(self.executor.buy_market, signal.token_id, stake, signal.limit_price)
        except Exception as exc:
            self.risk_manager.update_order_attempt(market.id, "unknown")
            log.error("[trade] order failed for %s (%s)", market.id, type(exc).__name__)
            self.logger.record_event("order_error", market_id=market.id, outcome=signal.outcome, error=type(exc).__name__)
            self.notify_error(f"Order outcome is unknown for market {market.id}; no retry will be made.")
            return

        self.logger.record_event(
            "order",
            market_id=market.id,
            outcome=signal.outcome,
            stake=stake,
            limit_price=signal.limit_price,
            expected_fill=signal.fill_price,
            dry_run=self.settings.dry_run,
            filled=result.filled,
            error_code="none" if result.filled else "order_not_filled",
        )
        self.risk_manager.update_order_attempt(market.id, "filled" if result.filled else "not_filled")
        if not result.filled:
            log.warning("[trade] %s order not filled", signal.outcome)
            self.notify(f"Order not filled for market {market.id}.")
            return

        expected_fill = signal.fill_price or signal.ask_price
        shares = result.shares or stake / expected_fill
        cost = result.cost_usdc or stake
        self.ledger.enter(
            market_id=market.id,
            outcome=signal.outcome,
            entry_price=cost / shares,
            stake=cost,
            model_probability=signal.model_probability,
            edge=signal.edge,
            shares=shares,
            end_ts=market.end_ts,
        )
        self.risk_manager.record_filled_position()
        log.info(
            "[trade] %s%s %.2f USDC for %.2f shares (avg %.3f, limit %s, p=%.3f)",
            "PAPER " if self.settings.dry_run else "",
            signal.outcome, cost, shares, cost / shares, fmt(signal.limit_price), signal.model_probability,
        )
        self.notify(
            f"Trade {signal.outcome}: {cost:.2f} USDC, {shares:.2f} shares, average price {cost / shares:.3f}."
        )

    async def resolve_positions(self, now: float) -> None:
        for market_id, end_ts in list(self.attempted_markets.items()):
            if end_ts < now - 3600:
                del self.attempted_markets[market_id]

        for trade in self.ledger.open_trades():
            if trade.end_ts is None or now < trade.end_ts + RESOLUTION_GRACE_SECONDS:
                continue
            if now < self.next_resolution_check.get(trade.market_id, 0.0):
                continue
            slow = now > trade.end_ts + 3600
            self.next_resolution_check[trade.market_id] = now + (RESOLUTION_SLOW_RETRY_SECONDS if slow else RESOLUTION_RETRY_SECONDS)
            try:
                winner = await asyncio.to_thread(self.gamma.fetch_resolution, trade.market_id)
            except Exception as exc:
                log.warning("[result] resolution lookup failed for %s: %s", trade.market_id, exc)
                self.notify_error(f"Market resolution lookup failed ({type(exc).__name__}).")
                continue
            if winner is None:
                if slow:
                    log.warning("[result] market %s still unresolved an hour after close", trade.market_id)
                continue

            resolved = self.ledger.resolve(trade.market_id, winner)
            if resolved is None or resolved.pnl is None:
                continue
            self.next_resolution_check.pop(trade.market_id, None)
            self.risk_manager.record_result(resolved.pnl)
            summary = self.ledger.summary()
            self.logger.record_event(
                "resolution",
                market_id=trade.market_id,
                outcome=resolved.outcome,
                winner=winner,
                stake=resolved.stake,
                shares=resolved.share_count,
                pnl=resolved.pnl,
                dry_run=self.settings.dry_run,
            )
            log.info(
                "[result] %s bought %s, resolved %s, pnl=%+.2f | resolved=%d win_rate=%.0f%% total_pnl=%+.2f daily_pnl=%+.2f",
                trade.market_id, resolved.outcome, winner, resolved.pnl, int(summary["resolved"]),
                summary["win_rate"] * 100, summary["total_pnl"], self.risk_manager.daily_pnl,
            )

    def log_decision(
        self,
        market: Market,
        signal: Signal,
        btc_price: float,
        open_price: float,
        samples: list[tuple[float, float]],
        current_timestamp: float | None,
        seconds_left: float,
        activity_measure: float,
        up_ask: float | None,
        down_ask: float | None,
        side_quote: BookQuote | None,
    ) -> None:
        features = self.feature_engine.compute_features(
            open_price=open_price,
            current_price=btc_price,
            recent_prices=[price for _, price in samples],
            recent_samples=samples,
            current_timestamp=current_timestamp,
            seconds_left=seconds_left,
            market_volume=activity_measure,
            up_ask=up_ask,
            down_ask=down_ask,
        )
        probability_up = signal.probability_up
        self.logger.record_decision(
            timestamp=datetime.now(DISPLAY_TZ).isoformat(),
            market_id=market.id,
            time_remaining=seconds_left,
            btc_price=btc_price,
            btc_open_price=open_price,
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
            polymarket_bid=side_quote.best_bid if side_quote and side_quote.best_bid is not None else 0.0,
            polymarket_ask=signal.ask_price,
            model_probability_up=probability_up,
            model_probability_down=None if probability_up is None else 1.0 - probability_up,
            estimated_edge=signal.edge,
            trade_decision="TRADE" if signal.should_trade else "NO_TRADE",
            decision_reason=signal.reason,
            position_size=self.settings.stake_usdc if signal.should_trade else 0.0,
        )


def setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def main() -> None:
    load_dotenv()
    setup_logging()
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        log.error("%s", exc)
        raise SystemExit(2) from None

    state = PriceState(
        max_age_seconds=settings.price_max_age_seconds,
        max_divergence_bps=settings.price_max_divergence_bps,
        coinbase_stale_after_seconds=settings.coinbase_stale_after_seconds,
    )
    feeds = [
        run_price_feed(
            state,
            feed,
            min_backoff=settings.price_reconnect_min_seconds,
            max_backoff=settings.price_reconnect_max_seconds,
            silence_timeout=settings.price_silence_reconnect_seconds,
        )
        for feed in build_feeds(settings)
    ]
    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    try:
        if main_task is not None:
            loop.add_signal_handler(signal.SIGTERM, main_task.cancel)
    except (NotImplementedError, RuntimeError):
        log.warning("[shutdown] SIGTERM handler unavailable on this platform")

    try:
        await asyncio.gather(*feeds, TradingBot(settings, state).run())
    except asyncio.CancelledError:
        log.info("[shutdown] SIGTERM received; stopping feeds and trading loop")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[shutdown] stopped")
