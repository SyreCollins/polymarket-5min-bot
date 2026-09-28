from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from statistics import median
from time import time
from typing import Callable
from zoneinfo import ZoneInfo

import websockets
from dotenv import load_dotenv

from polymarket_client import WINDOW_SECONDS, BookQuote, ClobMarketData, Executor, GammaClient, Market
from settings import ConfigError, Settings
from strategy import UNFILLABLE, FeatureEngine, PaperTradeLedger, RiskManager, Signal, TradeLogger, build_signal


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

    def _append(self, history: deque[tuple[float, float]], timestamp: float, price: float) -> None:
        history.append((timestamp, price))
        cutoff = timestamp - self.history_retention_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

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
        self.risk_manager = RiskManager(settings.risk, clock=clock)
        self.ledger = PaperTradeLedger()
        self.feature_engine = FeatureEngine()
        self.market: Market | None = None
        self.next_market_refresh = 0.0
        # market id -> end_ts; a market is never retried once an order was attempted.
        self.attempted_markets: dict[str, float] = {}
        self.open_prices: dict[tuple[str, str], float] = {}
        self.open_price_warned: set[str] = set()
        self.next_resolution_check: dict[str, float] = {}

    async def run(self) -> None:
        await asyncio.to_thread(self.executor.ensure_ready)
        mode = "DRY-RUN (paper)" if self.settings.dry_run else "LIVE"
        log.info("[startup] mode=%s stake=%.2f USDC", mode, self.settings.stake_usdc)
        while True:
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("[loop] unexpected error")
            await asyncio.sleep(self.settings.poll_seconds)

    async def step(self) -> None:
        now = self.clock()
        await self.resolve_positions(now)
        market = await self.current_market(now)
        if market is None:
            return

        start_ts = market.start_ts or market.end_ts - WINDOW_SECONDS
        if now < start_ts or market.end_ts <= now:
            return

        btc = self.state.get_current_btc_state(now=now)
        if btc.status != "OK" or btc.price is None or btc.primary_source is None:
            log.warning("[decision] %s: %s; new trades disabled", btc.status, btc.reason)
            return

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

        stake = self.settings.stake_usdc
        up_quote, down_quote = await asyncio.gather(
            asyncio.to_thread(self.clob.quote, market.up.token_id, stake),
            asyncio.to_thread(self.clob.quote, market.down.token_id, stake),
        )
        seconds_left = market.end_ts - self.clock()
        up_ask = up_quote.best_ask if up_quote else None
        down_ask = down_quote.best_ask if down_quote else None
        activity_measure = max(market.volume, market.liquidity)
        samples = self.state.samples(source)
        current_sample_timestamp = samples[-1][0] if samples else None

        signal = build_signal(
            open_price=open_price,
            current_price=btc.price,
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

    async def current_market(self, now: float) -> Market | None:
        previous = self.market
        refreshed = False
        if self.market is None or now >= self.market.end_ts or now >= self.next_market_refresh:
            refreshed = True
            try:
                markets = await asyncio.to_thread(self.gamma.find_btc_5m_markets)
            except Exception as exc:
                log.warning("[scan] market discovery failed: %s", exc)
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
        spread = quote.spread if quote and quote.spread is not None else float("inf")
        blocked = self.risk_manager.block_reason(spread=spread, liquidity=market.liquidity)
        if blocked:
            log.info("[risk] skipping %s %s: %s", market.slug or market.id, signal.outcome, blocked)
            return

        # Mark before sending: a timeout may still have filled, so never re-send for this market.
        self.attempted_markets[market.id] = market.end_ts
        try:
            result = await asyncio.to_thread(self.executor.buy_market, signal.token_id, stake, signal.limit_price)
        except Exception as exc:
            log.exception("[trade] order failed for %s", market.id)
            self.logger.record_event("order_error", market_id=market.id, outcome=signal.outcome, error=str(exc))
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
            error=result.error,
            response=result.raw,
        )
        if not result.filled:
            log.warning("[trade] %s order not filled: %s", signal.outcome, result.error)
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
        self.risk_manager.record_entry()
        log.info(
            "[trade] %s%s %.2f USDC for %.2f shares (avg %.3f, limit %s, p=%.3f)",
            "PAPER " if self.settings.dry_run else "",
            signal.outcome, cost, shares, cost / shares, fmt(signal.limit_price), signal.model_probability,
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
    await asyncio.gather(*feeds, TradingBot(settings, state).run())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[shutdown] stopped")
