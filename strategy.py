from __future__ import annotations

import json
import sqlite3
from collections import deque
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import erf, exp, isfinite, log, sqrt
from pathlib import Path
from statistics import pstdev
from time import time
from typing import Any, Callable, Iterable


# Passed as a fill price when the order book cannot absorb the whole stake.
UNFILLABLE = float("inf")


@dataclass(frozen=True)
class RiskConfig:
    trade_usdc: float = 5.0
    max_trades_per_day: int = 20
    min_edge: float = 0.07
    max_edge: float = 0.25
    min_seconds_left: float = 8.0
    max_seconds_left: float = 45.0
    min_market_volume: float = 1000.0
    max_buy_price: float = 0.92
    max_daily_loss: float = 50.0
    max_position_size: float = 25.0
    max_consecutive_losses: int = 3
    max_trades_per_hour: int = 10
    min_liquidity: float = 1000.0
    max_spread: float = 0.08
    min_edge_after_costs: float = 0.08
    # Per-share buffer for taker fees and slippage, in probability units.
    execution_cost: float = 0.0
    kill_switch: bool = False
    max_open_positions: int = 2
    min_probability: float = 0.01
    max_probability: float = 0.99
    min_price_history_seconds: float = 60.0


@dataclass(frozen=True)
class Signal:
    should_trade: bool
    outcome: str
    token_id: str
    model_probability: float
    ask_price: float
    edge: float
    reason: str
    probability_up: float | None = None
    fill_price: float | None = None
    edge_after_costs: float | None = None
    # Worst price the order may fill at while still clearing the after-cost edge threshold.
    limit_price: float | None = None


class FeatureEngine:
    def compute_features(
        self,
        *,
        open_price: float,
        current_price: float,
        recent_prices: Iterable[float],
        recent_samples: Iterable[tuple[float, float]] = (),
        current_timestamp: float | None = None,
        seconds_left: float,
        market_volume: float,
        up_ask: float | None = None,
        down_ask: float | None = None,
        vwap: float | None = None,
        order_imbalance: float = 0.0,
        volume_acceleration: float = 0.0,
    ) -> dict[str, float]:
        history = list(recent_prices)
        samples = sorted(
            (float(timestamp), float(price))
            for timestamp, price in recent_samples
            if timestamp is not None and price > 0
        )
        if not history:
            history = [current_price]

        btc_5m_return = (current_price / open_price - 1.0) if open_price > 0 else 0.0
        sample_end = current_timestamp if current_timestamp is not None else (samples[-1][0] if samples else None)
        btc_1m_return = self._elapsed_return(samples, sample_end, 60.0)
        btc_30s_return = self._elapsed_return(samples, sample_end, 30.0)
        history_span = price_history_span(samples, sample_end)

        if vwap is None:
            vwap = open_price
        distance_from_open = (current_price - open_price) / open_price if open_price > 0 else 0.0
        distance_from_vwap = (current_price - vwap) / vwap if vwap > 0 else 0.0

        momentum = btc_1m_return + 0.5 * btc_30s_return
        volatility = realized_volatility(history)
        normalized_volatility = min(max(volatility / 0.001, 0.0), 2.0)
        volume_feature = min(max((market_volume / 1000.0) - 1.0, -1.0), 2.0)
        time_component = max(0.0, 1.0 - (seconds_left / max(1.0, 300.0)))
        spread_component = 0.0
        if up_ask is not None and down_ask is not None and up_ask > 0 and down_ask > 0:
            spread_component = abs(up_ask - down_ask)

        return {
            "btc_5m_return": float(btc_5m_return),
            "btc_1m_return": float(btc_1m_return),
            "btc_30s_return": float(btc_30s_return),
            "price_history_seconds": float(history_span),
            "distance_from_open": float(distance_from_open),
            "distance_from_vwap": float(distance_from_vwap),
            "momentum": float(momentum),
            "volatility": float(normalized_volatility),
            "volume_feature": float(volume_feature),
            "order_flow": float(order_imbalance),
            "volume_acceleration": float(volume_acceleration),
            "time_component": float(time_component),
            "spread_component": float(spread_component),
        }

    @staticmethod
    def _elapsed_return(
        samples: list[tuple[float, float]],
        end_timestamp: float | None,
        duration_seconds: float,
    ) -> float:
        if len(samples) < 2 or end_timestamp is None:
            return 0.0
        start_timestamp = end_timestamp - duration_seconds
        previous_sample = next((sample for sample in reversed(samples) if sample[0] <= start_timestamp), None)
        current_sample = next((sample for sample in reversed(samples) if sample[0] <= end_timestamp), None)
        if previous_sample is None or current_sample is None:
            return 0.0
        previous = previous_sample[1]
        current = current_sample[1]
        if previous <= 0 or current <= 0:
            return 0.0
        return (current / previous) - 1.0


class ProbabilityModel:
    """Logistic score over engineered features.

    The default weights are unfitted placeholders: raw returns are ~1e-3, so the output stays
    within a hair of 0.5. It is kept for research only; build_signal uses the Brownian model.
    """

    def __init__(self, weights: dict[str, float] | None = None, min_probability: float = 0.01, max_probability: float = 0.99):
        self.weights = {
            "btc_5m_return": 0.35,
            "btc_1m_return": 0.25,
            "btc_30s_return": 0.2,
            "distance_from_open": 0.15,
            "distance_from_vwap": 0.1,
            "momentum": 0.2,
            "volatility": 0.0,
            "volume_feature": 0.0,
            "order_flow": 0.12,
            "volume_acceleration": 0.08,
            "time_component": 0.0,
            "spread_component": 0.0,
        }
        if weights:
            self.weights.update(weights)
        self.min_probability = min_probability
        self.max_probability = max_probability

    def estimate_probability(self, features: dict[str, float]) -> float:
        score = 0.0
        for key, weight in self.weights.items():
            score += weight * float(features.get(key, 0.0))
        probability = 1.0 / (1.0 + exp(-max(-60.0, min(60.0, score))))
        return min(max(probability, self.min_probability), self.max_probability)


class EdgeCalculator:
    def calculate_edge(
        self,
        *,
        model_probability_up: float,
        up_ask: float | None,
        model_probability_down: float | None = None,
        down_ask: float | None = None,
        execution_cost: float = 0.0,
    ) -> dict[str, float | None]:
        if model_probability_down is None:
            model_probability_down = 1.0 - model_probability_up
        up_edge = None if up_ask is None else model_probability_up - (up_ask + execution_cost)
        down_edge = None if down_ask is None else model_probability_down - (down_ask + execution_cost)
        return {"up_edge": up_edge, "down_edge": down_edge}


@dataclass
class DecisionLogger:
    """Logs every decision cycle within the execution window for calibration analysis."""
    path: Path | str = "decision_log.jsonl"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_decision(
        self,
        *,
        timestamp: str,
        market_id: str,
        market_slug: str,
        seconds_left: float,
        open_price: float,
        current_price: float,
        model_open_price: float | None = None,
        model_current_price: float | None = None,
        twap_open_price: float | None = None,
        twap_current_price: float | None = None,
        sigma: float,
        z_score: float,
        model_probability_up: float,
        trade_decision: str,
        trade_outcome: str | None,
    ) -> None:
        record = {
            "timestamp": timestamp,
            "market_id": market_id,
            "market_slug": market_slug,
            "seconds_left": seconds_left,
            "open_price": open_price,
            "current_price": current_price,
            "model_open_price": model_open_price,
            "model_current_price": model_current_price,
            "twap_open_price": twap_open_price,
            "twap_current_price": twap_current_price,
            "sigma": sigma,
            "z_score": z_score,
            "model_probability_up": model_probability_up,
            "trade_decision": trade_decision,  # "TRADE" or "NO_TRADE"
            "trade_outcome": trade_outcome,    # "UP", "DOWN", or None
        }
        self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


@dataclass
class TradeLogger:
    path: Path | str = "trade_log.jsonl"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_decision(
        self,
        *,
        timestamp: str,
        market_id: str,
        time_remaining: float,
        btc_price: float,
        btc_open_price: float,
        btc_return_30s: float,
        btc_return_1m: float,
        btc_return_5m: float,
        volume_features: dict[str, float],
        order_flow_features: dict[str, float],
        polymarket_bid: float,
        polymarket_ask: float,
        model_probability_up: float | None,
        model_probability_down: float | None,
        estimated_edge: float,
        trade_decision: str,
        decision_reason: str,
        position_size: float,
    ) -> None:
        record = {
            "timestamp": timestamp,
            "market_id": market_id,
            "time_remaining": time_remaining,
            "btc_price": btc_price,
            "btc_open_price": btc_open_price,
            "btc_return_30s": btc_return_30s,
            "btc_return_1m": btc_return_1m,
            "btc_return_5m": btc_return_5m,
            "volume_features": volume_features,
            "order_flow_features": order_flow_features,
            "polymarket_bid": polymarket_bid,
            "polymarket_ask": polymarket_ask,
            "model_probability_up": model_probability_up,
            "model_probability_down": model_probability_down,
            "estimated_edge": estimated_edge,
            "trade_decision": trade_decision,
            "decision_reason": decision_reason,
            "position_size": position_size,
        }
        self._append(record)

    def record_event(self, event: str, **fields: Any) -> None:
        self._append({"event": event, "logged_at": time(), **fields})

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


@dataclass
class PaperTrade:
    market_id: str
    outcome: str
    entry_price: float
    stake: float
    model_probability: float
    edge: float
    shares: float | None = None
    end_ts: float | None = None
    resolved_outcome: str | None = None
    pnl: float | None = None

    @property
    def share_count(self) -> float:
        if self.shares is not None:
            return self.shares
        return self.stake / self.entry_price if self.entry_price > 0 else 0.0


class PaperTradeLedger:
    def __init__(self) -> None:
        self.trades: list[PaperTrade] = []

    def enter(
        self,
        *,
        market_id: str,
        outcome: str,
        entry_price: float,
        stake: float,
        model_probability: float,
        edge: float,
        shares: float | None = None,
        end_ts: float | None = None,
    ) -> PaperTrade:
        trade = PaperTrade(
            market_id=market_id,
            outcome=outcome.upper(),
            entry_price=entry_price,
            stake=stake,
            model_probability=model_probability,
            edge=edge,
            shares=shares,
            end_ts=end_ts,
        )
        self.trades.append(trade)
        return trade

    def resolve(self, market_id: str, resolved_outcome: str) -> PaperTrade | None:
        normalized_outcome = resolved_outcome.upper()
        trade = next(
            (
                item
                for item in self.trades
                if item.market_id == market_id and item.resolved_outcome is None
            ),
            None,
        )
        if trade is None:
            return None

        trade.resolved_outcome = normalized_outcome
        payout = trade.share_count if trade.outcome == normalized_outcome else 0.0
        trade.pnl = payout - trade.stake
        return trade

    def open_trades(self) -> list[PaperTrade]:
        return [trade for trade in self.trades if trade.resolved_outcome is None]

    def summary(self) -> dict[str, float]:
        resolved = [trade for trade in self.trades if trade.pnl is not None]
        if not resolved:
            return {
                "trades": float(len(self.trades)),
                "resolved": 0.0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_pnl": 0.0,
                "average_edge": 0.0,
            }

        wins = sum(1 for trade in resolved if trade.pnl is not None and trade.pnl > 0)
        total_pnl = sum(trade.pnl or 0.0 for trade in resolved)
        return {
            "trades": float(len(self.trades)),
            "resolved": float(len(resolved)),
            "win_rate": wins / len(resolved),
            "total_pnl": total_pnl,
            "average_pnl": total_pnl / len(resolved),
            "average_edge": sum(trade.edge for trade in resolved) / len(resolved),
        }


@dataclass
class RiskManager:
    risk: RiskConfig = field(default_factory=RiskConfig)
    clock: Callable[[], float] = time
    daily_pnl: float = 0.0
    trades_today: int = 0
    open_positions: int = 0
    consecutive_losses: int = 0
    _entry_times: deque[float] = field(default_factory=deque, repr=False)
    _day: str | None = field(default=None, repr=False)
    state_path: str | Path | None = field(default=None, repr=False, compare=False)
    day_changed_on_load: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.state_path is None:
            return
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path = path
        today = datetime.fromtimestamp(self.clock(), tz=timezone.utc).date().isoformat()
        with closing(self._connect_state()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS daily_risk_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    utc_date TEXT NOT NULL,
                    trades_today INTEGER NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS order_attempts (
                    market_id TEXT PRIMARY KEY,
                    end_ts REAL NOT NULL,
                    attempted_at REAL NOT NULL,
                    status TEXT NOT NULL
                )"""
            )
            row = connection.execute(
                "SELECT utc_date, trades_today FROM daily_risk_state WHERE singleton = 1"
            ).fetchone()
            self.day_changed_on_load = bool(row and row[0] != today)
            self._day = today
            self.trades_today = int(row[1]) if row and row[0] == today else 0
            self._save_daily_count(connection)
            cutoff = self.clock() - 3600.0
            self._entry_times = deque(
                float(item[0])
                for item in connection.execute(
                    "SELECT attempted_at FROM order_attempts WHERE attempted_at > ? AND status != 'not_filled' ORDER BY attempted_at",
                    (cutoff,),
                )
            )

    def _connect_state(self) -> sqlite3.Connection:
        return sqlite3.connect(self.state_path, timeout=10.0)

    def _save_daily_count(self, connection: sqlite3.Connection) -> None:
        if self._day is None:
            return
        connection.execute(
            """INSERT INTO daily_risk_state (singleton, utc_date, trades_today)
               VALUES (1, ?, ?)
               ON CONFLICT(singleton) DO UPDATE SET
                   utc_date = excluded.utc_date,
                   trades_today = excluded.trades_today""",
            (self._day, self.trades_today),
        )

    def reserve_order_attempt(self, market_id: str, end_ts: float) -> bool:
        self._roll_day()
        now = self.clock()
        if self.state_path is None:
            self.trades_today += 1
            self._entry_times.append(now)
            return True
        with closing(self._connect_state()) as connection, connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO order_attempts (market_id, end_ts, attempted_at, status) VALUES (?, ?, ?, 'pending')",
                (market_id, end_ts, now),
            )
            if cursor.rowcount != 1:
                return False
            self.trades_today += 1
            self._entry_times.append(now)
            self._save_daily_count(connection)
            connection.execute("DELETE FROM order_attempts WHERE end_ts < ?", (now - 30 * 86400,))
            return True

    def active_order_attempts(self) -> dict[str, float]:
        if self.state_path is None:
            return {}
        now = self.clock()
        with closing(self._connect_state()) as connection:
            rows = connection.execute(
                "SELECT market_id, end_ts FROM order_attempts WHERE end_ts >= ?",
                (now - 3600.0,),
            ).fetchall()
        return {str(market_id): float(end_ts) for market_id, end_ts in rows}

    def update_order_attempt(self, market_id: str, status: str) -> None:
        if self.state_path is None:
            return
        with closing(self._connect_state()) as connection, connection:
            row = connection.execute(
                "SELECT attempted_at FROM order_attempts WHERE market_id = ?",
                (market_id,),
            ).fetchone()
            connection.execute(
                "UPDATE order_attempts SET status = ? WHERE market_id = ?",
                (status, market_id),
            )
            if status == "not_filled" and row is not None and self._day is not None:
                current = connection.execute(
                    "SELECT utc_date, trades_today FROM daily_risk_state WHERE singleton = 1"
                ).fetchone()
                if current is not None and current[0] == self._day:
                    self.trades_today = max(0, int(current[1]) - 1)
                    self._save_daily_count(connection)
                attempted_at = float(row[0])
                self._entry_times = deque(timestamp for timestamp in self._entry_times if timestamp != attempted_at)

    def record_filled_position(self) -> None:
        self._roll_day()
        self.open_positions += 1

    @property
    def daily_loss(self) -> float:
        self._roll_day()
        return max(0.0, -self.daily_pnl)

    @property
    def trades_last_hour(self) -> int:
        cutoff = self.clock() - 3600.0
        while self._entry_times and self._entry_times[0] <= cutoff:
            self._entry_times.popleft()
        return len(self._entry_times)

    def block_reason(self, *, spread: float | None = None, liquidity: float | None = None) -> str | None:
        self._roll_day()
        if self.risk.kill_switch:
            return "kill switch enabled"
        if self.daily_loss >= self.risk.max_daily_loss:
            return f"daily loss limit reached ({self.daily_loss:.2f})"
        if self.trades_today >= self.risk.max_trades_per_day:
            return "daily trade limit reached"
        if self.trades_last_hour >= self.risk.max_trades_per_hour:
            return "hourly trade limit reached"
        if self.open_positions >= self.risk.max_open_positions:
            return "max open positions reached"
        if self.consecutive_losses >= self.risk.max_consecutive_losses:
            return f"{self.consecutive_losses} consecutive losses; paused until next UTC day"
        if spread is not None and spread > self.risk.max_spread:
            return f"spread {spread:.3f} above max"
        if liquidity is not None and liquidity < self.risk.min_liquidity:
            return "market liquidity below threshold"
        return None

    def can_trade(self, market_id: str | None = None, *, spread: float | None = None, liquidity: float | None = None) -> bool:
        return self.block_reason(spread=spread, liquidity=liquidity) is None

    def record_entry(self) -> None:
        self._roll_day()
        self.trades_today += 1
        self.open_positions += 1
        self._entry_times.append(self.clock())
        self._persist_daily_count()

    def record_result(self, pnl: float) -> None:
        self._roll_day()
        self.open_positions = max(0, self.open_positions - 1)
        self.daily_pnl += float(pnl)
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0

    def record_trade(self, trade_size: float, outcome: str, entry_price: float, exit_price: float, pnl: float | None = None) -> None:
        """Record an entry, and its result too when pnl is known."""
        self.record_entry()
        if pnl is not None:
            self.record_result(pnl)

    def _roll_day(self) -> None:
        today = datetime.fromtimestamp(self.clock(), tz=timezone.utc).date().isoformat()
        if today != self._day:
            self._day = today
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.consecutive_losses = 0
            self._persist_daily_count()

    def _persist_daily_count(self) -> None:
        if self.state_path is None:
            return
        with closing(self._connect_state()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS daily_risk_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    utc_date TEXT NOT NULL,
                    trades_today INTEGER NOT NULL
                )"""
            )
            self._save_daily_count(connection)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def realized_volatility(prices: Iterable[float], fallback_per_second: float = 0.00008) -> float:
    values = list(prices)
    if len(values) < 4:
        return fallback_per_second

    returns: list[float] = []
    for prev, cur in zip(values, values[1:]):
        if prev > 0 and cur > 0:
            returns.append(log(cur / prev))

    if len(returns) < 3:
        return fallback_per_second
    return max(pstdev(returns), fallback_per_second)


def per_second_volatility(
    samples: Iterable[tuple[float, float]],
    *,
    end_timestamp: float | None = None,
    lookback_seconds: float = 300.0,
    min_returns: int = 30,
    floor: float = 0.00008,
) -> float:
    """Realized volatility of 1-second log returns.

    Tick-to-tick returns are not per-second returns (feeds tick anywhere from ~1 to ~50 times a
    second), so samples are resampled onto a 1-second grid with forward fill first.
    """
    points = sorted((float(ts), float(price)) for ts, price in samples if price > 0)
    if len(points) < 2:
        return floor
    end = points[-1][0] if end_timestamp is None else end_timestamp
    start = end - lookback_seconds
    last_by_second: dict[int, float] = {}
    for ts, price in points:
        if start <= ts <= end:
            last_by_second[int(ts)] = price
    if len(last_by_second) < 2:
        return floor

    seconds = sorted(last_by_second)
    previous = last_by_second[seconds[0]]
    squared_sum = 0.0
    count = 0
    for second in range(seconds[0] + 1, seconds[-1] + 1):
        current = last_by_second.get(second, previous)
        squared_sum += log(current / previous) ** 2
        count += 1
        previous = current
    if count < min_returns:
        return floor
    return max(sqrt(squared_sum / count), floor)


def price_history_span(samples: list[tuple[float, float]], end_timestamp: float | None) -> float:
    if not samples or end_timestamp is None:
        return 0.0
    return max(0.0, end_timestamp - samples[0][0])


def probability_up(
    open_price: float,
    current_price: float,
    seconds_left: float,
    recent_prices: Iterable[float] = (),
    sigma: float | None = None,
) -> float:
    """P(price at expiry >= open) under driftless Brownian motion with per-second volatility sigma."""
    if open_price <= 0 or current_price <= 0:
        return 0.5
    if seconds_left <= 0:
        return 1.0 if current_price >= open_price else 0.0

    if sigma is None:
        sigma = realized_volatility(recent_prices)
    distance = log(open_price / current_price)
    denom = sigma * sqrt(max(seconds_left, 1.0))
    if denom <= 0:
        return 1.0 if current_price >= open_price else 0.0

    return 1.0 - normal_cdf(distance / denom)


def build_signal(
    *,
    open_price: float,
    current_price: float,
    seconds_left: float,
    recent_prices: Iterable[float],
    recent_samples: Iterable[tuple[float, float]] = (),
    current_timestamp: float | None = None,
    up_token_id: str,
    down_token_id: str,
    up_ask: float | None,
    down_ask: float | None,
    market_volume: float,
    trades_today: int,
    risk: RiskConfig,
    up_fill_price: float | None = None,
    down_fill_price: float | None = None,
    sigma: float | None = None,
) -> Signal:
    """Pick the side with the best edge, if any clears every gate.

    up_ask/down_ask are best asks; *_fill_price are average fill prices for the full stake
    (defaulting to the best ask, or UNFILLABLE when the book is too thin).

    If sigma is provided, it is used directly (e.g., TWAP-based sigma for consistency with TWAP prices).
    Otherwise, sigma is computed from spot samples.
    """
    if trades_today >= risk.max_trades_per_day:
        return _skip("UP", up_token_id, "daily trade limit reached")

    if market_volume < risk.min_market_volume:
        return _skip("UP", up_token_id, "market volume below threshold")

    if seconds_left < risk.min_seconds_left or seconds_left > risk.max_seconds_left:
        return _skip("UP", up_token_id, "outside execution window")

    samples = sorted((float(ts), float(price)) for ts, price in recent_samples if ts is not None and price > 0)
    sample_end = current_timestamp if current_timestamp is not None else (samples[-1][0] if samples else None)
    if price_history_span(samples, sample_end) < risk.min_price_history_seconds:
        return _skip("UP", up_token_id, "insufficient BTC price history")

    if sigma is None:
        sigma = per_second_volatility(samples, end_timestamp=sample_end)
    up_probability = probability_up(open_price, current_price, seconds_left, sigma=sigma)
    up_probability = min(max(up_probability, risk.min_probability), risk.max_probability)
    candidates: list[Signal] = []

    if up_ask is not None:
        candidates.append(_candidate("UP", up_token_id, up_probability, up_ask, up_fill_price, up_probability, risk))
    if down_ask is not None:
        candidates.append(_candidate("DOWN", down_token_id, 1.0 - up_probability, down_ask, down_fill_price, up_probability, risk))

    tradable = [candidate for candidate in candidates if candidate.should_trade]
    if tradable:
        return max(tradable, key=lambda item: item.edge)
    best = max(candidates, key=lambda item: item.edge, default=None)
    if best:
        return best
    return _skip("UP", up_token_id, "missing CLOB ask prices", probability_up=up_probability)


def _candidate(
    outcome: str,
    token_id: str,
    probability: float,
    ask: float,
    fill_price: float | None,
    up_probability: float,
    risk: RiskConfig,
) -> Signal:
    fill = ask if fill_price is None else fill_price
    edge = probability - ask
    edge_after_costs = probability - fill - risk.execution_cost if isfinite(fill) else None
    limit_price = min(risk.max_buy_price, probability - risk.execution_cost - risk.min_edge_after_costs)

    def result(should_trade: bool, reason: str) -> Signal:
        return Signal(
            should_trade,
            outcome,
            token_id,
            probability,
            ask,
            edge,
            reason,
            probability_up=up_probability,
            fill_price=fill if isfinite(fill) else None,
            edge_after_costs=edge_after_costs,
            limit_price=limit_price if should_trade else None,
        )

    if ask > risk.max_buy_price:
        return result(False, "ask above max buy price")
    if edge < risk.min_edge:
        return result(False, "edge below threshold")
    if edge > risk.max_edge:
        return result(False, "edge above maximum threshold")
    if edge_after_costs is None:
        return result(False, "order book too thin for stake")
    if edge_after_costs < risk.min_edge_after_costs:
        return result(False, "edge after costs below threshold")
    if limit_price < fill:
        return result(False, "limit price below expected fill")
    return result(True, "edge threshold met")


def _skip(outcome: str, token_id: str, reason: str, probability_up: float | None = None) -> Signal:
    return Signal(False, outcome, token_id, 0.0, 0.0, 0.0, reason, probability_up=probability_up)
