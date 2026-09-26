from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import erf, exp, log, sqrt
from pathlib import Path
from statistics import pstdev
from time import time
from typing import Any, Iterable


@dataclass(frozen=True)
class RiskConfig:
    trade_usdc: float = 5.0
    max_trades_per_day: int = 20
    min_edge: float = 0.07
    min_seconds_left: float = 8.0
    max_seconds_left: float = 45.0
    min_market_volume: float = 1000.0
    max_buy_price: float = 0.92
    max_trade_size: float = 5.0
    bankroll_fraction: float = 0.01
    max_daily_loss: float = 50.0
    max_position_size: float = 25.0
    max_consecutive_losses: int = 3
    max_trades_per_hour: int = 10
    min_liquidity: float = 1000.0
    max_spread: float = 0.08
    min_time_remaining: float = 30.0
    max_time_remaining: float = 300.0
    min_edge_after_costs: float = 0.08
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
        history_span = max(0.0, sample_end - samples[0][0]) if samples and sample_end is not None else 0.0

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
        model_probability_up: float,
        model_probability_down: float,
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
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class PaperTrade:
    market_id: str
    outcome: str
    entry_price: float
    stake: float
    model_probability: float
    edge: float
    resolved_outcome: str | None = None
    pnl: float | None = None


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
    ) -> PaperTrade:
        trade = PaperTrade(
            market_id=market_id,
            outcome=outcome.upper(),
            entry_price=entry_price,
            stake=stake,
            model_probability=model_probability,
            edge=edge,
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
        shares = trade.stake / trade.entry_price if trade.entry_price > 0 else 0.0
        payout = shares if trade.outcome == normalized_outcome else 0.0
        trade.pnl = payout - trade.stake
        return trade

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
    daily_loss: float = 0.0
    trades_today: int = 0
    trades_last_hour: int = 0
    open_positions: int = 0
    consecutive_losses: int = 0
    last_hour_window: float = field(default_factory=time)

    def can_trade(self, market_id: str | None = None, *, spread: float | None = None, liquidity: float | None = None) -> bool:
        if self.risk.kill_switch:
            return False
        if self.daily_loss >= self.risk.max_daily_loss:
            return False
        if self.trades_today >= self.risk.max_trades_per_day:
            return False
        if self.open_positions >= self.risk.max_open_positions:
            return False
        if self.consecutive_losses >= self.risk.max_consecutive_losses:
            return False
        if self.trades_last_hour >= self.risk.max_trades_per_hour:
            return False
        if spread is not None and spread > self.risk.max_spread:
            return False
        if liquidity is not None and liquidity < self.risk.min_liquidity:
            return False
        return True

    def record_trade(self, trade_size: float, outcome: str, entry_price: float, exit_price: float, pnl: float | None = None) -> None:
        self.trades_today += 1
        self.trades_last_hour += 1
        loss_amount = abs(float(trade_size)) if trade_size else (abs(float(pnl)) if pnl is not None else 0.0)
        if pnl is not None:
            loss_amount = abs(float(pnl)) if float(pnl) < 0 else 0.0
        self.daily_loss += loss_amount
        if pnl is not None and float(pnl) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_hourly_window(self) -> None:
        self.trades_last_hour = 0
        self.last_hour_window = time()


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


def probability_up(
    open_price: float,
    current_price: float,
    seconds_left: float,
    recent_prices: Iterable[float],
) -> float:
    if open_price <= 0 or current_price <= 0:
        return 0.5
    if seconds_left <= 0:
        return 1.0 if current_price >= open_price else 0.0

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
) -> Signal:
    if trades_today >= risk.max_trades_per_day:
        return _skip("UP", up_token_id, "daily trade limit reached")

    if market_volume < risk.min_market_volume:
        return _skip("UP", up_token_id, "market volume below threshold")

    if seconds_left < risk.min_seconds_left or seconds_left > risk.max_seconds_left:
        return _skip("UP", up_token_id, "outside execution window")

    samples = list(recent_samples)
    features = FeatureEngine().compute_features(
        open_price=open_price,
        current_price=current_price,
        recent_prices=recent_prices,
        recent_samples=samples,
        current_timestamp=current_timestamp,
        seconds_left=seconds_left,
        market_volume=market_volume,
        up_ask=up_ask,
        down_ask=down_ask,
    )
    if features["price_history_seconds"] < risk.min_price_history_seconds:
        return _skip("UP", up_token_id, "insufficient BTC price history")

    up_probability = ProbabilityModel(min_probability=risk.min_probability, max_probability=risk.max_probability).estimate_probability(features)
    candidates: list[Signal] = []

    if up_ask is not None:
        candidates.append(_candidate("UP", up_token_id, up_probability, up_ask, risk))
    if down_ask is not None:
        candidates.append(_candidate("DOWN", down_token_id, 1.0 - up_probability, down_ask, risk))

    tradable = [candidate for candidate in candidates if candidate.should_trade]
    if not tradable:
        best = max(candidates, key=lambda item: item.edge, default=None)
        if best:
            return Signal(False, best.outcome, best.token_id, best.model_probability, best.ask_price, best.edge, best.reason)
        return _skip("UP", up_token_id, "missing CLOB ask prices")

    return max(tradable, key=lambda item: item.edge)


def _candidate(outcome: str, token_id: str, probability: float, ask: float, risk: RiskConfig) -> Signal:
    edge = probability - ask
    if ask > risk.max_buy_price:
        return Signal(False, outcome, token_id, probability, ask, edge, "ask above max buy price")
    if edge < risk.min_edge:
        return Signal(False, outcome, token_id, probability, ask, edge, "edge below threshold")
    return Signal(True, outcome, token_id, probability, ask, edge, "edge threshold met")


def _skip(outcome: str, token_id: str, reason: str) -> Signal:
    return Signal(False, outcome, token_id, 0.0, 0.0, 0.0, reason)
