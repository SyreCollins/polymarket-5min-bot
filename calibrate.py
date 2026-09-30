#!/usr/bin/env python3
"""
Calibration check for the Polymarket 5-minute bot's probability model.

Reads decision logs (decision_log.jsonl or legacy trade_log.jsonl),
fetches actual resolutions from Polymarket Gamma API, and prints
calibration tables binned by z-score (sufficient statistic) and by predicted probability.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from ssl import create_default_context
from typing import Any

try:
    import certifi
    SSL_CTX = create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = create_default_context()

GAMMA_URL = "https://gamma-api.polymarket.com"
DECISION_LOG_PATH = Path("decision_log.jsonl")
LEGACY_LOG_PATH = Path("paper_final.jsonl")  # fallback


@dataclass
class Decision:
    market_id: str
    market_slug: str
    timestamp: str
    seconds_left: float
    open_price: float
    current_price: float
    sigma: float
    z_score: float
    model_probability_up: float
    trade_decision: str
    trade_outcome: str | None


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"accept": "application/json", "user-agent": "calibrate/0.1"},
    )
    with urllib.request.urlopen(request, timeout=15, context=SSL_CTX) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_resolution(market_id: str) -> str | None:
    """Return 'UP'/'DOWN' if market has resolved, else None."""
    url = f"{GAMMA_URL}/markets/{urllib.parse.quote(market_id, safe='')}"
    try:
        payload = fetch_json(url)
        if not isinstance(payload, dict):
            return None
        if not payload.get("closed"):
            return None
        outcomes = payload.get("outcomes") or []
        prices = payload.get("outcomePrices") or []
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
        for name, price in zip(outcomes, prices):
            try:
                if float(price) >= 0.99:
                    label = str(name).strip().lower()
                    if label in {"up", "yes"}:
                        return "UP"
                    if label in {"down", "no"}:
                        return "DOWN"
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    return None


def parse_decision_log(path: Path) -> list[Decision]:
    """Parse decision events from JSONL log (new decision_log.jsonl format)."""
    decisions: list[Decision] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # New format has these fields
            if "z_score" not in record:
                continue  # skip legacy format records

            prob_up = record.get("model_probability_up")
            if prob_up is None:
                continue

            decisions.append(Decision(
                market_id=str(record.get("market_id", "")),
                market_slug=str(record.get("market_slug", "")),
                timestamp=record.get("timestamp", ""),
                seconds_left=float(record.get("seconds_left", 0.0)),
                open_price=float(record.get("open_price", 0.0)),
                current_price=float(record.get("current_price", 0.0)),
                sigma=float(record.get("sigma", 0.0)),
                z_score=float(record.get("z_score", 0.0)),
                model_probability_up=float(prob_up),
                trade_decision=record.get("trade_decision", ""),
                trade_outcome=record.get("trade_outcome"),
            ))
    return decisions


def parse_legacy_log(path: Path) -> list[Decision]:
    """Parse legacy trade_log.jsonl / paper_final.jsonl format."""
    decisions: list[Decision] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("event"):
                continue

            prob_up = record.get("model_probability_up")
            if prob_up is None:
                continue

            # Recompute z_score from legacy fields if possible
            open_price = record.get("btc_open_price", 0.0)
            current_price = record.get("btc_price", 0.0)
            seconds_left = record.get("time_remaining", 0.0)
            # Legacy log doesn't have sigma, so we can't compute exact z_score
            # Use a proxy: (current - open) / open as a rough distance measure
            z_score = 0.0
            if open_price > 0:
                z_score = (current_price - open_price) / open_price

            decisions.append(Decision(
                market_id=str(record.get("market_id", "")),
                market_slug="",
                timestamp=record.get("timestamp", ""),
                seconds_left=float(seconds_left),
                open_price=float(open_price),
                current_price=float(current_price),
                sigma=0.0,
                z_score=z_score,
                model_probability_up=float(prob_up),
                trade_decision=record.get("trade_decision", ""),
                trade_outcome=None,
            ))
    return decisions


def pick_best_decision_per_market(decisions: list[Decision]) -> dict[str, Decision]:
    """
    For each market, pick the most relevant decision for calibration.
    Prefer: TRADE decisions > last decision before window close (smallest seconds_left).
    """
    by_market: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        by_market[d.market_id].append(d)

    best: dict[str, Decision] = {}
    for market_id, items in by_market.items():
        trades = [d for d in items if d.trade_decision == "TRADE"]
        if trades:
            best[market_id] = min(trades, key=lambda d: d.seconds_left)
        else:
            best[market_id] = min(items, key=lambda d: d.seconds_left)
    return best


def bucket_z_score(z: float) -> str:
    """Return bucket label for z-score (standard normal quantile)."""
    # Bin edges at -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3
    edges = [-float("inf"), -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf")]
    for i in range(len(edges) - 1):
        if edges[i] <= z < edges[i + 1]:
            lo = "-inf" if edges[i] == -float("inf") else f"{edges[i]:.1f}"
            hi = "inf" if edges[i + 1] == float("inf") else f"{edges[i + 1]:.1f}"
            return f"[{lo}, {hi})"
    return "[3.0, inf)"


def bucket_probability(p: float) -> str:
    """Return bucket label like '0.40-0.50' for probability p in [0,1]."""
    if p >= 1.0:
        return "0.90-1.00"
    if p <= 0.0:
        return "0.00-0.10"
    idx = int(p * 10)
    lo = idx / 10.0
    hi = (idx + 1) / 10.0
    return f"{lo:.2f}-{hi:.2f}"


def print_calibration_table(buckets: dict[str, list[tuple[float, bool]]], title: str, x_label: str) -> int:
    """Print calibration table and return total sample count."""
    print("\n" + "=" * 70)
    print(f"CALIBRATION TABLE: {title}")
    print("=" * 70)
    print(f"{x_label:<18} | {'Count':>6} | {'Avg Prob':>10} | {'Avg Z':>8} | {'Actual Win Rate':>15}")
    print("-" * 70)

    total_samples = 0
    for bucket in sorted(buckets.keys()):
        samples = buckets[bucket]
        count = len(samples)
        if count == 0:
            continue
        avg_pred = sum(p for p, _ in samples) / count
        avg_z = sum(z for z, _ in samples) / count if samples else 0.0
        win_rate = sum(1 for _, w in samples if w) / count
        total_samples += count
        print(f"{bucket:<18} | {count:>6} | {avg_pred:>10.3f} | {avg_z:>8.3f} | {win_rate:>14.1%}")

    print("-" * 70)
    print(f"{'TOTAL':<18} | {total_samples:>6}")
    return total_samples


def main() -> int:
    # Try new decision log first, fall back to legacy
    if DECISION_LOG_PATH.exists():
        log_path = DECISION_LOG_PATH
        print(f"Reading decisions from {DECISION_LOG_PATH} (new format)...")
        decisions = parse_decision_log(DECISION_LOG_PATH)
    elif LEGACY_LOG_PATH.exists():
        log_path = LEGACY_LOG_PATH
        print(f"Reading decisions from {LEGACY_LOG_PATH} (legacy format)...")
        decisions = parse_legacy_log(LEGACY_LOG_PATH)
    else:
        print(f"Decision log not found: {DECISION_LOG_PATH} or {LEGACY_LOG_PATH}")
        print("Run the bot (dry-run) to generate logs.")
        return 1

    if not decisions:
        print("No decision records found.")
        return 1

    print(f"Found {len(decisions)} decision records across {len(set(d.market_id for d in decisions))} markets.")

    best_decisions = pick_best_decision_per_market(decisions)
    print(f"Selected {len(best_decisions)} unique markets for calibration.")

    # Fetch resolutions
    print("Fetching resolutions from Gamma API...")
    resolutions: dict[str, str] = {}
    for i, (market_id, decision) in enumerate(best_decisions.items(), 1):
        winner = fetch_resolution(market_id)
        if winner:
            resolutions[market_id] = winner
        if i % 10 == 0 or i == len(best_decisions):
            print(f"  {i}/{len(best_decisions)} markets checked...")

    print(f"Resolved {len(resolutions)} out of {len(best_decisions)} markets.")

    # Bucket by z-score and compute calibration
    z_buckets: dict[str, list[tuple[float, bool, float]]] = defaultdict(list)  # (z, actual_up, prob)
    p_buckets: dict[str, list[tuple[float, bool]]] = defaultdict(list)         # (prob, actual_up)

    for market_id, decision in best_decisions.items():
        winner = resolutions.get(market_id)
        if winner is None:
            continue
        actual_up = (winner == "UP")
        z = decision.z_score
        p = decision.model_probability_up
        z_bucket = bucket_z_score(z)
        p_bucket = bucket_probability(p)
        z_buckets[z_bucket].append((z, actual_up, p))
        p_buckets[p_bucket].append((p, actual_up))

    # Convert z_buckets to format for printing (need avg prob per z bucket)
    z_buckets_print: dict[str, list[tuple[float, bool]]] = {}
    for bucket, items in z_buckets.items():
        z_buckets_print[bucket] = [(z, actual) for z, actual, _ in items]

    total_z = print_calibration_table(z_buckets_print, "by Z-Score (sufficient statistic)", "Z-Score Bin")
    total_p = print_calibration_table(p_buckets, "by Predicted Probability", "Probability Bin")

    total_samples = max(total_z, total_p)

    if total_samples < 30:
        print("\nWARNING: Fewer than 30 resolved samples -- calibration unreliable.")
    elif total_samples < 100:
        print("\nCAUTION: Fewer than 100 resolved samples -- treat calibration as preliminary.")
    else:
        print("\nSufficient samples for meaningful calibration assessment.")

    # Quick diagnostic: show z-score vs probability mapping
    print("\n" + "=" * 70)
    print("Z-SCORE -> PROBABILITY MAPPING (model: p = 1 - Phi(z))")
    print("=" * 70)
    print(f"{'Z-Score':>10} | {'Model Prob':>12} | {'Phi(z)':>10}")
    print("-" * 40)
    for z in [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        p = 1.0 - phi
        print(f"{z:>10.1f} | {p:>12.4f} | {phi:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())