from __future__ import annotations

import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class OutcomeToken:
    name: str
    token_id: str


@dataclass(frozen=True)
class Market:
    id: str
    question: str
    slug: str
    end_ts: float
    volume: float
    up: OutcomeToken
    down: OutcomeToken
    start_ts: float | None = None
    event_slug: str = ""
    liquidity: float = 0.0


class GammaClient:
    def __init__(self, base_url: str, ssl_context: ssl.SSLContext | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.ssl_context = ssl_context

    def find_btc_5m_markets(self, limit: int = 200) -> list[Market]:
        markets_by_id: dict[str, Market] = {}
        for market in self._find_btc_5m_from_keyset(limit):
            markets_by_id.setdefault(market.id or market.slug, market)
        # Keep the legacy feeds as a compatibility fallback for older API deployments.
        if not markets_by_id:
            for market in self._find_btc_5m_from_events(limit):
                markets_by_id[market.id or market.slug] = market
            for market in self._find_btc_5m_from_markets(limit):
                markets_by_id.setdefault(market.id or market.slug, market)
        return sorted(markets_by_id.values(), key=lambda market: market.end_ts)

    def find_btc_5m_candidates(self, limit: int = 300) -> list[Market]:
        candidates: dict[str, Market] = {}
        for market in self._find_from_events(limit, require_strict_match=False):
            candidates[market.id or market.slug] = market
        for market in self._find_from_markets(limit, require_strict_match=False):
            candidates.setdefault(market.id or market.slug, market)
        return sorted(candidates.values(), key=lambda market: market.end_ts)

    def _find_btc_5m_from_events(self, limit: int) -> list[Market]:
        return self._find_from_events(limit, require_strict_match=True)

    def _find_btc_5m_from_markets(self, limit: int) -> list[Market]:
        return self._find_from_markets(limit, require_strict_match=True)

    def _find_btc_5m_from_keyset(self, limit: int) -> list[Market]:
        found: list[Market] = []
        seen: set[str] = set()
        target_matches = max(1, limit)

        # Gamma documents numeric tag IDs and cursor pagination for market feeds.
        for tag_id in ("235", "1312"):
            cursor: str | None = None
            pages = 0
            while len(found) < target_matches and pages < 50:
                page_limit = 100
                params = {
                    "tag_id": tag_id,
                    "closed": "false",
                    "limit": str(page_limit),
                }
                if cursor:
                    params["after_cursor"] = cursor
                payload = self._get_json("/markets/keyset", params)
                if not isinstance(payload, dict):
                    break
                raw_markets = payload.get("markets", [])
                if not isinstance(raw_markets, list) or not raw_markets:
                    break

                for item in raw_markets:
                    if not isinstance(item, dict):
                        continue
                    market = parse_market(item)
                    if market and market_matches_btc_5m(market, require_strict_match=True):
                        key = market.id or market.slug
                        if key not in seen:
                            seen.add(key)
                            found.append(market)
                cursor = payload.get("next_cursor")
                pages += 1
                if not cursor or len(raw_markets) < page_limit:
                    break
        return found

    def _find_from_events(self, limit: int, require_strict_match: bool) -> list[Market]:
        found: list[Market] = []
        for query in ("",):
            for offset in range(0, limit, 100):
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": "100",
                    "offset": str(offset),
                }
                raw_events = self._get_json("/events", params)
                if isinstance(raw_events, dict):
                    raw_events = raw_events.get("data", [])
                if not raw_events:
                    break
                for event in raw_events:
                    found.extend(parse_event_markets(event, require_strict_match))
                if len(raw_events) < 100:
                    break
        return found

    def _find_from_markets(self, limit: int, require_strict_match: bool) -> list[Market]:
        found: list[Market] = []
        for query in ("",):
            for offset in range(0, limit, 100):
                params = {
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "limit": "100",
                    "offset": str(offset),
                }
                raw_markets = self._get_json("/markets", params)
                if isinstance(raw_markets, dict):
                    raw_markets = raw_markets.get("data", [])
                if not raw_markets:
                    break
                for item in raw_markets:
                    market = parse_market(item)
                    if market and market_matches_btc_5m(market, require_strict_match):
                        found.append(market)
                if len(raw_markets) < 100:
                    break
        return found

    def _get_json(self, path: str, params: dict[str, str]) -> Any:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{self.base_url}{path}?{query}",
            headers={"accept": "application/json", "user-agent": "btc-5m-bot/0.1"},
        )
        with urllib.request.urlopen(request, timeout=10, context=self.ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))


class ClobMarketData:
    def __init__(self, host: str) -> None:
        self.host = host
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from py_clob_client.client import ClobClient

            self._client = ClobClient(self.host)
        return self._client

    def buy_price(self, token_id: str) -> float | None:
        try:
            book = self.client.get_order_book(token_id)
            asks = getattr(book, "asks", None) or []
            prices = []
            for level in asks:
                value = getattr(level, "price", None)
                if value is None and isinstance(level, dict):
                    value = level.get("price")
                if value is not None:
                    prices.append(float(value))
            return min(prices) if prices else None
        except Exception:
            return None


class Executor:
    def __init__(self, host: str, dry_run: bool) -> None:
        self.host = host
        self.dry_run = dry_run
        self._client = None

    def buy_market(self, token_id: str, amount_usdc: float) -> dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "token_id": token_id, "amount_usdc": amount_usdc}

        client = self._authenticated_client()
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        order = MarketOrderArgs(token_id=token_id, amount=amount_usdc, side=BUY, order_type=OrderType.FOK)
        signed = client.create_market_order(order)
        return client.post_order(signed, OrderType.FOK)

    def _authenticated_client(self):
        if self._client is not None:
            return self._client

        key = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
        funder = os.environ.get("POLYMARKET_FUNDER", "").strip()
        signature_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "1"))
        if not key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY is required when dry-run is disabled")

        from py_clob_client.client import ClobClient

        self._client = ClobClient(
            self.host,
            key=key,
            chain_id=137,
            signature_type=signature_type,
            funder=funder or None,
        )
        self._client.set_api_creds(self._client.create_or_derive_api_creds())
        return self._client


def parse_market(item: dict[str, Any]) -> Market | None:
    tokens = parse_tokens(item)
    up = next((token for token in tokens if token.name.lower() in {"up", "yes"}), None)
    down = next((token for token in tokens if token.name.lower() in {"down", "no"}), None)
    if not up or not down:
        return None

    end_ts = parse_ts(first_present(item, "endDate", "end_date", "endDateIso", "end_date_iso"))
    if not end_ts:
        return None
    start_ts = infer_start_ts(item, end_ts)

    return Market(
        id=str(first_present(item, "id", "conditionId", "condition_id", "slug") or ""),
        question=str(item.get("question") or item.get("title") or ""),
        slug=str(item.get("slug") or ""),
        end_ts=end_ts,
        volume=to_float(first_present(item, "volumeNum", "volume", "volume24hr")),
        liquidity=to_float(first_present(item, "liquidityNum", "liquidity")),
        up=up,
        down=down,
        start_ts=start_ts,
        event_slug=str(first_present(item, "eventSlug", "event_slug") or ""),
    )


def parse_event_markets(event: dict[str, Any], require_strict_match: bool) -> list[Market]:
    event_slug = str(event.get("slug") or "")
    event_title = str(event.get("title") or event.get("question") or "")
    raw_markets = event.get("markets") or []
    found: list[Market] = []
    for raw_market in raw_markets:
        if not isinstance(raw_market, dict):
            continue
        enriched = dict(raw_market)
        enriched.setdefault("eventSlug", event_slug)
        if not enriched.get("question"):
            enriched["question"] = event_title
        market = parse_market(enriched)
        if market and market_matches_btc_5m(market, require_strict_match):
            found.append(market)
    return found


def parse_tokens(item: dict[str, Any]) -> list[OutcomeToken]:
    clob_ids = decode_maybe_json(item.get("clobTokenIds") or item.get("clob_token_ids") or [])
    outcomes = decode_maybe_json(item.get("outcomes") or [])
    tokens: list[OutcomeToken] = []

    for idx, token_id in enumerate(clob_ids):
        name = str(outcomes[idx] if idx < len(outcomes) else idx)
        tokens.append(OutcomeToken(name=name, token_id=str(token_id)))
    return tokens


def decode_maybe_json(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def infer_start_ts(item: dict[str, Any], end_ts: float) -> float | None:
    slug = str(first_present(item, "slug", "eventSlug", "event_slug") or "")
    for part in reversed(slug.split("-")):
        if part.isdigit() and len(part) == 10:
            return float(part)

    explicit = parse_ts(first_present(item, "startDate", "start_date", "startDateIso", "start_date_iso"))
    if explicit:
        return explicit
    return end_ts - 300


def is_btc_5m_market(market: Market) -> bool:
    return market_matches_btc_5m(market, require_strict_match=True)


def market_matches_btc_5m(market: Market, require_strict_match: bool) -> bool:
    text = f"{market.question} {market.slug} {market.event_slug}".lower()
    btc = "bitcoin" in text or "btc" in text
    up_down = (
        "up or down" in text
        or "up/down" in text
        or "updown" in text
        or (" up " in f" {text} " and " down " in f" {text} ")
    )
    five_minute = bool(re.search(r"(?:^|[-_ ])5m(?:$|[-_ ])", text)) or "5-min" in text or "5 min" in text or "5-minute" in text
    still_open = market.end_ts > datetime.now(timezone.utc).timestamp()
    if require_strict_match:
        return btc and up_down and five_minute and still_open
    return btc and still_open and (up_down or five_minute)
