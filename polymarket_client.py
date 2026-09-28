from __future__ import annotations

import json
import logging
import math
import os
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from time import time
from typing import Any


log = logging.getLogger(__name__)
_clob_timeout_configured = False


def _configure_clob_timeout() -> None:
    global _clob_timeout_configured
    if _clob_timeout_configured:
        return
    from py_clob_client.http_helpers import helpers

    client = getattr(helpers, "_http_client", None)
    if client is None or not hasattr(client, "timeout"):
        raise RuntimeError("installed CLOB client does not expose a configurable HTTP timeout")
    client.timeout = 10.0
    _clob_timeout_configured = True

WINDOW_SECONDS = 300
BTC_5M_SLUG_PREFIX = "btc-updown-5m-"


def default_ssl_context() -> ssl.SSLContext:
    # python.org macOS builds ship without a CA bundle; certifi keeps verification on everywhere.
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def window_start(timestamp: float) -> int:
    return int(timestamp // WINDOW_SECONDS * WINDOW_SECONDS)


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
        self.ssl_context = ssl_context or default_ssl_context()

    def find_btc_5m_markets(self, limit: int = 200, now: float | None = None) -> list[Market]:
        # BTC 5m markets use deterministic slugs, so the current and next window cost two requests.
        now = time() if now is None else now
        by_slug: list[Market] = []
        for start in (window_start(now), window_start(now) + WINDOW_SECONDS):
            try:
                market = self.find_market_by_slug(f"{BTC_5M_SLUG_PREFIX}{start}")
            except (OSError, ValueError) as exc:
                log.warning("gamma slug lookup failed for window %s: %s", start, exc)
                continue
            if market and market_matches_btc_5m(market, require_strict_match=True):
                by_slug.append(market)
        if by_slug:
            return sorted(by_slug, key=lambda market: market.end_ts)
        return self.scan_btc_5m_markets(limit)

    def scan_btc_5m_markets(self, limit: int = 200) -> list[Market]:
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

    def find_market_by_slug(self, slug: str) -> Market | None:
        payload = self._get_json("/markets", {"slug": slug})
        items = payload.get("data", []) if isinstance(payload, dict) else payload
        for item in items or []:
            if isinstance(item, dict):
                market = parse_market(item)
                if market:
                    return market
        return None

    def fetch_resolution(self, market_id: str) -> str | None:
        """Return "UP"/"DOWN" once the market has resolved, else None."""
        payload = self._get_json(f"/markets/{urllib.parse.quote(market_id, safe='')}")
        return parse_resolution(payload) if isinstance(payload, dict) else None

    def _get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"accept": "application/json", "user-agent": "btc-5m-bot/0.1"},
        )
        with urllib.request.urlopen(request, timeout=10, context=self.ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class BookQuote:
    best_ask: float | None
    best_bid: float | None
    # Average and worst price to buy the requested USDC amount; None when the book is too thin.
    avg_fill_price: float | None = None
    worst_fill_price: float | None = None

    @property
    def spread(self) -> float | None:
        if self.best_ask is None or self.best_bid is None:
            return None
        return self.best_ask - self.best_bid


def _book_levels(levels: Any) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    if not isinstance(levels, (list, tuple)):
        return parsed
    for level in levels:
        price = getattr(level, "price", None)
        size = getattr(level, "size", None)
        if isinstance(level, dict):
            price = level.get("price", price)
            size = level.get("size", size)
        if price is None:
            continue
        try:
            parsed.append((float(price), float(size) if size is not None else 0.0))
        except (TypeError, ValueError):
            continue
    return parsed


def quote_from_book(book: Any, amount_usdc: float) -> BookQuote:
    asks = sorted(_book_levels(getattr(book, "asks", None)))
    bids = _book_levels(getattr(book, "bids", None))
    best_ask = asks[0][0] if asks else None
    best_bid = max((price for price, _ in bids), default=None)

    remaining = amount_usdc
    shares = 0.0
    worst = None
    for price, size in asks:
        if remaining <= 1e-9:
            break
        if price <= 0 or size <= 0:
            continue
        spend = min(remaining, price * size)
        shares += spend / price
        remaining -= spend
        worst = price
    if amount_usdc <= 0 or remaining > 1e-9 or shares <= 0:
        return BookQuote(best_ask, best_bid)
    return BookQuote(best_ask, best_bid, amount_usdc / shares, worst)


class ClobMarketData:
    def __init__(self, host: str) -> None:
        self.host = host
        self._client = None

    @property
    def client(self):
        if self._client is None:
            _configure_clob_timeout()
            from py_clob_client.client import ClobClient

            self._client = ClobClient(self.host)
        return self._client

    def quote(self, token_id: str, amount_usdc: float) -> BookQuote | None:
        try:
            return quote_from_book(self.client.get_order_book(token_id), amount_usdc)
        except Exception as exc:
            log.warning("order book fetch failed for token %s… (%s)", token_id[:12], type(exc).__name__)
            return None

    def buy_price(self, token_id: str) -> float | None:
        quote = self.quote(token_id, 0.0)
        return quote.best_ask if quote else None


@dataclass(frozen=True)
class OrderResult:
    filled: bool
    shares: float | None
    cost_usdc: float | None
    raw: Any
    error: str = ""


def parse_order_response(response: Any) -> OrderResult:
    if not isinstance(response, dict):
        return OrderResult(False, None, None, response, f"unexpected response: {response!r}")
    error = str(response.get("errorMsg") or response.get("error") or "")
    status = str(response.get("status") or "").lower()
    filled = bool(response.get("success")) and not error and status in ("", "matched")
    # For a BUY, makingAmount is USDC paid and takingAmount is shares received.
    shares = to_float(response.get("takingAmount")) or None
    cost = to_float(response.get("makingAmount")) or None
    return OrderResult(filled, shares, cost, response, error or ("" if filled else f"status={status or 'unknown'}"))


def floor_to_tick(price: float, tick: float = 0.01) -> float:
    return math.floor(price / tick + 1e-9) * tick


class Executor:
    def __init__(self, host: str, dry_run: bool, *, private_key: str = "", funder: str = "", signature_type: int = 1) -> None:
        self.host = host
        self.dry_run = dry_run
        self.private_key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
        self.funder = funder or os.environ.get("POLYMARKET_FUNDER", "").strip()
        self.signature_type = signature_type
        self._client = None

    def ensure_ready(self) -> None:
        """Authenticate up front so bad credentials fail at startup, not on the first trade."""
        if not self.dry_run:
            self._authenticated_client()

    def buy_market(self, token_id: str, amount_usdc: float, max_price: float | None = None) -> OrderResult:
        if self.dry_run:
            raw = {"dry_run": True, "token_id": token_id, "amount_usdc": amount_usdc, "max_price": max_price}
            return OrderResult(True, None, amount_usdc, raw)

        client = self._authenticated_client()
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        # A positive price caps the worst fill; without it the client sweeps the book at any price.
        price = floor_to_tick(max_price) if max_price is not None else 0.0
        if max_price is not None and price <= 0:
            return OrderResult(False, None, None, None, f"limit price {max_price:.4f} rounds to zero")
        order = MarketOrderArgs(token_id=token_id, amount=amount_usdc, side=BUY, price=price, order_type=OrderType.FOK)
        signed = client.create_market_order(order)
        return parse_order_response(client.post_order(signed, OrderType.FOK))

    def _authenticated_client(self):
        if self._client is not None:
            return self._client

        if not self.private_key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY is required when dry-run is disabled")

        _configure_clob_timeout()
        from py_clob_client.client import ClobClient

        client = ClobClient(
            self.host,
            key=self.private_key,
            chain_id=137,
            signature_type=self.signature_type,
            funder=self.funder or None,
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        self._client = client
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


def parse_resolution(item: dict[str, Any]) -> str | None:
    if not item.get("closed"):
        return None
    outcomes = decode_maybe_json(item.get("outcomes") or [])
    prices = decode_maybe_json(item.get("outcomePrices") or [])
    for name, price in zip(outcomes, prices):
        if to_float(price) >= 0.99:
            label = str(name).strip().lower()
            if label in {"up", "yes"}:
                return "UP"
            if label in {"down", "no"}:
                return "DOWN"
    return None


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

    # startDate is when the market was listed (often a day early), not when the window opens.
    explicit = parse_ts(first_present(item, "eventStartTime", "event_start_time"))
    if explicit:
        return explicit
    return end_ts - WINDOW_SECONDS


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
