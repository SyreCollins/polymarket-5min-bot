from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping

from strategy import RiskConfig


log = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    pass


class _Env:
    def __init__(self, env: Mapping[str, str]) -> None:
        self.env = env

    def raw(self, key: str, default: str) -> str:
        value = self.env.get(key)
        return default if value is None or value.strip() == "" else value.strip()

    def str(self, key: str, default: str = "") -> str:
        return self.raw(key, default)

    def bool(self, key: str, default: bool) -> bool:
        value = self.raw(key, "true" if default else "false").lower()
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
        raise ConfigError(f"{key} must be true/false, got {value!r}")

    def float(self, key: str, default: float) -> float:
        value = self.raw(key, str(default))
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"{key} must be a number, got {value!r}") from None

    def int(self, key: str, default: int) -> int:
        value = self.raw(key, str(default))
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"{key} must be an integer, got {value!r}") from None


@dataclass(frozen=True)
class Settings:
    dry_run: bool
    risk: RiskConfig
    poll_seconds: float = 2.0
    market_refresh_seconds: float = 30.0
    trade_log_path: str = "trade_log.jsonl"
    risk_state_path: str = "state.db"
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    private_key: str = ""
    funder: str = ""
    signature_type: int = 1

    enable_coinbase: bool = False
    enable_okx: bool = False
    enable_binance: bool = True
    coinbase_ws_url: str = "wss://advanced-trade-ws.coinbase.com"
    okx_ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"
    btc_symbol: str = "btcusdt"
    price_max_age_seconds: float = 10.0
    price_max_divergence_bps: float = 50.0
    coinbase_stale_after_seconds: float | None = None
    price_reconnect_min_seconds: float = 1.0
    price_reconnect_max_seconds: float = 30.0
    price_silence_reconnect_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = _Env(os.environ if env is None else env)

        # Live trading needs two explicit switches so a single typo cannot send real orders.
        dry_run_flag = e.bool("POLYMARKET_DRY_RUN", True)
        live_flag = e.bool("LIVE_TRADING", False)
        dry_run = dry_run_flag or not live_flag
        if dry_run_flag != (not live_flag):
            log.warning(
                "POLYMARKET_DRY_RUN=%s and LIVE_TRADING=%s disagree; running in DRY-RUN mode. "
                "Set POLYMARKET_DRY_RUN=false and LIVE_TRADING=true to trade live.",
                dry_run_flag,
                live_flag,
            )

        risk = RiskConfig(
            trade_usdc=e.float("TRADE_USDC", 5.0),
            max_trades_per_day=e.int("MAX_TRADES_PER_DAY", 20),
            max_trades_per_hour=e.int("MAX_TRADES_PER_HOUR", 10),
            min_edge=e.float("MIN_EDGE", 0.07),
            max_edge=e.float("MAX_EDGE", 0.25),
            min_edge_after_costs=e.float("MIN_EDGE_AFTER_COSTS", 0.08),
            execution_cost=e.float("EXECUTION_COST", 0.02),
            min_seconds_left=e.float("MIN_SECONDS_LEFT", 8.0),
            max_seconds_left=e.float("MAX_SECONDS_LEFT", 45.0),
            min_market_volume=e.float("MIN_MARKET_VOLUME", 1000.0),
            min_liquidity=e.float("MIN_LIQUIDITY", 1000.0),
            max_buy_price=e.float("MAX_BUY_PRICE", 0.92),
            max_spread=e.float("MAX_SPREAD", 0.08),
            max_daily_loss=e.float("MAX_DAILY_LOSS", 50.0),
            max_position_size=e.float("MAX_POSITION_SIZE", 25.0),
            max_open_positions=e.int("MAX_OPEN_POSITIONS", 2),
            max_consecutive_losses=e.int("MAX_CONSECUTIVE_LOSSES", 3),
            min_price_history_seconds=e.float("MIN_PRICE_HISTORY_SECONDS", 60.0),
            kill_switch=e.bool("KILL_SWITCH", False),
        )

        coinbase_stale = e.raw("COINBASE_STALE_AFTER_SECONDS", "")
        settings = cls(
            dry_run=dry_run,
            risk=risk,
            poll_seconds=e.float("POLL_SECONDS", 2.0),
            market_refresh_seconds=e.float("MARKET_REFRESH_SECONDS", 30.0),
            trade_log_path=e.str("TRADE_LOG_PATH", "trade_log.jsonl"),
            risk_state_path=e.str("RISK_STATE_PATH", "state.db"),
            gamma_url=e.str("GAMMA_URL", "https://gamma-api.polymarket.com"),
            clob_url=e.str("CLOB_URL", "https://clob.polymarket.com"),
            private_key=e.str("POLYMARKET_PRIVATE_KEY"),
            funder=e.str("POLYMARKET_FUNDER"),
            signature_type=e.int("POLYMARKET_SIGNATURE_TYPE", 1),
            enable_coinbase=e.bool("ENABLE_COINBASE_FEED", False),
            enable_okx=e.bool("ENABLE_OKX_FALLBACK", False),
            enable_binance=e.bool("ENABLE_BINANCE_FALLBACK", True),
            coinbase_ws_url=e.str("COINBASE_WS_URL", "wss://advanced-trade-ws.coinbase.com"),
            okx_ws_url=e.str("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public"),
            binance_ws_url=e.str("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws"),
            btc_symbol=e.str("BTC_SYMBOL", "btcusdt").lower(),
            price_max_age_seconds=e.float("PRICE_MAX_AGE_SECONDS", 10.0),
            price_max_divergence_bps=e.float("PRICE_MAX_DIVERGENCE_BPS", 50.0),
            coinbase_stale_after_seconds=e.float("COINBASE_STALE_AFTER_SECONDS", 0.0) if coinbase_stale else None,
            price_reconnect_min_seconds=e.float("PRICE_RECONNECT_MIN_SECONDS", 1.0),
            price_reconnect_max_seconds=e.float("PRICE_RECONNECT_MAX_SECONDS", 30.0),
            price_silence_reconnect_seconds=e.float("PRICE_SILENCE_RECONNECT_SECONDS", 30.0),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        r = self.risk
        errors: list[str] = []
        if r.trade_usdc <= 0:
            errors.append("TRADE_USDC must be > 0")
        if r.max_position_size <= 0:
            errors.append("MAX_POSITION_SIZE must be > 0")
        if not 0 < r.max_buy_price < 1:
            errors.append("MAX_BUY_PRICE must be between 0 and 1")
        if r.min_seconds_left < 0 or r.min_seconds_left >= r.max_seconds_left:
            errors.append("MIN_SECONDS_LEFT must be >= 0 and below MAX_SECONDS_LEFT")
        if r.min_edge < 0 or r.min_edge >= r.max_edge:
            errors.append("MIN_EDGE must be >= 0 and below MAX_EDGE")
        if r.execution_cost < 0:
            errors.append("EXECUTION_COST must be >= 0")
        if self.poll_seconds <= 0:
            errors.append("POLL_SECONDS must be > 0")
        if not self.risk_state_path.strip():
            errors.append("RISK_STATE_PATH must not be empty")
        if self.price_max_age_seconds <= 0:
            errors.append("PRICE_MAX_AGE_SECONDS must be > 0")
        if not (self.enable_coinbase or self.enable_okx or self.enable_binance):
            errors.append("enable at least one BTC price feed (ENABLE_COINBASE_FEED / ENABLE_OKX_FALLBACK / ENABLE_BINANCE_FALLBACK)")
        if self.signature_type not in (0, 1, 2):
            errors.append("POLYMARKET_SIGNATURE_TYPE must be 0 (EOA), 1 (email/Magic proxy) or 2 (browser wallet proxy)")
        if not self.dry_run and not self.private_key:
            errors.append("POLYMARKET_PRIVATE_KEY is required for live trading")
        if errors:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))

    @property
    def stake_usdc(self) -> float:
        return min(self.risk.trade_usdc, self.risk.max_position_size)
