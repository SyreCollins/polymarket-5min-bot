import asyncio
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from unittest.mock import Mock, patch

from btc_5m_bot import PriceState, TradingBot
from polymarket_client import (
    BookQuote,
    GammaClient,
    Market,
    OrderResult,
    OutcomeToken,
    floor_to_tick,
    infer_start_ts,
    parse_order_response,
    parse_resolution,
    quote_from_book,
)
from settings import ConfigError, Settings
from strategy import UNFILLABLE, RiskConfig, RiskManager, build_signal, per_second_volatility


def rising_samples(open_price=65000.0, move=0.0015, seconds=300, ticks_per_second=1):
    step = 1.0 / ticks_per_second
    count = int(seconds * ticks_per_second)
    return [(i * step, open_price * (1 + move * i / count)) for i in range(count + 1)]


def signal_for(samples, up_ask, down_ask, seconds_left=30.0, risk=None, **kwargs):
    return build_signal(
        open_price=samples[0][1],
        current_price=samples[-1][1],
        seconds_left=seconds_left,
        recent_prices=[price for _, price in samples],
        recent_samples=samples,
        current_timestamp=samples[-1][0],
        up_token_id="up",
        down_token_id="down",
        up_ask=up_ask,
        down_ask=down_ask,
        market_volume=5000.0,
        trades_today=0,
        risk=risk or RiskConfig(),
        **kwargs,
    )


class SignalTests(unittest.TestCase):
    def test_does_not_buy_the_near_certain_loser(self):
        # BTC is +0.15% over the open with 30s left; the market prices UP near certainty.
        signal = signal_for(rising_samples(), up_ask=0.97, down_ask=0.04)
        self.assertFalse(signal.should_trade)
        self.assertGreater(signal.probability_up, 0.95)

    def test_buys_side_the_market_underprices(self):
        signal = signal_for(rising_samples(), up_ask=0.80, down_ask=0.22)
        self.assertTrue(signal.should_trade)
        self.assertEqual(signal.outcome, "UP")
        self.assertIsNotNone(signal.limit_price)
        self.assertLessEqual(signal.limit_price, RiskConfig().max_buy_price)

    def test_thin_book_blocks_trade(self):
        signal = signal_for(rising_samples(), up_ask=0.80, down_ask=0.22, up_fill_price=UNFILLABLE)
        self.assertFalse(signal.should_trade)
        self.assertEqual(signal.reason, "order book too thin for stake")

    def test_slippage_and_costs_are_charged_against_edge(self):
        risk = RiskConfig(execution_cost=0.02)
        signal = signal_for(rising_samples(), up_ask=0.80, down_ask=0.22, up_fill_price=0.90, risk=risk)
        self.assertFalse(signal.should_trade)
        self.assertEqual(signal.reason, "edge after costs below threshold")

    def test_volatility_is_per_second_regardless_of_tick_rate(self):
        slow = per_second_volatility(rising_samples(ticks_per_second=1))
        fast = per_second_volatility(rising_samples(ticks_per_second=20))
        self.assertAlmostEqual(slow, fast, places=6)


class OrderBookTests(unittest.TestCase):
    def book(self, asks, bids=()):
        # The CLOB lists asks worst-first and bids worst-first.
        return Mock(
            asks=[{"price": p, "size": s} for p, s in asks],
            bids=[{"price": p, "size": s} for p, s in bids],
        )

    def test_quote_walks_levels_for_average_fill(self):
        quote = quote_from_book(self.book([("0.60", "100"), ("0.50", "4")], [("0.40", "10"), ("0.45", "10")]), 5.0)
        self.assertEqual(quote.best_ask, 0.50)
        self.assertEqual(quote.best_bid, 0.45)
        self.assertAlmostEqual(quote.spread, 0.05)
        # $2 buys 4 shares at 0.50, remaining $3 buys 5 shares at 0.60.
        self.assertAlmostEqual(quote.avg_fill_price, 5.0 / 9.0)
        self.assertEqual(quote.worst_fill_price, 0.60)

    def test_quote_without_enough_depth_has_no_fill(self):
        quote = quote_from_book(self.book([("0.50", "2")]), 5.0)
        self.assertEqual(quote.best_ask, 0.50)
        self.assertIsNone(quote.avg_fill_price)

    def test_limit_price_floors_to_tick(self):
        self.assertAlmostEqual(floor_to_tick(0.8299), 0.82)
        self.assertAlmostEqual(floor_to_tick(0.83), 0.83)

    def test_order_response_parsing(self):
        ok = parse_order_response({"success": True, "status": "matched", "makingAmount": "5", "takingAmount": "6.25", "errorMsg": ""})
        self.assertTrue(ok.filled)
        self.assertEqual((ok.shares, ok.cost_usdc), (6.25, 5.0))
        rejected = parse_order_response({"success": False, "errorMsg": "not enough balance"})
        self.assertFalse(rejected.filled)
        self.assertEqual(rejected.error, "not enough balance")


class MarketDataTests(unittest.TestCase):
    def test_resolution_requires_closed_market(self):
        self.assertIsNone(parse_resolution({"closed": False, "outcomes": '["Up","Down"]', "outcomePrices": '["0.995","0.005"]'}))
        self.assertEqual(parse_resolution({"closed": True, "outcomes": '["Up","Down"]', "outcomePrices": '["0","1"]'}), "DOWN")
        self.assertIsNone(parse_resolution({"closed": True, "outcomes": '["Up","Down"]', "outcomePrices": '["0.5","0.5"]'}))

    def test_start_uses_event_start_not_listing_date(self):
        item = {"slug": "bitcoin-up-or-down", "startDate": "2026-09-27T15:53:15Z", "eventStartTime": "2026-09-28T15:45:00Z"}
        self.assertEqual(infer_start_ts(item, 1790610600.0), 1790610300.0)
        self.assertEqual(infer_start_ts({"startDate": "2026-09-27T15:53:15Z"}, 1790610600.0), 1790610300.0)

    def test_gamma_looks_up_current_window_by_slug(self):
        client = GammaClient("https://gamma-api.polymarket.com")
        client._get_json = Mock(return_value=[{
            "id": "m1",
            "slug": "btc-updown-5m-1790610300",
            "question": "Bitcoin Up or Down",
            "endDate": "2099-01-01T00:00:00Z",
            "outcomes": '["Up", "Down"]',
            "clobTokenIds": '["u", "d"]',
        }])
        markets = client.find_btc_5m_markets(now=1790610400.0)
        self.assertEqual(markets[0].id, "m1")
        first_call = client._get_json.call_args_list[0]
        self.assertEqual(first_call.args, ("/markets", {"slug": "btc-updown-5m-1790610300"}))


class PriceStateTests(unittest.TestCase):
    def test_busy_feed_keeps_window_open_sample(self):
        state = PriceState()
        for i in range(300 * 50):  # 50 ticks/s for five minutes
            ts = 1000.0 + i / 50
            state.update_source("binance", 65000.0 + i * 0.001, ts, ts)
        self.assertAlmostEqual(state.market_open_price(1000.0, source="binance"), 65000.0)


class RiskManagerTests(unittest.TestCase):
    def setUp(self):
        self.now = 1790610000.0
        self.manager = RiskManager(
            RiskConfig(max_daily_loss=10.0, max_consecutive_losses=3, max_open_positions=5, max_trades_per_hour=100),
            clock=lambda: self.now,
        )

    def test_daily_loss_blocks_then_resets_next_day(self):
        self.manager.record_trade(5.0, "UP", 0.5, 0.0, -6.0)
        self.assertIsNone(self.manager.block_reason())
        self.manager.record_trade(5.0, "UP", 0.5, 0.0, -5.0)
        self.assertIn("daily loss", self.manager.block_reason())
        self.now += 86400
        self.assertIsNone(self.manager.block_reason())

    def test_consecutive_losses_block(self):
        for _ in range(3):
            self.manager.record_trade(1.0, "UP", 0.5, 0.0, -1.0)
        self.assertIn("consecutive losses", self.manager.block_reason())

    def test_open_positions_and_spread_block(self):
        self.assertIn("spread", self.manager.block_reason(spread=0.2))
        for _ in range(5):
            self.manager.record_entry()
        self.assertEqual(self.manager.block_reason(), "max open positions reached")

    def test_daily_count_and_order_intent_survive_restart_and_utc_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            now = datetime(2026, 9, 28, 23, 59, 50, tzinfo=timezone.utc).timestamp()
            db_path = Path(temp_dir) / "state.db"
            manager = RiskManager(
                RiskConfig(max_trades_per_day=1),
                clock=lambda: now,
                state_path=db_path,
            )
            self.assertTrue(manager.reserve_order_attempt("market-1", now + 300))

            restarted = RiskManager(
                RiskConfig(max_trades_per_day=1),
                clock=lambda: now,
                state_path=db_path,
            )
            self.assertEqual(restarted.trades_today, 1)
            self.assertEqual(restarted.active_order_attempts(), {"market-1": now + 300})
            self.assertFalse(restarted.reserve_order_attempt("market-1", now + 300))

            now += 10
            next_day = RiskManager(
                RiskConfig(max_trades_per_day=1),
                clock=lambda: now,
                state_path=db_path,
            )
            self.assertEqual(next_day.trades_today, 0)
            self.assertTrue(next_day.day_changed_on_load)

    def test_rejected_order_does_not_consume_daily_trade_cap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RiskManager(
                RiskConfig(max_trades_per_day=1),
                clock=lambda: self.now,
                state_path=Path(temp_dir) / "state.db",
            )
            self.assertTrue(manager.reserve_order_attempt("market-rejected", self.now + 300))
            manager.update_order_attempt("market-rejected", "not_filled")
            self.assertEqual(manager.trades_today, 0)


class SettingsTests(unittest.TestCase):
    def test_live_requires_both_switches(self):
        self.assertTrue(Settings.from_env({"POLYMARKET_DRY_RUN": "false"}).dry_run)
        self.assertTrue(Settings.from_env({"LIVE_TRADING": "true"}).dry_run)
        live = Settings.from_env({"POLYMARKET_DRY_RUN": "false", "LIVE_TRADING": "true", "POLYMARKET_PRIVATE_KEY": "0xabc"})
        self.assertFalse(live.dry_run)

    def test_live_without_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            Settings.from_env({"POLYMARKET_DRY_RUN": "false", "LIVE_TRADING": "true"})

    def test_bad_values_are_reported(self):
        with self.assertRaisesRegex(ConfigError, "TRADE_USDC"):
            Settings.from_env({"TRADE_USDC": "five"})
        with self.assertRaisesRegex(ConfigError, "price feed"):
            Settings.from_env({"ENABLE_BINANCE_FALLBACK": "false"})

    def test_stake_is_capped_by_max_position_size(self):
        self.assertEqual(Settings.from_env({"TRADE_USDC": "40", "MAX_POSITION_SIZE": "25"}).stake_usdc, 25.0)


class FakeClob:
    def __init__(self, quotes):
        self.quotes = quotes

    def quote(self, token_id, amount_usdc):
        return self.quotes[token_id]


class FakeExecutor:
    def __init__(self):
        self.orders = []

    def ensure_ready(self):
        pass

    def buy_market(self, token_id, amount_usdc, max_price=None):
        self.orders.append((token_id, amount_usdc, max_price))
        return OrderResult(True, None, amount_usdc, {"dry_run": True})


class TradingBotFlowTests(unittest.TestCase):
    def test_paper_trade_is_placed_once_and_resolved(self):
        start, end = 1790610300.0, 1790610600.0
        clock = {"now": end - 30}
        market = Market("m1", "Bitcoin Up or Down", "btc-updown-5m-1790610300", end, 5000.0,
                        OutcomeToken("Up", "up"), OutcomeToken("Down", "down"), start, liquidity=5000.0)

        state = PriceState()
        for ts, price in rising_samples(seconds=270):
            state.update_source("binance", price, start + ts, start + ts)

        gamma = Mock()
        gamma.find_btc_5m_markets.return_value = [market]
        gamma.fetch_resolution.return_value = "UP"
        executor = FakeExecutor()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        settings = replace(
            Settings.from_env({}),
            trade_log_path=str(Path(tmp.name) / "log.jsonl"),
            risk_state_path=str(Path(tmp.name) / "state.db"),
        )
        bot = TradingBot(
            settings,
            state,
            gamma=gamma,
            clob=FakeClob({
                "up": BookQuote(best_ask=0.80, best_bid=0.78, avg_fill_price=0.80, worst_fill_price=0.80),
                "down": BookQuote(best_ask=0.22, best_bid=0.20, avg_fill_price=0.22, worst_fill_price=0.22),
            }),
            executor=executor,
            clock=lambda: clock["now"],
        )
        bot.geoblock_blocked = False
        bot.next_geoblock_check = monotonic() + 7200

        asyncio.run(bot.step())
        clock["now"] += 2
        asyncio.run(bot.step())
        self.assertEqual(len(executor.orders), 1)
        token, stake, limit = executor.orders[0]
        self.assertEqual((token, stake), ("up", 5.0))
        self.assertLessEqual(limit, settings.risk.max_buy_price)
        self.assertEqual(bot.risk_manager.open_positions, 1)
        self.assertTrue(any("Trade UP" in bot.telegram_outbox.get_nowait() for _ in range(bot.telegram_outbox.qsize())))

        clock["now"] = end + 20
        asyncio.run(bot.resolve_positions(clock["now"]))
        trade = bot.ledger.trades[0]
        self.assertEqual(trade.resolved_outcome, "UP")
        self.assertAlmostEqual(trade.pnl, 5.0 / 0.80 - 5.0)
        self.assertEqual(bot.risk_manager.open_positions, 0)
        self.assertAlmostEqual(bot.risk_manager.daily_pnl, trade.pnl)


class GeoBlockTests(unittest.TestCase):
    def test_blocked_pauses_and_false_resumes_with_country_alert(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(Settings.from_env({}), risk_state_path=str(Path(temp_dir) / "state.db"))
            bot = TradingBot(settings, PriceState(), gamma=Mock(), clob=Mock(), executor=Mock())
        with patch("btc_5m_bot.check_geoblock", return_value=(True, "US")):
            asyncio.run(bot.check_geoblock_status(force=True))
        self.assertTrue(bot.geoblock_blocked)
        self.assertIn("US", bot.telegram_outbox.get_nowait())

        with patch("btc_5m_bot.check_geoblock", return_value=(False, "US")):
            asyncio.run(bot.check_geoblock_status(force=True))
        self.assertFalse(bot.geoblock_blocked)
        self.assertIn("resumed", bot.telegram_outbox.get_nowait())


class TelegramTests(unittest.TestCase):
    def test_commands_require_configured_chat_and_sender(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {"TELEGRAM_CHAT_ID": "12345"}):
            settings = replace(Settings.from_env({}), risk_state_path=str(Path(temp_dir) / "state.db"))
            bot = TradingBot(settings, PriceState(), gamma=Mock(), clob=Mock(), executor=Mock())
            updates = [
                {"update_id": 1, "message": {"chat": {"id": 999}, "from": {"id": 999}, "text": "/stop"}},
                {"update_id": 2, "message": {"chat": {"id": 12345}, "from": {"id": 999}, "text": "/stop"}},
                {"update_id": 3, "message": {"chat": {"id": 12345}, "from": {"id": 12345}, "text": "/stop"}},
            ]
            bot.process_telegram_updates(updates)
            self.assertTrue(bot.telegram_stopped)
            self.assertEqual(bot.telegram_offset, 4)
            self.assertIn("stop command accepted", bot.telegram_outbox.get_nowait())
            self.assertTrue(bot.telegram_outbox.empty())

            bot.process_telegram_updates([
                {"update_id": 4, "message": {"chat": {"id": 12345}, "from": {"id": 12345}, "text": "/resume"}}
            ])
            self.assertFalse(bot.telegram_stopped)

    def test_telegram_outage_does_not_kill_alert_worker(self):
        async def exercise_worker():
            with tempfile.TemporaryDirectory() as temp_dir:
                settings = replace(Settings.from_env({}), risk_state_path=str(Path(temp_dir) / "state.db"))
                bot = TradingBot(settings, PriceState(), gamma=Mock(), clob=Mock(), executor=Mock())
            bot.notify("test alert")
            worker = asyncio.create_task(bot.telegram_sender_loop())
            try:
                await asyncio.wait_for(bot.telegram_outbox.join(), timeout=1.0)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

        with patch("btc_5m_bot.send_telegram_alert", return_value=False):
            asyncio.run(exercise_worker())


if __name__ == "__main__":
    unittest.main()
