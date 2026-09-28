import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from btc_5m_bot import (
    COINBASE_SUBSCRIPTION,
    PriceState,
    choose_current_market,
    parse_coinbase_tickers,
    parse_binance_trades,
    parse_okx_tickers,
)
from polymarket_client import ClobMarketData, GammaClient, Market, OutcomeToken, infer_start_ts, market_matches_btc_5m
from strategy import FeatureEngine, PaperTradeLedger, ProbabilityModel, RiskConfig, RiskManager, TradeLogger, build_signal


class StrategyImprovementTests(unittest.TestCase):
    def test_binance_trade_parser_extracts_price_and_millisecond_timestamp(self):
        ticks = parse_binance_trades(
            {"e": "trade", "p": "84156.01", "T": 1790438432240},
            1790438432.250,
        )
        self.assertEqual(ticks, [(84156.01, 1790438432.24)])

    def test_coinbase_ticker_message_parser(self):
        ticks = parse_coinbase_tickers(
            {
                "channel": "ticker",
                "timestamp": "2026-09-26T12:00:00Z",
                "events": [{"tickers": [{"product_id": "BTC-USD", "price": "65000.25"}]}],
            },
            1790424000.0,
        )
        self.assertEqual(ticks, [(65000.25, 1790424000.0)])

    def test_coinbase_subscription_is_public_ticker_channel(self):
        self.assertEqual(
            COINBASE_SUBSCRIPTION,
            {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"},
        )

    def test_coinbase_unrelated_channel_is_ignored(self):
        self.assertEqual(
            parse_coinbase_tickers(
                {"channel": "heartbeats", "events": [{"tickers": [{"product_id": "BTC-USD", "price": "65000"}]}]},
                100.0,
            ),
            [],
        )

    def test_coinbase_healthy_status_requires_valid_ticker(self):
        state = PriceState()
        state.set_feed_status("coinbase", "CONNECTED")
        state.set_feed_status("coinbase", "SUBSCRIBED")
        self.assertEqual(state.feed_status["coinbase"], "SUBSCRIBED")
        state.update_source("coinbase", 65000.0, 100.0, 100.0)
        self.assertEqual(state.feed_status["coinbase"], "HEALTHY")
        self.assertEqual(state.last_update_timestamp["coinbase"], 100.0)

    def test_okx_ticker_message_parser(self):
        ticks = parse_okx_tickers(
            {"data": [{"instId": "BTC-USDT", "last": "65001.5", "ts": "1790424000000"}]},
            1790424000.0,
        )
        self.assertEqual(ticks, [(65001.5, 1790424000.0)])

    def test_price_state_falls_back_to_fresh_secondary_feed(self):
        state = PriceState(max_age_seconds=5.0)
        state.update_source("coinbase", 65000.0, 100.0, 100.0)
        state.update_source("okx", 65002.0, 101.0, 101.0)
        snapshot = state.get_current_btc_state(now=106.0)
        self.assertEqual(snapshot.status, "OK")
        self.assertEqual(snapshot.sources, ("okx",))
        self.assertEqual(snapshot.price, 65002.0)

    def test_price_state_prefers_coinbase_over_binance(self):
        state = PriceState(max_age_seconds=5.0)
        state.update_source("binance", 65010.0, 100.0, 100.0)
        state.update_source("coinbase", 65000.0, 101.0, 101.0)
        snapshot = state.get_current_btc_state(now=102.0)
        self.assertEqual(snapshot.status, "OK")
        self.assertEqual(snapshot.price, 65000.0)

    def test_price_state_uses_receive_age_not_exchange_clock_skew(self):
        state = PriceState(max_age_seconds=10.0)
        state.update_source("binance", 65000.0, 84.0, 100.0)
        snapshot = state.get_current_btc_state(now=105.0)
        self.assertEqual(snapshot.status, "OK")
        self.assertEqual(snapshot.price, 65000.0)

    def test_price_state_rejects_silent_feed_by_receive_age(self):
        state = PriceState(max_age_seconds=10.0)
        state.update_source("binance", 65000.0, 84.0, 100.0)
        self.assertEqual(state.get_current_btc_state(now=111.0).status, "DATA_UNAVAILABLE")

    def test_price_state_rejects_all_stale_feeds(self):
        state = PriceState(max_age_seconds=5.0)
        state.update_source("coinbase", 65000.0, 100.0, 100.0)
        self.assertEqual(state.get_current_btc_state(now=106.0).status, "DATA_UNAVAILABLE")

    def test_price_state_rejects_abnormal_exchange_divergence(self):
        state = PriceState(max_age_seconds=5.0, max_divergence_bps=10.0)
        state.update_source("coinbase", 65000.0, 100.0, 100.0)
        state.update_source("okx", 65100.0, 100.0, 100.0)
        snapshot = state.get_current_btc_state(now=100.0)
        self.assertEqual(snapshot.status, "DATA_UNAVAILABLE")
        self.assertGreater(snapshot.divergence_bps, 10.0)

    def test_price_state_records_window_open_from_feed_history(self):
        state = PriceState()
        state.update_source("coinbase", 65000.0, 100.0, 100.0)
        state.update_source("okx", 65002.0, 101.0, 101.0)
        self.assertEqual(state.market_open_price(100.0), 65000.0)
        self.assertEqual(state.market_open_price(100.0, source="okx"), 65002.0)

    def test_open_price_is_unknown_without_sample_near_start(self):
        state = PriceState(max_age_seconds=5.0)
        state.update_source("binance", 65000.0, 200.0, 200.0)
        self.assertIsNone(state.market_open_price(100.0))

    def test_probability_model_clamps_output(self):
        model = ProbabilityModel()
        features = FeatureEngine().compute_features(
            open_price=60000.0,
            current_price=61000.0,
            recent_prices=[60000.0, 60150.0, 60600.0, 61000.0],
            seconds_left=120.0,
            market_volume=2000.0,
            up_ask=0.47,
            down_ask=0.53,
        )
        probability = model.estimate_probability(features)
        self.assertGreaterEqual(probability, 0.01)
        self.assertLessEqual(probability, 0.99)

    def test_probability_model_is_neutral_without_directional_features(self):
        model = ProbabilityModel()
        self.assertEqual(model.estimate_probability({}), 0.5)
        self.assertEqual(
            model.estimate_probability({
                "volatility": 2.0,
                "volume_feature": 2.0,
                "time_component": 1.0,
                "spread_component": 1.0,
            }),
            0.5,
        )

    def test_features_use_elapsed_time_not_tick_count(self):
        engine = FeatureEngine()
        samples = [(float(ts), 100.0 + ts / 60.0) for ts in range(0, 121, 5)]
        features = engine.compute_features(
            open_price=100.0,
            current_price=102.0,
            recent_prices=[sample[1] for sample in samples],
            recent_samples=samples,
            current_timestamp=120.0,
            seconds_left=30.0,
            market_volume=1000.0,
        )
        self.assertAlmostEqual(features["btc_1m_return"], (102.0 / 101.0) - 1.0)
        self.assertAlmostEqual(features["btc_30s_return"], (102.0 / 101.5) - 1.0)
        self.assertEqual(features["price_history_seconds"], 120.0)

    def test_signal_skips_without_full_timestamped_lookback(self):
        signal = build_signal(
            open_price=100.0,
            current_price=100.0,
            seconds_left=30.0,
            recent_prices=[100.0, 100.1],
            recent_samples=[(100.0, 100.0), (110.0, 100.1)],
            current_timestamp=110.0,
            up_token_id="up",
            down_token_id="down",
            up_ask=0.01,
            down_ask=0.98,
            market_volume=2000.0,
            trades_today=0,
            risk=RiskConfig(),
        )
        self.assertFalse(signal.should_trade)
        self.assertEqual(signal.reason, "insufficient BTC price history")

    def test_risk_manager_blocks_hourly_trade_limit(self):
        risk = RiskConfig(
            max_daily_loss=10.0,
            max_position_size=25.0,
            min_edge=0.08,
            max_trades_per_hour=3,
        )
        manager = RiskManager(risk)
        manager.record_trade(5.0, "up", 0.55, 0.50, 0.25)
        manager.record_trade(5.0, "up", 0.55, 0.50, 0.25)
        manager.record_trade(5.0, "up", 0.55, 0.50, 0.25)
        manager.record_trade(5.0, "up", 0.55, 0.50, 0.25)
        self.assertEqual(manager.can_trade("market-1"), False)
        self.assertEqual(manager.block_reason(), "hourly trade limit reached")

    def test_logger_records_decisions(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        logger = TradeLogger(path=Path(tmp.name) / "trade_log.jsonl")
        logger.record_decision(
            timestamp="2026-09-21T00:00:00Z",
            market_id="market-1",
            time_remaining=120.0,
            btc_price=60000.0,
            btc_open_price=59800.0,
            btc_return_30s=0.002,
            btc_return_1m=0.003,
            btc_return_5m=0.004,
            volume_features={"relative_volume": 1.1},
            order_flow_features={"imbalance": 0.3},
            polymarket_bid=0.49,
            polymarket_ask=0.51,
            model_probability_up=0.62,
            model_probability_down=0.38,
            estimated_edge=0.11,
            trade_decision="NO_TRADE",
            decision_reason="insufficient edge",
            position_size=0.0,
        )
        self.assertTrue(logger.path.exists())

    def test_paper_trade_ledger_resolves_binary_payout(self):
        ledger = PaperTradeLedger()
        ledger.enter(
            market_id="market-1",
            outcome="UP",
            entry_price=0.50,
            stake=5.0,
            model_probability=0.65,
            edge=0.15,
        )
        trade = ledger.resolve("market-1", "UP")
        self.assertIsNotNone(trade)
        self.assertAlmostEqual(trade.pnl, 5.0)
        self.assertEqual(ledger.summary()["win_rate"], 1.0)

    def test_gamma_client_uses_keyset_market_envelope(self):
        client = GammaClient("https://gamma-api.polymarket.com")
        client._get_json = Mock(
            return_value={
                "markets": [
                    {
                        "id": "market-1",
                        "slug": "btc-updown-5m-9999999999",
                        "question": "Bitcoin Up or Down - test",
                        "endDate": "2099-01-01T00:00:00Z",
                        "outcomes": '["Up", "Down"]',
                        "clobTokenIds": '["up-token", "down-token"]',
                    }
                ],
                "next_cursor": None,
            }
        )
        markets = client._find_btc_5m_from_keyset(limit=1)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].up.token_id, "up-token")
        self.assertEqual(client._get_json.call_args.args[0], "/markets/keyset")
        self.assertEqual(client._get_json.call_args.args[1]["tag_id"], "235")

    def test_clob_buy_price_uses_best_order_book_ask(self):
        class Level:
            def __init__(self, price):
                self.price = price

        client = ClobMarketData("https://clob.polymarket.com")
        client._client = Mock()
        client._client.get_order_book.return_value = Mock(
            asks=[Level("0.61"), Level("0.54")]
        )
        self.assertEqual(client.buy_price("token-1"), 0.54)
        client._client.get_order_book.assert_called_once_with("token-1")

    def test_market_filter_does_not_confuse_15m_with_5m(self):
        market = Market(
            id="market-15m",
            question="Bitcoin Up or Down",
            slug="btc-updown-15m-9999999999",
            end_ts=4102444800.0,
            volume=1000.0,
            up=OutcomeToken("Up", "up-token"),
            down=OutcomeToken("Down", "down-token"),
        )
        self.assertFalse(market_matches_btc_5m(market, require_strict_match=True))

    def test_market_parser_preserves_liquidity_when_volume_is_missing(self):
        from polymarket_client import parse_market

        market = parse_market({
            "id": "market-liquidity",
            "slug": "btc-updown-5m-9999999999",
            "question": "Bitcoin Up or Down",
            "endDate": "2099-01-01T00:00:00Z",
            "volumeNum": None,
            "liquidityNum": 22023.1065,
            "outcomes": '["Up", "Down"]',
            "clobTokenIds": '["up-token", "down-token"]',
        })
        self.assertIsNotNone(market)
        self.assertEqual(market.volume, 0.0)
        self.assertEqual(market.liquidity, 22023.1065)

    def test_activity_uses_liquidity_when_reported_volume_is_small(self):
        market_volume = 5.0
        market_liquidity = 16245.0
        self.assertEqual(max(market_volume, market_liquidity), 16245.0)

    def test_market_selection_prefers_running_window(self):
        running = Market("running", "", "", 200.0, 0.0, OutcomeToken("Up", "u"), OutcomeToken("Down", "d"), 100.0)
        upcoming = Market("upcoming", "", "", 150.0, 0.0, OutcomeToken("Up", "u"), OutcomeToken("Down", "d"), 140.0)
        import btc_5m_bot
        original_time = btc_5m_bot.time
        btc_5m_bot.time = lambda: 130.0
        try:
            self.assertEqual(choose_current_market([upcoming, running]).id, "running")
        finally:
            btc_5m_bot.time = original_time

    def test_slug_epoch_is_preferred_for_market_start(self):
        self.assertEqual(infer_start_ts({"slug": "btc-updown-5m-1790262300", "startDate": "2026-09-23T15:18:30Z"}, 1790262600.0), 1790262300.0)


if __name__ == "__main__":
    unittest.main()
