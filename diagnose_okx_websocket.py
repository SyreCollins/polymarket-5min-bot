from __future__ import annotations

import asyncio
import json
import time

import websockets


URL = "wss://ws.okx.com:8443/ws/v5/public"
SUBSCRIPTION = {
    "id": "btc-ticker",
    "op": "subscribe",
    "args": [{"channel": "tickers", "instId": "BTC-USDT"}],
}
TEST_SECONDS = 65.0


def parse_ticker(message: object) -> tuple[float, float | None, float | None, int] | None:
    if not isinstance(message, dict):
        return None
    arg = message.get("arg")
    if not isinstance(arg, dict):
        return None
    if arg.get("channel") != "tickers" or arg.get("instId") != "BTC-USDT":
        return None
    data = message.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None

    ticker = data[0]
    try:
        price = float(ticker["last"])
        timestamp = int(ticker["ts"])
        bid = float(ticker["bidPx"]) if ticker.get("bidPx") else None
        ask = float(ticker["askPx"]) if ticker.get("askPx") else None
    except (KeyError, TypeError, ValueError):
        return None
    return price, bid, ask, timestamp


async def main() -> int:
    started = time.monotonic()
    count = 0
    subscribed = False
    print("[okx-test] connecting...", flush=True)
    try:
        async with websockets.connect(URL, ping_interval=None, open_timeout=20, close_timeout=5) as ws:
            print("[okx-test] connected", flush=True)
            print("[okx-test] subscribing...", flush=True)
            await ws.send(json.dumps(SUBSCRIPTION, separators=(",", ":")))

            while time.monotonic() - started < TEST_SECONDS:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=20)
                except asyncio.TimeoutError:
                    await ws.send("ping")
                    continue

                if raw == "pong":
                    continue
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError) as exc:
                    print(f"[okx-test] JSON parse error raw={raw!r} exception_type={type(exc).__name__} exception={exc}", flush=True)
                    continue

                if isinstance(message, dict) and message.get("event") == "error":
                    print("[okx-test] subscription error", flush=True)
                    print("code=", message.get("code"), flush=True)
                    print("message=", message.get("msg"), flush=True)
                    return 1

                if isinstance(message, dict) and message.get("event") == "subscribe":
                    arg = message.get("arg") or {}
                    print("[okx-test] subscription response:", json.dumps(message, sort_keys=True), flush=True)
                    if arg.get("channel") == "tickers" and arg.get("instId") == "BTC-USDT":
                        subscribed = True
                        print("[okx-test] subscription confirmed", flush=True)
                    else:
                        print("[okx-test] unexpected subscription acknowledgement", flush=True)
                    continue

                ticker = parse_ticker(message)
                if ticker is None:
                    continue
                if not subscribed:
                    print("[okx-test] received ticker before matching subscription acknowledgement", flush=True)
                    continue

                price, bid, ask, timestamp = ticker
                count += 1
                if count <= 5:
                    print(f"[okx-test] BTC-USDT last={price} bid={bid} ask={ask} timestamp={timestamp}", flush=True)

            elapsed = time.monotonic() - started
            healthy = subscribed and count >= 5 and elapsed >= 60.0
            print(f"[okx-test] {'feed healthy' if healthy else 'test incomplete'} valid_tickers={count} open_seconds={elapsed:.1f}", flush=True)
            return 0 if healthy else 1
    except websockets.exceptions.ConnectionClosed as exc:
        print("[okx-test] connection closed", flush=True)
        print("close_code=", exc.code, flush=True)
        print("close_reason=", exc.reason, flush=True)
        return 1
    except Exception as exc:
        print("[okx-test] websocket exception", flush=True)
        print("exception_type=", type(exc).__name__, flush=True)
        print("exception=", str(exc), flush=True)
        print("close_code=", getattr(exc, "code", None), flush=True)
        print("close_reason=", getattr(exc, "reason", None), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
