import socket
import urllib.request

from polymarket_client import default_ssl_context

# Verified TLS via certifi, the same trust store the bot uses.
context = default_ssl_context()

def test_url(name, url):
    try:
        print(f"Testing {name} ({url})...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5, context=context):
            print(f"  [SUCCESS] {name} is reachable!")
            return True
    except Exception as e:
        print(f"  [FAILED] {name} error: {e}")
        return False

def test_dns(hostname):
    try:
        print(f"Testing DNS for {hostname}...")
        ip = socket.gethostbyname(hostname)
        print(f"  [SUCCESS] {hostname} resolved to {ip}")
    except Exception as e:
        print(f"  [FAILED] {hostname} DNS error: {e}")

if __name__ == "__main__":
    print("=== Network & DNS Diagnosis ===")
    test_dns("api.binance.com")
    test_dns("api.coinbase.com")
    test_dns("gamma-api.polymarket.com")
    test_dns("clob.polymarket.com")
    test_dns("google.com")
    print("-" * 30)
    test_url("Binance", "https://api.binance.com/api/v3/ping")
    test_url("Coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot")
    test_url("Polymarket Gamma", "https://gamma-api.polymarket.com/markets?limit=1")
    test_url("Polymarket CLOB", "https://clob.polymarket.com/time")
    test_url("Google", "https://www.google.com")
