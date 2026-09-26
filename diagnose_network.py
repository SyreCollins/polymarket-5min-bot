import urllib.request
import json
import ssl
import socket

# Create a "dangerous" context that ignores SSL errors
context = ssl._create_unverified_context()

def test_url(name, url):
    try:
        print(f"Testing {name} ({url})...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5, context=context) as response:
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
    test_dns("google.com")
    print("-" * 30)
    test_url("Binance", "https://api.binance.com/api/v3/ping")
    test_url("Coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot")
    test_url("Google", "https://www.google.com")
