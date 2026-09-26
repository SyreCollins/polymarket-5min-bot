import json
import urllib.request
import urllib.parse

def inspect_recent_btc():
    params = {
        "active": "false",
        "closed": "true",
        "limit": "20",
        "order": "endDate",
        "ascending": "false",
        "query": "BTC"
    }
    url = f"https://gamma-api.polymarket.com/markets?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    for m in data:
        print(f"Title: {m.get('question')}")
        print(f"End: {m.get('endDate')}")
        print(f"Group ID: {m.get('group_id')}")
        # Check for specific price data in the description or elsewhere
        print("-" * 20)

if __name__ == "__main__":
    inspect_recent_btc()
