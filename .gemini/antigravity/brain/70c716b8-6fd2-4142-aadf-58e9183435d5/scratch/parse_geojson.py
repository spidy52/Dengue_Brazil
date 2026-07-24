import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://raw.githubusercontent.com/luizpedone/municipal-brazilian-geodata/master/minified/Brasil.min.json"
print("Downloading and parsing...")
try:
    with urllib.request.urlopen(url, context=ctx, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
        print("Total features:", len(data["features"]))
        print("\nFirst feature keys:", data["features"][0].keys())
        print("First feature properties:", data["features"][0]["properties"])
        print("First feature id:", data["features"][0].get("id"))
        
        print("\nSecond feature properties:", data["features"][1]["properties"])
        print("Second feature id:", data["features"][1].get("id"))
except Exception as e:
    print("Error:", e)
