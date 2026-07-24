import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://raw.githubusercontent.com/luizpedone/municipal-brazilian-geodata/master/minified/Brasil.min.json"
print("Downloading a chunk of Brazil municipalities GeoJSON...")
try:
    with urllib.request.urlopen(url, context=ctx, timeout=10) as response:
        # Read first 5000 bytes to inspect
        chunk = response.read(15000).decode("utf-8")
        # Since it's minified, the first 15000 bytes should contain the first few features
        # Let's clean it up to see if we can find a feature block
        print("Chunk preview:")
        print(chunk[:1000])
except Exception as e:
    print("Error:", e)
