import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://raw.githubusercontent.com/filipemeneses/geojson-brazil/master/geojson/municipios.json"
print("Checking municipios.json URL...")
try:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
        # Get content length
        length = response.getheader("Content-Length")
        if length:
            size_mb = int(length) / (1024 * 1024)
            print(f"File exists. Size: {size_mb:.2f} MB")
        else:
            print("File exists, but Content-Length header is not available.")
except Exception as e:
    print("Error:", e)
