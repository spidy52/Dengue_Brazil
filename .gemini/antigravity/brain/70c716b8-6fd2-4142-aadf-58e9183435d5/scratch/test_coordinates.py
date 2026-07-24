import urllib.request
import pandas as pd
import ssl
import io

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
print("Downloading coordinates CSV...")
try:
    with urllib.request.urlopen(url, context=ctx, timeout=10) as response:
        csv_data = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_data))
        print("Success! Downloaded.")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Head:")
        print(df.head())
except Exception as e:
    print("Error:", e)
