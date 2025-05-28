import requests


r = requests.get("https://github.com/NaniOP12/ml/archive/refs/heads/main.zip")

open("temp.zip", "wb").write(r.content)