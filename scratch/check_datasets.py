import urllib.request
import json

for ds in ['movies_1', 'tax', 'toy']:
    try:
        url = f'https://api.github.com/repos/BigDaMa/raha/contents/datasets/{ds}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            files = [f"{f['name']} ({f['size']} bytes)" for f in data]
            print(ds, files)
    except Exception as e:
        print(ds, e)
