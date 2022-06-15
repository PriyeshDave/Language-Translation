import requests
import json

data = {
  'sentence': 'My name is Priyesh'
}

url = "http://127.0.0.1:8000/translate"

data_json = json.dumps(data)
output = requests.post(url, data=data_json)
output = json.loads(output.text)
print(output['translation'])