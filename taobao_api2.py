import requests
import pandas as pd
import json
import pprint
import config
url = "https://taobao-api.p.rapidapi.com/api"

querystring = {"api":"item_search","page_size":"40","q":"shoes"}

headers = {
	"X-RapidAPI-Key": config.taobao_api_key,
	"X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)
response = response.text
response = json.loads(response)
pprint.pprint(response)
response = response['result']['item']
# print(response.text)
df = pd.DataFrame(response)
df.to_csv('taobao_test.csv')
