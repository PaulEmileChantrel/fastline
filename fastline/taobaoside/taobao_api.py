import requests
import pandas as pd
import json
import pprint
from fastline.config import taobao_api_key


API_ON = True


def call_taobao_api(querystring):
	url = "https://taobao-api.p.rapidapi.com/api"

	#querystring = {"api":"item_search","page_size":"40","q":search_query,"page":page}

	headers = {
		"X-RapidAPI-Key": taobao_api_key,
		"X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
	}

	response = requests.request("GET", url, headers=headers, params=querystring)
	response = response.text
	response = json.loads(response)
	#pprint.pprint(response)
	#response = response['result']['item']
	return response

def taobao_search(search_query,page):

	if API_ON:# we avoid using the api for tests; we might want to check for cache in the future
		querystring = {"api":"item_search","page_size":"40","q":search_query,"page":page}

		response = call_taobao_api(querystring)
		response = response['result']['item']

		# print(response.text)
		df = pd.DataFrame(response)
		df.to_csv('taobao_test.csv',index=False)

		return response
	else:

		df = pd.read_csv('taobao_test.csv')
		d = df.to_dict('records')
		return d

def taobao_item_image(item_id):
	querystring = {"api":"item_desc","num_iid":item_id}
	response = call_taobao_api(querystring)
	response = response['result']['item']

	# print(response.text)
	df = pd.DataFrame(response)
	df.to_csv('taobao_item_image.csv',index=False)
	return response

def taobao_item_details(item_id):
	querystring = {"api":"item_detail_simple","num_iid":item_id}
	response = call_taobao_api(querystring)
	response = response['result']['item']

	# print(response.text)
	df = pd.DataFrame(response)
	df.to_csv('taobao_item_test.csv',index=False)
	return response

def taobao_item_search(item_id):

	if API_ON:
		images = taobao_item_image(item_id)
		details = taobao_item_details(item_id)

	else:
		images = pd.read_csv('taobao_item_test.csv')
		details = pd.read_csv('taobao_item_image.csv')
	return {'images':images,'details':details}
