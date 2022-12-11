# Import the necessary libraries
from taobao.api.taobao_api import TaoBaoAPI

# Set your Taobao API key and secret
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'

# Initialize the TaoBaoAPI client
tb = TaoBaoAPI(api_key, api_secret)

# Set the keyword to search for
keyword = 'phone case'

# Search for items on Taobao using the keyword
response = tb.taobao_item_search(q=keyword, fields='num_iid,title,price')

# Print the number of items found
print(f'Found {response["total_results"]} items')

# Print the title and price of each item
for item in response['items']:
    print(f'{item["title"]}: {item["price"]}')
