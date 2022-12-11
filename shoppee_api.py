import pyshopee


shopid = 'lily840824'
partnerid = 
API_key =

client = pyshopee.Client( shopid, partnerid, API_key )

# get_order_by_status (UNPAID/READY_TO_SHIP/SHIPPED/COMPLETED/CANCELLED/ALL)
resp = client.order.get_order_by_status(order_status="READY_TO_SHIP")
print(resp)
