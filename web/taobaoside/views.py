from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from taobaoside.taobao_api import *

# Create your views here.
class Home(generic.ListView):
    #get -> http with a box
    def get(self,request, *args, **kwargs):
        return render(request, 'taobaoside/home.html')



class ProductSearch(generic.ListView):
    #load product page
    def post(self,request, *args, **kwargs):
        search_query = request.POST['search_query']
        print(search_query)

        #function to search the key word with taobao api
        items_found = taobao_search(search_query,0)
        #print(items_found)
        context = {'items':items_found}
        #Load the data into a dict
        return render(request, 'taobaoside/product_search_result.html',context=context)


class ItemDetails(generic.ListView):
    #load product detail
    # With an add shoppee button
    def get(self,request, *args, **kwargs):
        #print(kwargs)
        item_id = kwargs['slug']
        context = taobao_item_search(item_id)
        print('ok')
        print(context)
        print('ok')
        #print(context['images'][0])
        return render(request, 'taobaoside/item_details.html',context=context)


class AddShopee(generic.ListView):

    def get(self,request, *args, **kwargs):
        item_id = kwargs['slug']
        context = taobao_item_search(item_id)
        return render(request, 'taobaoside/add_item.html',context=context)

    def post(self,request, *args, **kwargs):
        # Add to shopee
        return render(request, 'taobaoside/product_added.html')






# average_BMI = 0
# name_list = list(d.keys())
# for key in name_list:
#     height = d[key]['h']
#     weight = d[key]['w']
#     BMI = w/(h/100)**2
#     average_BMI += BMI
#
# average_BMI = average_BMI /len(d)
