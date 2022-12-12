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
        print(context['images'][0])
        return render(request, 'taobaoside/item_details.html',context=context)
