from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic


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

        #Load the data into a dict


        return render(request, 'taobaoside/product_search.html')


def product_details(request):
    #load product detail
    # With an add shoppee button
    pass
