from django.urls import path

from . import views
app_name = 'taobaoside'
urlpatterns = [
    path('', views.Home.as_view(), name='home'),
    path('search-result',views.ProductSearch.as_view(),name='search_item'),
    path('search-result/<slug:slug>/detail',views.ItemDetails.as_view(),name='item_detail'),
    path('search-result/<slug:slug>/add-shopee',views.AddShopee.as_view(),name='add_shopee'),
    path('product-added',views.AddShopee.as_view(),name='product_added')
]
