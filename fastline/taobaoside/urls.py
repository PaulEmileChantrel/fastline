from django.urls import path

from . import views
app_name = 'taobaoside'
urlpatterns = [
    path('', views.Home.as_view(), name='home'),
    path('search',views.ProductSearch.as_view(),name='search_item'),
]
