from django.contrib import admin
from django.urls import path
from .views import DataSetListView, DataSetCreateView, DataSetUpdateView
from . import views


urlpatterns = [
    path('data_sets/', DataSetListView.as_view(), name='data_sets'),
    path('data_sets/create/', DataSetCreateView.as_view(), name='create_data_set'),
    path('data_sets/<int:pk>/update/', DataSetUpdateView.as_view(), name='update_data_set'),
    path('data_sets/<int:dataset_id>/', views.show_json, name='show_json'),
]