from django.urls import path

from . import views

urlpatterns=[
    path('', views.index, name = 'board_list'),
    path('comments/', views.board_comment, name = 'board_comment'),
    path('<int:board_id>/', views.board_info, name = 'board_info'),
]