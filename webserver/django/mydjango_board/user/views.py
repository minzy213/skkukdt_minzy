from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
import json
from json2html import *

# request : <WSGIRequest: GET '/user/'>
# <WSGIRequest: GET '/user/'>
# <class 'django.core.handlers.wsgi.WSGIRequest'>
def index(request):
    data={'h1':'나의 프로필',
         'ul' : {
             'li':['이름: 박민지', '별명: 민디띠'],
            }
        }
    # json.load(data)
    
    
    # print('request ======')
    # for d in dir(request):
    #     print(d)
    #     print(eval("request." + d))
    #     print('='*20)
    # print('==============')
    
    # request_attrs = dir(request)
    # for attr in request_attrs:
    #     print(attr, getattr(request, attr))
    #     print('-'*10)
    
    return HttpResponse("<h1>나의 프로필</h1>\
    <ul>\
        <li>이름: 박민지</li>\
        <li>별명: 민디띠</li>\
    </ul>")