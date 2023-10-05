from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
import json
from json2html import *


# request : <WSGIRequest: GET '/user/'>
# <WSGIRequest: GET '/user/'>
# <class 'django.core.handlers.wsgi.WSGIRequest'>
def index(request):
    sample_list = range(10)
    return render(request, "board/index.html", {"name": "유저", "sample": sample_list})
