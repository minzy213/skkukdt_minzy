from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def index(request):
    return HttpResponse('<h1>나의 프로필</h1><ul></ul><li>이름 : 박민지</li><li>별명 : 민디띠</li>')