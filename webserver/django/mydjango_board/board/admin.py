from django.contrib import admin

# Register your models here.

from .models import Board, Comment
# admin이 Board, Comment에 권한을 가지도록 한다.
admin.site.register(Board)
admin.site.register(Comment)