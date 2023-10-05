from django.contrib import admin

# Register your models here.

from .models import Board, Comment

admin.site.register(Board)
admin.site.register(Comment)