from django.contrib import admin

# Register your models here.
from .models import Students, Score


@admin.register(Students)
class StudentAdmin(admin.ModelAdmin):
    list_display = ["name", "address", "email"]
    list_display_links = ["name"]


@admin.register(Score)
class ScoreAdmin(admin.ModelAdmin):
    list_display = ["student", "english", "math", "science"]
    list_display_links = ["student"]
