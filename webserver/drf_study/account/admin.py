from django.contrib import admin

# Register your models here.

from .models import User

@admin.register(User)
# class UserAdmin(admin.ModelAdmin):
#     # admin 페이지 리스트에서 보여줄 항목 선택
#    list_display = ["username", "email", "phone_number", "is_staff"]
#    list_display_links = ["username"]
