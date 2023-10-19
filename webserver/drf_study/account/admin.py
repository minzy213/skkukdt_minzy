from django.contrib import admin

# Register your models here.
from .models import User

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
   list_display = ["username", "email", "phone_number"]
   list_display_links = ["username"]
