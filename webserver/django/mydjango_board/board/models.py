from django.db import models

# Create your models here.

class Board(models.Model):
    id = models.AutoField(primary_key=True) # integer(auto-increment)
    title = models.CharField(max_length=255) # varchar(255)
    author = models.CharField(max_length=255)
    content = models.TextField() # text
    created_at = models.DateTimeField(auto_now_add=True) # 추가될 때 default로 현재시간
    updated_at = models.DateTimeField(auto_now=True) # 추가/업데이트 될 때 default로 현재시간
    
class Comment(models.Model):
    id = models.AutoField(primary_key=True) # integer(auto-increment)
    b_id = models.ForeignKey(Board, null=True, on_delete=models.SET_NULL)
    content = models.CharField(max_length=255) # varchar(255)
    created_at = models.DateTimeField(auto_now_add=True) # 추가될 때 default로 현재시간
    updated_at = models.DateTimeField(auto_now=True) # 추가/업데이트 될 때 default로 현재시간