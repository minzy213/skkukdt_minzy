from django.db import models

# Create your models here.

class Board(models.Model):
    id = models.AutoField(primary_key = True)
    title = models.CharField(max_length = 255)
    content = models.TextField()
    author = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
class Comment(models.Model):
    id = models.AutoField(primary_key = True)
    board = models.ForeignKey(Board, null=True, on_delete = models.SET_NULL)
    content = models.CharField(max_length = 255)
    author = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)