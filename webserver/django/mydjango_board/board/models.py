from django.db import models

# Create your models here.

# python manage.py migrate board 0005
# 0005로 rollback 해준다.


class Board(models.Model):
    id = models.AutoField(primary_key=True)  # integer(auto-increment)
    title = models.CharField(max_length=255)  # varchar(255)
    author = models.CharField(max_length=255)
    content = models.TextField()  # text
    created_at = models.DateTimeField(auto_now_add=True)  # 추가될 때 default로 현재시간
    updated_at = models.DateTimeField(
        auto_now=True)  # 추가/업데이트 될 때 default로 현재시간
    
    def __str__(self):
        return self.title
    
    # @property  # https://www.daleseo.com/python-property/
    # def to_html(self):
    #     return f"""
    # """
    # def __repr__(self):
    #     return self.title


class Comment(models.Model):
    # id 없어도 자동으로 만들어준다.
    # id = models.AutoField(primary_key=True)  # integer(auto-increment)
    # related_name="comment_set"은 기본값, 그래서 board.comment_set 하면 이게 불러와진다.
    # 이 이름은 바꿀 수 있다
    board = models.ForeignKey(Board, null=True, on_delete=models.SET_NULL, related_name="comment_set")
    content = models.CharField(max_length=255)  # varchar(255)
    author = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)  # 추가될 때 default로 현재시간
    updated_at = models.DateTimeField(
        auto_now=True)  # 추가/업데이트 될 때 default로 현재시간
