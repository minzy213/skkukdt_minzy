from django.db import models
from django.utils import timezone

from django.core import validators

# Create your models here.

# python manage.py migrate board 0005
# 0005로 rollback 해준다.


class Board(models.Model):
    id = models.AutoField(primary_key=True)  # integer(auto-increment)
    title = models.CharField(
        "제목",
        max_length=255,  # varchar(255)
        validators=[validators.MinLengthValidator(2, "최소 세 글자 이상은 입력해주셔야 합니다.")],
    )
    author = models.CharField(max_length=255)
    content = models.TextField(
        "내용", validators=[validators.MinLengthValidator(10, "최소 10글자 이상은 입력해주셔야 합니다.")]
    )  # text
    created_at = models.DateTimeField(auto_now_add=True)  # 추가될 때 default로 현재시간
    updated_at = models.DateTimeField(auto_now=True)  # 추가/업데이트 될 때 default로 현재시간

    # 이렇게 하면 admin에서 board title 보임
    def __str__(self):
        return self.title

    # 아래꺼로 해도 title 보여야 하는데... 잘 모르겠음
    # def __repr__(self):
    #     return self.title

    # @property  # https://www.daleseo.com/python-property/
    # def to_html(self):
    #     return f"""
    # """


class Comment(models.Model):
    # id 없어도 자동으로 만들어준다.
    # id = models.AutoField(primary_key=True)  # integer(auto-increment)

    # related_name="comment_set"은 기본값, 그래서 board.comment_set 하면 이게 불러와진다.
    # Board.comment_set 하면 해당 board의 comment들 불러온다.
    # n에서 1을 불러오는것보다 1에서 n을 불러오는 경우가 많기 때문에 있는 기능.
    # 이 이름은 바꿀 수 있다
    board = models.ForeignKey(
        Board, null=True, on_delete=models.SET_NULL, related_name="comment_set"
    )
    content = models.CharField(max_length=255)  # varchar(255)
    author = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)  # 추가될 때 default로 현재시간
    updated_at = models.DateTimeField(auto_now=True)  # 추가/업데이트 될 때 default로 현재시간
