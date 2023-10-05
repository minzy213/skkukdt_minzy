from django.urls import path

from . import views

app_name = "board"


urlpatterns = [
    path("", views.board_list_html, name="board_list_html"),
    path("comments/", views.comment_list, name="comment_list"),
    # django에서는 <>를 변수로 인식하고, 함수의 인자로 전달된다. 이름은 동일해야한다.
    path("<int:board_id>/", views.board_detail_html, name="board_detail_html"),
    path("write/", views.board_create, name="write"),
    path("<int:board_id>/modify", views.board_modify_html, name="modify"),
    path("<int:board_id>/delete", views.board_delete, name="delete"),
]
