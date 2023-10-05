from django.shortcuts import render, redirect, reverse
from .models import Board, Comment

# Create your views here.

from django.http import HttpResponse


def index(request):
    # 인자로 전달하는 변수가 흐름 내에서 유지되면 context라고 부른다.
    sample_list = range(10)
    return render(request, "board/index.html", {"name": "유저", "sample": sample_list})


def comment_list(request):
    comment_list = Comment.objects.all()
    ret = "<ul>"
    for comment in comment_list:
        ret += f"<li>{comment.id} | {comment.content} | {comment.board_id}"
    ret += "</ul>"
    return HttpResponse(ret)


# ===========================================


def board_list_html(request):
    board_list = Board.objects.prefetch_related("comment_set").all()
    li = []
    for board in board_list:
        li.append(
            {
                "id": board.id,
                "title": board.title,
                "content": board.content,
                "com_cnt": len(board.comment_set.all()),
                "date": board.created_at,
            }
        )
    return render(request, "board/boardList.html", {"info": li})


def board_detail_html(request, board_id):
    if request.method == "POST":
        data = request.POST
        comment = data.get("comment")
        if len(comment) > 0:
            Comment.objects.create(
                board_id=board_id, content=comment, author=request.user
            )
        # index 페이지로 이동
        return redirect(reverse("board:board_detail_html", args=[board_id]))

    board = Board.objects.prefetch_related("comment_set").get(pk=board_id)
    date = board.created_at
    print(board.updated_at)
    if board.created_at != board.updated_at:
        date = board.updated_at
    info = {
        "id": board.id,
        "title": board.title,
        "content": board.content,
        "comment": board.comment_set.all(),
        "com_len": len(board.comment_set.all()),
        "date": date,
    }
    return render(request, "board/boardDetail.html", {"info": info})


def board_modify_html(request, board_id):
    if request.method == "POST":
        data = request.POST
        title, content = data["title"], data["content"]

        board = Board.objects.get(id=board_id)
        board.title = title
        board.content = content
        board.save()
        return redirect(reverse("board:modify", args=[board_id]))

    # elif request.methd == 'GET':
    board = Board.objects.prefetch_related("comment_set").get(id=board_id)

    info = {
        "id": board.id,
        "title": board.title,
        "content": board.content,
        "comment": board.comment_set.all(),
        "com_len": len(board.comment_set.all()),
        "date": board.created_at,
    }
    return render(request, "board/boardModify.html", {"info": info})


def board_delete(request, board_id):
    board = Board.objects.get(pk=board_id)
    board.delete()
    return redirect(reverse("board:board_list_html"))


def write_html(request):
    return render(request, "board/boardWrite.html")


################################################
# 게시글 등록하기
from .forms import BoardForm


def board_create(request):
    form = BoardForm()
    if request.method == "POST":
        form = BoardForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse("board:board_list_html"))
    return render(request, "board/boardwrite.html", {"form": form})


# def board_create(request):
#     if request.method == "POST":
#         data = request.POST
#         title, content = data["title"], data.get("content")
#         Board.objects.create(title=title, content=content, author=request.user)
#         # index 페이지로 이동
#         return redirect(
#             reverse(
#                 "board:board_list_html",
#             )
#         )
#     return render(request, "board/boardwrite.html")
