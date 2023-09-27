from django.shortcuts import render
from board.models import Board, Comment

# Create your views here.

from django.http import HttpResponse


def index(request):
    return render(request, 'board/boardList.html')

def board_list(request):
    board_list = Board.objects.all()
    ret = "<ul>"
    for board in board_list:
        ret += f"<li>{board.id} | \
            <a href=\"{board.id}\">{board.title}</a></li>"
    ret += "</ul>"
    return HttpResponse(ret)

def board_comment(request):
    comment_list = Comment.objects.all()
    ret = "<ul>"
    for comment in comment_list:
        ret += f"<li>{comment.id} | {comment.content} | {comment.board_id}"
    ret += "</ul>"
    return HttpResponse(ret)

def board_info(request, board_id):
    board = Board.objects.prefetch_related('comment_set').get(id = board_id)
    ret = f'<h2>{board.id} | {board.title}</h2><a>{board.content}</a>'
    ret += "<ul>"
    for comment in board.comment_set.all():
        ret += f"<li>{comment.id} | {comment.content} | {comment.board_id}"
    ret += "</ul>"
    return HttpResponse(ret)

# ===========================================

def board_list_html(request):
    board_list = Board.objects.all()
    b_dic = {}
    ret = "<ul>"
    for board in board_list:
        b_dic[board.id] = {'href' : f'\"{board.id}\"'}
        ret += f"<li>{board.id} | \
            <a href=\"{board.id}\">{board.title}</a></li>"
    ret += "</ul>"
    return render(request, 'board/boardList.html')

def board_comment_html(request):
    comment_list = Comment.objects.all()
    ret = "<ul>"
    for comment in comment_list:
        ret += f"<li>{comment.id} | {comment.content} | {comment.board_id}"
    ret += "</ul>"
    return HttpResponse(ret)

def board_info_html(request, board_id):
    board = Board.objects.prefetch_related('comment_set').get(id = board_id)
    ret = f'<h2>{board.id} | {board.title}</h2><a>{board.content}</a>'
    ret += "<ul>"
    for comment in board.comment_set.all():
        ret += f"<li>{comment.id} | {comment.content} | {comment.board_id}"
    ret += "</ul>"
    return render(request, 'board/boardDetail.html')