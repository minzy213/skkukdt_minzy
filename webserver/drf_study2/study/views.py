from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from .models import Students, Score
from .serializers import StudentSerializer, ScoreSerializer
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404

## FBV(Function Based View)
# @api_view(['GET', 'POST'])
# def StudentView(request):
#     """학생들 조회하는 API"""
#     if request.method == 'GET':
#         qs = Students.objects.all()
#         serializer = StudentSerializer(qs, many=True)
#         return Response(serializer.data)
#
#     elif request.method == 'POST':
#         serializer = StudentSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#
#
# @api_view(['GET', 'PUT', 'DELETE'])
# def StudentDetailView(request, pk):
#     qs = get_object_or_404(Students, pk=pk)
#
#     if request.method == 'GET':
#         serializer = StudentSerializer(qs)
#         return Response(serializer.data)
#
#     elif request.method == 'PUT':
#         serializer = StudentSerializer(qs, data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#
#     elif request.method == 'DELETE':
#         qs.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)
#
#
# @api_view(['GET', "POST"])
# def ScoreView(request):
#     if request.method == 'GET':
#         qs = Score.objects.all()
#         serializer = ScoreSerializer(qs, many=True)
#         return Response(serializer.data)
#     elif request.method == 'POST':
#         serializer = ScoreSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)
#
#
# @api_view(['GET', 'PUT', 'DELETE'])
# def ScoreDetailView(request, pk):
#     qs = get_object_or_404(Score, pk=pk)
#     print(qs)
#     if request.method == 'GET':
#         serializer = ScoreSerializer(qs)
#         return Response(serializer.data)
#     elif request.method == 'PUT':
#         serializer = ScoreSerializer(qs, data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#     elif request.method == 'DELETE':
#         qs.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)


## CBV(Class Based View)
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from rest_framework import status
#
# class StudentView(APIView):
#     def get(self, request):
#         qs = Students.objects.filter()
#         serializer = StudentSerializer(qs, many=True)
#         return Response(serializer.data)
#
#     def post(self, request):
#         serializer = StudentSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#
# class StudentDetailView(APIView):
#     def get_object(self, pk):
#         return get_object_or_404(Students, pk=pk)
#
#     def get(self, request, pk):
#         qs = self.get_object(pk)
#         serializer = StudentSerializer(qs)
#         return Response(serializer.data)
#
#     def put(self, request, pk):
#         qs = self.get_object(pk)
#         serializer = StudentSerializer(qs, data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#
#     def delete(self, request, pk):
#         qs = self.get_object(pk)
#         qs.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)
#
#
# class ScoreView(APIView):
#     def get(self, request):
#         qs = Score.objects.filter()
#         serializer = ScoreSerializer(qs, many=True)
#         return Response(serializer.data)
#
#     def post(self, request):
#         serializer = ScoreSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#
# class ScoreDetailView(APIView):
#     def get_object(self, pk):
#         return get_object_or_404(Score, pk=pk)
#
#     def get(self, request, pk):
#         qs = self.get_object(pk)
#         serializer = ScoreSerializer(qs)
#         return Response(serializer.data)
#
#     def put(self, request, pk):
#         qs = self.get_object(pk)
#         serializer = ScoreSerializer(qs, data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#
#     def delete(self, request, pk):
#         qs = self.get_object(pk)
#         qs.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)

from rest_framework import viewsets


## ViewSet
class StudentViewSet(viewsets.ModelViewSet):
    queryset = Students.objects.all()
    serializer_class = StudentSerializer


class ScoreViewSet(viewsets.ModelViewSet):
    queryset = Score.objects.all()
    serializer_class = ScoreSerializer
