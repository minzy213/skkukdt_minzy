from django.urls import path, include
from . import views
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register('students', views.StudentViewSet)
router.register('score', views.ScoreViewSet)
print(router.urls)
urlpatterns = [
   ## FBV
   # path('students/', views.StudentView),
   # path('score/', views.ScoreView),
   # path('students/<int:pk>', views.StudentDetailView),
   # path('score/<int:pk>', views.ScoreDetailView),

   ## CBV
   # path('students/', views.StudentView.as_view()),
   # path('students/<int:pk>', views.StudentDetailView.as_view()),
   # path('score/', views.ScoreView.as_view()),
   # path('score/<int:pk>', views.ScoreDetailView.as_view()),
   path('', include(router.urls))
]
