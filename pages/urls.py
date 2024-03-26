from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.home, name="index"),
    path('firstever/', views.firstever, name="firstever"),
    path('download_resume/', views.download_resume, name="download_resume"),
]