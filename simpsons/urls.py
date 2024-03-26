from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.simpsons, name="simpsons"),
    path('predict/', views.predict, name="simpsons_predict"),
]