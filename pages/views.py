from django.shortcuts import redirect, render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import logout as lg
from django.http import Http404
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.contrib import messages, auth
from django.contrib import messages
from django.db.models import Count
import os
from django.http import HttpResponse
from django.conf import settings


def home(request):
    #forums = Category.objects.all()
    user = request.user
    context = {
        #"forums": forums,
        "user": user,
    }
    return render(request, 'main.html', context)

def firstever(request):
    print("Downloading the file...  ")
    file_path = os.path.join(settings.MEDIA_ROOT, 'Game.zip')
    if os.path.exists(file_path):
        print("The file exists. Start downloading...")
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/zip")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            print("The download completed successfully...")
            return response
    raise Http404

def download_resume(request):
    print("Downloading the resume...  ")
    file_path = os.path.join(settings.MEDIA_ROOT, 'resume.docx')
    if os.path.exists(file_path):
        print("The resume exists. Start downloading...")
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/docx")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            print("The download completed successfully...")
            return response
    raise Http404

