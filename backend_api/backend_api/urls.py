"""backend_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from extraction.views import extract_regions_view, extract_regions_dual_view, extract_regions_front_view
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.conf.urls.static import static
from facial_recognition.views import verify_faces, clear_media_dirs, verify_faces_advanced, register_face, verify_face1, train_model

urlpatterns = [
    path('extraction/', csrf_exempt(extract_regions_view), name='extraction_api'),
    path('extraction_dual/', csrf_exempt(extract_regions_dual_view),
         name='extraction_api1'),
    path('extraction_front/', csrf_exempt(extract_regions_front_view),
         name='extraction_api_front'),
    path('face_verification/', csrf_exempt(verify_faces),
         name='face_verification_api'),
    path('face_verification_advanced/', csrf_exempt(verify_faces_advanced),
         name='face_verification_advanced_api'),
    path('register_face/', csrf_exempt(register_face),
         name='regiter_face_api'),
    path('search_face/', csrf_exempt(verify_face1),
         name='search_face_api'),
    path('train_model/', csrf_exempt(train_model),
         name='train_model_api'),
    path('clear-directories/', csrf_exempt(clear_media_dirs), name='clear_media_dirs')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
