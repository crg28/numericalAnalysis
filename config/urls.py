# config/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

# auth views exposed from base.py
from methods.views.base import AnalysisLoginView, AnalysisLogoutView, signup

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('methods.urls')),

    # Auth
    path('login/',  AnalysisLoginView.as_view(),  name='login'),
    path('logout/', AnalysisLogoutView.as_view(), name='logout'),
    path('signup/', signup,                     name='signup'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
