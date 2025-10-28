from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from methods import views as mviews  



urlpatterns = [
    path('admin/', admin.site.urls),                   
    path('', include('methods.urls')),
    # Auth built-in:
    path('login/',  mviews.AnalysisLoginView.as_view(),  name='login'),
    path('logout/', mviews.AnalysisLogoutView.as_view(), name='logout'),
    path('signup/', mviews.signup,                       name='signup'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)