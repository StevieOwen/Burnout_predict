from django.urls import path
from . import views

app_name = 'workloads'

urlpatterns = [
    path('dashboard/', views.lecturer_dashboard, name='lecturer_dashboard'),
    path('hod/dashboard/', views.hod_dashboard, name='hod_dashboard'),
    path('submit-log/', views.submit_log, name='submit_log'),
]