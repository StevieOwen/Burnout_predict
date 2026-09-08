from django.urls import path
from . import views


app_name = 'workloads'

urlpatterns = [
    # Auth Views
    path('login/', views.UTBLoginView.as_view(), name='login'),
    
    # HTML Template Dashboards & Forms
    path('dashboard/', views.lecturer_dashboard, name='lecturer_dashboard'),
    path('hod/dashboard/', views.hod_dashboard, name='hod_dashboard'),
    path('submit-log/', views.submit_log, name='submit_log'),
    
    # REST API Endpoints for Dynamic Dashboards & State Store
    path('api/v1/dashboard/lecturer/<str:staff_id>/', views.lecturer_dashboard_api, name='lecturer_dashboard_api'),
    path('api/v1/dashboard/executive/<str:dept_id>/', views.executive_dashboard_api, name='executive_dashboard_api'),
    path('api/v1/reallocate/accept/', views.accept_reallocation_api, name='accept_reallocation_api'),
    
]