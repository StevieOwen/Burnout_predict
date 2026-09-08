# workloads/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, WorkloadLog, PrivacyConsent, UserProfile

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        ('UTB Academic Profile', {'fields': ('role', 'department')}),
    )
    list_display = ['username', 'email', 'first_name', 'last_name', 'role', 'department']
    list_filter = ['role', 'department']

@admin.register(WorkloadLog)
class WorkloadLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'week_starting', 'predicted_risk_level', 'confidence_score', 'primary_risk_driver']
    list_filter = ['predicted_risk_level', 'user__department']
    search_fields = ['user__username', 'user__first_name', 'user__last_name']

@admin.register(PrivacyConsent)
class PrivacyConsentAdmin(admin.ModelAdmin):
    list_display = ['user', 'allow_hod_alerting', 'updated_at']
    list_filter = ['allow_hod_alerting']
    search_fields = ['user__username', 'user__email']

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'role', 'department']
    list_filter = ['role', 'department']
    search_fields = ['user__username', 'user__email']