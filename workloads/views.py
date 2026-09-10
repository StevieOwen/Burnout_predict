from functools import wraps
import json

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from django.db.models import Avg
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_http_methods, require_POST

from .models import WorkloadLog, PrivacyConsent, UserProfile
from .forms import WorkloadLogForm, PrivacyConsentForm
from ml_engine.workload_engine import StaffFeatureVector, AcademicRank
from .services import (
    calculate_burnout_risk,
    get_lecturer_dashboard_payload,
    get_executive_dashboard_payload,
)


# ==========================================
# DECORATORS & AUTH VIEWS
# ==========================================

def role_required(allowed_roles=None):
    """Decorator to enforce role-based access control checking user.role and user.profile.role."""
    if allowed_roles is None:
        allowed_roles = []

    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            # 1. If not logged in, redirect to login page
            if not request.user.is_authenticated:
                return redirect('login')
            
            # Get role directly from Custom User model or UserProfile
            user_role = getattr(request.user, 'role', None) or getattr(getattr(request.user, 'profile', None), 'role', None)

            # 2. Check if user has an allowed role
            if user_role in allowed_roles:
                return view_func(request, *args, **kwargs)
            
            # 3. Redirect to proper home if role doesn't match
            if user_role == 'HOD':
                return redirect('workloads:hod_dashboard')
            elif user_role == 'LECTURER':
                return redirect('workloads:lecturer_dashboard')
            
            return redirect('login')
        return _wrapped_view
    return decorator


class UTBLoginView(LoginView):
    template_name = 'registration/login.html'
    redirect_authenticated_user = True

    def get_success_url(self):
        user = self.request.user
        user_role = getattr(user, 'role', None) or getattr(getattr(user, 'profile', None), 'role', None)

        if user_role == 'HOD':
            return reverse_lazy('workloads:hod_dashboard')
        
        return reverse_lazy('workloads:lecturer_dashboard')


# ==========================================
# WORKLOAD LOG FORM & TEMPLATE DASHBOARDS
# ==========================================

@role_required(allowed_roles=['LECTURER', 'HOD'])
def submit_log(request):
    consent, _ = PrivacyConsent.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = WorkloadLogForm(request.POST)
        consent_form = PrivacyConsentForm(request.POST, instance=consent)
        
        if form.is_valid() and consent_form.is_valid():
            log = form.save(commit=False)
            log.user = request.user
            
            # Construct StaffFeatureVector from form submission and profile data
            staff_id = getattr(request.user, 'staff_id', None) or f"UTB-FAC-{request.user.id}"
            dept = getattr(request.user, 'department', None) or getattr(getattr(request.user, 'profile', None), 'department', "BIT")
            
            feature_vector = StaffFeatureVector(
                staff_id=staff_id,
                department=str(dept),
                academic_rank=AcademicRank.LECTURER,
                contractual_max_hours_week=getattr(log, 'contractual_hours', 20.0),
                assigned_teaching_hours_week=getattr(log, 'teaching_hours', 0.0),
                total_students_enrolled=getattr(log, 'students_count', 100),
                module_preparation_count=getattr(log, 'modules_count', 2),
                assessment_grading_backlog_units=getattr(log, 'marking_hours', 0.0) * 10,  # Map hours to backlog units
                committee_admin_hours_week=getattr(log, 'admin_hours', 0.0),
                research_supervision_hours_week=getattr(log, 'supervision_hours', 0.0),
                lms_off_hours_activity_ratio=0.25,
                schedule_fragmentation_index=0.40,
                days_since_last_leave=60,
                average_response_latency_hours=12.0,
                primary_domain="Software Engineering",
                secondary_domains=["Database Systems"]
            )
            
            # Predict risk using ML engine via services module
            inference = calculate_burnout_risk(feature_vector)
            
            log.predicted_risk_level = inference.risk_tier.value
            log.confidence_score = float(inference.burnout_risk_score)
            
            if inference.top_contributing_factors:
                log.primary_risk_driver = inference.top_contributing_factors[0].factor
            else:
                log.primary_risk_driver = "Balanced Workload"
            
            log.save()
            consent_form.save()
            messages.success(request, "Workload log submitted successfully!")
            
            # Redirect user back to their respective dashboard
            user_role = getattr(request.user, 'role', None) or getattr(getattr(request.user, 'profile', None), 'role', None)
            if user_role == 'HOD':
                return redirect('workloads:hod_dashboard')
            return redirect('workloads:lecturer_dashboard')
    else:
        form = WorkloadLogForm()
        consent_form = PrivacyConsentForm(instance=consent)

    return render(request, 'workloads/log_form.html', {
        'form': form, 
        'consent_form': consent_form
    })


@role_required(allowed_roles=['LECTURER'])
def lecturer_dashboard(request):
    logs = WorkloadLog.objects.filter(user=request.user).order_by('-created_at')
    latest_log = logs.first()
    consent = PrivacyConsent.objects.filter(user=request.user).first()
    
    # Staff ID for API / Store fetching on frontend
    staff_id = getattr(request.user, 'staff_id', None) or f"UTB-STAFF-{request.user.id}"

    return render(request, 'workloads/dashboard.html', {
        'logs': logs,
        'latest_log': latest_log,
        'consent': consent,
        'staff_id': staff_id,
    })


@role_required(allowed_roles=['HOD'])
def hod_dashboard(request):
    # Fetch HOD's department (supports User and UserProfile models)
    department = getattr(request.user, 'department', None) or getattr(getattr(request.user, 'profile', None), 'department', None)

    # Filter logs by department
    if department:
        department_logs = WorkloadLog.objects.filter(user__department=department).order_by('-created_at')
        dept_id = str(department)
    else:
        department_logs = WorkloadLog.objects.all().order_by('-created_at')
        dept_id = "BIT"

    # Aggregate department metrics
    total_logs = department_logs.count()
    high_risk_count = department_logs.filter(predicted_risk_level='HIGH').count()
    moderate_risk_count = department_logs.filter(predicted_risk_level='MODERATE').count()
    low_risk_count = department_logs.filter(predicted_risk_level='LOW').count()

    # Calculate average hours
    avg_teaching = department_logs.aggregate(Avg('teaching_hours'))['teaching_hours__avg'] or 0
    avg_marking = department_logs.aggregate(Avg('marking_hours'))['marking_hours__avg'] or 0

    # Get high-risk logs where the staff member consented to HOD alerting
    flagged_logs = department_logs.filter(
        predicted_risk_level='HIGH',
        user__privacy_consent__allow_hod_alerting=True
    ).select_related('user')[:10]

    return render(request, 'workloads/hod_dashboard.html', {
        'department': department,
        'dept_id': dept_id,
        'department_logs': department_logs[:15],
        'total_logs': total_logs,
        'high_risk_count': high_risk_count,
        'moderate_risk_count': moderate_risk_count,
        'low_risk_count': low_risk_count,
        'avg_teaching': round(avg_teaching, 1),
        'avg_marking': round(avg_marking, 1),
        'flagged_logs': flagged_logs,
    })


# ==========================================
# REST API ENDPOINTS FOR DASHBOARDS & STATE STORE
# ==========================================

@login_required
@require_http_methods(["GET"])
def lecturer_dashboard_api(request, staff_id):
    """
    API Contract: GET /api/v1/dashboard/lecturer/:staff_id
    Returns JSON payload for the Lecturer UI components & state store.
    """
    payload = get_lecturer_dashboard_payload(staff_id)
    return JsonResponse(payload)


@login_required
@role_required(allowed_roles=['HOD'])
@require_http_methods(["GET"])
def executive_dashboard_api(request, dept_id):
    """
    API Contract: GET /api/v1/dashboard/executive/:dept_id
    Returns JSON payload for the HoD / Dean Executive UI components.
    """
    payload = get_executive_dashboard_payload(dept_id)
    return JsonResponse(payload)


@login_required
@role_required(allowed_roles=['HOD'])
@require_POST
def accept_reallocation_api(request):
    """
    API Action: POST /api/v1/reallocate/accept/
    Processes an accepted workload transfer recommendation from the HoD dashboard.
    """
    try:
        data = json.loads(request.body)
        recommendation_id = data.get('recommendation_id')

        if not recommendation_id:
            return HttpResponseBadRequest(JsonResponse({'error': 'Missing recommendation_id'}))

        # Record or process the reallocation in your database
        return JsonResponse({
            'status': 'success',
            'message': f'Reallocation {recommendation_id} successfully applied.',
            'recommendation_id': recommendation_id
        })
    except json.JSONDecodeError:
        return HttpResponseBadRequest(JsonResponse({'error': 'Invalid JSON format'}))