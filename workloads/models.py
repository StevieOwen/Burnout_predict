from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db.models.signals import post_save
from django.dispatch import receiver


class User(AbstractUser):
    ROLE_CHOICES = (
        ('LECTURER', 'Lecturer / Academic Staff'),
        ('HOD', 'Head of Department / Dean'),
        ('ADMIN_STAFF', 'Administrative Officer'),
        ('HR_MANAGER', 'HR Manager / Governance'),
        ('SYSTEM_ADMIN', 'System Administrator'),
    )
    
    DEPARTMENT_CHOICES = (
        ('BIT', 'Business Information Technology'),
        ('HTM', 'Hospitality & Tourism Management'),
        ('BA', 'Business Administration'),
        ('ADM', 'Central Administration'),
    )

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='LECTURER')
    department = models.CharField(max_length=10, choices=DEPARTMENT_CHOICES, default='BIT')

    def is_lecturer(self):
        return self.role == 'LECTURER'

    def is_hod(self):
        return self.role == 'HOD'


class UserProfile(models.Model):
    RANK_CHOICES = (
        ('TUTOR', 'Tutorial Assistant'),
        ('LECTURER', 'Lecturer'),
        ('SENIOR_LECTURER', 'Senior Lecturer'),
        ('ASSOC_PROF', 'Associate Professor'),
        ('PROFESSOR', 'Professor'),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=User.ROLE_CHOICES, default='LECTURER')
    department = models.CharField(max_length=10, choices=User.DEPARTMENT_CHOICES, default='BIT')
    academic_rank = models.CharField(max_length=20, choices=RANK_CHOICES, default='LECTURER')
    
    # Contractual and Capacity Parameters
    contractual_max_hours = models.FloatField(default=20.0, validators=[MinValueValidator(0.0)])
    active_student_count = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    grading_backlog_count = models.IntegerField(default=0, validators=[MinValueValidator(0)])

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} ({self.get_academic_rank_display()})"


class WorkloadLog(models.Model):
    RISK_LEVELS = (
        ('LOW', 'Low Risk'),
        ('MODERATE', 'Moderate Risk'),
        ('HIGH', 'High Risk'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='workload_logs')
    week_starting = models.DateField(help_text="Start date of the logged week")
    
    # Machine Learning Input Features
    teaching_hours = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])
    marking_hours = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])
    supervisory_hours = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])
    administrative_hours = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])
    meeting_hours = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])
    module_prep_hours = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])
    
    # Digital Footprint & Off-Hours Metrics (for UI trend charts)
    lms_off_hours_activity = models.FloatField(default=0.0, help_text="Hours logged on LMS between 8 PM and 6 AM or weekends.")
    upcoming_deadlines_count = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    perceived_complexity_score = models.IntegerField(
        default=3, 
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )

    # ML Prediction & Optimization Outputs
    burnout_risk_score = models.FloatField(default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(100.0)], help_text="BRS 0-100%")
    predicted_risk_level = models.CharField(max_length=10, choices=RISK_LEVELS, blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)
    primary_risk_driver = models.CharField(max_length=100, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def total_hours(self):
        return (
            self.teaching_hours + 
            self.marking_hours + 
            self.supervisory_hours + 
            self.administrative_hours + 
            self.meeting_hours + 
            self.module_prep_hours
        )

    class Meta:
        ordering = ['-week_starting']
        unique_together = ['user', 'week_starting']


class PrivacyConsent(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='privacy_consent')
    allow_hod_alerting = models.BooleanField(
        default=False, 
        help_text="Allow HoD to view my identity when high burnout risk is detected."
    )
    updated_at = models.DateTimeField(auto_now=True)


class AssistanceRequest(models.Model):
    STATUS_CHOICES = (
        ('PENDING', 'Pending Review'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Declined'),
    )
    
    REQUEST_TYPES = (
        ('FATIGUE', 'Fatigue / Mental Health Flag'),
        ('MARKING_SUPPORT', 'Marking / Grading Assistant Request'),
        ('LEAVE', 'Short-term Workload Adjustment / Leave'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='assistance_requests')
    request_type = models.CharField(max_length=20, choices=REQUEST_TYPES)
    reason = models.TextField()
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)


class TaskReallocation(models.Model):
    STATUS_CHOICES = (
        ('SUGGESTED', 'AI Suggested'),
        ('ACCEPTED', 'Accepted'),
        ('MODIFIED', 'Modified'),
        ('REJECTED', 'Overridden / Rejected'),
    )

    from_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reallocations_out')
    to_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reallocations_in')
    task_description = models.CharField(max_length=255)
    hours_reallocated = models.FloatField(validators=[MinValueValidator(0.5)])
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='SUGGESTED')
    hod_comment = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(
            user=instance, 
            role=instance.role, 
            department=instance.department
        )
    else:
        profile, _ = UserProfile.objects.get_or_create(user=instance)
        if profile.role != instance.role or profile.department != instance.department:
            profile.role = instance.role
            profile.department = instance.department
            profile.save()