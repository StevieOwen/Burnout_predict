# workloads/forms.py
from django import forms
from .models import WorkloadLog, PrivacyConsent

class WorkloadLogForm(forms.ModelForm):
    class Meta:
        model = WorkloadLog
        fields = [
            'week_starting', 
            'teaching_hours', 
            'marking_hours', 
            'supervisory_hours', 
            'administrative_hours', 
            'meeting_hours', 
            'upcoming_deadlines_count', 
            'perceived_complexity_score'
        ]
        widgets = {
            'week_starting': forms.DateInput(attrs={'type': 'date', 'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none'}),
            'teaching_hours': forms.NumberInput(attrs={'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none', 'step': '0.5'}),
            'marking_hours': forms.NumberInput(attrs={'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none', 'step': '0.5'}),
            'supervisory_hours': forms.NumberInput(attrs={'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none', 'step': '0.5'}),
            'administrative_hours': forms.NumberInput(attrs={'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none', 'step': '0.5'}),
            'meeting_hours': forms.NumberInput(attrs={'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none', 'step': '0.5'}),
            'upcoming_deadlines_count': forms.NumberInput(attrs={'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none'}),
            'perceived_complexity_score': forms.Select(
                choices=[(i, f"{i} - {'Low' if i<=2 else 'Moderate' if i<=4 else 'High'}") for i in range(1, 6)],
                attrs={'class': 'w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-utb-navy/20 outline-none'}
            ),
        }

class PrivacyConsentForm(forms.ModelForm):
    class Meta:
        model = PrivacyConsent
        fields = ['allow_hod_alerting']
        widgets = {
            'allow_hod_alerting': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-utb-navy rounded border-slate-300 focus:ring-utb-navy'
            })
        }