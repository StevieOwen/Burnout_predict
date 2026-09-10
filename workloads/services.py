import math
from typing import Dict, List, Optional

from ml_engine.workload_engine import (
    StaffFeatureVector,
    ModelInferenceResult,
    RiskTier,
    AcademicRank,
    evaluate_staff_workload,
)


def calculate_burnout_risk(
    feature_vector: StaffFeatureVector,
    department_colleagues: Optional[List[StaffFeatureVector]] = None,
) -> ModelInferenceResult:
    """
    Executes trained ML inference model and SHAP explainer to calculate Burnout 
    Risk Score (BRS), risk tier, drivers, and prescriptive MILP interventions.
    """
    return evaluate_staff_workload(
        staff_input=feature_vector,
        department_colleagues=department_colleagues
    )


def get_lecturer_dashboard_payload(staff_id: str) -> Dict:
    """
    GET /api/v1/dashboard/lecturer/:staff_id implementation
    Ingests staff metrics, runs XGBoost/SHAP evaluation, and formats response.
    """
    # Feature Ingestion for target lecturer (Can be mapped from Django ORM models)
    fv = StaffFeatureVector(
        staff_id=staff_id,
        department="BIT",
        academic_rank=AcademicRank.LECTURER,
        contractual_max_hours_week=20.0,
        assigned_teaching_hours_week=16.0,
        total_students_enrolled=185,
        module_preparation_count=2,
        assessment_grading_backlog_units=42,
        committee_admin_hours_week=4.5,
        research_supervision_hours_week=3.0,
        lms_off_hours_activity_ratio=0.38,
        schedule_fragmentation_index=0.65,
        days_since_last_leave=140,
        average_response_latency_hours=18.5,
        primary_domain="Software Engineering",
        secondary_domains=["Database Systems", "Web Development"]
    )

    inference = calculate_burnout_risk(fv)

    if inference.top_contributing_factors:
        top_driver = inference.top_contributing_factors[0].factor
    else:
        top_driver = "Balanced Workload"

    status_badge = "Optimal"
    if inference.risk_tier == RiskTier.HIGH:
        status_badge = "High Burnout Risk"
    elif inference.risk_tier == RiskTier.MODERATE:
        status_badge = "Moderate Load"

    weekly_load = round(
        fv.assigned_teaching_hours_week + fv.committee_admin_hours_week + fv.research_supervision_hours_week,
        1
    )

    return {
        "profile": {
            "staff_id": fv.staff_id,
            "name": "Dr. Eric Manirakiza",
            "rank": fv.academic_rank.value.title(),
            "department": "Business Information Technology",
            "status_badge": status_badge
        },
        "kpis": {
            "weekly_load_hours": weekly_load,
            "contractual_max_hours": fv.contractual_max_hours_week,
            "burnout_risk_score": inference.burnout_risk_score,
            "active_students": fv.total_students_enrolled,
            "pending_assessments": fv.assessment_grading_backlog_units
        },
        "charts": {
            "workload_breakdown": [
                {"category": "Teaching", "hours": fv.assigned_teaching_hours_week},
                {"category": "Grading", "hours": round(fv.assessment_grading_backlog_units * 0.15, 1)},
                {"category": "Admin", "hours": fv.committee_admin_hours_week},
                {"category": "Research", "hours": fv.research_supervision_hours_week},
                {"category": "Prep", "hours": round(fv.module_preparation_count * 2.5, 1)}
            ],
            "off_hours_trend": [
                {"week": "Wk 1", "scheduled_hours": 20.0, "off_hours_lms_logs": 3},
                {"week": "Wk 2", "scheduled_hours": 21.0, "off_hours_lms_logs": 6},
                {"week": "Wk 3", "scheduled_hours": 23.5, "off_hours_lms_logs": 14},
                {"week": "Wk 4", "scheduled_hours": 23.5, "off_hours_lms_logs": 11}
            ]
        },
        "ai_advisor_message": (
            f"Your Burnout Risk Index is currently evaluated at {inference.burnout_risk_score}%. "
            f"The primary stress driver is '{top_driver}'. "
            f"Recommendation: Request a task offload of 3.5 grading hours or shift office hours to consolidate schedule gaps."
        )
    }


def get_executive_dashboard_payload(dept_id: str) -> Dict:
    """
    GET /api/v1/dashboard/executive/:dept_id implementation
    Evaluates department faculty using the ML + MILP optimization engine.
    """
    # Department faculty pool
    faculty_members = [
        StaffFeatureVector(
            staff_id="UTB-FAC-101",
            department=dept_id,
            academic_rank=AcademicRank.LECTURER,
            contractual_max_hours_week=18.0,
            assigned_teaching_hours_week=18.0,
            total_students_enrolled=250,
            module_preparation_count=4,
            assessment_grading_backlog_units=180,
            committee_admin_hours_week=8.0,
            research_supervision_hours_week=4.0,
            lms_off_hours_activity_ratio=0.45,
            schedule_fragmentation_index=0.75,
            days_since_last_leave=120,
            average_response_latency_hours=24.0,
            primary_domain="Software Engineering",
            secondary_domains=["Database Systems"]
        ),
        StaffFeatureVector(
            staff_id="UTB-FAC-102",
            department=dept_id,
            academic_rank=AcademicRank.PROFESSOR,
            contractual_max_hours_week=18.0,
            assigned_teaching_hours_week=16.0,
            total_students_enrolled=190,
            module_preparation_count=3,
            assessment_grading_backlog_units=120,
            committee_admin_hours_week=6.0,
            research_supervision_hours_week=5.0,
            lms_off_hours_activity_ratio=0.35,
            schedule_fragmentation_index=0.60,
            days_since_last_leave=90,
            average_response_latency_hours=18.0,
            primary_domain="Database Systems",
            secondary_domains=["Software Engineering"]
        ),
        StaffFeatureVector(
            staff_id="UTB-FAC-103",
            department=dept_id,
            academic_rank=AcademicRank.LECTURER,
            contractual_max_hours_week=20.0,
            assigned_teaching_hours_week=12.0,
            total_students_enrolled=120,
            module_preparation_count=2,
            assessment_grading_backlog_units=40,
            committee_admin_hours_week=3.0,
            research_supervision_hours_week=2.5,
            lms_off_hours_activity_ratio=0.18,
            schedule_fragmentation_index=0.35,
            days_since_last_leave=45,
            average_response_latency_hours=10.0,
            primary_domain="Web Development",
            secondary_domains=["Software Engineering"]
        ),
        StaffFeatureVector(
            staff_id="UTB-FAC-104",
            department=dept_id,
            academic_rank=AcademicRank.LECTURER,
            contractual_max_hours_week=20.0,
            assigned_teaching_hours_week=8.0,
            total_students_enrolled=80,
            module_preparation_count=1,
            assessment_grading_backlog_units=20,
            committee_admin_hours_week=2.0,
            research_supervision_hours_week=2.0,
            lms_off_hours_activity_ratio=0.10,
            schedule_fragmentation_index=0.20,
            days_since_last_leave=15,
            average_response_latency_hours=6.0,
            primary_domain="Software Engineering",
            secondary_domains=["Web Development"]
        ),
        StaffFeatureVector(
            staff_id="UTB-FAC-105",
            department=dept_id,
            academic_rank=AcademicRank.TUTOR,
            contractual_max_hours_week=20.0,
            assigned_teaching_hours_week=6.0,
            total_students_enrolled=60,
            module_preparation_count=1,
            assessment_grading_backlog_units=10,
            committee_admin_hours_week=1.0,
            research_supervision_hours_week=0.0,
            lms_off_hours_activity_ratio=0.05,
            schedule_fragmentation_index=0.15,
            days_since_last_leave=10,
            average_response_latency_hours=4.0,
            primary_domain="Database Systems",
            secondary_domains=["Software Engineering"]
        ),
    ]

    staff_names = {
        "UTB-FAC-101": "Dr. Eric Manirakiza",
        "UTB-FAC-102": "Prof. Jean Ndayisaba",
        "UTB-FAC-103": "Dr. Paul Kwizera",
        "UTB-FAC-104": "Lecturer Marie Uwase",
        "UTB-FAC-105": "Tutor Aimable Mussa",
    }

    heatmap_data = []
    prescriptive_panel = []
    high_risk_count = 0
    total_brs = 0.0

    # Evaluate each faculty member against whole department
    for staff in faculty_members:
        res = calculate_burnout_risk(staff, department_colleagues=faculty_members)
        total_brs += res.burnout_risk_score

        if res.risk_tier == RiskTier.HIGH:
            high_risk_count += 1

        workload_hours = round(
            staff.assigned_teaching_hours_week + staff.committee_admin_hours_week + staff.research_supervision_hours_week,
            1
        )

        heatmap_data.append({
            "staff_id": staff.staff_id,
            "name": staff_names.get(staff.staff_id, staff.staff_id),
            "workload_hours": workload_hours,
            "brs": res.burnout_risk_score,
            "risk_zone": res.risk_tier.value
        })

        # Process prescriptive interventions generated by MILP solver
        for idx, interv in enumerate(res.recommended_interventions, start=1):
            for sub in interv.suggested_substitutes:
                prescriptive_panel.append({
                    "recommendation_id": f"REC-{dept_id}-{staff.staff_id[-3:]}-{idx}",
                    "target_staff_id": staff.staff_id,
                    "target_staff_name": staff_names.get(staff.staff_id, staff.staff_id),
                    "current_brs": res.burnout_risk_score,
                    "proposed_action": interv.task_name,
                    "hours_to_transfer": interv.hours_to_offload,
                    "projected_brs_after": round(max(5.0, res.burnout_risk_score - (interv.hours_to_offload * 3.5)), 1),
                    "substitute_staff_id": sub.staff_id,
                    "substitute_staff_name": staff_names.get(sub.staff_id, sub.name),
                    "substitute_current_brs": sub.current_brs,
                    "substitute_projected_brs": round(sub.current_brs + (interv.hours_to_offload * 3.5), 1),
                    "status": "SUGGESTED"
                })

    avg_brs = round(total_brs / len(faculty_members), 1) if faculty_members else 0.0

    return {
        "department_summary": {
            "department_id": dept_id,
            "department_name": f"{dept_id} Department",
            "average_burnout_risk_score": avg_brs,
            "high_risk_faculty_count": high_risk_count,
            "workload_imbalance_index": 0.38,
            "pending_assistance_requests": len(prescriptive_panel)
        },
        "heatmap_data": heatmap_data,
        "prescriptive_optimization_panel": prescriptive_panel
    }