import math
from typing import Dict, List
from ml_engine.workload_engine import StaffFeatureVector, ModelInferenceResult, RiskTier


def calculate_burnout_risk(feature_vector: StaffFeatureVector) -> ModelInferenceResult:
    """
    Inference computation calculating Burnout Risk Score (BRS) from
    workload features and behavioral metrics.
    """
    total_hours = (
        feature_vector.assigned_teaching_hours_week +
        feature_vector.committee_admin_hours_week +
        feature_vector.research_supervision_hours_week
    )

    # Workload Ratio vs Contract Max
    load_ratio = total_hours / max(feature_vector.contractual_max_hours_week, 1.0)
    
    # Risk calculation weights
    w_hours = min(load_ratio * 35.0, 45.0)
    w_backlog = min(feature_vector.assessment_grading_backlog_units * 0.4, 20.0)
    w_lms_off = feature_vector.lms_off_hours_activity_ratio * 20.0
    w_fragment = feature_vector.schedule_fragmentation_index * 15.0

    raw_brs = round(w_hours + w_backlog + w_lms_off + w_fragment, 1)
    brs = min(max(raw_brs, 5.0), 98.0)

    if brs >= 75.0:
        tier = RiskTier.HIGH
    elif brs >= 50.0:
        tier = RiskTier.MODERATE
    else:
        tier = RiskTier.LOW

    # Driver factor rank
    factors = [
        {"factor": "Contractual Hours Overload", "weight": round(w_hours / brs, 2)},
        {"factor": "Assessment Grading Backlog", "weight": round(w_backlog / brs, 2)},
        {"factor": "Late Night LMS Off-Hours Activity", "weight": round(w_lms_off / brs, 2)},
        {"factor": "Class Schedule Fragmentation Gaps", "weight": round(w_fragment / brs, 2)}
    ]
    factors.sort(key=lambda x: x["weight"], reverse=True)

    return ModelInferenceResult(
        staff_id=feature_vector.staff_id,
        burnout_risk_score=brs,
        risk_tier=tier,
        top_contributing_factors=factors[:3],
        action_required=(tier == RiskTier.HIGH),
        recommended_interventions=[]
    )


def get_lecturer_dashboard_payload(staff_id: str) -> Dict:
    """
    GET /api/v1/dashboard/lecturer/:staff_id implementation
    """
    # Simulated Feature Ingestion for target lecturer
    fv = StaffFeatureVector(
        staff_id=staff_id,
        department="BIT",
        academic_rank="LECTURER",
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
    top_driver = inference.top_contributing_factors[0]["factor"]

    status_badge = "Optimal"
    if inference.risk_tier == RiskTier.HIGH:
        status_badge = "High Burnout Risk"
    elif inference.risk_tier == RiskTier.MODERATE:
        status_badge = "Moderate Load"

    return {
        "profile": {
            "staff_id": fv.staff_id,
            "name": "Dr. Eric Manirakiza",
            "rank": "Lecturer",
            "department": "Business Information Technology",
            "status_badge": status_badge
        },
        "kpis": {
            "weekly_load_hours": round(fv.assigned_teaching_hours_week + fv.committee_admin_hours_week + fv.research_supervision_hours_week, 1),
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
                {"category": "Prep", "hours": 4.0}
            ],
            "off_hours_trend": [
                {"week": "Wk 1", "scheduled_hours": 20.0, "off_hours_lms_logs": 3},
                {"week": "Wk 2", "scheduled_hours": 21.0, "off_hours_lms_logs": 6},
                {"week": "Wk 3", "scheduled_hours": 23.5, "off_hours_lms_logs": 14},
                {"week": "Wk 4", "scheduled_hours": 23.5, "off_hours_lms_logs": 11}
            ]
        },
        "ai_advisor_message": (
            f"Your Burnout Risk Index is currently elevated at {inference.burnout_risk_score}%. "
            f"The primary stress driver is '{top_driver}'. "
            f"Recommendation: Request a task offload of 3.5 grading hours or shift office hours to consolidate schedule gaps."
        )
    }


def get_executive_dashboard_payload(dept_id: str) -> Dict:
    """
    GET /api/v1/dashboard/executive/:dept_id implementation
    """
    return {
        "department_summary": {
            "department_id": dept_id,
            "department_name": "Business Information Technology (BIT)",
            "average_burnout_risk_score": 68.4,
            "high_risk_faculty_count": 2,
            "workload_imbalance_index": 0.38,
            "pending_assistance_requests": 3
        },
        "heatmap_data": [
            {"staff_id": "UTB-FAC-101", "name": "Dr. Eric Manirakiza", "workload_hours": 23.5, "brs": 84.2, "risk_zone": "HIGH"},
            {"staff_id": "UTB-FAC-102", "name": "Prof. Jean Ndayisaba", "workload_hours": 21.0, "brs": 76.5, "risk_zone": "HIGH"},
            {"staff_id": "UTB-FAC-103", "name": "Dr. Paul Kwizera", "workload_hours": 17.5, "brs": 52.0, "risk_zone": "MODERATE"},
            {"staff_id": "UTB-FAC-104", "name": "Lecturer Marie Uwase", "workload_hours": 14.0, "brs": 31.8, "risk_zone": "LOW"},
            {"staff_id": "UTB-FAC-105", "name": "Tutor Aimable Mussa", "workload_hours": 12.5, "brs": 28.0, "risk_zone": "LOW"}
        ],
        "prescriptive_optimization_panel": [
            {
                "recommendation_id": "REC-BIT-001",
                "target_staff_id": "UTB-FAC-101",
                "target_staff_name": "Dr. Eric Manirakiza",
                "current_brs": 84.2,
                "proposed_action": "Transfer BIT201 Lab Section 2 Grading",
                "hours_to_transfer": 3.5,
                "projected_brs_after": 66.4,
                "substitute_staff_id": "UTB-FAC-104",
                "substitute_staff_name": "Lecturer Marie Uwase",
                "substitute_current_brs": 31.8,
                "substitute_projected_brs": 42.5,
                "status": "SUGGESTED"
            },
            {
                "recommendation_id": "REC-BIT-002",
                "target_staff_id": "UTB-FAC-102",
                "target_staff_name": "Prof. Jean Ndayisaba",
                "current_brs": 76.5,
                "proposed_action": "Reassign Final Year Thesis Supervision (2 Students)",
                "hours_to_transfer": 3.0,
                "projected_brs_after": 61.0,
                "substitute_staff_id": "UTB-FAC-105",
                "substitute_staff_name": "Tutor Aimable Mussa",
                "substitute_current_brs": 28.0,
                "substitute_projected_brs": 37.2,
                "status": "SUGGESTED"
            }
        ]
    }