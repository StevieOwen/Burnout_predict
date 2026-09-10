from django.test import TestCase
from ml_engine.workload_engine import (
    evaluate_staff_workload,
    StaffFeatureVector,
    AcademicRank,
)


class MILPReallocationTestCase(TestCase):
    def setUp(self):
        self.high_risk_staff = StaffFeatureVector(
            staff_id="UTB-FAC-101",
            department="BIT",
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
            secondary_domains=["Database Systems"],
        )

        self.qualified_colleague = StaffFeatureVector(
            staff_id="UTB-FAC-102",
            department="BIT",
            academic_rank=AcademicRank.SENIOR_LECTURER,
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
            secondary_domains=["Web Development"],
        )

        self.unqualified_colleague = StaffFeatureVector(
            staff_id="UTB-FAC-103",
            department="BIT",
            academic_rank=AcademicRank.LECTURER,
            contractual_max_hours_week=20.0,
            assigned_teaching_hours_week=6.0,
            total_students_enrolled=50,
            module_preparation_count=1,
            assessment_grading_backlog_units=10,
            committee_admin_hours_week=1.0,
            research_supervision_hours_week=1.0,
            lms_off_hours_activity_ratio=0.05,
            schedule_fragmentation_index=0.10,
            days_since_last_leave=10,
            average_response_latency_hours=4.0,
            primary_domain="Cybersecurity",
            secondary_domains=["Networking"],
        )

    def test_reallocation_generates_interventions(self):
        """Ensure MILP engine generates non-empty interventions when qualified peers exist."""
        result = evaluate_staff_workload(
            self.high_risk_staff,
            department_colleagues=[self.qualified_colleague]
        )

        self.assertGreater(len(result.recommended_interventions), 0)
        intervention = result.recommended_interventions[0]
        self.assertEqual(intervention.intervention_type, "TASK_REASSIGNMENT")
        self.assertGreater(intervention.hours_to_offload, 0.0)

    def test_reallocation_substitute_matching(self):
        """Ensure MILP engine selects candidate with domain matching and low BRS."""
        result = evaluate_staff_workload(
            self.high_risk_staff,
            department_colleagues=[self.qualified_colleague, self.unqualified_colleague]
        )

        intervention = result.recommended_interventions[0]
        substitute_ids = [sub.staff_id for sub in intervention.suggested_substitutes]

        self.assertIn("UTB-FAC-102", substitute_ids)
        self.assertNotIn("UTB-FAC-103", substitute_ids)