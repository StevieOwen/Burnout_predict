from django.test import TestCase
from ml_engine.workload_engine import (
    evaluate_staff_workload,
    StaffFeatureVector,
    AcademicRank,
    RiskTier,
)


class MLEngineTestCase(TestCase):
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
            max_supervision_capacity=10,
            current_supervised_students=8,
        )

        self.low_risk_staff = StaffFeatureVector(
            staff_id="UTB-FAC-102",
            department="BIT",
            academic_rank=AcademicRank.SENIOR_LECTURER,
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
            primary_domain="Software Engineering",
            secondary_domains=["Web Development"],
            max_supervision_capacity=10,
            current_supervised_students=2,
        )

    def test_high_burnout_risk_inference(self):
        """Verify high-risk feature vector produces HIGH risk tier and action_required=True."""
        result = evaluate_staff_workload(self.high_risk_staff)

        self.assertEqual(result.staff_id, "UTB-FAC-101")
        self.assertGreaterEqual(result.burnout_risk_score, 75.0)
        self.assertEqual(result.risk_tier, RiskTier.HIGH)
        self.assertTrue(result.action_required)
        self.assertGreater(len(result.top_contributing_factors), 0)

    def test_low_burnout_risk_inference(self):
        """Verify low-risk feature vector produces LOW risk tier and action_required=False."""
        result = evaluate_staff_workload(self.low_risk_staff)

        self.assertEqual(result.staff_id, "UTB-FAC-102")
        self.assertLess(result.burnout_risk_score, 50.0)
        self.assertEqual(result.risk_tier, RiskTier.LOW)
        self.assertFalse(result.action_required)

    def test_shap_factors_sorting(self):
        """Ensure top contributing factors are returned in descending order of impact."""
        result = evaluate_staff_workload(self.high_risk_staff)
        weights = [f.weight for f in result.top_contributing_factors]
        
        self.assertEqual(weights, sorted(weights, reverse=True))