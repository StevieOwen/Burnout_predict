from django.test import TestCase
from workloads.services import get_lecturer_dashboard_payload, get_executive_dashboard_payload


class WorkloadServicesTestCase(TestCase):
    def test_lecturer_dashboard_payload_structure(self):
        """Verify get_lecturer_dashboard_payload contract key types."""
        payload = get_lecturer_dashboard_payload("UTB-FAC-101")

        self.assertIn("profile", payload)
        self.assertIn("kpis", payload)
        self.assertIn("charts", payload)
        self.assertIn("ai_advisor_message", payload)

        self.assertIsInstance(payload["kpis"]["burnout_risk_score"], float)
        self.assertIsInstance(payload["charts"]["workload_breakdown"], list)

    def test_executive_dashboard_payload_structure(self):
        """Verify get_executive_dashboard_payload structure and prescriptive panel."""
        payload = get_executive_dashboard_payload("BIT")

        self.assertIn("department_summary", payload)
        self.assertIn("heatmap_data", payload)
        self.assertIn("prescriptive_optimization_panel", payload)

        summary = payload["department_summary"]
        self.assertEqual(summary["department_id"], "BIT")
        self.assertGreaterEqual(summary["average_burnout_risk_score"], 0.0)
        self.assertIsInstance(payload["heatmap_data"], list)