/**
 * UTB Workload Optimization State Store
 */
class WorkloadOptimizationStore {
  constructor() {
    this.executiveData = null;
    this.lecturerData = null;
    this.subscribers = [];
  }

  // Subscribe UI render callbacks
  subscribe(callback) {
    this.subscribers.push(callback);
  }

  notify() {
    this.subscribers.forEach(cb => cb(this.getSnapshot()));
  }

  getSnapshot() {
    return {
      executive: this.executiveData,
      lecturer: this.lecturerData
    };
  }

  // Initial load
  async initialize(deptId = "BIT", staffId = "UTB-FAC-101") {
    try {
      // In production, replaced with actual fetch(`/api/v1/...`)
      this.executiveData = await this.fetchExecutiveMock(deptId);
      this.lecturerData = await this.fetchLecturerMock(staffId);
      this.notify();
    } catch (err) {
      console.error("Failed to initialize workload store:", err);
    }
  }

  /**
   * ACTION CONTROL: Accept Recommendation & Reallocate Workload
   * Executes dynamic BRS reduction and state synchronization
   */
  acceptReallocation(recommendationId) {
    if (!this.executiveData) return;

    const rec = this.executiveData.prescriptive_optimization_panel.find(
      r => r.recommendation_id === recommendationId
    );

    if (!rec || rec.status === 'ACCEPTED') return;

    // 1. Mark recommendation as ACCEPTED
    rec.status = 'ACCEPTED';

    // 2. Update Target Staff in Heatmap
    const targetStaff = this.executiveData.heatmap_data.find(s => s.staff_id === rec.target_staff_id);
    if (targetStaff) {
      targetStaff.workload_hours = Math.max(0, targetStaff.workload_hours - rec.hours_to_transfer);
      targetStaff.brs = rec.projected_brs_after;
      targetStaff.risk_zone = rec.projected_brs_after >= 75 ? 'HIGH' : (rec.projected_brs_after >= 50 ? 'MODERATE' : 'LOW');
    }

    // 3. Update Substitute Staff in Heatmap
    const subStaff = this.executiveData.heatmap_data.find(s => s.staff_id === rec.substitute_staff_id);
    if (subStaff) {
      subStaff.workload_hours += rec.hours_to_transfer;
      subStaff.brs = rec.substitute_projected_brs;
      subStaff.risk_zone = rec.substitute_projected_brs >= 75 ? 'HIGH' : (rec.substitute_projected_brs >= 50 ? 'MODERATE' : 'LOW');
    }

    // 4. Update Departmental Summary KPIs
    const highRiskCount = this.executiveData.heatmap_data.filter(s => s.risk_zone === 'HIGH').length;
    this.executiveData.department_summary.high_risk_faculty_count = highRiskCount;
    
    const sumBrs = this.executiveData.heatmap_data.reduce((acc, curr) => acc + curr.brs, 0);
    this.executiveData.department_summary.average_burnout_risk_score = +(sumBrs / this.executiveData.heatmap_data.length).toFixed(1);

    // 5. Synchronize Lecturer Dashboard if viewing the affected staff member
    if (this.lecturerData && this.lecturerData.profile.staff_id === rec.target_staff_id) {
      this.lecturerData.kpis.burnout_risk_score = rec.projected_brs_after;
      this.lecturerData.kpis.weekly_load_hours = Math.max(0, this.lecturerData.kpis.weekly_load_hours - rec.hours_to_transfer);
      this.lecturerData.profile.status_badge = rec.projected_brs_after >= 75 ? 'High Burnout Risk' : (rec.projected_brs_after >= 50 ? 'Moderate Load' : 'Optimal');
      this.lecturerData.ai_advisor_message = `Success: ${rec.hours_to_transfer} hours offloaded to ${rec.substitute_staff_name}. Your Burnout Risk Index dropped to ${rec.projected_brs_after}%.`;
    }

    // Trigger state change update
    this.notify();
  }

  // API Call abstractions
  async fetchLecturerMock(staffId) {
    return {
      profile: { staff_id: staffId, name: "Dr. Eric Manirakiza", rank: "Lecturer", department: "Business Information Technology", status_badge: "High Burnout Risk" },
      kpis: { weekly_load_hours: 23.5, contractual_max_hours: 20.0, burnout_risk_score: 84.2, active_students: 185, pending_assessments: 42 },
      charts: {
        workload_breakdown: [{ category: "Teaching", hours: 16.0 }, { category: "Grading", hours: 4.5 }, { category: "Admin", hours: 2.0 }, { category: "Research", hours: 1.0 }],
        off_hours_trend: [{ week: "Wk 1", scheduled_hours: 20, off_hours_lms_logs: 3 }, { week: "Wk 2", scheduled_hours: 21, off_hours_lms_logs: 6 }, { week: "Wk 3", scheduled_hours: 23.5, off_hours_lms_logs: 14 }]
      },
      ai_advisor_message: "Your Burnout Risk Index is high (84.2%). Recommendation: Offload 3.5 hours of grading."
    };
  }

  async fetchExecutiveMock(deptId) {
    return {
      department_summary: { department_id: deptId, department_name: "BIT", average_burnout_risk_score: 54.5, high_risk_faculty_count: 2, workload_imbalance_index: 0.38, pending_assistance_requests: 3 },
      heatmap_data: [
        { staff_id: "UTB-FAC-101", name: "Dr. Eric Manirakiza", workload_hours: 23.5, brs: 84.2, risk_zone: "HIGH" },
        { staff_id: "UTB-FAC-102", name: "Prof. Jean Ndayisaba", workload_hours: 21.0, brs: 76.5, risk_zone: "HIGH" },
        { staff_id: "UTB-FAC-104", name: "Lecturer Marie Uwase", workload_hours: 14.0, brs: 31.8, risk_zone: "LOW" }
      ],
      prescriptive_optimization_panel: [
        {
          recommendation_id: "REC-BIT-001",
          target_staff_id: "UTB-FAC-101",
          target_staff_name: "Dr. Eric Manirakiza",
          current_brs: 84.2,
          proposed_action: "Transfer BIT201 Lab Section 2 Grading",
          hours_to_transfer: 3.5,
          projected_brs_after: 66.4,
          substitute_staff_id: "UTB-FAC-104",
          substitute_staff_name: "Lecturer Marie Uwase",
          substitute_current_brs: 31.8,
          substitute_projected_brs: 42.5,
          status: "SUGGESTED"
        }
      ]
    };
  }
}

// Global Singleton Instance
window.workloadStore = new WorkloadOptimizationStore();