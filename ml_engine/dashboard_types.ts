// ==========================================
// 1. Feature Vector Interface
// ==========================================
export type AcademicRank = 'TUTOR' | 'LECTURER' | 'SENIOR_LECTURER' | 'ASSOC_PROF' | 'PROFESSOR';
export type RiskTier = 'LOW' | 'MODERATE' | 'HIGH';
export type InterventionType = 'TASK_REASSIGNMENT' | 'SCHEDULE_CONSOLIDATION';

export interface StaffFeatureVector {
  staff_id: string;
  department: string;
  academic_rank: AcademicRank;
  contractual_max_hours_week: number;
  assigned_teaching_hours_week: number;
  total_students_enrolled: number;
  module_preparation_count: number;
  assessment_grading_backlog_units: number;
  committee_admin_hours_week: number;
  research_supervision_hours_week: number;
  lms_off_hours_activity_ratio: number;
  schedule_fragmentation_index: number;
  days_since_last_leave: number;
  average_response_latency_hours: number;
  primary_domain: string;
  secondary_domains: string[];
  max_supervision_capacity: number;
  current_supervised_students: number;
}

// ==========================================
// 2. Lecturer Dashboard API Contract
// ==========================================
export interface LecturerDashboardPayload {
  profile: {
    staff_id: string;
    name: string;
    rank: string;
    department: string;
    status_badge: 'Optimal' | 'Moderate Load' | 'High Burnout Risk';
  };
  kpis: {
    weekly_load_hours: number;
    contractual_max_hours: number;
    burnout_risk_score: number;
    active_students: number;
    pending_assessments: number;
  };
  charts: {
    workload_breakdown: Array<{
      category: 'Teaching' | 'Grading' | 'Admin' | 'Research' | 'Prep';
      hours: number;
    }>;
    off_hours_trend: Array<{
      week: string;
      scheduled_hours: number;
      off_hours_lms_logs: number;
    }>;
  };
  ai_advisor_message: string;
}

// ==========================================
// 3. HoD / Dean Executive Dashboard API Contract
// ==========================================
export interface HeatmapItem {
  staff_id: string;
  name: string;
  workload_hours: number;
  brs: number;
  risk_zone: RiskTier;
}

export interface PrescriptiveOption {
  recommendation_id: string;
  target_staff_id: string;
  target_staff_name: string;
  current_brs: number;
  proposed_action: string;
  hours_to_transfer: number;
  projected_brs_after: number;
  substitute_staff_id: string;
  substitute_staff_name: string;
  substitute_current_brs: number;
  substitute_projected_brs: number;
  status: 'SUGGESTED' | 'ACCEPTED' | 'REJECTED';
}

export interface ExecutiveDashboardPayload {
  department_summary: {
    department_id: string;
    department_name: string;
    average_burnout_risk_score: number;
    high_risk_faculty_count: number;
    workload_imbalance_index: number;
    pending_assistance_requests: number;
  };
  heatmap_data: HeatmapItem[];
  prescriptive_optimization_panel: PrescriptiveOption[];
}