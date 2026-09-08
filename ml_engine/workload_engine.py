from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class AcademicRank(str, Enum):
    TUTOR = "TUTOR"
    LECTURER = "LECTURER"
    SENIOR_LECTURER = "SENIOR_LECTURER"
    ASSOC_PROF = "ASSOC_PROF"
    PROFESSOR = "PROFESSOR"


class RiskTier(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class InterventionType(str, Enum):
    TASK_REASSIGNMENT = "TASK_REASSIGNMENT"
    SCHEDULE_CONSOLIDATION = "SCHEDULE_CONSOLIDATION"


# ==========================================
# 1. Feature Vector Schema (Model Input)
# ==========================================
class StaffFeatureVector(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # Identity & Baseline
    staff_id: str = Field(..., description="Unique UTB Staff ID (e.g. UTB-FAC-102)")
    department: str = Field(..., description="Department Code (BIT, HTM, BA)")
    academic_rank: AcademicRank
    contractual_max_hours_week: float = Field(default=20.0, ge=0.0, le=40.0)

    # Workload Drivers
    assigned_teaching_hours_week: float = Field(..., ge=0.0)
    total_students_enrolled: int = Field(..., ge=0)
    module_preparation_count: int = Field(..., ge=0)
    assessment_grading_backlog_units: int = Field(..., ge=0)
    committee_admin_hours_week: float = Field(..., ge=0.0)
    research_supervision_hours_week: float = Field(..., ge=0.0)

    # Behavioral Stress Indicators
    lms_off_hours_activity_ratio: float = Field(
        ..., ge=0.0, le=1.0, 
        description="Ratio of LMS actions recorded between 10 PM and 6 AM"
    )
    schedule_fragmentation_index: float = Field(
        ..., ge=0.0, le=1.0, 
        description="Index measuring idle gaps between scheduled classes (0.0 to 1.0)"
    )
    days_since_last_leave: int = Field(..., ge=0)
    average_response_latency_hours: float = Field(..., ge=0.0)

    # Constraints & Domains
    primary_domain: str = Field(..., description="Primary area of academic expertise")
    secondary_domains: List[str] = Field(default_factory=list)
    max_supervision_capacity: int = Field(default=10, ge=0)
    current_supervised_students: int = Field(default=0, ge=0)


# ==========================================
# 2. Model Inference Engine Outputs
# ==========================================
class FactorContribution(BaseModel):
    factor: str = Field(..., description="Feature name driving burnout risk score")
    weight: float = Field(..., ge=0.0, le=1.0, description="Relative impact percentage")


class SubstituteCandidate(BaseModel):
    staff_id: str
    name: str
    current_brs: float = Field(..., ge=0.0, le=100.0)
    match_score: float = Field(..., ge=0.0, le=1.0, description="Domain and capacity compatibility score")


class PrescriptiveIntervention(BaseModel):
    intervention_type: InterventionType
    task_name: str = Field(..., description="Task or section to offload, e.g., BIT201 Lab Section 2")
    hours_to_offload: float = Field(..., ge=0.5)
    suggested_substitutes: List[SubstituteCandidate] = Field(default_factory=list)


class ModelInferenceResult(BaseModel):
    staff_id: str
    burnout_risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_tier: RiskTier
    top_contributing_factors: List[FactorContribution]
    action_required: bool
    recommended_interventions: List[PrescriptiveIntervention] = Field(default_factory=list)