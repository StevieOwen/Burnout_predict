import os
import joblib
from enum import Enum
from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ml_engine.optimizer import solve_workload_redistribution


# ==========================================
# Enums
# ==========================================
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


# ==========================================
# 3. Runtime Engine Class & Cached Loader
# ==========================================
_MODEL_CACHE = None
_EXPLAINER_CACHE = None


def _load_ml_artifacts():
    """Lazy-loads and caches model and SHAP explainer artifacts."""
    global _MODEL_CACHE, _EXPLAINER_CACHE

    if _MODEL_CACHE is None or _EXPLAINER_CACHE is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "trained_models")

        model_path = os.path.join(models_dir, "burnout_xgb_v1.joblib")
        explainer_path = os.path.join(models_dir, "shap_explainer_v1.joblib")

        if not os.path.exists(model_path) or not os.path.exists(explainer_path):
            raise FileNotFoundError(
                f"Missing ML artifacts in {models_dir}. Please run 'python ml_engine/train_model.py' first."
            )

        _MODEL_CACHE = joblib.load(model_path)
        _EXPLAINER_CACHE = joblib.load(explainer_path)

    return _MODEL_CACHE, _EXPLAINER_CACHE


def evaluate_staff_workload(
    staff_input: StaffFeatureVector,
    department_colleagues: Optional[List[StaffFeatureVector]] = None
) -> ModelInferenceResult:
    """
    Core execution engine:
    1. Converts Pydantic input feature vector to SWU & Stress Index
    2. Runs XGBoost inference for continuous Burnout Risk Score (BRS)
    3. Computes SHAP feature importance for top contributing drivers
    4. Triggers MILP prescriptive optimization via PuLP if BRS >= 75.0 (HIGH tier)
    """
    model, explainer = _load_ml_artifacts()

    # Calculate Standardised Workload Units (SWU)
    swu_total = (
        (staff_input.assigned_teaching_hours_week * 1.00) +
        (staff_input.module_preparation_count * 2.50) +
        (staff_input.assessment_grading_backlog_units * 0.15) +
        (staff_input.current_supervised_students * 0.50) +
        (staff_input.committee_admin_hours_week * 0.80)
    )

    wsi = swu_total / staff_input.contractual_max_hours_week if staff_input.contractual_max_hours_week > 0 else 1.0

    # Build DataFrame aligning with trained feature layout
    input_df = pd.DataFrame([{
        "workload_stress_index": wsi,
        "assigned_teaching_hours_week": staff_input.assigned_teaching_hours_week,
        "module_preparation_count": staff_input.module_preparation_count,
        "assessment_grading_backlog": staff_input.assessment_grading_backlog_units,
        "current_supervised_students": staff_input.current_supervised_students,
        "committee_admin_hours_week": staff_input.committee_admin_hours_week,
        "total_students_enrolled": staff_input.total_students_enrolled,
        "lms_off_hours_activity_ratio": staff_input.lms_off_hours_activity_ratio,
        "schedule_fragmentation_index": staff_input.schedule_fragmentation_index,
        "days_since_last_leave": float(staff_input.days_since_last_leave),
        "average_response_latency_hours": staff_input.average_response_latency_hours
    }])

    # Model Inference
    raw_brs = float(model.predict(input_df)[0])
    brs_predicted = round(float(np.clip(raw_brs, 0.0, 100.0)), 1)

    # Determine Risk Tier & Override Rules
    if brs_predicted >= 75.0 or wsi >= 1.35:
        tier = RiskTier.HIGH
        action_req = True
    elif brs_predicted >= 50.0:
        tier = RiskTier.MODERATE
        action_req = False
    else:
        tier = RiskTier.LOW
        action_req = False

    # Extract SHAP Contributions
    shap_vals = explainer(input_df)
    shap_contributions = np.abs(shap_vals.values[0])
    total_shap_sum = np.sum(shap_contributions) if np.sum(shap_contributions) > 0 else 1.0

    factor_weights = []
    feature_names = input_df.columns.tolist()
    for name, abs_val in zip(feature_names, shap_contributions):
        normalized_weight = round(float(abs_val / total_shap_sum), 2)
        factor_weights.append(
            FactorContribution(
                factor=name.replace("_", " ").title(),
                weight=min(1.0, max(0.0, normalized_weight))
            )
        )

    # Sort and take top 3 contributing drivers
    top_factors = sorted(factor_weights, key=lambda x: x.weight, reverse=True)[:3]

    # Generate Prescriptive Interventions if HIGH tier
    interventions = []
    if tier == RiskTier.HIGH and department_colleagues:
        # Break down backlog into manageable chunks (max 30 units per task)
        backlog_units = staff_input.assessment_grading_backlog_units
        chunk_size = min(30, backlog_units) if backlog_units > 0 else 0

        if chunk_size > 0:
            tasks_to_reallocate = [
                {
                    "task_id": f"TASK-GRADING-{staff_input.staff_id}-CHUNK1",
                    "task_type": "GRADING_BACKLOG",
                    "source_staff_id": staff_input.staff_id,
                    "required_domain": staff_input.primary_domain,
                    "swu_value": round(chunk_size * 0.15, 2)
                }
            ]

            # Dynamically calculate real current BRS for eligible colleagues
            colleague_profiles = []
            colleague_eval_map = {}

            for col in department_colleagues:
                if col.staff_id != staff_input.staff_id:
                    # Evaluate candidate to retrieve real BRS (department_colleagues=None prevents recursion)
                    col_res = evaluate_staff_workload(col, department_colleagues=None)
                    colleague_eval_map[col.staff_id] = col_res

                    colleague_profiles.append({
                        "staff_id": col.staff_id,
                        "name": getattr(col, 'name', col.staff_id),
                        "primary_domain": col.primary_domain,
                        "secondary_domains": col.secondary_domains,
                        "burnout_risk_score": col_res.burnout_risk_score,
                        "risk_tier": col_res.risk_tier.value
                    })

            # Run MILP Optimization
            recommendations = solve_workload_redistribution(colleague_profiles, tasks_to_reallocate)

            substitutes = []
            for rec in recommendations:
                cand_id = rec["recommended_target_staff_id"]
                cand_eval = colleague_eval_map.get(cand_id)

                substitutes.append(
                    SubstituteCandidate(
                        staff_id=cand_id,
                        name=cand_id,
                        current_brs=cand_eval.burnout_risk_score if cand_eval else 45.0,
                        match_score=0.92
                    )
                )

            if substitutes:
                interventions.append(
                    PrescriptiveIntervention(
                        intervention_type=InterventionType.TASK_REASSIGNMENT,
                        task_name=f"Reassign Grading Backlog Batch ({chunk_size} Units)",
                        hours_to_offload=round(chunk_size * 0.15, 1),
                        suggested_substitutes=substitutes
                    )
                )

    return ModelInferenceResult(
        staff_id=staff_input.staff_id,
        burnout_risk_score=brs_predicted,
        risk_tier=tier,
        top_contributing_factors=top_factors,
        action_required=action_req,
        recommended_interventions=interventions
    )