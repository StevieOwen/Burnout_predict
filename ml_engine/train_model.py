import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap


def generate_synthetic_utb_relational_data(n_samples=1200):
    """
    Generates synthetic UTB Kigali workload data incorporating SWU 
    calculations and continuous Burnout Risk Scores (BRS).
    """
    np.random.seed(42)

    # 1. Feature Generation
    teaching_hours = np.random.uniform(4, 20, n_samples)
    module_prep_count = np.random.randint(1, 5, n_samples)
    grading_backlog = np.random.randint(10, 250, n_samples)
    supervised_students = np.random.randint(1, 12, n_samples)
    admin_hours = np.random.uniform(1, 10, n_samples)
    total_enrolled = np.random.randint(60, 350, n_samples)

    contractual_max = np.random.choice([14.0, 18.0, 20.0, 24.0], size=n_samples)

    # Behavioral Features
    lms_off_hours = np.random.uniform(0.05, 0.60, n_samples)
    schedule_fragmentation = np.random.uniform(0.10, 0.90, n_samples)
    days_since_leave = np.random.randint(10, 200, n_samples)
    response_latency = np.random.uniform(2.0, 48.0, n_samples)

    # 2. SWU and Stress Index Calculations
    swu_total = (
        (teaching_hours * 1.00) +
        (module_prep_count * 2.50) +
        (grading_backlog * 0.15) +
        (supervised_students * 0.50) +
        (admin_hours * 0.80)
    )

    wsi = swu_total / contractual_max

    # 3. Target Continuous Burnout Risk Score (BRS: 0 - 100)
    brs_raw = (
        (wsi * 35.0) +
        (lms_off_hours * 25.0) +
        (schedule_fragmentation * 15.0) +
        ((grading_backlog / 250.0) * 15.0) +
        ((days_since_leave / 200.0) * 10.0) +
        np.random.normal(0, 2.0, n_samples)
    )

    brs_score = np.clip(brs_raw, 10.0, 98.0)

    data = pd.DataFrame({
        'workload_stress_index': wsi,
        'assigned_teaching_hours_week': teaching_hours,
        'module_preparation_count': module_prep_count,
        'assessment_grading_backlog': grading_backlog,
        'current_supervised_students': supervised_students,
        'committee_admin_hours_week': admin_hours,
        'total_students_enrolled': total_enrolled,
        'lms_off_hours_activity_ratio': lms_off_hours,
        'schedule_fragmentation_index': schedule_fragmentation,
        'days_since_last_leave': days_since_leave,
        'average_response_latency_hours': response_latency,
        'burnout_risk_score': brs_score
    })

    return data


def load_or_generate_dataset():
    """
    Attempts to merge existing UTB relational CSV files from the data/ directory
    at the project root; falls back to synthetic generation if files are missing.
    """
    # Dynamically locate project root and data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")

    profiles_path = os.path.join(data_dir, "faculty_profiles.csv")
    workloads_path = os.path.join(data_dir, "workload_activities.csv")
    behavioral_path = os.path.join(data_dir, "behavioral_logs.csv")
    ground_truth_path = os.path.join(data_dir, "burnout_ground_truth.csv")

    csv_paths = [profiles_path, workloads_path, behavioral_path, ground_truth_path]

    if all(os.path.exists(p) for p in csv_paths):
        print(f"Found local UTB relational CSVs in '{data_dir}'. Merging datasets...")
        profiles = pd.read_csv(profiles_path)
        workloads = pd.read_csv(workloads_path)
        behavioral = pd.read_csv(behavioral_path)
        ground_truth = pd.read_csv(ground_truth_path)

        df = profiles.merge(workloads, on="staff_id") \
                     .merge(behavioral, on="staff_id") \
                     .merge(ground_truth, on="staff_id")

        # Calculate WSI feature if missing
        if "workload_stress_index" not in df.columns:
            swu = (
                (df["assigned_teaching_hours_week"] * 1.0) +
                (df["module_preparation_count"] * 2.5) +
                (df["assessment_grading_backlog"] * 0.15) +
                (df["current_supervised_students"] * 0.5) +
                (df["committee_admin_hours_week"] * 0.8)
            )
            df["workload_stress_index"] = swu / df["contractual_max_hours_week"]

        feature_cols = [
            "workload_stress_index", "assigned_teaching_hours_week",
            "module_preparation_count", "assessment_grading_backlog",
            "current_supervised_students", "committee_admin_hours_week",
            "total_students_enrolled", "lms_off_hours_activity_ratio",
            "schedule_fragmentation_index", "days_since_last_leave",
            "average_response_latency_hours"
        ]
        return df[feature_cols], df["burnout_risk_score"]
    else:
        print(f"Relational CSVs not found in '{data_dir}'. Generating in-memory synthetic dataset...")
        df = generate_synthetic_utb_relational_data(1200)
        X = df.drop("burnout_risk_score", axis=1)
        y = df["burnout_risk_score"]
        return X, y


def train_and_save_model():
    X, y = load_or_generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("\nTraining XGBoost Regressor for Burnout Risk Scoring...")
    xgb_model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Performance Metrics
    y_pred = xgb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "=" * 50)
    print("      XGBOOST MODEL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  • R² Score (Variance Explained) : {r2:.4f}")
    print(f"  • Mean Absolute Error (MAE)    : {mae:.2f} BRS points")
    print(f"  • Root Mean Sq. Error (RMSE)   : {rmse:.2f} BRS points")
    print("=" * 50 + "\n")

    # Fit SHAP Explainer
    print("Fitting TreeSHAP Explainer for Model Explainability...")
    explainer = shap.TreeExplainer(xgb_model)

    # Save artifacts inside ml_engine/trained_models/
    models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'burnout_xgb_v1.joblib')
    explainer_path = os.path.join(models_dir, 'shap_explainer_v1.joblib')

    joblib.dump(xgb_model, model_path)
    joblib.dump(explainer, explainer_path)

    print(f"Successfully saved XGBoost model to: {model_path}")
    print(f"Successfully saved SHAP explainer to: {explainer_path}")


if __name__ == '__main__':
    train_and_save_model()