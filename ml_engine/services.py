# ml_engine/services.py
import os
import joblib
import numpy as np
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_engine', 'trained_models', 'burnout_rf_v1.joblib')

def predict_workload_risk(log_instance):
    features = np.array([[
        log_instance.teaching_hours,
        log_instance.marking_hours,
        log_instance.supervisory_hours,
        log_instance.administrative_hours,
        log_instance.meeting_hours,
        log_instance.upcoming_deadlines_count,
        log_instance.perceived_complexity_score
    ]])

    if not os.path.exists(MODEL_PATH):
        # Fallback if model file is missing
        return 'LOW', 0.50, 'Baseline Calculation'

    model = joblib.load(MODEL_PATH)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    # Identify primary risk driver
    feature_values = [
        ('Teaching Load', log_instance.teaching_hours),
        ('Marking Load', log_instance.marking_hours),
        ('Supervisory Load', log_instance.supervisory_hours),
        ('Administrative Load', log_instance.administrative_hours),
        ('Meetings/Committees', log_instance.meeting_hours),
        ('Upcoming Deadlines', log_instance.upcoming_deadlines_count * 5),  # weighted for comparison
        ('Task Complexity', log_instance.perceived_complexity_score * 4)
    ]
    
    primary_driver = sorted(feature_values, key=lambda x: x[1], reverse=True)[0][0]

    return prediction, confidence, f"Elevated {primary_driver}"