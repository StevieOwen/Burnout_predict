# ml_engine/train_model.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def generate_synthetic_utb_data(n_samples=1200):
    """Generates an evenly balanced synthetic workload dataset for UTB Kigali staff."""
    np.random.seed(42)

    # Features
    teaching_hours = np.random.uniform(0, 20, n_samples)
    marking_hours = np.random.uniform(0, 15, n_samples)
    supervisory_hours = np.random.uniform(0, 10, n_samples)
    administrative_hours = np.random.uniform(1, 15, n_samples)
    meeting_hours = np.random.uniform(1, 10, n_samples)
    deadlines_count = np.random.randint(0, 6, n_samples)
    complexity_score = np.random.randint(1, 6, n_samples)

    total_hours = teaching_hours + marking_hours + supervisory_hours + administrative_hours + meeting_hours

    risk_labels = []
    for i in range(n_samples):
        hours = total_hours[i]
        d_count = deadlines_count[i]
        c_score = complexity_score[i]

        # Calibrated thresholds for even distribution:
        # HIGH: Over 44 total hours OR heavy workload with multiple deadlines
        if hours > 44 or (hours > 38 and d_count >= 3) or (hours > 35 and c_score >= 4 and d_count >= 3):
            risk_labels.append('HIGH')
        # LOW: Standard/Light workload under 28 total hours with low deadlines/complexity
        elif hours < 28 and d_count <= 2 and c_score <= 3:
            risk_labels.append('LOW')
        # MODERATE: Moderate workload falling in between
        else:
            risk_labels.append('MODERATE')

    data = pd.DataFrame({
        'teaching_hours': teaching_hours,
        'marking_hours': marking_hours,
        'supervisory_hours': supervisory_hours,
        'administrative_hours': administrative_hours,
        'meeting_hours': meeting_hours,
        'deadlines_count': deadlines_count,
        'complexity_score': complexity_score,
        'risk_level': risk_labels
    })

    return data

def train_and_save_model():
    print("Generating balanced synthetic UTB workload dataset...")
    df = generate_synthetic_utb_data(1200)

    # Display dataset balance
    print("\nClass Distribution in Generated Dataset:")
    print(df['risk_level'].value_counts())
    print("-" * 50)

    X = df.drop('risk_level', axis=1)
    y = df['risk_level']

    # Stratified split ensures equal ratio of LOW, MODERATE, HIGH in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'burnout_rf_v1.joblib')
    joblib.dump(rf_model, model_path)
    print(f"Successfully saved updated model to: {model_path}")

if __name__ == '__main__':
    train_and_save_model()