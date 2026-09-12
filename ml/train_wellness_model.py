import json
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

FEATURE_NAMES = [
    'sleep_hours',
    'work_study_hours',
    'stress_level',
    'physical_activity_minutes',
    'mood_rating',
    'journal_sentiment',
    'negative_emotion_frequency'
]

TARGET_NAME = 'risk_class'

def train_model():
    print("Loading synthetic wellness dataset...")
    df = pd.read_csv('data/synthetic_wellness_dataset.csv', comment='#')
    
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    
    # Pipeline with SimpleImputer (median) + RandomForestClassifier
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Fitting Random Forest pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Predictions on test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro'))
    rec_macro = float(recall_score(y_test, y_pred, average='macro'))
    f1_macro = float(f1_score(y_test, y_pred, average='macro'))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract feature importance from the fitted RF
    rf_model = pipeline.named_steps['rf']
    imputer = pipeline.named_steps['imputer']
    importances = rf_model.feature_importances_
    
    feature_importance_list = []
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        feature_importance_list.append({
            'feature': name,
            'importance': round(float(imp), 4)
        })
    
    print("\n--- MODEL EVALUATION (ON SYNTHETIC DATA) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec_macro:.4f} (macro)")
    print(f"Recall:    {rec_macro:.4f} (macro)")
    print(f"F1-Score:  {f1_macro:.4f} (macro)")
    print("\nConfusion Matrix [Low, Moderate, Elevated]:")
    print(np.array(cm))
    print("\nFeature Importances:")
    for item in feature_importance_list:
        print(f"  {item['feature']:<28}: {item['importance']:.4f}")
    
    # Save Pipeline
    os.makedirs('models', exist_ok=True)
    model_path = 'models/wellness_risk_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\nModel artifact saved to: {model_path}")
    
    # Save Feature Schema
    imputer_medians = dict(zip(FEATURE_NAMES, [float(m) for m in imputer.statistics_]))
    schema = {
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "target_classes": {
            0: "Low",
            1: "Moderate",
            2: "Elevated"
        },
        "feature_descriptions": {
            "sleep_hours": "Average daily sleep duration in hours (valid range 0-24)",
            "work_study_hours": "Average daily study/work duration in hours (valid range 0-24)",
            "stress_level": "Self-reported perceived stress on 1-10 scale",
            "physical_activity_minutes": "Daily physical activity in minutes (valid range 0-1440)",
            "mood_rating": "Recent mood check-in average on 1-5 scale",
            "journal_sentiment": "Recent journal sentiment polarity score (-1.0 to +1.0)",
            "negative_emotion_frequency": "Frequency ratio of negative emotions (sadness, fear, anger, disgust) in recent journals (0.0 to 1.0)"
        },
        "imputer_medians": imputer_medians
    }
    with open('models/feature_schema.json', 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)
    print("Feature schema saved to: models/feature_schema.json")
    
    # Save Model Metrics Report
    metrics_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "synthetic_prototype",
        "dataset_path": "data/synthetic_wellness_dataset.csv",
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution_train": {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "class_distribution_test": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "model_type": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "random_state": 42
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "confusion_matrix": cm,
            "detailed_report": report_dict
        },
        "feature_importances": feature_importance_list,
        "disclaimer": "This model is trained on synthetic data for technical prototyping only. It is not clinically diagnostic."
    }
    with open('models/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2)
    print("Model metrics report saved to: models/model_metrics.json")

if __name__ == '__main__':
    train_model()
