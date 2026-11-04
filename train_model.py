"""
Smart Irrigation Prediction System - Model Training & Evaluation Pipeline
Algorithms: Decision Tree, Random Forest, Logistic Regression
Metrics: Accuracy, Precision, Recall, F1-Score
Visualizations: Matplotlib & Seaborn (Model Comparison, Feature Importance, Confusion Matrix)
Artifacts: models/best_model.pkl, models/scaler.pkl, models/model_metrics.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def train_and_evaluate():
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset', 'irrigation_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    static_img_dir = os.path.join(base_dir, 'static', 'images')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)

    # 2. Check and load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please make sure 'irrigation_data.csv' exists."
        )

    print("=" * 70)
    print("🌾 SMART IRRIGATION PREDICTION SYSTEM - MODEL TRAINING PIPELINE 🌾")
    print("=" * 70)
    print(f"[1/6] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    required_cols = ['soil_moisture', 'temperature', 'humidity', 'rainfall', 'irrigation_required']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset missing expected column: {col}")

    print(f"      Total records loaded: {len(df)}")
    print(f"      Class distribution (0=No, 1=Yes):\n{df['irrigation_required'].value_counts().to_string()}")

    # 3. Train/Test Split & Feature Scaling
    print("\n[2/6] Preprocessing and feature scaling...")
    X = df[['soil_moisture', 'temperature', 'humidity', 'rainfall']]
    y = df['irrigation_required']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Instantiate Models
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'use_scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_split=4),
            'use_scaled': False
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=3),
            'use_scaled': False
        }
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    print("\n[3/6] Training and evaluating algorithms...")
    print("-" * 70)
    header = f"{'Algorithm':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}"
    print(header)
    print("-" * 70)

    for name, conf in models.items():
        clf = conf['model']
        if conf['use_scaled']:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'confusion_matrix': cm,
            'uses_scaler': conf['use_scaled']
        }

        print(f"{name:<22} | {acc * 100:>8.2f}% | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}%")

        # Select best model based on F1-Score (balanced measure)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf

    print("-" * 70)
    print(f"\n🏆 Best Selected Model: {best_model_name} (F1-Score: {best_f1 * 100:.2f}%)")

    # 5. Generate and Save Visualizations (Matplotlib & Seaborn)
    print("\n[4/6] Generating evaluation visualizations...")

    # Plot 1: Model Comparison Bar Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bar_width = 0.2
    x = np.arange(len(metric_labels))

    colors = ['#10b981', '#3b82f6', '#f59e0b']
    for idx, (m_name, m_data) in enumerate(results.items()):
        vals = [m_data[m] * 100 for m in metrics_names]
        offset = (idx - 1) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width, label=m_name, color=colors[idx % len(colors)], alpha=0.9)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Machine Learning Algorithm Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    comp_plot_path = os.path.join(static_img_dir, 'model_comparison.png')
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()
    print(f"      Saved: {comp_plot_path}")

    # Plot 2: Best Model Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    best_cm = np.array(results[best_model_name]['confusion_matrix'])
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['No Irrigation (0)', 'Irrigate (1)'],
                yticklabels=['No Irrigation (0)', 'Irrigate (1)'],
                annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
    ax.set_title(f'Confusion Matrix: {best_model_name}', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    plt.tight_layout()
    cm_plot_path = os.path.join(static_img_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"      Saved: {cm_plot_path}")

    # Plot 3: Feature Importance (From Tree / Forest Model)
    fig, ax = plt.subplots(figsize=(8, 5))
    # Extract feature importance from Random Forest or Decision Tree
    tree_clf = models['Random Forest']['model']
    importances = tree_clf.feature_importances_
    features = ['Soil Moisture', 'Temperature', 'Humidity', 'Rainfall']
    indices = np.argsort(importances)[::-1]

    sns.barplot(x=[importances[i] * 100 for i in indices],
                y=[features[i] for i in indices],
                palette='viridis', ax=ax)
    for i, v in enumerate([importances[idx] * 100 for idx in indices]):
        ax.text(v + 1, i, f"{v:.1f}%", color='black', va='center', fontweight='bold', fontsize=10)

    ax.set_xlabel('Relative Importance (%)', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance for Smart Irrigation Decision', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(0, max(importances) * 100 + 15)
    plt.tight_layout()
    feat_plot_path = os.path.join(static_img_dir, 'feature_importance.png')
    plt.savefig(feat_plot_path, dpi=200)
    plt.close()
    print(f"      Saved: {feat_plot_path}")

    # 6. Save Model Artifacts & Metrics Metadata
    print("\n[5/6] Persisting artifacts to 'models/' directory...")
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    metrics_path = os.path.join(models_dir, 'model_metrics.json')

    joblib.dump(best_model_obj, best_model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        'best_model': best_model_name,
        'features': ['soil_moisture', 'temperature', 'humidity', 'rainfall'],
        'models_evaluated': results,
        'feature_importance': {
            features[i]: round(float(importances[i]), 4) for i in range(len(features))
        },
        'dataset_size': len(df),
        'test_size': len(y_test)
    }

    with open(metrics_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"      Saved Best Model: {best_model_path}")
    print(f"      Saved Scaler: {scaler_path}")
    print(f"      Saved Metrics: {metrics_path}")

    print("\n[6/6] ✅ Training pipeline completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    train_and_evaluate()
