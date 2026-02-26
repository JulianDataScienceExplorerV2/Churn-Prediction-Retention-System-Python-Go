"""
01_churn_model.py
=================
EN: Trains a Logistic Regression model to predict customer churn.
    Outputs model performance metrics, a ROC Curve, Feature Importance plots,
    and saves risk probabilities to predictions.csv.

ES: Entrena un modelo de Regresion Logistica para predecir fuga de clientes.
    Mide el desempeno, genera Curva ROC y graficos de importancia de variables,
    y guarda las probabilidades de riesgo en predictions.csv.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
import os

np.random.seed(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ── 1. GENERATE REALISTIC TELECOM DATASET ──────────────────────────────────────
print("Generating synthetic Telecom dataset (2500 customers)...")
N = 2500
df = pd.DataFrame({
    "customer_id": [f"TEL-{i:04d}" for i in range(1, N+1)],
    "tenure_months": np.random.randint(1, 72, N),
    "monthly_charges": np.random.uniform(20.0, 120.0, N),
    "support_calls_ltm": np.random.poisson(1.5, N),
    "internat_plan": np.random.choice([0, 1], N, p=[0.85, 0.15]),
    "contract_type": np.random.choice([0, 1], N, p=[0.5, 0.5]) # 0=Month-to-month, 1=1/2 Year
})
df["total_charges"] = df["tenure_months"] * df["monthly_charges"] * np.random.uniform(0.9, 1.1, N)

# True Churn probability follows logistic function
intercept = -2.5
z = (
    intercept 
    - 0.05 * df["tenure_months"]             # Longer tenure = lower churn
    + 0.015 * df["monthly_charges"]          # High price = higher churn
    + 0.8 * df["support_calls_ltm"]          # More complaints = huge churn driver
    - 1.5 * df["contract_type"]              # Long contract = lower churn
)
prob = 1 / (1 + np.exp(-z))
df["churn_actual"] = np.random.binomial(1, prob)

df.to_csv(os.path.join(DATA_DIR, "telecom_customers.csv"), index=False)
print(f"Dataset generated. Churn Rate: {df['churn_actual'].mean():.1%}\n")

# ── 2. TRAIN LOGISTIC REGRESSION ───────────────────────────────────────────────
print("Training Logistic Regression model...")
features = ["tenure_months", "monthly_charges", "total_charges", "support_calls_ltm", "internat_plan", "contract_type"]
X = df[features]
y = df["churn_actual"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(class_weight="balanced", random_state=42)
lr.fit(X_train_s, y_train)

# ── 3. PREDICTIONS AND METRICS ────────────────────────────────────────────────
y_pred_prob = lr.predict_proba(X_test_s)[:, 1]
y_pred_class = (y_pred_prob >= 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"Model AUC-ROC: {roc_auc:.4f}\n")
print(classification_report(y_test, y_pred_class))

# Save predictions for Go CLI
pred_df = df.copy()
pred_df["churn_risk_prob"] = lr.predict_proba(scaler.transform(X))[:, 1]
pred_df = pred_df[["customer_id", "tenure_months", "monthly_charges", "support_calls_ltm", "churn_risk_prob", "churn_actual"]]
pred_df.to_csv(os.path.join(DATA_DIR, "churn_predictions.csv"), index=False)
print("Saved predictions to data/churn_predictions.csv (Go CLI will read this).")


# ── 4. PROFESSIONAL PLOTS FOR README ─────────────────────────────────────────
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
ax1.plot(fpr, tpr, color='#00e5ff', lw=3, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='#444444', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('Model Performance (ROC Curve)', fontsize=14, pad=15)
ax1.legend(loc="lower right")
ax1.grid(alpha=0.2)

# Feature Importance (Absolute Coefficients)
coefs = pd.Series(lr.coef_[0], index=features).sort_values()
colors = ['#ff3366' if c > 0 else '#00ff9d' for c in coefs] # Red for churn drivers, Green for retention
coefs.plot(kind='barh', ax=ax2, color=colors, width=0.7)
ax2.set_title('Feature Importance (Logistic Regression Coefs)', fontsize=14, pad=15)
ax2.set_xlabel('Impact on Churn (< 0 Protects, > 0 Causes Churn)', fontsize=12)
ax2.grid(axis='x', alpha=0.2)

# Highlight interpretation
ax2.text(0.1, 4.5, "↑ Support calls drive churn", color='#ff3366', fontsize=10, weight='bold')
ax2.text(-1.5, 0.5, "↑ Long contract protects", color='#00ff9d', fontsize=10, weight='bold')

plt.tight_layout()
fig.savefig(os.path.join(DATA_DIR, "model_results.png"), dpi=300, transparent=True)
print("\nGenerated high-quality plots for README: data/model_results.png")
