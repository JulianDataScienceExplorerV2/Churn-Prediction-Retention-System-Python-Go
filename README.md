# Churn Prediction & Retention System

An end-to-end customer churn prediction pipeline. Uses **Python (Scikit-Learn) and Logistic Regression** to train the prediction model, calculate ROC curves, and evaluate feature importance. Uses a high-performance **Go CLI** to ingest the risk scores and build daily action lists for the retention team.

Un pipeline de extremo a extremo para la prediccion de fuga de clientes. Utiliza **Python (Scikit-Learn) y Regresion Logistica** para entrenar el modelo, calcular curvas ROC y evaluar factores de riesgo. Utiliza una veloz herramienta **CLI en Go** para leer los puntajes de riesgo y generar listas diarias de accion para retencion.

---

## 🏗️ Project Architecture / Arquitectura del Proyecto

| Language  | Component | Libraries/Frameworks |
|-----------|-----------|----------------------|
| **Python** | Model training, Evaluation, Visualizations | `pandas`, `numpy`, `scikit-learn`, `matplotlib` |
| **Go**     | High-speed Retentional Action CLI | `stdlib only` (Zero dependencies) |

## 📊 Model Performance / Rendimiento del Modelo

The Logistic Regression model was trained on a synthetic telecom dataset of **2,500 customers**.

* **Base Churn Rate**: 12.4%
* **Model AUC-ROC**: **0.8883** (Excellent predictive ability)

### ROC Curve & Feature Impact

![Model Results](https://raw.githubusercontent.com/JulianDataScienceExplorerV2/Churn-Prediction-Python-Go/main/data/model_results.png)

> **Business Insights:** 
> - Increased **support calls** are the strongest warning sign of impending churn.
> - Customers on **longer contracts** (1-2 years) are highly protected against churn.
> 
> **Insights de Negocio:**
> - El incremento en **llamadas a soporte** es la mayor señal de advertencia de fuga.
> - Los clientes con **contratos largos** tienen alta proteccion contra la fuga.

---

## 🚀 How to Run / Como Ejecutar

### 1. Train the Python Model
Generates the synthetic dataset, trains the ML model, and outputs the `churn_predictions.csv` file for Go to consume.

```bash
pip install pandas numpy scikit-learn matplotlib

python python/01_churn_model.py
```

### 2. Run the Go Retention CLI
Reads the output probabilities and generates a targeted list of actions based on risk threshold.

```bash
cd go

# Default run (Outputs Top 15 users with Risk > 70%)
go run retention_cli.go -file ../data/churn_predictions.csv

# Custom Filter (Outputs Top 50 users with Risk > 85%)
go run retention_cli.go -file ../data/churn_predictions.csv -threshold 0.85 -limit 50
```

> **CLI Example Output / Salida de Ejemplo del CLI:**
> ```
> DAILY RETENTION ACTION LIST
> Model Threshold: > 85% Risk Probability
> Total Customers Analyzed: 2500
> 
> CUSTOMER ID     TENURE (mo)  SUPPORT CALLS  RISK PROB    RECOMMENDATION 
> ----------------------------------------------------------------------------
> TEL-1108        3            7              99.4%        Agent Call ASAP
> TEL-0082        8            6              99.2%        Agent Call ASAP
> TEL-1587        1            5              98.9%        Onboarding Flow
> TEL-1381        34           4              86.7%        Offer 10% Discount
> ----------------------------------------------------------------------------
> Found 115 customers needing immediate retention action.
> ```

---

## 👨‍💻 Author / Autor

**Julian David Urrego Lancheros**
Data Analyst · Python Developer · Marketing Science
[GitHub](https://github.com/JulianDataScienceExplorerV2) · [juliandavidurrego2011@gmail.com](mailto:juliandavidurrego2011@gmail.com)
