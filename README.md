# Credit Card Fraud Detection
# 💳 Credit Card Fraud Detection System

### 🧠 Machine Learning | Deep Learning | Flask | React | Streamlit

---

## 🔹 Abstract  
Credit card fraud is a major concern in today’s digital payment systems. This project builds a machine learning–based fraud detection model that identifies suspicious transactions using anonymized financial data. The dataset is highly imbalanced, with very few fraudulent cases. The proposed model combines data preprocessing, SMOTE oversampling, and an optimized XGBoost classifier to achieve strong precision-recall performance. A Flask API and React-based web interface are developed for real-time prediction, along with an optional Streamlit dashboard for rapid model visualization.

---

## 🔹 Dataset  

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions  
- **Frauds:** 492 (≈ 0.172% of total)  
- **Features:**  
  - 30 total columns:
    - 28 anonymized PCA-transformed numerical features (`V1–V28`)
    - `Time` (seconds since first transaction)
    - `Amount` (transaction amount)
    - `Class` (target: `1 = Fraud`, `0 = Genuine`)  
- **Characteristics:** Highly imbalanced binary classification dataset. No personally identifiable information (PII) is included.

---

## 🔹 Preprocessing Steps  

1. **Data split:** 80/20 train-test split (before resampling to avoid data leakage).  
2. **Feature scaling:** Standardized the `Amount` feature using `StandardScaler` (fitted on training data only).  
3. **Imbalance handling:** Applied **SMOTE (Synthetic Minority Oversampling Technique)** on the training set to generate synthetic fraud samples.  
4. **Feature transformation:** Used helper functions from `ml/utils.py` for scaling and transformation.  
5. **Model saving:** Trained model (`xgb_model.joblib`) and scaler (`scaler.joblib`) saved in `models/` directory.

---

## 🔹 Models Tried  

| Model | Description | ROC-AUC | PR-AUC | Remarks |
|--------|--------------|----------|---------|----------|
| Logistic Regression | Baseline linear classifier | ~0.93 | ~0.70 | Simple, interpretable |
| Random Forest | Ensemble of trees | ~0.96 | ~0.82 | Handles imbalance better |
| **XGBoost (Final)** | Gradient boosting decision trees | **0.979** | **0.871** | Best precision-recall tradeoff |

**Final model hyperparameters:**
n_estimators=200
max_depth=5
learning_rate=0.1
scale_pos_weight=tuned for imbalance
random_state=42

---

## 🔹 Evaluation Metrics  

- **ROC-AUC:** 0.979  
- **PR-AUC (Average Precision):** 0.871  
- **Recall (Fraud Class):** 0.867  
- **Precision (Fraud Class):** 0.802  
- **F1-score (Fraud Class):** 0.833  

📊 **Saved Evaluation Plots (from `ml/evaluate.py`):**
- `models/roc_curve.png`
- `models/pr_curve.png`
- `models/confusion_matrix.png`

These results indicate a strong ability to identify fraudulent transactions with low false positives.

---

## 🔹 Demo Instructions  

### Commands used to run project  
```bash
🧠 Train Model
python -m ml.train
📈 Evaluate Model
python -m ml.evaluate
⚙️ Start Flask API
python api/app.py
API Endpoints:
http://127.0.0.1:5000/health
/health – API status check

/predict – Accepts a single JSON transaction

/predict_csv – Accepts CSV file and returns fraud probabilities
##Open another terminal
💻 Start React Frontend
cd frontend
npm install
npm start
Visit http://localhost:3000
Enter transaction fields or upload a CSV file for batch prediction.
📊 Optional Streamlit Demo
streamlit run streamlit_app.py
Visit http://localhost:8501
Upload CSV or input fields interactively to view results.

🔹 Limitations & Future Work
The dataset is anonymized, so feature meanings are unknown (limits explainability).

No user or card-level identifiers — cannot model behavioral patterns.

Time dependencies ignored; transactions treated independently.

🔮 Future Enhancements:

Add time-series / rolling features (transactions per user per hour).

Deploy model using real-time streaming (Kafka / Spark Streaming).

Integrate SHAP explainability for transparency.

Implement LSTM / Autoencoder models for anomaly detection.

Build admin dashboard for fraud monitoring and threshold tuning.
fraud_detection/
├─ api/                 ← Flask backend (API endpoints)
├─ ml/                  ← Machine Learning scripts
│  ├─ train.py          ← Model training
│  ├─ evaluate.py       ← Evaluation & plots
│  └─ utils.py          ← Data preprocessing utilities
├─ models/              ← Saved models & scalers
├─ data/                ← Dataset (.csv)
├─ frontend/            ← React web app (UI)
├─ streamlit_app.py     ← Optional Streamlit dashboard
├─ requirements.txt     ← Python dependencies
└─ README.md            ← Project documentation

🔹 Tools & Technologies

Languages: Python, JavaScript

Libraries: pandas, scikit-learn, imbalanced-learn, XGBoost, Flask, Streamlit, React.js, Axios

Visualization: matplotlib, seaborn

Environment: VS Code

Dataset: Kaggle (Credit Card Fraud Detection, 2013)
Download the dataset (creditcard.csv) from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it in the `data/` folder.
