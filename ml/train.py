# ml/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib
from ml.utils import load_data, create_features, fit_and_save_scaler, transform_amount_with_scaler

def main():
    os.makedirs('models', exist_ok=True)
    print("Loading data...")
    df = load_data('data/creditcard.csv')
    X, y = create_features(df)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("Fitting/scaling Amount...")
    scaler = fit_and_save_scaler(X_train, out_path='models/scaler.joblib')
    X_train = transform_amount_with_scaler(X_train, scaler)
    X_test  = transform_amount_with_scaler(X_test, scaler)

    # Optionally drop Time or create Time features; we keep Time for now
    print("Resampling training set with SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("Resampled training distribution:", y_train_res.sum(), "fraud out of", len(y_train_res))

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    joblib.dump(model, 'models/xgb_model.joblib')
    print("Saved model to models/xgb_model.joblib")

    # Quick evaluation on test set
    print("Evaluating on test set...")
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC (avg precision):", average_precision_score(y_test, y_proba))
    print("Classification report (test):")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()
