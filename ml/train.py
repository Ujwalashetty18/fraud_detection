# ml/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib
from ml.utils import load_data, create_features, fit_and_save_scaler, transform_amount_with_scaler

def evaluate_and_save(model, name, X_test, y_test, out_dir='models'):
    """Evaluate model on test set, print metrics, and save model to disk."""
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        # If predict_proba not available, try decision_function then scale to [0,1]
        try:
            df = model.decision_function(X_test)
            # simple min-max to 0-1 for ranking (only for evaluation; not ideal but fallback)
            y_proba = (df - df.min()) / (df.max() - df.min() + 1e-12)
        except Exception:
            y_proba = None

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Could not get predict for model {name}: {e}")
        y_pred = None

    print(f"\n--- Evaluation for {name} ---")
    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        pr = average_precision_score(y_test, y_proba)
        print(f"ROC-AUC: {roc:.6f}")
        print(f"PR-AUC (avg precision): {pr:.6f}")
    else:
        print("No probability output available; skipping ROC/PR scores.")

    if y_pred is not None:
        print("Classification report (test):")
        print(classification_report(y_test, y_pred, digits=4))
    else:
        print("No predictions available; skipping classification report.")

    # Save model
    out_path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")

    return {'name': name, 'roc_auc': (roc if (y_proba is not None) else -1), 'pr_auc': (pr if (y_proba is not None) else -1)}

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

    print("Resampling training set with SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("Resampled training distribution:", int(y_train_res.sum()), "fraud out of", len(y_train_res))

    # Dictionary to collect evaluation results
    eval_results = []

    # 1) Logistic Regression
    try:
        print("\nTraining Logistic Regression...")
        logreg = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        logreg.fit(X_train_res, y_train_res)
        res = evaluate_and_save(logreg, 'logreg', X_test, y_test)
        eval_results.append(res)
    except Exception as e:
        print("Error training Logistic Regression:", e)

    # 2) Decision Tree
    try:
        print("\nTraining Decision Tree Classifier...")
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_res, y_train_res)
        res = evaluate_and_save(dt, 'decision_tree', X_test, y_test)
        eval_results.append(res)
    except Exception as e:
        print("Error training Decision Tree:", e)

    # 3) Random Forest
    try:
        print("\nTraining Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_res, y_train_res)
        res = evaluate_and_save(rf, 'random_forest', X_test, y_test)
        eval_results.append(res)
    except Exception as e:
        print("Error training Random Forest:", e)

    # 4) XGBoost (your original model)
    try:
        print("\nTraining XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train_res, y_train_res)
        res = evaluate_and_save(xgb, 'xgb_model', X_test, y_test)
        eval_results.append(res)
    except Exception as e:
        print("Error training XGBoost:", e)

    # Choose best model by ROC-AUC (if available)
    best = None
    for r in eval_results:
        if best is None or r['roc_auc'] > best['roc_auc']:
            best = r
    if best and best['roc_auc'] > 0:
        print(f"\nBest model by ROC-AUC: {best['name']} (ROC-AUC={best['roc_auc']:.6f})")
        # copy best model file to models/best_model.joblib
        best_path = os.path.join('models', f"{best['name']}.joblib")
        best_dest = os.path.join('models', 'best_model.joblib')
        try:
            # load then save to standard best slot (ensures consistent format)
            m = joblib.load(best_path)
            joblib.dump(m, best_dest)
            print(f"Saved best model to {best_dest}")
        except Exception as e:
            print("Could not copy best model:", e)
    else:
        print("\nNo valid ROC-AUC computed; no best model saved.")

if __name__ == "__main__":
    main()
