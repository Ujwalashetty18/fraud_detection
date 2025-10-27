# ml/evaluate.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import os
from ml.utils import load_data, create_features, transform_amount_with_scaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs('models', exist_ok=True)
    print("Loading data...")
    df = load_data('data/creditcard.csv')
    X, y = create_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = joblib.load('models/scaler.joblib')
    model = joblib.load('models/xgb_model.joblib')

    X_test = transform_amount_with_scaler(X_test, scaler)

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    print("Average precision (PR-AUC):", average_precision_score(y_test, y_proba))
    print("Classification report (test):")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('models/confusion_matrix.png')
    print("Saved models/confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('models/roc_curve.png')
    print("Saved models/roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('models/pr_curve.png')
    print("Saved models/pr_curve.png")

if __name__ == "__main__":
    main()
