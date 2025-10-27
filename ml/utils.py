# ml/utils.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(path='data/creditcard.csv'):
    """
    Load dataset from CSV. Returns pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df

def create_features(df):
    """
    Split features and target.
    Assumes target column is 'Class'.
    """
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y

def fit_and_save_scaler(X_train, out_path='models/scaler.joblib'):
    """
    Fit scaler to the Amount column of X_train and save.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[['Amount']])
    joblib.dump(scaler, out_path)
    return scaler

def transform_amount_with_scaler(X, scaler):
    """
    Add Amount_scaled and drop original Amount.
    Returns a copy (doesn't mutate input).
    """
    Xc = X.copy()
    Xc['Amount_scaled'] = scaler.transform(Xc[['Amount']])
    Xc = Xc.drop(columns=['Amount'])
    return Xc
