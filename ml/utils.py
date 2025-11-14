# ml/utils.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Default country -> (lat, lon) map — extend as needed
DEFAULT_COORDS: Dict[str, Tuple[float, float]] = {
    "India": (20.5937, 78.9629),
    "Australia": (-25.2744, 133.7751),
    "USA": (37.0902, -95.7129),
    # add other countries or regions you need
}


def load_data(path: str = 'data/creditcard.csv') -> pd.DataFrame:
    """
    Load dataset from CSV. Returns pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split features and target.
    Assumes target column is 'Class'.
    Returns (X, y) where X contains all feature columns (including Time and Amount) and
    y is the 'Class' Series.
    """
    if 'Class' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Class' column as target.")
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y


def fit_and_save_scaler(X_train: pd.DataFrame, out_path: str = 'models/scaler.joblib') -> StandardScaler:
    """
    Fit a StandardScaler on the 'Amount' column of X_train and save to out_path.
    Returns the fitted scaler.
    """
    if 'Amount' not in X_train.columns:
        raise ValueError("X_train must contain 'Amount' column to fit scaler.")

    scaler = StandardScaler()
    scaler.fit(X_train[['Amount']])
    joblib.dump(scaler, out_path)
    return scaler


def transform_amount_with_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Transform Amount using the provided scaler.
    - Adds 'Amount_scaled' column
    - Drops original 'Amount' column
    Returns a NEW DataFrame (does not mutate input).
    """
    Xc = X.copy()
    if 'Amount' not in Xc.columns:
        raise ValueError("Input DataFrame must contain 'Amount' column to transform.")
    # ensure shape for scaler
    Xc['Amount_scaled'] = scaler.transform(Xc[['Amount']])
    Xc = Xc.drop(columns=['Amount'])
    return Xc


# -----------------------------
# Timestamp & Location helpers
# -----------------------------
def parse_iso_timestamp(ts: Optional[Any]) -> Optional[datetime]:
    """
    Accepts:
      - datetime object -> returns as-is
      - ISO-formatted string (e.g., '2025-11-12T10:30:00') -> parsed datetime
      - other common formats like 'YYYY-MM-DD HH:MM:SS' -> parsed
    Returns datetime or None if input is None.
    Raises ValueError if format is unrecognized.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        # treat as epoch seconds
        try:
            return datetime.fromtimestamp(ts)
        except Exception:
            raise ValueError(f"Unrecognized numeric timestamp: {ts}")
    if isinstance(ts, str):
        # try fromisoformat first
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            # try common fallback formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    return datetime.strptime(ts, fmt)
                except Exception:
                    continue
    raise ValueError(f"Unrecognized timestamp format: {ts}")


def ensure_latlon(row: Dict[str, Any], coords_map: Optional[Dict[str, Tuple[float, float]]] = None
                  ) -> Optional[Tuple[float, float]]:
    """
    Return (lat, lon) for a given row dict-like object.
    Priority:
      1) If 'lat' and 'lon' keys present and non-empty -> return as floats
      2) If 'Location' present and found in coords_map (or DEFAULT_COORDS) -> return mapped coords
      3) Otherwise -> return None
    """
    if coords_map is None:
        coords_map = DEFAULT_COORDS

    # direct lat/lon columns
    if 'lat' in row and 'lon' in row and row.get('lat') is not None and row.get('lon') is not None:
        try:
            return float(row['lat']), float(row['lon'])
        except Exception:
            return None

    # country name mapping
    loc = row.get('Location') or row.get('location')
    if loc:
        return coords_map.get(loc)

    return None


def add_latlon_from_location(df: pd.DataFrame, location_col: str = 'Location',
                             lat_col: str = 'lat', lon_col: str = 'lon',
                             coords_map: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """
    For every row in df, if lat/lon are missing and a Location country exists,
    fill lat_col and lon_col from coords_map (DEFAULT_COORDS).
    Returns a new DataFrame (does not mutate original).
    """
    if coords_map is None:
        coords_map = DEFAULT_COORDS

    out = df.copy()
    # ensure columns exist
    if lat_col not in out.columns:
        out[lat_col] = None
    if lon_col not in out.columns:
        out[lon_col] = None

    def _fill_latlon(row):
        if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
            return row[lat_col], row[lon_col]
        loc = row.get(location_col)
        if loc and loc in coords_map:
            return coords_map[loc]
        return row[lat_col], row[lon_col]

    filled = out.apply(lambda r: pd.Series(_fill_latlon(r), index=[lat_col, lon_col]), axis=1)
    out[lat_col] = filled[lat_col]
    out[lon_col] = filled[lon_col]
    return out


# -----------------------------
# Prediction preparation helper
# -----------------------------
def prepare_for_prediction(df: pd.DataFrame, scaler: StandardScaler,
                           feature_cols: Optional[list] = None
                           ) -> pd.DataFrame:
    """
    Prepare input DataFrame for model.predict_proba:
    - Expects df to contain 'Amount' (or 'Amount_scaled' already).
    - Uses transform_amount_with_scaler to create Amount_scaled and drop Amount.
    - Keeps only ML feature columns (default: Time, V1..V28, Amount_scaled).
      If feature_cols provided, uses that exact list (must match training features).
    Returns a new DataFrame ready for model.predict_proba.
    """
    if feature_cols is None:
        # default features used in training: Time, V1..V28, plus Amount_scaled
        feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']

    # copy to avoid mutating
    X = df.copy()

    # If Amount_scaled not present, create it using scaler
    if 'Amount_scaled' not in X.columns:
        if 'Amount' not in X.columns:
            raise ValueError("Input must contain either 'Amount' or 'Amount_scaled' to prepare for prediction.")
        X = transform_amount_with_scaler(X, scaler)

    # Ensure all required feature columns are present
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for prediction: {missing}")

    # Return only the selected features in order
    return X[feature_cols].copy()


# -----------------------------
# Small utility: safe load scaler
# -----------------------------
def load_scaler(path: str = 'models/scaler.joblib') -> StandardScaler:
    """
    Load a saved scaler from disk.
    """
    return joblib.load(path)


# -----------------------------
# Optional: augment small demo dataset (for testing)
# -----------------------------
def augment_with_demo_location_and_timestamps(df: pd.DataFrame,
                                              start_ts: Optional[str] = None,
                                              countries: Optional[list] = None) -> pd.DataFrame:
    """
    Create demo Timestamp and Location columns for quick prototyping.
    - start_ts: ISO string for first timestamp (default now)
    - countries: list of country names to randomly assign (defaults to keys of DEFAULT_COORDS)
    Returns a copy of df with 'Timestamp', 'Location', 'lat', 'lon' columns added.
    """
    out = df.copy().reset_index(drop=True)
    if start_ts:
        start = parse_iso_timestamp(start_ts)
    else:
        start = datetime.utcnow()

    if countries is None:
        countries = list(DEFAULT_COORDS.keys())

    import numpy as np
    n = len(out)
    out['Timestamp'] = [(start + pd.Timedelta(minutes=int(i))).isoformat() for i in range(n)]
    out['Location'] = list(pd.Series(np.random.choice(countries, size=n)))
    latlons = out['Location'].map(lambda c: DEFAULT_COORDS.get(c))
    out['lat'] = latlons.map(lambda t: t[0] if t else None)
    out['lon'] = latlons.map(lambda t: t[1] if t else None)
    return out
