# api/app.py
import sys, os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import traceback
from ml.utils import transform_amount_with_scaler, parse_iso_timestamp, ensure_latlon, load_scaler, prepare_for_prediction
from datetime import datetime
from geopy.distance import great_circle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# MongoDB
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# -----------------------
# Config / paths
# -----------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_model.joblib')
# fallback to xgb_model if best_model missing
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xgb_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.joblib')

MODEL_PATH = os.path.abspath(MODEL_PATH)
SCALER_PATH = os.path.abspath(SCALER_PATH)

# -----------------------
# Load model & scaler
# -----------------------
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Could not load model from {MODEL_PATH}: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded scaler from {SCALER_PATH}")
except Exception as e:
    print(f"❌ Could not load scaler from {SCALER_PATH}: {e}")
    scaler = None

# Required ML columns
REQUIRED = set(['Time','Amount'] + [f'V{i}' for i in range(1,29)])

# -----------------------
# MongoDB setup
# -----------------------
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB = os.getenv('MONGO_DB', 'fraudDB')
MONGO_COLLECTION = os.getenv('MONGO_COLLECTION', 'last_txns')

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB]
    last_txns = mongo_db[MONGO_COLLECTION]
    print(f"Connected to MongoDB: {MONGO_URI} DB:{MONGO_DB} COL:{MONGO_COLLECTION}")
except Exception as e:
    print("Warning: Could not connect to MongoDB:", e)
    mongo_client = None
    last_txns = None

# -----------------------
# Email helper (optional)
# -----------------------
def send_alert_email(subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    alert_from = os.getenv("ALERT_FROM")
    alert_to = os.getenv("ALERT_TO")

    if not (smtp_host and smtp_user and smtp_pass and alert_from and alert_to):
        print("Email creds/recipients not fully set; skipping email.")
        return False, "Email config not set"

    try:
        msg = MIMEMultipart()
        msg["From"] = alert_from
        msg["To"] = alert_to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(alert_from, alert_to.split(','), msg.as_string())

        return True, "Email sent"
    except Exception as e:
        print("Error sending email:", e)
        return False, str(e)

# -----------------------
# Location check helper
# -----------------------
DEFAULT_COORDS = {
    "India": (20.5937, 78.9629),
    "Australia": (-25.2744, 133.7751),
    "USA": (37.0902, -95.7129),
}

def compute_travel_speed_and_flag(loc1, t1, loc2, t2, coords_map=DEFAULT_COORDS, max_speed_kmph=1000):
    # Accept loc as country string or (lat,lon) tuple
    def _get_coords(loc):
        if isinstance(loc, (list, tuple)) and len(loc) == 2:
            return tuple(loc)
        return coords_map.get(loc)

    c1 = _get_coords(loc1)
    c2 = _get_coords(loc2)
    if c1 is None or c2 is None:
        return False, "Insufficient coordinate info to compute distance"

    # parse times if strings
    if isinstance(t1, str):
        t1 = parse_iso_timestamp(t1)
    if isinstance(t2, str):
        t2 = parse_iso_timestamp(t2)

    if t1 is None or t2 is None:
        return False, "Missing timestamps for travel check"

    try:
        distance_km = great_circle(c1, c2).km
    except Exception as e:
        return False, f"Error computing distance: {e}"

    time_diff_hours = abs((t2 - t1).total_seconds()) / 3600.0
    if time_diff_hours == 0:
        speed = float("inf")
    else:
        speed = distance_km / time_diff_hours

    if speed > max_speed_kmph:
        return True, f"⚠️ Suspicious! Travel speed {speed:.1f} km/h between {loc1} and {loc2} (distance {distance_km:.1f} km; Δ {time_diff_hours:.2f} h)."
    return False, f"✅ Normal: Travel speed {speed:.1f} km/h between {loc1} and {loc2}."

# -----------------------
# Helper: fetch and upsert last txn
# -----------------------
def get_last_txn_for_card(card_id: str):
    if last_txns is None:
        return None
    return last_txns.find_one({"card_id": card_id})

def upsert_last_txn(card_id: str, doc: dict):
    if last_txns is None:
        return False
    doc_copy = doc.copy()
    doc_copy['updated_at'] = datetime.utcnow()
    last_txns.update_one({"card_id": card_id}, {"$set": doc_copy}, upsert=True)
    return True

# -----------------------
# Routes
# -----------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts single JSON transaction or list.
    If card_id present, the API will fetch last txn from MongoDB and compare automatically.
    If notify=true in payload, and suspicious found, it will try to send an email alert.
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON received'}), 400

        rows = data if isinstance(data, list) else [data]
        # Build DataFrame for ML prediction
        df_ml = pd.DataFrame(rows)
        missing = REQUIRED - set(df_ml.columns)
        if missing:
            return jsonify({'error': f'Missing ML columns: {sorted(list(missing))}'}), 400

        # Prepare features for model predict
        df_for_model = prepare_for_prediction(df_ml, scaler)  # uses Amount -> Amount_scaled
        probs = model.predict_proba(df_for_model)[:,1]
        df_ml['fraud_probability'] = probs

        results = []
        for i, row in enumerate(rows):
            rec = dict(df_ml.iloc[i].to_dict())
            rec['location_flag'] = False
            rec['location_message'] = None

            # Resolve lat/lon pair for current row
            cur_latlon = ensure_latlon(row)
            cur_ts = row.get('Timestamp') or row.get('timestamp') or None
            # parse timestamp to datetime if present
            cur_ts_parsed = None
            try:
                cur_ts_parsed = parse_iso_timestamp(cur_ts) if cur_ts else None
            except Exception as e:
                # invalid timestamp format
                rec['location_message'] = f"Invalid Timestamp format: {e}"

            # If 'prev' explicitly provided, prefer that
            if isinstance(row, dict) and row.get('prev'):
                prev = row.get('prev')
                prev_latlon = ensure_latlon(prev)
                prev_ts = prev.get('Timestamp') or prev.get('timestamp')
                try:
                    prev_ts_parsed = parse_iso_timestamp(prev_ts) if prev_ts else None
                    flag, msg = compute_travel_speed_and_flag(prev_latlon, prev_ts_parsed, cur_latlon, cur_ts_parsed)
                    rec['location_flag'] = bool(flag)
                    rec['location_message'] = msg
                except Exception as e:
                    rec['location_message'] = f"Location check error: {e}"
            else:
                # Try to get prev from the list (previous entry in provided array)
                if i > 0:
                    prev = rows[i-1]
                    prev_latlon = ensure_latlon(prev)
                    prev_ts = prev.get('Timestamp') or prev.get('timestamp')
                    try:
                        prev_ts_parsed = parse_iso_timestamp(prev_ts) if prev_ts else None
                        flag, msg = compute_travel_speed_and_flag(prev_latlon, prev_ts_parsed, cur_latlon, cur_ts_parsed)
                        rec['location_flag'] = bool(flag)
                        rec['location_message'] = msg
                    except Exception as e:
                        rec['location_message'] = f"Location check error: {e}"
                # If not in-list prev, try Mongo last_txn for same card_id
                elif isinstance(row, dict) and row.get('card_id'):
                    card_id = row.get('card_id')
                    prev_doc = get_last_txn_for_card(card_id)
                    if prev_doc:
                        try:
                            prev_latlon = (prev_doc.get('lat'), prev_doc.get('lon')) if prev_doc.get('lat') else prev_doc.get('Location')
                            prev_ts_parsed = parse_iso_timestamp(prev_doc.get('Timestamp')) if prev_doc.get('Timestamp') else None
                            flag, msg = compute_travel_speed_and_flag(prev_latlon, prev_ts_parsed, cur_latlon, cur_ts_parsed)
                            rec['location_flag'] = bool(flag)
                            rec['location_message'] = msg
                        except Exception as e:
                            rec['location_message'] = f"Last txn location check error: {e}"

            # After checks, persist current txn as last for the card_id (if provided)
            if isinstance(row, dict) and row.get('card_id'):
                card_id = row.get('card_id')
                store_doc = {
                    "card_id": card_id,
                    "Timestamp": row.get('Timestamp') or row.get('timestamp'),
                    "lat": row.get('lat'),
                    "lon": row.get('lon'),
                    "Location": row.get('Location') or row.get('location'),
                    "ml_prob": rec.get('fraud_probability'),
                    "stored_at": datetime.utcnow()
                }
                try:
                    upsert_last_txn(card_id, store_doc)
                except Exception as e:
                    print("Warning: could not upsert last txn:", e)

            results.append(rec)

        # Notification handling (if notify true in single payload or first list item)
                # --- Automatic notification handling (auto-send when suspicious) ---
        from datetime import timedelta

        FRAUD_THRESH = float(os.getenv('FRAUD_PROB_THRESHOLD', 0.8))
        ALERT_COOLDOWN_MINUTES = int(os.getenv('ALERT_COOLDOWN_MINUTES', 60))

        def should_send_alert_for_card(card_id: str) -> bool:
            if last_txns is None:
                return True
            doc = last_txns.find_one({"card_id": card_id})
            if not doc:
                return True
            last_alert = doc.get('last_alert_at')
            if not last_alert:
                return True
            # last_alert may be string or datetime
            if isinstance(last_alert, str):
                try:
                    last_alert_dt = parse_iso_timestamp(last_alert)
                except Exception:
                    return True
            else:
                last_alert_dt = last_alert
            if not last_alert_dt:
                return True
            delta = datetime.utcnow() - last_alert_dt
            return delta.total_seconds() > ALERT_COOLDOWN_MINUTES * 60

        def mark_alert_sent(card_id: str):
            if last_txns is None:
                return
            last_txns.update_one({"card_id": card_id}, {"$set": {"last_alert_at": datetime.utcnow()}}, upsert=True)

        # Build list of suspicious records
        suspicious_recs = [r for r in results if r.get('location_flag') or (r.get('fraud_probability', 0) >= FRAUD_THRESH)]

        # Determine which suspicious recs should actually be alerted (respect per-card cooldown)
        alerts_to_send = []
        for r in suspicious_recs:
            card = r.get('card_id')
            if card:
                if should_send_alert_for_card(card):
                    alerts_to_send.append(r)
                    mark_alert_sent(card)
                else:
                    r['_alert_skipped'] = True
            else:
                # no card_id -> include in alerts by default (but this may spam)
                alerts_to_send.append(r)

        if alerts_to_send:
            subject = "Fraud Alert: Suspicious transaction(s) detected"
            body_lines = []
            for r in alerts_to_send:
                lines = [
                    f"card_id: {r.get('card_id')}",
                    f"fraud_probability: {r.get('fraud_probability')}",
                    f"location_flag: {r.get('location_flag')}",
                    f"location_message: {r.get('location_message')}",
                    f"timestamp: {r.get('Timestamp') or r.get('timestamp')}",
                ]
                body_lines.append("\n".join(lines))
            body = "Suspicious transactions detected:\n\n" + "\n\n---\n\n".join(body_lines)
            sent, info = send_alert_email(subject, body)
            return jsonify({'results': results, 'notification': {'sent': sent, 'info': info}})

        # otherwise just return results
        return jsonify(results)


        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        f = request.files['file']
        df = pd.read_csv(f)
        missing = REQUIRED - set(df.columns)
        if missing:
            return jsonify({'error': f'Missing columns: {sorted(list(missing))}'}), 400
        df_t = transform_amount_with_scaler(df, scaler)
        probs = model.predict_proba(df_t)[:,1]
        df['fraud_probability'] = probs
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)

