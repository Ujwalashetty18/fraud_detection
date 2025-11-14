// frontend/src/components/PredictForm.js
import React, { useState } from "react";
import axios from "axios";

const defaultValues = { Time: 0, Amount: 0 };
for (let i = 1; i <= 28; i++) defaultValues[`V${i}`] = 0;

function probabilityColor(p) {
  // p is the raw decimal probability (e.g. 1e-6)
  const pct = Math.min(Math.max(p * 100, 0), 100);
  if (pct < 10) return { variant: "success", pct };
  if (pct < 40) return { variant: "warning", pct };
  return { variant: "danger", pct };
}

/* Real fraud example from the dataset - load it using "Load fraud example" button */
const fraudExample = {
  Time: 406,
  V1: -1.3598071336738,
  V2: -0.0727811733098497,
  V3: 2.53634673796914,
  V4: 1.37815522427443,
  V5: -0.338320769942518,
  V6: 0.462387777762292,
  V7: 0.239598554061257,
  V8: 0.0986979012610507,
  V9: 0.363786969611213,
  V10: 0.0907941719789316,
  V11: -0.551599533260813,
  V12: -0.617800855762348,
  V13: -0.991389847235408,
  V14: -0.311169353699879,
  V15: 1.46817697209427,
  V16: -0.470400525259478,
  V17: 0.207971241929242,
  V18: 0.0257905801985591,
  V19: 0.403992960255733,
  V20: 0.251412098239705,
  V21: -0.018306777944153,
  V22: 0.277837575558899,
  V23: -0.110473910188767,
  V24: 0.0669280749146731,
  V25: 0.128539358273528,
  V26: -0.189114843888824,
  V27: 0.133558376740387,
  V28: -0.0210530534538215,
  Amount: 149.62,
};

export default function PredictForm() {
  const [form, setForm] = useState(defaultValues);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // tweak this to change how visible the UI bar is.
  // scaleFactor multiplies raw prob (decimal) to create a percent-like UI score.
  // default 1e6: 1e-6 -> 1% on the bar, 1e-4 -> 100% (capped)
  const SCALE_FACTOR = 1_000_000;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value === "" ? "" : Number(value) }));
  };

  const submit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const resp = await axios.post("http://localhost:5000/predict", form, {
        headers: { "Content-Type": "application/json" },
      });
      const data = Array.isArray(resp.data) ? resp.data[0] : resp.data;
      setResult(data);
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setForm(defaultValues);
    setResult(null);
  };

  const loadExample = () => {
    setForm({ ...defaultValues, ...fraudExample });
    setResult(null);
  };

  return (
    <div className="container py-4">
      <div className="row justify-content-center">
        <div className="col-xl-8 col-lg-10 col-md-11">
          <div className="card shadow-sm">
            <div className="card-body">
              <div className="d-flex align-items-center mb-3">
                <img
                  src="https://img.icons8.com/emoji/48/000000/credit-card-emoji.png"
                  alt="card"
                  style={{ width: 36, marginRight: 10 }}
                />
                <h3 className="mb-0">Credit Card Fraud Detection</h3>
              </div>
              <p className="text-muted small">
                Enter a single transaction (Time, Amount, V1–V28) or upload a
                CSV below.
              </p>

              <div className="row g-2">
                <div className="col-sm-6 col-md-3">
                  <label className="form-label small">Time</label>
                  <input
                    type="number"
                    step="any"
                    name="Time"
                    value={form.Time}
                    onChange={handleChange}
                    className="form-control form-control-sm"
                    placeholder="e.g. 406"
                  />
                </div>
                <div className="col-sm-6 col-md-3">
                  <label className="form-label small">Amount</label>
                  <input
                    type="number"
                    step="any"
                    name="Amount"
                    value={form.Amount}
                    onChange={handleChange}
                    className="form-control form-control-sm"
                    placeholder="e.g. 149.62"
                  />
                </div>

                {Array.from({ length: 28 }, (_, i) => i + 1).map((i) => (
                  <div key={i} className="col-6 col-md-2 col-lg-1">
                    <label className="form-label small">{`V${i}`}</label>
                    <input
                      type="number"
                      step="any"
                      name={`V${i}`}
                      value={form[`V${i}`]}
                      onChange={handleChange}
                      className="form-control form-control-sm"
                      placeholder="0.0"
                    />
                  </div>
                ))}
              </div>

              <div className="d-flex gap-2 mt-3">
                <button
                  className="btn btn-primary btn-sm"
                  onClick={submit}
                  disabled={loading}
                >
                  {loading ? "Predicting..." : "Predict"}
                </button>
                <button
                  className="btn btn-outline-secondary btn-sm"
                  onClick={reset}
                  disabled={loading}
                >
                  Reset
                </button>
                <button
                  className="btn btn-outline-info btn-sm"
                  onClick={loadExample}
                  disabled={loading}
                >
                  Load fraud example
                </button>
              </div>

              {result &&
                (() => {
                  // raw probability from API (decimal)
                  const prob = Number(result.fraud_probability) || 0;

                  // raw percentage (very small usually)
                  const rawPct = prob * 100;

                  // scaled percentage for UI visibility (tweak SCALE_FACTOR)
                  const scaledPct = Math.min(100, prob * SCALE_FACTOR);

                  // color decision still based on raw prob (keeps color meaningful)
                  const { variant } = probabilityColor(prob);

                  // example threshold (you can change): treat as suspicious if raw prob >
                  // e.g. 1e-4. This is optional — choose threshold after validation.
                  const THRESHOLD = 1e-4;
                  const suspicious = prob >= THRESHOLD;

                  return (
                    <div className="mt-4">
                      <h5>Prediction Result</h5>

                      <div
                        className={`badge text-bg-${
                          suspicious ? "danger" : "success"
                        } mb-2`}
                      >
                        {suspicious ? "⚠️ Suspicious" : "✅ Looks Safe"}
                      </div>

                      <div className="card border-0 bg-light p-3 mb-3">
                        <div className="d-flex justify-content-between align-items-center gap-3">
                          <div>
                            <small className="text-muted">
                              Fraud Probability
                            </small>
                            <div className="h5 mb-0">
                              {rawPct < 0.0001
                                ? `${rawPct.toFixed(6)}%`
                                : `${rawPct.toFixed(4)}%`}
                            </div>
                            <div className="small text-muted">
                              (decimal: {prob.toExponential(6)})
                            </div>
                          </div>

                          <div style={{ minWidth: 260 }}>
                            <div className="progress" style={{ height: 14 }}>
                              <div
                                className={`progress-bar bg-${variant}`}
                                role="progressbar"
                                style={{ width: `${scaledPct}%` }}
                                aria-valuenow={scaledPct}
                                aria-valuemin="0"
                                aria-valuemax="100"
                              />
                            </div>
                            <div className="small text-muted mt-1">
                              {scaledPct.toFixed(2)}% (UI suspicion score)
                              {` — threshold: ${THRESHOLD}`}
                            </div>
                          </div>
                        </div>
                      </div>

                      <details>
                        <summary
                          className="fw-semibold"
                          style={{ cursor: "pointer" }}
                        >
                          Full response (JSON)
                        </summary>
                        <pre
                          style={{
                            maxHeight: 240,
                            overflow: "auto",
                            background: "#f8f9fa",
                            padding: 12,
                            borderRadius: 6,
                          }}
                        >
                          {JSON.stringify(result, null, 2)}
                        </pre>
                      </details>
                    </div>
                  );
                })()}

              <hr />

              <div>
                <h6 className="mb-2">Batch Prediction (Upload CSV)</h6>
                <p className="text-muted small">
                  Upload a CSV with columns: Time, Amount, V1..V28
                </p>
                <div className="d-flex gap-2">
                  <input type="file" className="form-control form-control-sm" />
                  <button className="btn btn-secondary btn-sm">
                    Upload & Predict
                  </button>
                </div>
                <small className="text-muted d-block mt-2">
                  Tip: add a small sample CSV for demo — do not add the full
                  dataset to GitHub.
                </small>
              </div>
            </div>

            <div className="card-footer text-muted small">
              Local demo • Flask API @ http://127.0.0.1:5000
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
