import React, { useState } from "react";
import axios from "axios";

export default function UploadCSV() {
  const [file, setFile] = useState(null);
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    if (!file) {
      alert("Please select a CSV file first");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const resp = await axios.post(
        "http://localhost:5000/predict_csv",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setRows(resp.data);
    } catch (err) {
      alert(err.response?.data?.error || err.message);
    }
    setLoading(false);
  };

  return (
    <div>
      <h3>Batch Prediction (Upload CSV)</h3>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={submit} style={{ marginLeft: 8 }}>
        {loading ? "Uploading..." : "Upload & Predict"}
      </button>

      {rows.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <h4>Flagged Transactions (probability > 0.5)</h4>
          <pre>
            {JSON.stringify(
              rows.filter((r) => r.fraud_probability > 0.5).slice(0, 10),
              null,
              2
            )}
          </pre>
        </div>
      )}
    </div>
  );
}
