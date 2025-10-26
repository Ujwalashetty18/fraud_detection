import React, { useState } from "react";
import axios from "axios";

const defaultValues = { Time: 0, Amount: 0 };
for (let i = 1; i <= 28; i++) defaultValues[`V${i}`] = 0;

export default function PredictForm() {
  const [form, setForm] = useState(defaultValues);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const submit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const resp = await axios.post("http://localhost:5000/predict", form);
      setResult(resp.data[0]);
    } catch (err) {
      alert(err.response?.data?.error || err.message);
    }
    setLoading(false);
  };

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4,1fr)",
          gap: 8,
        }}
      >
        <div>
          <label>Time</label>
          <br />
          <input name="Time" value={form.Time} onChange={handleChange} />
        </div>
        <div>
          <label>Amount</label>
          <br />
          <input name="Amount" value={form.Amount} onChange={handleChange} />
        </div>
        {Array.from({ length: 28 }, (_, i) => i + 1).map((i) => (
          <div key={i}>
            <label>{`V${i}`}</label>
            <br />
            <input
              name={`V${i}`}
              value={form[`V${i}`]}
              onChange={handleChange}
            />
          </div>
        ))}
      </div>
      <button onClick={submit} disabled={loading} style={{ marginTop: 12 }}>
        Predict
      </button>

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Result</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
