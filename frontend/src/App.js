import React from "react";
import "./App.css";
import PredictForm from "./components/PredictForm";

function App() {
  const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:5000";
  const SCALE_FACTOR = Number(process.env.REACT_APP_SCALE_FACTOR || 3);
  const THRESHOLD = Number(process.env.REACT_APP_THRESHOLD || 0.5);
  const autoAlert = (process.env.REACT_APP_AUTO_ALERT || "true") === "true";

  return (
    <div
      className="App"
      style={{ padding: 18, background: "#f5f6fb", minHeight: "100vh" }}
    >
      <header
        style={{
          maxWidth: 980,
          margin: "0 auto 18px",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 20 }}>Fraud Detection Dashboard</h1>
        <div style={{ marginLeft: "auto", fontSize: 13, color: "#666" }}>
          Connected to: <strong>{apiUrl}</strong>
        </div>
      </header>

      <main style={{ maxWidth: 980, margin: "0 auto" }}>
        <PredictForm
          apiUrl={apiUrl}
          SCALE_FACTOR={SCALE_FACTOR}
          THRESHOLD={THRESHOLD}
          autoAlert={autoAlert}
        />
      </main>
    </div>
  );
}

export default App;
