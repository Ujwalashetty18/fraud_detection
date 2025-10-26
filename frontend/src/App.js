import React from "react";
import PredictForm from "./components/PredictForm";
import UploadCSV from "./components/UploadCSV";

function App() {
  return (
    <div
      style={{
        maxWidth: 1000,
        margin: "24px auto",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <h1>💳 Credit Card Fraud Detection</h1>
      <p>
        Enter a single transaction (Time, Amount, V1–V28) or upload a CSV file.
      </p>

      <PredictForm />
      <hr style={{ margin: "24px 0" }} />
      <UploadCSV />
    </div>
  );
}

export default App;
