// src/components/Dashboard.jsx
import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  ResponsiveContainer,
} from "recharts";
import {
  FaBars,
  FaDownload,
  FaRecycle,
  FaHome,
  FaChartBar,
  FaFilePdf,
} from "react-icons/fa";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

/*
  Assumptions:
  - Your API root: http://127.0.0.1:5000
  - Endpoint for predict is POST /predict (already exists)
  - Endpoint for history per card: GET /history?card_id=<id>
  - Endpoint for distribution data (optional) GET /distribution or you can compute from history.
*/

const API_ROOT = process.env.REACT_APP_API_ROOT || "http://127.0.0.1:5000";

function RiskBadge({ score, threshold = 0.5 }) {
  // score is 0..1 raw or scaled to 0..1 (we expect scaled to UI suspicion 0..1)
  let color = "bg-green-500";
  let label = "Looks Safe";

  if (score >= 0.7) {
    color = "bg-red-500";
    label = "High Risk";
  } else if (score >= 0.3) {
    color = "bg-yellow-500 text-gray-800";
    label = "Medium Risk";
  }

  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-white ${color} font-semibold`}
    >
      {label}
    </span>
  );
}

function AnimatedProgress({ pct }) {
  // pct = 0..100
  const barRef = useRef();
  useEffect(() => {
    if (!barRef.current) return;
    // animate width
    barRef.current.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  }, [pct]);

  // color by tiers
  const colorClass =
    pct >= 70 ? "bg-red-500" : pct >= 30 ? "bg-yellow-400" : "bg-green-500";

  return (
    <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden shadow-inner">
      <div
        ref={barRef}
        className={`h-4 ${colorClass} transition-all duration-1000`}
        style={{ width: `0%` }}
      />
    </div>
  );
}

function PredictCard({ onPredictResult }) {
  const [form, setForm] = useState(() => {
    const base = {
      Time: 0,
      Amount: 0.0,
      card_id: "card_demo",
      lat: 12.9716,
      lon: 77.5946,
    };
    for (let i = 1; i <= 28; i++) base[`V${i}`] = 0.0;
    return base;
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  function setField(k, v) {
    setForm((s) => ({ ...s, [k]: v }));
  }

  async function handlePredict() {
    setLoading(true);
    try {
      const res = await axios.post(`${API_ROOT}/predict`, form);
      setResult(res.data);
      if (onPredictResult) onPredictResult(res.data);
    } catch (err) {
      alert(
        "Prediction failed: " + (err?.response?.data?.message || err.message)
      );
    } finally {
      setLoading(false);
    }
  }

  function loadExample() {
    const ex = {
      ...form,
      Time: 406,
      Amount: 149.62,
      V1: 2.53,
      V2: 1.37,
      V3: -0.34,
    };
    setForm(ex);
  }

  return (
    <div className="bg-white rounded-2xl shadow-md p-6 hover:shadow-xl transition-shadow">
      <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
        <span>💳</span> Credit Card Fraud Detection
      </h2>

      <div className="grid grid-cols-6 gap-3 mb-3">
        <div className="col-span-1">
          <label className="text-sm block">Time</label>
          <input
            type="number"
            value={form.Time}
            onChange={(e) => setField("Time", Number(e.target.value))}
            className="w-full border rounded px-2 py-1"
          />
        </div>
        <div className="col-span-1">
          <label className="text-sm block">Amount</label>
          <input
            type="number"
            value={form.Amount}
            onChange={(e) => setField("Amount", Number(e.target.value))}
            className="w-full border rounded px-2 py-1"
          />
        </div>

        {/* V1-V4 example (you can add grid mapping or collapse) */}
        {Array.from({ length: 4 }).map((_, i) => {
          const key = `V${i + 1}`;
          return (
            <div key={key}>
              <label className="text-sm block">{key}</label>
              <input
                type="number"
                value={form[key]}
                onChange={(e) => setField(key, Number(e.target.value))}
                className="w-full border rounded px-2 py-1"
              />
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-3 mb-4">
        <button
          className="bg-blue-600 text-white px-4 py-1 rounded"
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
        <button
          className="border px-3 py-1 rounded"
          onClick={() => setForm((f) => ({ ...f, Amount: 0 }))}
        >
          Reset
        </button>
        <button
          className="text-blue-600 px-3 py-1 rounded border"
          onClick={loadExample}
        >
          Load fraud example
        </button>
      </div>

      {result && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-medium">Prediction Result</h3>
            <RiskBadge
              score={result.scaled_score ?? result.fraud_probability ?? 0}
            />
          </div>

          <div className="p-4 bg-gray-50 rounded">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="text-sm text-gray-500">
                  Fraud Probability (raw)
                </div>
                <div className="text-lg font-bold">
                  {result.fraud_probability ?? 0}
                </div>
                <div className="text-xs text-gray-400">decimal</div>
              </div>

              <div style={{ flex: 2 }}>
                <div className="text-sm text-gray-500 mb-1">
                  UI suspicion score
                </div>
                <AnimatedProgress
                  pct={
                    (result.scaled_score ?? result.fraud_probability * 100) *
                    100
                  }
                />
                <div className="text-xs text-gray-400 mt-1">
                  {(
                    (result.scaled_score ?? result.fraud_probability * 100) *
                    100
                  ).toFixed(2)}
                  % — threshold: 0.0001
                </div>
              </div>
            </div>
          </div>

          <details className="mt-3 p-3 bg-white border rounded text-sm text-gray-700">
            <summary className="cursor-pointer font-medium">
              ▾ Full response (JSON)
            </summary>
            <pre className="text-xs mt-2 overflow-auto max-h-40">
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}

function ChartsPanel({ history, distribution }) {
  // history = [{time, amount, fraud_probability}, ...]
  const histData = (history || []).map((r, i) => ({
    name: i + 1,
    amount: r.Amount ?? r.amount ?? 0,
    prob: (r.fraud_probability ?? 0) * 100,
  }));
  const distData = distribution || histData;

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-white p-4 rounded shadow-sm hover:shadow-md transition-shadow">
        <h4 className="font-semibold mb-2">Transaction amount distribution</h4>
        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={distData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="amount" fill="#3182ce" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white p-4 rounded shadow-sm hover:shadow-md transition-shadow">
        <h4 className="font-semibold mb-2">
          Fraud probability over recent txns
        </h4>
        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={histData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="prob"
                stroke="#e53e3e"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [history, setHistory] = useState([]);
  const [distribution, setDistribution] = useState([]);
  const [lastResult, setLastResult] = useState(null);
  const printRef = useRef();

  useEffect(() => {
    // load demo history (if you have endpoint)
    async function fetchHistory() {
      try {
        const res = await axios.get(`${API_ROOT}/history?card_id=card_demo`);
        setHistory(res.data || []);
        // simple distribution mapping
        setDistribution(
          (res.data || []).map((r, i) => ({
            name: i + 1,
            amount: r.Amount || r.amount || 0,
          }))
        );
      } catch (err) {
        // ignore if endpoint not present
        // console.warn("history fetch failed", err);
      }
    }
    fetchHistory();
  }, []);

  async function handleExportPDF() {
    if (!printRef.current) return;
    const node = printRef.current;
    // capture
    const canvas = await html2canvas(node, { scale: 2 });
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF({
      orientation: "portrait",
      unit: "px",
      format: [canvas.width, canvas.height],
    });
    pdf.addImage(imgData, "PNG", 0, 0, canvas.width, canvas.height);
    pdf.save("fraud_report.pdf");
  }

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* Side nav */}
      <aside className="w-64 bg-white border-r p-4 hidden md:block">
        <div className="mb-6 flex items-center gap-2">
          <FaChartBar className="text-2xl text-blue-600" />
          <div>
            <h3 className="font-bold">Fraud Dashboard</h3>
            <div className="text-xs text-gray-500">Connected: {API_ROOT}</div>
          </div>
        </div>

        <nav className="flex flex-col gap-2">
          <a
            className="flex items-center gap-2 px-3 py-2 rounded hover:bg-gray-50"
            href="#"
          >
            <FaHome /> Overview
          </a>
          <a
            className="flex items-center gap-2 px-3 py-2 rounded hover:bg-gray-50"
            href="#"
          >
            <FaChartBar /> Analytics
          </a>
          <a
            className="flex items-center gap-2 px-3 py-2 rounded hover:bg-gray-50"
            href="#"
          >
            <FaFilePdf /> Reports
          </a>
        </nav>

        <div className="mt-6">
          <button
            onClick={handleExportPDF}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
          >
            <FaDownload /> Export report (PDF)
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 p-8">
        <div ref={printRef}>
          <div className="max-w-3xl mx-auto">
            <PredictCard onPredictResult={setLastResult} />
            <div className="mt-6">
              <ChartsPanel history={history} distribution={distribution} />
            </div>
          </div>
        </div>

        {/* Floating quick actions */}
        <div className="fixed right-6 bottom-6 flex flex-col gap-2">
          <button
            title="Export PDF"
            onClick={handleExportPDF}
            className="bg-white p-3 rounded-full shadow hover:shadow-lg"
          >
            <FaFilePdf />
          </button>
          <button
            title="Refresh history"
            onClick={() => window.location.reload()}
            className="bg-white p-3 rounded-full shadow hover:shadow-lg"
          >
            <FaRecycle />
          </button>
        </div>
      </main>
    </div>
  );
}
