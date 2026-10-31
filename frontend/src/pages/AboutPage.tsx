import { useEffect, useState } from "react";
import { fetchMetrics, getErrorMessage, type DatasetMetrics } from "../services/api";
import { DATASET_LABELS, type DatasetKey } from "../types";
import "./about.css";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { PageHeading, LoadingState, ErrorState } from "../components/ui";
const base = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const percent = (value: number) => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "Unavailable";
export function AboutPage() {
  const [metrics, setMetrics] = useState<DatasetMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    fetchMetrics().then(response => {
      if (active) setMetrics(response.metrics);
    }).catch(reason => {
      if (active) setError(getErrorMessage(reason));
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [revision]);
  return (
    <div className="about-layout">
      <PageHeading
        eyebrow="ABOUT THE PROJECT"
        title="About TrafficFlow"
        description="CNN-Based Traffic Flow Classification in 5G Core Networks."
      />
      <div className="about-intro">
        <h2>Traffic classification in 5G core networks</h2>
        <p>
          This project explores one-dimensional convolutional neural networks
          for traffic-flow classification. It brings dataset-specific
          preprocessing, trained CNN checkpoints, and held-out evaluation into
          one workspace for examining network behavior.
        </p>
        <p>
          UNSW-NB15 covers normal and attack traffic. ISCX VPN-nonVPN covers
          application traffic and VPN variants using 15-, 30-, 60-, and
          120-second flow configurations.
        </p>
      </div>
      <section className="about-model-section" aria-labelledby="model-evaluation-title">
        <div className="about-model-heading">
          <h2 id="model-evaluation-title">Model Architecture &amp; Evaluation</h2>
          <span className="tag">TrafficCNN-1D</span>
        </div>
        <ol className="about-model-pipeline" aria-label="Classification pipeline">
          {["Dataset", "Preprocessing", "CNN", "Classification"].map((stage, index) => (
            <li key={stage}><span className="mono">0{index + 1}</span><strong>{stage}</strong>{index < 3 && <ArrowRight size={17} aria-hidden="true" />}</li>
          ))}
        </ol>
      <div className="about-columns">
        <section className="about-section">
          <h3>TrafficCNN-1D architecture</h3>
          <ul>
            <li>Three Conv1d → BatchNorm → ReLU blocks</li>
            <li>Max pooling and adaptive max pooling</li>
            <li>128 → 64 → class-count linear layers</li>
            <li>Dropout at 30% and 20%</li>
            <li>Separate fitted preprocessor for each dataset</li>
          </ul>
        </section>
        <section className="about-section">
          <h3>How to read the results</h3>
          <p>
            Confidence is a softmax score, not a calibrated probability. A high
            score does not establish that a prediction is correct.
          </p>
          <p>Macro precision, recall and F1 weight each class equally, highlighting performance beyond the most common traffic classes.</p>
        </section>
      </div>
        <div className="about-evaluation">
          <h3>Held-out test results</h3>
          <p className="about-methodology">Saved dataset-specific checkpoints are evaluated on their corresponding held-out test splits, using the fitted training preprocessors and original class mappings. Training and validation records are excluded from these test metrics. Different datasets represent different tasks; their scores are not directly comparable.</p>
          {loading ? <LoadingState text="Loading evaluation metrics…" /> : error ? <ErrorState title="Evaluation unavailable" message={error} onRetry={() => setRevision(n => n + 1)} /> : metrics.length ? (
            <div className="table-scroll" tabIndex={0} role="region" aria-label="Dataset evaluation metrics, scroll horizontally on smaller screens">
              <table className="data-table about-metrics-table">
                <caption>UNSW-NB15 and ISCX VPN-nonVPN · 100% training-label checkpoints</caption>
                <thead><tr><th scope="col">Dataset</th><th scope="col">Accuracy</th><th scope="col">Macro Precision</th><th scope="col">Macro Recall</th><th scope="col">Macro F1</th></tr></thead>
                <tbody>{metrics.map(metric => <tr key={metric.dataset}>
                  <th scope="row">{DATASET_LABELS[metric.dataset as DatasetKey] ?? metric.dataset}<small>{metric.test_samples.toLocaleString()} test samples</small></th>
                  <td>{percent(metric.accuracy)}</td><td>{percent(metric.precision_macro)}</td><td>{percent(metric.recall_macro)}</td><td>{percent(metric.f1_macro)}</td>
                </tr>)}</tbody>
              </table>
            </div>
          ) : <p className="muted">No evaluation metrics are available.</p>}
        </div>
      </section>
      <section className="about-section panel">
        <div className="section-header">
          <div>
            <h3>Backend API reference</h3>
            <p className="muted small">
              Existing FastAPI capabilities used by this workspace.
            </p>
          </div>
          <a
            className="text-link"
            href={base.replace(/\/$/, "") + "/docs"}
            target="_blank"
            rel="noreferrer"
          >
            Open API docs <ArrowUpRight size={14} />
          </a>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Endpoint</th>
                <th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["GET", "/health", "Backend and artifact readiness"],
                ["GET", "/datasets", "Configured datasets and availability"],
                [
                  "GET",
                  "/schema?dataset=…",
                  "Dataset-specific raw input schema",
                ],
                ["POST", "/predict", "Classify one feature record"],
                ["POST", "/predict/batch", "Classify 1–100 feature records"],
                ["GET", "/metrics", "Held-out evaluation summary"],
                [
                  "GET",
                  "/metrics/{dataset}/classes",
                  "Per-class precision, recall, F1, support",
                ],
                [
                  "GET",
                  "/metrics/{dataset}/confusion-matrix",
                  "Actual and predicted class counts",
                ],
              ].map(([method, path, desc]) => (
                <tr key={path}>
                  <td>
                    <span
                      className={`badge ${method === "GET" ? "badge-neutral" : "badge-accent"}`}
                    >
                      {method}
                    </span>
                  </td>
                  <td className="mono">{path}</td>
                  <td className="muted">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="about-section">
        <h3>Scope and limitations</h3>
        <p>
          The prediction service uses 100% training-label checkpoints. The
          repository also includes 10%, 25%, and 50% label-fraction experiments.
          This workspace uses the evaluation endpoints exposed by the backend.
        </p>
        <p>
          The application classifies supplied feature records. Live traffic
          capture, authentication, operational monitoring, and production
          infrastructure are separate work. Dataset access and redistribution
          remain subject to their respective terms.
        </p>
      </section>
    </div>
  );
}
