import { InferencePipeline } from "../components/visualizations/InferencePipeline";
import { ArrowUpRight } from "lucide-react";
import { PageHeading } from "../components/ui";
const base = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
export function AboutPage() {
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
      <InferencePipeline />
      <div className="about-columns">
        <section className="about-section">
          <h3>Model architecture</h3>
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
          <p>
            Evaluation metrics describe held-out test splits. Class support and
            per-class recall give essential context, especially where
            minority-class performance differs from aggregate scores.
          </p>
        </section>
      </div>
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
