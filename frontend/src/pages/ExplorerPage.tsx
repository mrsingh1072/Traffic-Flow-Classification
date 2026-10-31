import { useEffect, useState } from "react";
import { Database, ArrowUpRight } from "lucide-react";
import {
  fetchDatasets,
  fetchSchema,
  fetchClassMetrics,
  fetchMetrics,
  getErrorMessage,
  type DatasetInfo,
  type SchemaResponse,
  type ClassMetric,
  type DatasetMetrics,
} from "../services/api";
import {
  DATASET_LABELS,
  DATASET_DESCRIPTIONS,
  type DatasetKey,
} from "../types";
import { PageHeading, ErrorState, LoadingState } from "../components/ui";
export function ExplorerPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [dataset, setDataset] = useState<DatasetKey>("UNSW_NB15");
  const [catalogError, setCatalogError] = useState("");
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [classes, setClasses] = useState<ClassMetric[] | null>(null);
  const [metrics, setMetrics] = useState<DatasetMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState<string[]>([]);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    fetchDatasets()
      .then((d) => {
        if (active) {
          setDatasets(d.datasets);
          setCatalogError("");
        }
      })
      .catch((e) => {
        if (active) setCatalogError(getErrorMessage(e));
      });
    return () => {
      active = false;
    };
  }, [revision]);
  useEffect(() => {
    let active = true;
    setLoading(true);
    setSchema(null);
    setClasses(null);
    setMetrics(null);
    setErrors([]);
    Promise.allSettled([
      fetchSchema(dataset),
      fetchClassMetrics(dataset),
      fetchMetrics(dataset),
    ] as const).then(([s, c, m]) => {
      if (!active) return;
      const err: string[] = [];
      if (s.status === "fulfilled") setSchema(s.value);
      else err.push("Feature schema: " + getErrorMessage(s.reason));
      if (c.status === "fulfilled") setClasses(c.value.classes);
      else err.push("Class information: " + getErrorMessage(c.reason));
      if (m.status === "fulfilled") setMetrics(m.value.metrics[0] ?? null);
      else err.push("Evaluation: " + getErrorMessage(m.reason));
      setErrors(err);
      setLoading(false);
    });
    return () => {
      active = false;
    };
  }, [dataset, revision]);
  const info = datasets.find((d) => d.dataset === dataset);
  return (
    <div>
      <PageHeading
        eyebrow="DATASET EXPLORER"
        title="Dataset Explorer"
        description="Inspect dataset configurations, feature schemas, traffic classes, and artifact availability."
      />
      {catalogError && (
        <ErrorState
          message={catalogError}
          onRetry={() => setRevision((n) => n + 1)}
        />
      )}
      <div className="explorer-layout">
        <nav className="explorer-list" aria-label="Dataset configurations">
          {datasets.map((d) => (
            <button
              className={`explorer-item ${dataset === d.dataset ? "selected" : ""}`}
              key={d.dataset}
              aria-pressed={dataset === d.dataset}
              onClick={() => setDataset(d.dataset as DatasetKey)}
            >
              <Database
                size={14}
                style={{
                  display: "inline",
                  marginRight: 9,
                  verticalAlign: "middle",
                }}
              />
              {DATASET_LABELS[d.dataset as DatasetKey] || d.dataset}
              <small>{d.dataset}</small>
            </button>
          ))}
        </nav>
        <section className="panel">
          <div className="section-header">
            <div>
              
              <h2 style={{ marginTop: 8, fontSize: 30 }}>
                {DATASET_LABELS[dataset]}
              </h2>
            </div>
            {info && (
              <span
                className={`badge ${info.model_available && info.preprocessor_available ? "badge-neutral" : "badge-accent"}`}
              >
                {info.model_available && info.preprocessor_available
                  ? "Artifacts ready"
                  : "Artifacts incomplete"}
              </span>
            )}
          </div>
          <p
            className="muted"
            style={{ fontSize: 18, maxWidth: 680, lineHeight: 1.9 }}
          >
            {DATASET_DESCRIPTIONS[dataset]}
          </p>
          {loading ? (
            <LoadingState text="Reading feature and class information…" />
          ) : (
            <>
              {errors.map((e) => (
                <ErrorState
                  key={e}
                  message={e}
                  onRetry={() => setRevision((n) => n + 1)}
                />
              ))}
              <div className="explorer-facts">
                <div>
                  <small>Raw features</small>
                  <strong>{schema?.required_feature_count ?? "—"}</strong>
                </div>
                <div>
                  <small>Traffic classes</small>
                  <strong>{classes?.length ?? "—"}</strong>
                </div>
                <div>
                  <small>Test samples</small>
                  <strong>
                    {metrics?.test_samples.toLocaleString() ?? "—"}
                  </strong>
                </div>
                <div>
                  <small>Training labels used</small>
                  <strong>{info?.label_fraction ?? "—"}</strong>
                </div>
              </div>
              {info && (
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 18,
                    marginBottom: 26,
                  }}
                >
                  <span className="availability">
                    <i
                      className={`status-dot ${info.model_available ? "healthy" : "degraded"}`}
                    />
                    CNN checkpoint{" "}
                    {info.model_available ? "available" : "missing"}
                  </span>
                  <span className="availability">
                    <i
                      className={`status-dot ${info.preprocessor_available ? "healthy" : "degraded"}`}
                    />
                    Preprocessor{" "}
                    {info.preprocessor_available ? "available" : "missing"}
                  </span>
                </div>
              )}
              {classes && (
                <div>
                  <h3>Traffic classes</h3>
                  <p className="muted small" style={{ marginTop: 5 }}>
                    Class labels reported by the evaluation endpoint.
                  </p>
                  <div className="class-tags">
                    {classes.map((c) => (
                      <span className="tag" key={c.class_name}>
                        {c.class_name}
                      </span>
                    ))}
                  </div>
                  {!classes.length && (
                    <p className="empty-message">No class labels returned.</p>
                  )}
                </div>
              )}
              {schema && (
                <section className="feature-section">
                  <div className="section-header">
                    <div>
                      <h3>Raw feature schema</h3>
                      <p className="muted small">
                        Exact input keys and types from the fitted preprocessing
                        pipeline.
                      </p>
                    </div>
                    <a className="text-link" href="#/classifier">
                      Classify a record <ArrowUpRight size={13} />
                    </a>
                  </div>
                  <div className="feature-table">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Feature name</th>
                          <th>Input type</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(schema.features).map(([name, type]) => (
                          <tr key={name}>
                            <td className="mono">{name}</td>
                            <td>
                              <span className="badge badge-muted">{type}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="input-help">{schema.note}</p>
                </section>
              )}
            </>
          )}
        </section>
      </div>
      <p className="chart-caption">
        These benchmark datasets contain recorded traffic. The application
        classifies supplied feature records; it does not capture live 5G
        traffic.
      </p>
    </div>
  );
}
