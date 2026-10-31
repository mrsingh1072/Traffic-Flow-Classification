import { motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchMetrics,
  fetchClassMetrics,
  fetchConfusionMatrix,
  getErrorMessage,
  type DatasetMetrics,
  type ClassMetric,
  type ConfusionMatrixResponse,
} from "../services/api";
import { DatasetPicker } from "../components/DatasetPicker";
import { PageHeading, LoadingState, ErrorState } from "../components/ui";
import { ConfusionMatrix } from "../components/evaluation/ConfusionMatrix";
import { DATASET_LABELS, type DatasetKey } from "../types";
const pct = (v: number) => (v * 100).toFixed(2) + "%";
export function EvaluationPage() {
  const reduced = useReducedMotion();
  const [dataset, setDataset] = useState<DatasetKey>("UNSW_NB15");
  const [metrics, setMetrics] = useState<DatasetMetrics | null>(null);
  const [classes, setClasses] = useState<ClassMetric[]>([]);
  const [matrix, setMatrix] = useState<ConfusionMatrixResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState<string[]>([]);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    setLoading(true);
    setMetrics(null);
    setClasses([]);
    setMatrix(null);
    setErrors([]);
    Promise.allSettled([
      fetchMetrics(dataset),
      fetchClassMetrics(dataset),
      fetchConfusionMatrix(dataset),
    ] as const).then(([m, c, x]) => {
      if (!active) return;
      const issues: string[] = [];
      if (m.status === "fulfilled") setMetrics(m.value.metrics[0] ?? null);
      else issues.push("Summary: " + getErrorMessage(m.reason));
      if (c.status === "fulfilled") setClasses(c.value.classes);
      else issues.push("Class report: " + getErrorMessage(c.reason));
      if (x.status === "fulfilled") setMatrix(x.value);
      else issues.push("Confusion matrix: " + getErrorMessage(x.reason));
      setErrors(issues);
      setLoading(false);
    });
    return () => {
      active = false;
    };
  }, [dataset, revision]);
  const chart = classes.map((c) => ({
    name: c.class_name,
    Precision: c.precision,
    Recall: c.recall,
    F1: c.f1_score,
  }));
  return (
    <div>
      <PageHeading
        eyebrow="MODEL EVALUATION"
        title="Model Evaluation"
        description="Inspect held-out test results, class-level behavior, and classification errors."
      />
      <DatasetPicker value={dataset} onChange={setDataset} />
      {loading ? (
        <LoadingState text="Loading evaluation evidence…" />
      ) : (
        <>
          {errors.map((e) => (
            <ErrorState
              key={e}
              message={e}
              onRetry={() => setRevision((n) => n + 1)}
            />
          ))}
          {metrics && (
            <>
              <div className="section-header">
                <h3>
                  {DATASET_LABELS[dataset]}{" "}
                  <span className="muted small"> / Held-out test split</span>
                </h3>
                <span className="quiet-tag">
                  {metrics.test_samples.toLocaleString()} test samples
                </span>
              </div>
              <div className="metric-strip">
                {[
                  { label: "Accuracy", value: metrics.accuracy },
                  { label: "Macro precision", value: metrics.precision_macro },
                  { label: "Macro recall", value: metrics.recall_macro },
                  { label: "Macro F1", value: metrics.f1_macro },
                  { label: "Weighted F1", value: metrics.f1_weighted },
                ].map((m) => (
                  <div key={m.label}>
                    <small>{m.label}</small>
                    <strong>{pct(m.value)}</strong>
                  </div>
                ))}
              </div>
              <p className="muted small" style={{ marginBottom: 24 }}>
                Macro metrics weight each class equally. Weighted F1 accounts
                for class support. Test samples:{" "}
                {metrics.test_samples.toLocaleString()}.
              </p>
            </>
          )}
          {classes.length > 0 ? (
            <div className="evaluation-analysis">
              <section className="panel">
                <div className="section-header">
                  <div>
                    <h3>Class-level performance</h3>
                    <p className="muted small">
                      Class-level precision, recall, and F1-score
                    </p>
                  </div>
                </div>
                <div className="chart-legend">
                  <span>
                    <i style={{ background: "var(--accent)" }} />
                    Precision
                  </span>
                  <span>
                    <i style={{ background: "var(--chart-secondary)" }} />
                    Recall
                  </span>
                  <span>
                    <i style={{ background: "var(--chart-tertiary)" }} />
                    F1-score
                  </span>
                </div>
                <div
                  className="table-scroll"
                  role="img"
                  aria-label="Class metrics chart. Exact values are provided in the following table."
                >
                  <motion.div
                    key={dataset}
                    initial={reduced ? false : {opacity:0,clipPath:"inset(0 100% 0 0)"}}
                    animate={{opacity:1,clipPath:"inset(0 0% 0 0)"}}
                    transition={{duration:.4,ease:"easeOut"}}
                    className="evaluation-chart"
                    style={{ minWidth: 600, height: Math.max(480, classes.length * 58) }}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={chart}
                        layout="vertical"
                        margin={{ bottom: 10, left: 0, right: 15 }}
                        barGap={2}
                      >
                        <CartesianGrid
                          horizontal={false}
                          stroke="var(--border)"
                          strokeDasharray="3 4"
                        />
                        <XAxis
                          type="number"
                          domain={[0, 1]}
                          tickFormatter={(v) => Math.round(v * 100) + "%"}
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "var(--muted)", fontSize: 16 }}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={165}
                          interval={0}
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "var(--muted)", fontSize: 16 }}
                        />
                        <Tooltip
          itemStyle={{ color: "var(--text)" }}
          labelStyle={{ color: "var(--text)" }}
                          cursor={{ fill: "var(--surface)" }}
                          formatter={(v) => pct(Number(v))}
                          contentStyle={{
                            background: "var(--surface)",
                            border: "1px solid var(--border)",
                            borderRadius: 5,
                            color: "var(--text)",
                          }}
                        />
                        <Bar
                          isAnimationActive={false}
                          dataKey="Precision"
                          fill="var(--accent)"
                          radius={[2, 2, 0, 0]}
                        />
                        <Bar
                          isAnimationActive={false}
                          dataKey="Recall"
                          fill="var(--chart-secondary)"
                          radius={[2, 2, 0, 0]}
                        />
                        <Bar
                          isAnimationActive={false}
                          dataKey="F1"
                          fill="var(--chart-tertiary)"
                          radius={[2, 2, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </motion.div>
                </div>
              </section>
              <section className="panel evaluation-table">
                <div className="section-header">
                  <h3>Class performance report</h3>
                  <span className="muted small">Sorted by F1-score</span>
                </div>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {[
                          "Traffic class",
                          "Precision",
                          "Recall",
                          "F1-score",
                          "Support",
                        ].map((h) => (
                          <th key={h}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[...classes]
                        .sort((a, b) => b.f1_score - a.f1_score)
                        .map((c) => (
                          <tr key={c.class_name}>
                            <td>{c.class_name}</td>
                            <td className="mono">{pct(c.precision)}</td>
                            <td className="mono">{pct(c.recall)}</td>
                            <td
                              className="mono"
                              style={{ color: "var(--text)" }}
                            >
                              {pct(c.f1_score)}
                            </td>
                            <td className="mono muted">
                              {c.support.toLocaleString()}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          ) : (
            <p className="empty-message">
              No class performance rows available for this configuration.
            </p>
          )}
          {matrix && (
            <section className="panel matrix-panel">
              <div className="section-header">
                <div>
                  <h3>Confusion matrix</h3>
                  <p className="muted small">
                    {matrix.labels.length} classes · Exact held-out sample
                    counts
                  </p>
                </div>
              </div>
              <ConfusionMatrix data={matrix} />
            </section>
          )}
          <p className="chart-caption">
            Evaluation uses the saved 100% training-label checkpoint. These
            datasets represent different tasks and are not directly comparable.
          </p>
        </>
      )}
    </div>
  );
}
