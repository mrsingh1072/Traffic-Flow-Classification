import { motion, useReducedMotion } from "motion/react";
import "./overview.css";
import { useEffect, useState } from "react";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import {
  fetchDatasets,
  fetchMetrics,
  getErrorMessage,
  type DatasetInfo,
  type DatasetMetrics,
} from "../services/api";
import { InferencePipeline } from "../components/visualizations/InferencePipeline";
import { SignalNetwork } from "../components/visualizations/SignalNetwork";
import { ErrorState } from "../components/ui";

export function OverviewPage() {
  const reduced = useReducedMotion();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [metrics, setMetrics] = useState<DatasetMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState<string[]>([]);
  const load = async () => {
    setLoading(true);
    setErrors([]);
    const [catalog, evaluation] = await Promise.allSettled([
      fetchDatasets(),
      fetchMetrics(),
    ] as const);
    const issues: string[] = [];
    if (catalog.status === "fulfilled") setDatasets(catalog.value.datasets);
    else {
      setDatasets([]);
      issues.push("Dataset inventory: " + getErrorMessage(catalog.reason));
    }
    if (evaluation.status === "fulfilled") setMetrics(evaluation.value.metrics);
    else {
      setMetrics([]);
      issues.push("Evaluation results: " + getErrorMessage(evaluation.reason));
    }
    setErrors(issues);
    setLoading(false);
  };
  useEffect(() => {
    void load();
  }, []);
  const ready = datasets.filter(
    (d) => d.model_available && d.preprocessor_available,
  ).length;
  const samples = metrics.reduce((n, m) => n + m.test_samples, 0);


  return (
    <div className="overview-page">
      <div className="signal-hero">
        <div className="signal-intro">
          <div className="signal-kicker"><span />5G NETWORK INTELLIGENCE</div>
          <h1><span className="hero-line">Every packet.</span><br />A new <em>perspective.</em></h1>
          <p>Understand the network. Classify the signal. Turn traffic features into insight with dataset-specific CNN models.</p>
          <div className="signal-actions">
            <a href="#/classifier" className="btn btn-primary">Start Classification <ArrowRight size={17}/></a>
            <a href="#/about" className="signal-secondary">View Performance <ArrowUpRight size={16}/></a>
          </div>
        </div>
        <SignalNetwork />
      </div>
      {errors.map((e) => (
        <ErrorState key={e} message={e} onRetry={load} />
      ))}
      <div className="overview-workspace">
        <section className="overview-stats" aria-label="Workspace summary">
          {[
            {
              label: "Dataset configurations",
              value: loading
                ? "—"
                : datasets.length
                  ? String(datasets.length)
                  : "Unavailable",
              note: "UNSW-NB15 / ISCX VPN-nonVPN",
            },
            {
              label: "Available models",
              value: loading
                ? "—"
                : datasets.length
                  ? `${ready} / ${datasets.length}`
                  : "Unavailable",
              note: "Checkpoint + preprocessor",
            },
            {
              label: "Held-out test samples",
              value: loading
                ? "—"
                : metrics.length
                  ? samples.toLocaleString()
                  : "Unavailable",
              note: "Across evaluated test splits",
            },
          ].map((s) => (
            <motion.div className="overview-stat" key={s.label} initial={reduced ? false : {opacity:0,y:10}} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{duration:.35}}>
              <div className="stat-label">{s.label}</div>
              <div className={`stat-number${s.value === "Unavailable" ? " stat-unavailable" : ""}`}>{s.value}</div>
              <div className="overview-stat-note">{s.note}</div>
            </motion.div>
          ))}
        </section>
      </div>
      <motion.div className="overview-pipeline" initial={reduced ? false : {opacity:0,y:8}} whileInView={{opacity:1,y:0}} viewport={{once:true,amount:.2}} transition={{duration:.35}}><InferencePipeline /></motion.div>

    </div>
  );
}
