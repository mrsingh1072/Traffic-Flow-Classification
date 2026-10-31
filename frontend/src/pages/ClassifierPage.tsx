import { Fragment, useEffect, useRef, useState, type FormEvent } from "react";
import {
  ArrowRight,
  Download,
  FileJson,
  ScanLine,
  RotateCcw,
  Upload,
  AlertCircle,
  ChevronDown,
} from "lucide-react";
import {
  fetchSchema,
  predict,
  predictBatch,
  getErrorMessage,
  type SchemaResponse,
  type PredictResponse,
} from "../services/api";
import { DATASET_LABELS, type DatasetKey } from "../types";
import { DatasetPicker } from "../components/DatasetPicker";
import { ErrorState, LoadingState, PageHeading } from "../components/ui";
import exampleData from "../data/classifier-examples.json";
import "./classifier.css";
type RecordInput = Record<string, string | number | null>;
const examples: Record<DatasetKey, { source: string; examples: { label: string; testIndex: number; features: RecordInput }[] }> = exampleData;
function download(name: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function ClassifierPage() {
  const [dataset, setDataset] = useState<DatasetKey>("UNSW_NB15");
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [schemaError, setSchemaError] = useState("");
  const [revision, setRevision] = useState(0);
  const [values, setValues] = useState<Record<string, string>>({});
  const [json, setJson] = useState("");
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<PredictResponse[]>([]);
  const [selected, setSelected] = useState(0);
  const [exampleIndex, setExampleIndex] = useState(0);
  const [sampleLabels, setSampleLabels] = useState<string[]>([]);
  const request = useRef(0);
  const formRef = useRef<HTMLFormElement>(null);
  const [requestFailed, setRequestFailed] = useState(false);
  useEffect(() => {
    const id = ++request.current;
    setSchema(null);
    setLoading(true);
    setSchemaError("");
    setValues({});
    setJson("");
    setResults([]);
    setRequestFailed(false);
    setError("");
    setFieldErrors({});
    setExampleIndex(0);
    setSampleLabels([]);
    fetchSchema(dataset)
      .then((s) => {
        if (id === request.current) setSchema(s);
      })
      .catch((e) => {
        if (id === request.current) setSchemaError(getErrorMessage(e));
      })
      .finally(() => {
        if (id === request.current) setLoading(false);
      });
    return () => {
      request.current++;
    };
  }, [dataset, revision]);
  const reset = () => {
    setValues({});
    setResults([]);
    setRequestFailed(false);
    setError("");
    setFieldErrors({});
    setJson("");
    setSampleLabels([]);
  };
  const samples = examples[dataset].examples;
  const examplesMatchSchema = !!schema && samples.every(({ features }) =>
    Object.keys(features).length === Object.keys(schema.features).length &&
    Object.entries(schema.features).every(([key, kind]) => key in features &&
      (features[key] === null || (kind === "numeric" ? typeof features[key] === "number" : typeof features[key] === "string"))),
  );
  const loadExample = () => {
    if (!examplesMatchSchema) {
      setError("The demo examples do not match the current dataset schema. Use your own matching feature records.");
      return;
    }
    if (mode === "single") {
      const sample = samples[exampleIndex] ?? samples[0];
      setValues(Object.fromEntries(Object.entries(sample.features).map(([key, value]) => [key, value === null ? "" : String(value)])));
      setSampleLabels([sample.label]);
    } else {
      setJson(JSON.stringify(samples.map(sample => sample.features), null, 2));
      setSampleLabels(samples.map(sample => sample.label));
    }
    setResults([]);
    setFieldErrors({});
    setRequestFailed(false);
    setError("");
  };
  const template = () =>
    schema
      ? Object.fromEntries(Object.keys(schema.features).map((k) => [k, null]))
      : {};
  const getRecords = (): RecordInput[] => {
    if (!schema) throw new Error("Wait for the dataset schema to load.");
    let records: RecordInput[] = [];
    if (mode === "batch") {
      let parsed: unknown;
      try {
        parsed = JSON.parse(json);
      } catch {
        throw new Error("Enter a valid JSON array of feature objects.");
      }
      if (!Array.isArray(parsed) || parsed.length < 1 || parsed.length > 100)
        throw new Error(
          "A batch must contain between 1 and 100 feature objects.",
        );
      records = parsed;
    } else {
      const invalid: Record<string, string> = {};
      for (const [key, type] of Object.entries(schema.features)) {
        const v = values[key]?.trim();
        if (v && type === "numeric" && !Number.isFinite(Number(v)))
          invalid[key] = "Enter a finite number.";
      }
      if (Object.keys(invalid).length || Object.values(fieldErrors).some(Boolean)) {
        throw new Error("Correct the highlighted numeric fields.");
      }
      records = [
        Object.fromEntries(
          Object.entries(schema.features).map(([key, type]) => {
            const v = values[key]?.trim();
            return [key, !v ? null : type === "numeric" ? Number(v) : v];
          }),
        ),
      ];
    }
    records.forEach((row, index) => {
      const prefix = mode === "batch" ? `Record ${index + 1}: ` : "";
      if (!row || typeof row !== "object" || Array.isArray(row))
        throw new Error(prefix + "Each record must be a feature object.");
      const expected = Object.keys(schema.features);
      const missing = expected.filter((k) => !(k in row));
      const unknown = Object.keys(row).filter((k) => !(k in schema.features));
      if (missing.length || unknown.length)
        throw new Error(
          prefix +
            `Schema mismatch.${missing.length ? " Missing: " + missing.join(", ") + "." : ""}${unknown.length ? " Unknown: " + unknown.join(", ") + "." : ""}`,
        );
      if (Object.values(row).every((v) => v === null))
        throw new Error(
          prefix + (mode === "single" ? "Load a demo example or enter measured feature values before classifying." : "Supply at least one non-null feature value."),
        );
      for (const [key, type] of Object.entries(schema.features)) {
        const v = row[key];
        if (v === null) continue;
        if (
          type === "numeric" &&
          (typeof v !== "number" || !Number.isFinite(v))
        )
          throw new Error(
            prefix + key + " must be a finite JSON number or null.",
          );
        if (type !== "numeric" && typeof v !== "string")
          throw new Error(prefix + key + " must be a string or null.");
      }
    });
    return records;
  };
  let inputMessage = "";
  try { getRecords(); } catch (e) { inputMessage = getErrorMessage(e); }
  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!schema || busy) return;
    setRequestFailed(false);
    setError("");
    setFieldErrors({});
    setResults([]);
    let records: RecordInput[];
    try { records = getRecords(); }
    catch (e) { setError(getErrorMessage(e)); return; }
    setBusy(true);
    const id = request.current;
    try {
      const out =
        mode === "batch"
          ? (await predictBatch(dataset, records)).predictions
          : [await predict(dataset, records[0])];
      if (id === request.current) {
        setResults(out);
        setSelected(0);
      }
    } catch (e) {
      if (id === request.current) { setRequestFailed(true); setError(getErrorMessage(e)); }
    } finally {
      setBusy(false);
    }
  };
  const result = results[selected];
  const featureEntries = schema ? Object.entries(schema.features).sort((a, b) => a[1].localeCompare(b[1])) : [];
  return (
    <div className="classifier-page">
      <PageHeading
        eyebrow="INFERENCE WORKSPACE"
        title="Traffic Classifier"
        description="Select a traffic dataset, load a sample record or provide your own features, and run CNN inference."
      />
      <DatasetPicker value={dataset} onChange={setDataset} disabled={busy} />
      {loading ? (
        <LoadingState text="Loading the dataset-specific feature schema…" />
      ) : schemaError ? (
        <ErrorState
          message={schemaError}
          onRetry={() => setRevision((n) => n + 1)}
        />
      ) : (
        schema && (
          <div className="classifier-layout">
            <section className="panel classifier-input">
              <div className="section-header">
                <div>
                  <h3>Traffic features</h3>
                  <p className="muted small">
                    {schema.required_feature_count} raw features ·{" "}
                    {DATASET_LABELS[dataset]}
                  </p>
                </div>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={reset}
                  disabled={busy}
                >
                  <RotateCcw size={13} />
                  Clear
                </button>
              </div>
              <div className="mode-tabs" aria-label="Prediction mode">
                {(["single", "batch"] as const).map((m) => (
                  <button
                    key={m}
                    className={mode === m ? "selected" : ""}
                    disabled={busy}
                    aria-pressed={mode === m}
                    onClick={() => {
                      setMode(m);
                      setRequestFailed(false);
                      setError("");
                      setResults([]);
                      setSampleLabels([]);
                    }}
                  >
                    {m === "single" ? (
                      <ScanLine size={14} />
                    ) : (
                      <FileJson size={14} />
                    )}{" "}
                    {m === "single" ? "Single record" : "Batch JSON"}
                  </button>
                ))}
              </div>
              <form ref={formRef} onSubmit={submit} noValidate>
                {mode === "single" ? (
                  <>
                    <div className="classifier-example">
                      <div className="example-copy">
                        <h4>Load a dataset sample</h4>
                        <p>Choose a real held-out record to get started.</p>
                      </div>
                      <div className="example-actions">
                        <label htmlFor="demo-example" className="input-label">Demo record</label>
                        <select id="demo-example" className="select" value={exampleIndex} disabled={busy} onChange={e => setExampleIndex(Number(e.target.value))}>
                          {samples.map((sample, index) => <option key={sample.label} value={index}>{sample.label}</option>)}
                        </select>
                        <button type="button" className="btn btn-secondary" onClick={loadExample} disabled={busy || !examplesMatchSchema}>Load example <ArrowRight size={15} /></button>
                      </div>
                    </div>
                    {sampleLabels.length > 0 && <p className="example-status" role="status">Example loaded · {schema.required_feature_count} features · Dataset label: <strong>{sampleLabels[0]}</strong>.</p>}
                    <details className="manual-features">
                      <summary>Review or edit features <ChevronDown size={16} aria-hidden="true" /><span>{schema.required_feature_count} inputs</span></summary>
                    <p className="input-help">
                      These are measured network-flow statistics, not an IP address or website URL.
                      Use a complete dataset record rather than guessing values. Blank fields are imputed by the fitted preprocessing pipeline.
                    </p>
                    <div
                      className="feature-scroll"
                      tabIndex={0}
                      aria-label="Dataset feature inputs, scroll for all fields"
                    >
                      <fieldset disabled={busy} className="form-grid">
                        {featureEntries.map(([name, type], index) => (
                          <Fragment key={name}>
                            {(index === 0 ||
                              featureEntries[index - 1][1] !== type) && (
                              <h4 className="feature-group-heading">
                                {type === "numeric"
                                  ? "Numeric inputs"
                                  : type === "categorical"
                                    ? "Categorical inputs"
                                    : "Other inputs"}
                                <span>
                                  {
                                    featureEntries.filter(
                                      ([, kind]) => kind === type,
                                    ).length
                                  }{" "}
                                  features
                                </span>
                              </h4>
                            )}
                            <div className="input-group">
                              <label
                                className="input-label"
                                htmlFor={`feature-${name}`}
                              >
                                {name.replaceAll("_", " ")}
                                <span>{type}</span>
                              </label>
                              <input
                                id={`feature-${name}`}
                                className={`input ${fieldErrors[name] ? "input-error" : ""}`}
                                type={type === "numeric" ? "number" : "text"}
                                step="any"
                                value={values[name] ?? ""}
                                placeholder="null"
                                onChange={(e) => {
                                  setFieldErrors(previous => ({ ...previous, [name]: e.target.validity.badInput ? "Enter a finite number." : "" }));
                                  setResults([]);
                                  setSampleLabels([]);
                                  setError("");
                                  setRequestFailed(false);
                                  setValues((v) => ({
                                    ...v,
                                    [name]: e.target.value,
                                  }));
                                }}
                                aria-invalid={!!fieldErrors[name]}
                                aria-describedby={
                                  fieldErrors[name]
                                    ? `error-${name}`
                                    : undefined
                                }
                              />
                              {fieldErrors[name] && (
                                <small
                                  className="field-error"
                                  id={`error-${name}`}
                                >
                                  {fieldErrors[name]}
                                </small>
                              )}
                            </div>
                          </Fragment>
                        ))}
                      </fieldset>
                    </div>
                    </details>
                  </>
                ) : (
                  <>
                    <div className="batch-toolbar">
                      <button type="button" className="btn btn-secondary btn-sm" disabled={busy || !examplesMatchSchema} onClick={loadExample}><FileJson size={14} />Load sample batch</button>
                      <button type="button" className="btn btn-secondary btn-sm" disabled={busy || !examplesMatchSchema} onClick={() => download(`${dataset}-sample-batch.json`, samples.map(sample => sample.features))}><Download size={14} />Download sample JSON</button>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        disabled={busy}
                        onClick={() =>
                          download(`${dataset}-batch-template.json`, [
                            template(),
                          ])
                        }
                      >
                        <Download size={13} />
                        Blank template (advanced)
                      </button>
                      <label className="btn btn-secondary btn-sm">
                        <Upload size={13} />
                        Import JSON
                        <input
                          type="file"
                          accept=".json,application/json"
                          disabled={busy}
                          className="file-input"
                          onChange={async (e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;
                            if (file.size > 2 * 1024 * 1024) {
                              setError("Choose a JSON file smaller than 2 MB.");
                              return;
                            }
                            try {
                              setJson(await file.text());
                              setRequestFailed(false);
                      setError("");
                              setResults([]);
                              setSampleLabels([]);
                            } catch {
                              setError("Could not read this file.");
                            }
                            e.target.value = "";
                          }}
                        />
                      </label>
                    </div>
                    <p className="input-help batch-guidance">For a demo, choose Load sample batch, or download this dataset’s sample JSON and import that same file. Files from another dataset use different features.</p>
                    {sampleLabels.length > 0 && <p className="example-status" role="status">{sampleLabels.length} held-out examples loaded. Classify batch to compare the model’s predictions with the dataset labels.</p>}
                    <label className="input-label" htmlFor="batch-json">
                      Feature records <span>1–100 records</span>
                    </label>
                    <textarea
                      aria-invalid={!!json.trim() && !!inputMessage}
                      aria-describedby="classifier-validation"
                      id="batch-json"
                      className="input json-editor mono"
                      value={json}
                      disabled={busy}
                      onChange={(e) => { setJson(e.target.value); setResults([]); setSampleLabels([]); setError(""); setRequestFailed(false); }}
                      placeholder="Paste a JSON array of feature objects…"
                      spellCheck={false}
                    />
                    <p className="input-help">
                      Each object must include every feature in this dataset’s
                      schema. Use null for missing values. The whole batch must
                      be valid.
                    </p>
                  </>
                )}
                {error && !requestFailed && (
                  <ErrorState
                    message={error}
                    title="Classification not completed"
                  />
                )}
                <p id="classifier-validation" className={`input-validation${inputMessage ? " invalid" : ""}`} role="status">{inputMessage || (mode === "batch" ? `${getRecords().length} records ready for inference` : "Record ready for inference")}</p>
                <div className="form-submit">
                  <span className="muted small">
                    TrafficCNN-1D · 100% training labels
                  </span>
                  <button
                    className="btn btn-primary"
                    type="submit"
                    disabled={busy || !!inputMessage}
                    aria-describedby="classifier-validation"
                  >
                    {busy ? (
                      <>
                        <span className="spinner" />
                        Classifying…
                      </>
                    ) : (
                      <>
                        Classify Traffic
                        <ArrowRight size={14} />
                      </>
                    )}
                  </button>
                </div>
              </form>
            </section>
            <aside className="classifier-result" aria-live="polite" aria-busy={busy}>
              <h3 className="results-heading">Prediction results</h3>
              {result ? (
                <section className="panel result-panel">
                  <div className="eyebrow">Model prediction</div>
                  {mode === "batch" && <p className="batch-success" role="status">Batch complete · {results.length} records classified</p>}
                  {results.length > 1 && (
                    <>
                      <label htmlFor="result-index" className="input-label">
                        Batch record
                      </label>
                      <select
                        className="select"
                        id="result-index"
                        value={selected}
                        onChange={(e) => setSelected(Number(e.target.value))}
                      >
                        {results.map((r, i) => (
                          <option key={i} value={i}>
                            {i + 1} · {r.predicted_class}
                          </option>
                        ))}
                      </select>
                    </>
                  )}
                  <div className="result-class">{result.predicted_class}</div>
                  <p className="muted small">
                    {result.dataset} · Class {result.class_id}
                  </p>
                  {sampleLabels[selected] && <div className="sample-comparison"><span>Dataset label</span><strong>{sampleLabels[selected]}</strong><p>{result.predicted_class === sampleLabels[selected] ? "Prediction matches this example’s label." : "Prediction differs from this example’s label. Models can make mistakes; review the evaluation summary in About for overall performance."}</p></div>}
                  <div className="confidence">
                    <span>Softmax confidence</span>
                    <strong>
                      {(result.confidence * 100).toFixed(2)}
                      <small>%</small>
                    </strong>
                  </div>
                  {result.probabilities && Object.keys(result.probabilities).length > 0 && <>
                  <h4 className="eyebrow">Class scores</h4>
                  <div className="score-distribution">
                    {Object.entries(result.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([name, value]) => (
                        <div className="score-row" key={name}>
                          <div>
                            <span>{name}</span>
                            <span className="mono">
                              {(value * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="prob-bar-track">
                            <div
                              className="prob-bar-fill"
                              style={{
                                width: `${value * 100}%`,
                                background:
                                  name === result.predicted_class
                                    ? "var(--accent)"
                                    : "var(--chart-secondary)",
                              }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                  </>}
                  <p className="result-note">{result.note}</p>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() =>
                      download(`${dataset}-predictions.json`, results)
                    }
                  >
                    <Download size={13} />
                    Export{" "}
                    {results.length > 1
                      ? `${results.length} results`
                      : "result"}
                  </button>
                </section>
              ) : (
                <section className="result-empty" role={requestFailed ? "alert" : undefined}>
                  <div className="result-empty-icon">
                    {busy ? <span className="spinner" /> : requestFailed ? <AlertCircle size={25} strokeWidth={1.5} /> : <ScanLine size={25} strokeWidth={1.5} />}
                  </div>
                  <h3>{busy ? "Running inference" : requestFailed ? "Classification failed" : "Awaiting classification"}</h3>
                  <p>{busy ? "Processing your traffic features with the selected CNN." : requestFailed ? error : "Your prediction and confidence results will appear here after inference."}</p>
                  {requestFailed && <button className="btn btn-secondary btn-sm" disabled={busy || !!inputMessage} onClick={() => formRef.current?.requestSubmit()}><RotateCcw size={14} />Retry inference</button>}
                </section>
              )}
            </aside>
          </div>
        )
      )}
    </div>
  );
}
