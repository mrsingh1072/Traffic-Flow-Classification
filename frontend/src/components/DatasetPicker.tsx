import { useEffect, useState } from "react";
import {
  fetchDatasets,
  getErrorMessage,
  type DatasetInfo,
} from "../services/api";
import { DATASET_LABELS, type DatasetKey } from "../types";
import { LoadingState } from "./ui";
export function DatasetPicker({
  value,
  onChange,
  disabled = false,
}: {
  value: DatasetKey;
  onChange: (v: DatasetKey) => void;
  disabled?: boolean;
}) {
  const [items, setItems] = useState<DatasetInfo[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setItems((await fetchDatasets()).datasets);
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    void load();
  }, []);
  return (
    <div className="dataset-picker">
      {loading ? (
        <LoadingState text="Checking dataset availability…" />
      ) : error ? (
        <div className="dataset-list-error" role="status">
          <span>Dataset list unavailable.</span>
          <button className="btn btn-secondary btn-sm" onClick={load} title={error}>Reload datasets</button>
        </div>
      ) : (
        <>
          <span className="eyebrow">Dataset</span>
          <div className="dataset-tabs">
            {items.map((d) => (
              <button
                key={d.dataset}
                disabled={disabled}
                aria-pressed={value === d.dataset}
                className={`dataset-tab ${value === d.dataset ? "selected" : ""}`}
                onClick={() => onChange(d.dataset as DatasetKey)}
              >
                {DATASET_LABELS[d.dataset as DatasetKey] || d.dataset}
                <span
                  className={`status-dot ${d.model_available && d.preprocessor_available ? "healthy" : "degraded"}`}
                  title={
                    d.model_available && d.preprocessor_available
                      ? "Artifacts available"
                      : "Artifacts incomplete"
                  }
                />
              </button>
            ))}
          </div>
          {items.length === 0 && (
            <p className="muted">No datasets configured.</p>
          )}
        </>
      )}
    </div>
  );
}
