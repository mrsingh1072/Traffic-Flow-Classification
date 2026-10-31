// ─── API Service ─────────────────────────────────────────────────────────────
// Centralized API client for the TrafficFlow backend.
// All requests use the VITE_API_BASE_URL env variable.

import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

// ─── Response Types ───────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "healthy" | "degraded";
  device: string;
  datasets_configured: number;
  missing_artifacts: string[];
}

export interface DatasetInfo {
  dataset: string;
  model: string;
  label_fraction: string;
  preprocessor_available: boolean;
  model_available: boolean;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
}

export interface SchemaResponse {
  dataset: string;
  required_feature_count: number;
  features: Record<string, "numeric" | "categorical" | "unknown">;
  note: string;
}

export interface PredictResponse {
  dataset: string;
  model: string;
  predicted_class: string;
  class_id: number;
  confidence: number;
  probabilities: Record<string, number>;
  note: string;
}

export interface DatasetMetrics {
  dataset: string;
  test_samples: number;
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  f1_weighted: number;
}

export interface MetricsResponse {
  metrics: DatasetMetrics[];
}

export interface ClassMetric {
  class_name: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface ClassMetricsResponse {
  dataset: string;
  classes: ClassMetric[];
}

export interface ConfusionMatrixResponse {
  dataset: string;
  labels: string[];
  matrix: number[][];
}

// ─── API Functions ────────────────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await api.get<HealthResponse>("/health", { timeout: 5_000 });
  return res.data;
}

export async function fetchDatasets(): Promise<DatasetsResponse> {
  const res = await api.get<DatasetsResponse>("/datasets");
  return res.data;
}

export async function fetchSchema(dataset: string): Promise<SchemaResponse> {
  const res = await api.get<SchemaResponse>("/schema", { params: { dataset } });
  return res.data;
}

export async function fetchMetrics(dataset?: string): Promise<MetricsResponse> {
  const res = await api.get<MetricsResponse>("/metrics", {
    params: dataset ? { dataset } : {},
  });
  return res.data;
}

export async function fetchClassMetrics(
  dataset: string,
): Promise<ClassMetricsResponse> {
  const res = await api.get<ClassMetricsResponse>(
    `/metrics/${dataset}/classes`,
  );
  return res.data;
}

export async function fetchConfusionMatrix(
  dataset: string,
): Promise<ConfusionMatrixResponse> {
  const res = await api.get<ConfusionMatrixResponse>(
    `/metrics/${dataset}/confusion-matrix`,
  );
  return res.data;
}

export async function predict(
  dataset: string,
  features: Record<string, string | number | null>,
): Promise<PredictResponse> {
  const res = await api.post<PredictResponse>("/predict", {
    dataset,
    features,
  });
  return res.data;
}

export interface BatchPredictResponse {
  dataset: string;
  count: number;
  predictions: PredictResponse[];
}

export async function predictBatch(
  dataset: string,
  records: Record<string, string | number | null>[],
): Promise<BatchPredictResponse> {
  return (
    await api.post<BatchPredictResponse>("/predict/batch", { dataset, records })
  ).data;
}

// ─── Error helpers ────────────────────────────────────────────────────────────

export function getErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail))
      return detail
        .map((item) => `${item.loc?.join(".") ?? "Input"}: ${item.msg}`)
        .join("; ");
    if (detail?.message) return detail.message;
    if (err.response?.status === 503)
      return "Backend service unavailable. Is the API server running?";
    if (err.code === "ECONNABORTED" || err.code === "ERR_NETWORK")
      return (
        "Cannot reach the backend. Check that the API server is running at " +
        BASE_URL
      );
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return "An unexpected error occurred.";
}
