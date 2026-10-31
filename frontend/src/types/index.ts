// ─── Shared Types ─────────────────────────────────────────────────────────────

export type DatasetKey =
  "UNSW_NB15" | "ISCX_15s" | "ISCX_30s" | "ISCX_60s" | "ISCX_120s";

export const DATASET_LABELS: Record<DatasetKey, string> = {
  UNSW_NB15: "UNSW-NB15",
  ISCX_15s: "ISCX 15s",
  ISCX_30s: "ISCX 30s",
  ISCX_60s: "ISCX 60s",
  ISCX_120s: "ISCX 120s",
};

export const DATASET_DESCRIPTIONS: Record<DatasetKey, string> = {
  UNSW_NB15:
    "Classification of normal network traffic vs. nine attack categories including DoS, Exploits, Fuzzers, Reconnaissance, and more.",
  ISCX_15s:
    "VPN vs. non-VPN traffic classification using 15-second flow aggregation windows across 14 application classes.",
  ISCX_30s:
    "VPN vs. non-VPN traffic classification using 30-second flow aggregation windows across 14 application classes.",
  ISCX_60s:
    "VPN vs. non-VPN traffic classification using 60-second flow aggregation windows across 14 application classes.",
  ISCX_120s:
    "VPN vs. non-VPN traffic classification using 120-second flow aggregation windows across 14 application classes.",
};

export type Page =
  "overview" | "classifier" | "datasets" | "about";
