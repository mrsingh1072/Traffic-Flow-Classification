import type { Page } from "./types";

export const PAGE_PATHS: Record<Page, string> = {
  overview: "#/overview",
  classifier: "#/classifier",
  evaluation: "#/evaluation",
  datasets: "#/datasets",
  about: "#/about",
};

export function pageFromHash(hash: string): Page {
  const route = hash.replace(/^#\/?/, "").replace(/\/$/, "");
  if (route === "explorer") return "datasets";
  return Object.hasOwn(PAGE_PATHS, route) ? (route as Page) : "overview";
}
