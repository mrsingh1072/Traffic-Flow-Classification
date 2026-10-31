import type { ReactNode } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
export function PageHeading({
  title,
  description,
  aside,
}: {
  eyebrow: string;
  title: string;
  description: string;
  aside?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {aside}
    </div>
  );
}
export function LoadingState({ text = "Loading…" }: { text?: string }) {
  return (
    <div className="loading-line" role="status">
      <span className="spinner" />
      {text}
    </div>
  );
}
export function ErrorState({
  message,
  onRetry,
  title = "Data unavailable",
}: {
  message: string;
  onRetry?: () => void;
  title?: string;
}) {
  return (
    <div className="error-state" role="alert">
      <AlertCircle size={18} />
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
      {onRetry && (
        <button className="btn btn-secondary btn-sm" onClick={onRetry}>
          <RefreshCw size={13} />
          Retry
        </button>
      )}
    </div>
  );
}
