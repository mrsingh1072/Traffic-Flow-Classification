// ─── useHealth hook ───────────────────────────────────────────────────────────
import { useState, useEffect, useCallback, useRef } from "react";
import { fetchHealth, type HealthResponse } from "../services/api";

interface UseHealthResult {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useHealth(): UseHealthResult {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const request = useRef(0);

  const refetch = useCallback(() => {
    const id = ++request.current;
    setLoading(true);
    fetchHealth()
      .then((data) => {
        if (id !== request.current) return;
        setHealth(data);
        setError(null);
      })
      .catch(() => {
        if (id !== request.current) return;
        setHealth(null);
        setError("Cannot reach backend");
      })
      .finally(() => {
        if (id === request.current) setLoading(false);
      });
  }, []);

  useEffect(() => {
    refetch();
    const interval = setInterval(refetch, 30_000);
    window.addEventListener("online", refetch);
    window.addEventListener("focus", refetch);
    return () => {
      request.current++;
      clearInterval(interval);
      window.removeEventListener("online", refetch);
      window.removeEventListener("focus", refetch);
    };
  }, [refetch]);

  return { health, loading, error, refetch };
}
