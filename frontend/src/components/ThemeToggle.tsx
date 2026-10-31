import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Moon, Sun } from "lucide-react";

type Theme = "dark" | "light";
const ThemeContext = createContext<{ theme: Theme; toggle: () => void } | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => document.documentElement.dataset.theme === "light" ? "light" : "dark");
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#080808" : "#F4F1EB");
    try { localStorage.setItem("trafficflow-theme", theme); } catch { /* Theme remains available when browser storage is disabled. */ }
  }, [theme]);
  useEffect(() => {
    const sync = (event: StorageEvent) => {
      if (event.key === "trafficflow-theme") setTheme(event.newValue === "light" ? "light" : "dark");
    };
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);
  return <ThemeContext.Provider value={{ theme, toggle: () => setTheme(value => value === "dark" ? "light" : "dark") }}>{children}</ThemeContext.Provider>;
}

export function ThemeToggle({ className = "" }: { className?: string }) {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("ThemeToggle requires ThemeProvider");
  const label = context.theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
  return <button type="button" className={`theme-toggle ${className}`} onClick={context.toggle} aria-label={label} title={label}>
    {context.theme === "dark" ? <Sun size={19} strokeWidth={1.7} aria-hidden="true" /> : <Moon size={19} strokeWidth={1.7} aria-hidden="true" />}
  </button>;
}
