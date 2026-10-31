import { ThemeToggle } from "../ThemeToggle";
import {
  LayoutDashboard,
  ScanLine,
  Database,
  Info,
  X,
} from "lucide-react";
import { useEffect, useRef } from "react";
import { motion } from "motion/react";
import type { Page } from "../../types";
import { PAGE_PATHS } from "../../routes";

const items = [
  { id: "overview", label: "Overview", Icon: LayoutDashboard },
  { id: "classifier", label: "Traffic Classifier", Icon: ScanLine },
  { id: "datasets", label: "Dataset Explorer", Icon: Database },
  { id: "about", label: "About", Icon: Info },
] as const;

export function Sidebar({
  activePage,
  mobileOpen,
  onMobileClose,
}: {
  activePage: Page;
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}) {
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!mobileOpen) return;
    const previous = document.activeElement as HTMLElement;
    panel.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const handle = (e: KeyboardEvent) => {
      if (e.key === "Escape") onMobileClose?.();
      if (e.key === "Tab") {
        const els = panel.current?.querySelectorAll<HTMLElement>("button, a");
        if (!els?.length) return;
        if (e.shiftKey && document.activeElement === els[0]) {
          e.preventDefault();
          els[els.length - 1].focus();
        } else if (
          !e.shiftKey &&
          document.activeElement === els[els.length - 1]
        ) {
          e.preventDefault();
          els[0].focus();
        }
      }
    };
    document.addEventListener("keydown", handle);
    return () => {
      document.removeEventListener("keydown", handle);
      previous?.focus();
    };
  }, [mobileOpen, onMobileClose]);

  return (
    <>
      {mobileOpen && <div className="nav-backdrop" onClick={onMobileClose} />}
      <aside
        ref={panel}
        className={`sidebar ${mobileOpen ? "open" : ""}`}
        aria-label="Main navigation"
      >
        <a
          className="brand"
          href="#/overview"
          onClick={onMobileClose}
          aria-label="TrafficFlow overview"
        >
          <span className="brand-mark">
            <svg
              width="26"
              height="28"
              viewBox="0 0 26 28"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M5 7H21M5 7V21H21M21 7V21"
                stroke="currentColor"
                strokeWidth="1.5"
              />
              <path d="M10 14H16" stroke="currentColor" strokeWidth="1.5" />
              <rect
                x="1"
                y="3"
                width="8"
                height="8"
                fill="var(--surface)"
                stroke="currentColor"
                strokeWidth="1.5"
              />
              <rect
                x="17"
                y="3"
                width="8"
                height="8"
                fill="var(--surface)"
                stroke="currentColor"
                strokeWidth="1.5"
              />
              <rect
                x="17"
                y="17"
                width="8"
                height="8"
                fill="var(--surface)"
                stroke="currentColor"
                strokeWidth="1.5"
              />
              <rect
                x="1"
                y="17"
                width="8"
                height="8"
                fill="var(--surface)"
                stroke="currentColor"
                strokeWidth="1.5"
              />
            </svg>
          </span>
          <span className="brand-word">
            <strong>Traffic</strong>Flow
          </span>
        </a>
        <button
          className="mobile-close btn btn-ghost btn-icon"
          onClick={onMobileClose}
          aria-label="Close navigation"
        >
          <X size={18} />
        </button>
        <div className="sidebar-rule" />
        <nav>
          {items.map(({ id, label, Icon }) => (
            <a
              key={id}
              className={`nav-item ${activePage === id ? "active" : ""}`}
              href={PAGE_PATHS[id]}
              onClick={onMobileClose}
              aria-current={activePage === id ? "page" : undefined}
            >
              {activePage === id && <motion.span className="nav-line" layoutId="navigation-line" transition={{ duration: 0.22 }} />}
              <Icon size={18} strokeWidth={1.6} />
              {label}
            </a>
          ))}
        </nav>
        <ThemeToggle className="desktop-theme-toggle" />
      </aside>
    </>
  );
}
