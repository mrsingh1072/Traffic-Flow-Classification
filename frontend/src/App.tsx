import { lazy, Suspense, useEffect, useState } from "react";
import { MotionConfig, motion, useReducedMotion } from "motion/react";
import { Sidebar } from "./components/layout/Sidebar";
import { TopBar } from "./components/layout/TopBar";
import { OverviewPage } from "./pages/OverviewPage";
import { useHealth } from "./hooks/useHealth";
import type { Page } from "./types";
import { pageFromHash, PAGE_PATHS } from "./routes";
const ClassifierPage = lazy(() =>
  import("./pages/ClassifierPage").then((m) => ({ default: m.ClassifierPage })),
);
const ExplorerPage = lazy(() =>
  import("./pages/ExplorerPage").then((m) => ({ default: m.ExplorerPage })),
);
const AboutPage = lazy(() =>
  import("./pages/AboutPage").then((m) => ({ default: m.AboutPage })),
);
const pages = {
  overview: OverviewPage,
  classifier: ClassifierPage,
  datasets: ExplorerPage,
  about: AboutPage,
};
function currentPage(): Page {
  return pageFromHash(location.hash);
}
export default function App() {
  const reduced = useReducedMotion();
  const [activePage, setActivePage] = useState<Page>(currentPage);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { health, loading, error } = useHealth();
  useEffect(() => {
    const update = () => {
      const page = currentPage();
      if (location.hash !== PAGE_PATHS[page])
        history.replaceState(null, "", PAGE_PATHS[page]);
      setActivePage(page);
      window.scrollTo(0, 0);
      document.getElementById("main-content")?.focus({ preventScroll: true });
    };
    window.addEventListener("hashchange", update);
    update();
    return () => window.removeEventListener("hashchange", update);
  }, []);
  const ActivePage = pages[activePage];
  return (
    <MotionConfig reducedMotion="user">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main-content")?.focus();
        }}
      >
        Skip to content
      </a>
      <div className="app-shell">
        <TopBar
          activePage={activePage}
          health={health}
          healthLoading={loading}
          healthError={error}
          onMobileMenu={() => setMobileOpen(true)}
        />
        <Sidebar
          activePage={activePage}
          mobileOpen={mobileOpen}
          onMobileClose={() => setMobileOpen(false)}
        />
        <div className="main-content">
          <main className="page-body" id="main-content" tabIndex={-1}>
            <Suspense
              fallback={
                <div className="loading-line" role="status">
                  <span className="spinner" />
                  Loading workspace…
                </div>
              }
            >
              <motion.div
                key={activePage}
                initial={{ opacity: 0, y: reduced ? 0 : 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: reduced ? 0.12 : 0.28, ease: [0.22, 1, 0.36, 1] }}
              >
                <ActivePage />
              </motion.div>
            </Suspense>
          </main>
          <footer className="app-footer">
            <span>
              TrafficFlow <span className="footer-divider">/</span> CNN-based
              traffic classification
            </span>
            {activePage !== "classifier" && activePage !== "overview" && (
              <span lang="sa" className="cultural-signature" title="The world is one family">वसुधैव कुटुम्बकम्</span>
            )}
          </footer>
        </div>
      </div>
    </MotionConfig>
  );
}
