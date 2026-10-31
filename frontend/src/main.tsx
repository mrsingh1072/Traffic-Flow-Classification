import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import "./workspace.css";
import "./refinement.css";
import App from "./App";
import { ThemeProvider } from "./components/ThemeToggle";
import "./theme.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider><App /></ThemeProvider>
  </StrictMode>,
);
