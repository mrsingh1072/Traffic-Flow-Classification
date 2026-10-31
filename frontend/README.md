# TrafficFlow frontend

React 19, TypeScript, Vite, Tailwind CSS, Lucide, Recharts, Motion for React. This frontend uses the existing FastAPI service without changing its models, preprocessing, or endpoints.

## Run locally

From the repository root, start the API in one PowerShell terminal:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open http://127.0.0.1:5173. Configuration defaults to `VITE_API_BASE_URL=http://127.0.0.1:8000`. Restart Vite after changing environment variables. For another origin, configure the backend's existing `FRONTEND_ORIGINS` setting.

## Build and preview

```powershell
npm run test:routes
npm run build
npm run preview -- --host 127.0.0.1 --port 5173 --strictPort
```

Build output is in `dist/`. A preview on port 5173 matches the backend's default CORS allowlist. The frontend needs a reachable API; a static deployment does not host the Python backend. Production configuration should use an HTTPS API URL accessible to the user's browser.

Canonical routes: `#/overview`, `#/classifier`, `#/evaluation`, `#/datasets`, and `#/about`. Native links preserve refresh and browser back/forward behavior.

## Workspaces

- Overview: API-reported dataset availability, evaluated test sample totals, accuracy and macro F1, and navigation into the tools.
- Traffic Classifier: dataset-specific raw schema, single-record forms, batch JSON entry/import, schema-template download, record selection, class scores, and JSON result export. Batches accept 1–100 objects, matching the existing API.
- Model Evaluation: accuracy, macro precision/recall/F1, weighted F1, per-class table/chart, and keyboard-accessible confusion matrix with exact counts.
- Dataset Explorer: configured dataset inventory, independent checkpoint/preprocessor availability, raw feature counts and types, test samples, and evaluated class labels.
- About: research scope, architecture, limitations, and existing API reference.

Single-record blanks are sent as null and handled by the saved imputer. At least one non-null value is required. Batch objects must include all schema keys with finite numbers, categorical strings, or null. A downloaded template contains nulls and requires values before submission. Confidence is the backend's softmax score, not a calibrated probability.

All dataset counts and performance values come from the API. The application does not capture live network traffic. The network hero is explicitly illustrative.

## Visual and accessibility behavior

Black Signal design: near-black canvas, grayscale workspaces, restrained signal-red accents, Space Grotesk headings, Inter body, and JetBrains Mono technical values. Noto Sans Devanagari renders वसुधैव कुटुम्बकम् at the bottom-right of every page's footer.

Overview uses a custom interactive SVG logical globe with selectable UE/gNB/core/destination nodes, a contextual inspector, orbit control, control-plane visibility, reset, and a finite user-triggered packet trace. It is explicitly illustrative, not geographical observations or live monitoring. Motion for React handles controlled path reveals, packet movement, navigation, and page transitions; reduced-motion preferences disable spatial animation. No reference screenshots are embedded in the product.

All five workspaces share dark fields, tables, charts, and errors. Evaluation pairs a horizontal class chart with the exact-value report and a dark confusion matrix. Desktop navigation is horizontal; mobile has an accessible menu. Large headings adapt to smaller viewports; tables scroll within their own containers.

See [design references and decisions](DESIGN.md) and the screenshots in `previews/`.

## Architecture

`src/routes.ts` defines canonical hash routes and legacy aliases. `src/index.css` holds shared primitives; `src/workspace.css` implements Black Signal; `src/refinement.css` improves typography, spacing, visualization scale, and responsive readability. `src/services/api.ts` owns typed HTTP contracts. `src/components/` holds navigation, shared states, dataset selection, matrix, and interactive topology rendering. `src/pages/` contains the five workspaces. Non-overview pages and analytical charts load separately. TypeScript strict mode and unused-code checks run in every production build.

## Verification

With the API running, from the repository root:

```powershell
.\venv\Scripts\python.exe backend/test_backend_ready.py
.\venv\Scripts\python.exe backend/test_api_validation.py
.\venv\Scripts\python.exe frontend/scripts/verify_api.py
```

The frontend contract check uses synthetic zero/null fixtures solely for testing. It verifies all five schemas, single versus batch consistency, softmax normalization, evaluation/class/matrix consistency, CORS, and batch size limits. It does not present test fixtures as captured traffic.

Validation completed during implementation: 15 readiness checks, 8 API validation checks, and 48 cross-dataset contract checks. Five route regression tests also passed. Browser checks covered all five routes and refreshes, back/forward, all five dataset predictions, batch validation/inference/selection/export, evaluation switching, backend online/offline states, and mobile navigation. Responsive inspection covered 1920px, 1440px, 1366px, and 390px. See `previews/browser-checks.json`. Infrastructure deployment, authentication, and production security hardening remain separate work.


