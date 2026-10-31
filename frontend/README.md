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

Canonical routes: `#/overview`, `#/classifier`, and `#/about`. Native links preserve refresh and browser back/forward behavior. Retired `#/evaluation` bookmarks redirect to About.

## Workspaces

- Overview: API-reported dataset availability, evaluated test sample totals, accuracy and macro F1, and navigation into the tools.
- Traffic Classifier: dataset-specific raw schema, single-record forms, batch JSON entry/import, schema-template download, record selection, class scores, and JSON result export. Batches accept 1–100 objects, matching the existing API.
- About: TrafficCNN-1D architecture, the classification pipeline, held-out methodology, and API-backed accuracy and macro precision/recall/F1 for the five dataset configurations. Detailed per-class reports and confusion matrices remain available through the backend API.

## Presenting the classifier

1. Open **Traffic Classifier** and choose a dataset.
2. In **Single record**, select a demo record and choose **Load sample**.
3. Choose **Classify traffic**. The result displays the model prediction, class scores, and the record's known dataset label. Predictions can disagree with that label.
4. Expand **Review or edit features** only when you want to inspect or change the measured inputs. Editing removes the known-label comparison, since the edited record is no longer the original example.

For file uploads, switch to **Batch JSON**, choose **Download sample JSON**, and import the downloaded `<dataset>-sample-batch.json` with the same dataset selected. Then choose **Classify batch**. **Load sample batch** provides the same feature records without needing a file. A file is an array of raw feature objects, with all schema keys and no target labels or API-request wrapper. A blank template is for advanced use and cannot be submitted unchanged.

The example catalog in `src/data/classifier-examples.json` contains one record per class, selected with NumPy random seed 42 from each saved held-out test split. Raw feature values were matched back to the source CSV/ARFF and verified against the saved test arrays using the corresponding fitted preprocessor. Examples were not filtered by prediction correctness and are not a representative accuracy benchmark. Download the dataset-specific batch from the interface; the internal catalog itself is not an upload file.

Single-record blanks are sent as null and handled by the saved imputer. At least one non-null value is required. Batch objects must include all schema keys with finite numbers, categorical strings, or null. A downloaded template contains nulls and requires values before submission. Confidence is the backend's softmax score, not a calibrated probability.

All dataset counts and performance values come from the API. The application does not capture live network traffic. The network hero is explicitly illustrative.

## Visual and accessibility behavior

Black Signal design: near-black canvas, grayscale workspaces, restrained signal-red accents, Space Grotesk headings, Inter body, and JetBrains Mono technical values. Noto Sans Devanagari renders वसुधैव कुटुम्बकम् at the bottom-right of every page's footer.

Overview uses a custom interactive SVG logical globe with selectable UE/gNB/core/destination nodes, a contextual inspector, orbit control, control-plane visibility, reset, and a finite user-triggered packet trace. It is explicitly illustrative, not geographical observations or live monitoring. Motion for React handles controlled path reveals, packet movement, navigation, and page transitions; reduced-motion preferences disable spatial animation. No reference screenshots are embedded in the product.

All three workspaces support dark and light themes. About presents a concise evaluation summary from the backend API. Desktop navigation is horizontal; mobile has an accessible menu. Large headings adapt to smaller viewports; tables scroll within their own containers.

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


