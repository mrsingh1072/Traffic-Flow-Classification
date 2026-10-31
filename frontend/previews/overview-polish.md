# Overview polish and rotating globe — 31 October 2026

Only OverviewPage.tsx, its OverviewChart, SignalNetwork, the new networkGeometry module, Overview-specific CSS, and the related tests/review artifacts changed. The other four pages, application shell, navigation, API client, shared pipeline, existing shared CSS, and backend main.py match their pre-task SHA-256 hashes. See overview-scope-check.json.

The approved Black Signal composition, headline, CTAs, actual API metrics, dataset values, and bottom-right Devanagari signature remain intact. Overview-only spacing, metric typography, pipeline reveal/hover direction, chart axes/legend, and inventory row feedback were polished.

## Globe implementation

The existing SVG implementation now uses a tilted orthographic 3D projection with depth-aware nodes and great-circle connection arcs. Node centers, path endpoints, interface labels, and packet positions share that projection. Labels remain upright, and rear nodes/links appear faintly. A persistent inspector selector keeps every node accessible when it is on the rear hemisphere.

Default rotation is 25 seconds per revolution. The speed slider varies it between approximately 20 and 29 seconds. Pause/resume and reset are explicit controls. Dragging, arrow-key orbit, and selection temporarily hold automatic rotation; it resumes after 3.2 seconds of inactivity unless explicitly paused. Pointer hover movement also holds rotation while the user explores the globe.

Selected connections carry restrained packet indicators; the explicit user-plane trace follows UE → gNB → UPF → DN. User-plane paths are solid and directional; control-plane paths are dashed. The selected node has a subtle pulse. Full forms, roles, plane classification, and relevant interfaces are shown for all seven nodes.

An IntersectionObserver and document visibility handler gate the frame loop. Moving SVG attributes does not rerender the inspector or controls every frame. No Three.js, WebGL, or new package dependencies were introduced. Reduced motion follows the system preference and can also be enabled explicitly in the visualization. It stops automatic rotation, packets, and pulsing while preserving inspection and manual orbit.

## Verification

- Browser-measured natural speed: 14.3879 degrees/second, approximately 25.02 seconds per turn.
- Pause held the angle unchanged. The slider changed the displayed turn time from 25 to 20 seconds; reset restored the defaults.
- A pointer drag changed the orbit by 42 degrees. Keyboard orbit worked, and automatic rotation resumed after inactivity.
- All eight rendered connection endpoints matched their node centers exactly in a live DOM measurement.
- Offscreen angle and frame count remained unchanged across separate observations.
- Reduced-motion mode held the angle still, hid all packets, disabled automatic motion controls, and preserved all seven inspectors. The automatic gating policy is also tested directly for reduced-motion and hidden-document states.
- Inspector checks passed for UE, gNB, AMF, SMF, UDM, UPF, and DN, including full names, planes, and interfaces.
- Actual API inventory and evaluation data rendered in the compiled Overview. Routes/API contracts were not edited.
- Responsive document bounds passed at 1920, 1440, 1366, 768, 390, 360, and 320 pixels. Tables retain their own horizontal scroll container on mobile.
- The exact Hindi signature remains in the footer at the bottom-right.
- Six geometry/motion tests and five existing route regression tests passed. Strict TypeScript and Vite production build passed.
- The compiled page was inspected in the browser, its angle advanced, and its console had no warnings or errors.

Run the related tests from frontend:

    node --experimental-strip-types --test scripts/overview-globe.test.mjs scripts/routes.test.mjs

Screenshots: overview-1920.png, overview-1440.png, overview-1366.png, overview-mobile.png, and overview-detail.png. Full desktop capture: ../preview.png. Full-page captures use a paused pose for inspection; the running page starts rotating by default.
