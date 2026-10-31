# TrafficFlow — Black Signal

TrafficFlow has been recomposed around dark monochromatic surfaces, a custom interactive spatial network, horizontal desktop navigation, and a bottom-right Hindi signature. The existing API, models, preprocessing artifacts, and Python scripts are preserved.

## Four visual references studied

1. [VPN Service Website Design — Conceptzilla](https://dribbble.com/shots/21571426-VPN-Service-Website-Design): opened the actual artwork at an enlarged size. Applied the composition of expressive typography beside a detailed network visual, fine connection lines, and restrained interactive accents. The reference's green palette was not adopted.
2. [Future of AI Network — UX SAJIB](https://dribbble.com/shots/27073385-Future-of-AI-Network): inspected the enlarged artwork. Applied near-black foundations, grayscale depth, precise negative space, and a focal network object; no AI-head imagery was copied.
3. User-provided JT-GELFLOW screenshot: studied the globe, arcs, node labels, and adjacent inspector. Applied a rotatable logical globe and node-specific context, without fabricated geolocation or monitoring counters.
4. User-provided network-operations screenshot: studied connected infrastructure, grouped panels, and information hierarchy. Applied separate user/control planes and analytical workspaces; no CPU, bandwidth, alerts, or operational statistics were invented.

No reference screenshot is shipped as an application image or background. All network artwork is code-native SVG. The architecture includes the requested UE, gNB, AMF, SMF, UPF, UDM, and DN components.

## Identity and composition

Palette: #030303 canvas, #111111 surfaces, #191919 elevated fill, #E2E2E0 text, #9F9D9A secondary text, #696763 muted details, #817E78 supporting tones, #303030 borders, #D16A52 signal accent, and #E18A70 interactive highlight. Space Grotesk headings, Inter body, JetBrains Mono metadata, and Noto Sans Devanagari signature.

Overview pairs expressive type with the large spatial topology, followed by real API metrics, a separate inference pipeline, and a chart/table composition. Classification retains schema-driven forms, dataset selection, batch JSON/import/templates, validation, class scores, selection, and export. Evaluation now pairs horizontal class metrics with the exact-value report; its confusion matrix uses the same dark-to-signal scale. Dataset Explorer uses an inventory rail and configuration detail surface. About uses a split technical introduction and the CNN pipeline.

Desktop navigation is horizontal. Mobile retains an accessible menu with focus management and Escape handling. All canonical routes remain #/overview, #/classifier, #/evaluation, #/datasets, and #/about, with legacy aliases, refresh preservation, and native browser history.

The exact signature वसुधैव कुटुम्बकम् appears at the bottom-right of the footer on every page, following the latest refinement request.

## Interactive network

SignalNetwork.tsx draws its own orthographic globe, grid, directional paths, device/function glyphs, and selectable nodes. Orbit changes the projection; selecting a node updates its inspector and related interfaces. Control-plane visibility and reset are real controls. Trace user plane sends finite, sequential packet markers along UE → gNB → UPF → DN. The AMF/SMF/UDM control relationships are distinct from user traffic. Related interface buttons navigate between logical nodes.

Motion for React drives path reveals, packet movement, page fades, and the shared navigation indicator. Reduced-motion preferences disable packet travel and path-reveal movement. There is no continuous autonomous rotation or endless animation. SVG avoids a WebGL dependency and preserves accessible node selection.

The globe explicitly says Illustrative / not live and Logical placement, not geolocation. It represents a simplified logical architecture, not actual infrastructure or geographical observations. Actual analytics come only from the dataset evaluation endpoints. Confidence is a softmax score, not calibrated probability.

## Verification

- Strict TypeScript and Vite production build passed.
- Route regression suite: 5/5 passed.
- API suites: 71/71 passed (15 readiness, 8 validation, 48 cross-dataset contract assertions).
- Browser: all five desktop routes, active states, reload preservation, back/forward, and all five mobile routes passed.
- All five dataset-specific forms and real prediction requests passed (42 UNSW features; 23 for each ISCX configuration).
- Batch: empty array rejected; two-record inference and second-record selection passed. Existing export/import/template functionality retained.
- Evaluation switching verified UNSW 71.96% accuracy and ISCX 120s 57.48%, with the corresponding real confusion matrices.
- Node selection by click and keyboard, orbit rotation, layer toggle, reset, and actual CSS offset-path packet travel verified in the browser.
- Actual backend-offline state verified using an isolated frontend directed to unused local port 65534. The existing backend process was left running.
- Desktop captures at 1920px, 1440px, 1366px and mobile at 390px; all showed matching document client and scroll widths.
- Hindi signature text, Noto Sans Devanagari font, bottom-right footer placement, and bounds verified across all five mobile pages.
- All five compiled workspaces inspected; final compiled browser console had no warnings or errors.

Screenshots and structured evidence are in previews/. preview.png contains the final desktop overview. Prediction checks use synthetic features solely as test input, not captured traffic.

Backend main.py SHA-256 before and after: F74079FC35618B6E2DB5D7BD7F27075933A50BD7DEFECDC7FFA0E3F026A50D43.


## Readability and interaction refinement

The Black Signal identity and existing pages are retained. src/refinement.css defines the refinement separately from the established workspace composition. Body text is now 18px, desktop page headings 52px, navigation 17px, metric values 40–48px, chart labels 16px, and table contents 16–17px. Mobile type scales appropriately without shrinking data values into tiny metadata.

The globe now occupies a larger share of the hero, with a dedicated inspector below it. At 1440px the SVG is approximately 706px tall, compared with the previous 495px maximum. Nodes use larger device glyphs, 20–25-unit circles, larger hit regions, and 21px logical labels. Mobile enlarges the globe within its clipped orbital canvas while preserving node visibility. Control paths retain dashed styling; directional markers stop at node boundaries. UDM relationships now explicitly show N8 and N10, consistent with [ETSI TS 123 501, reference points](https://www.etsi.org/deliver/etsi_TS/123500_123599/123501/19.06.00_60/ts_123501v190600p.pdf).

Motion for React adds smooth node-panel entry, radius feedback on hover/selection, page entry, packet travel, trace-button feedback, and data-chart reveals. Packet traces remain finite and user-triggered. Reduced-motion preferences remove spatial movement. Charts keep their exact endpoint values throughout the transition.

Sections now have more separation. Overview chart and inventory stack below 1600px to preserve readable axes and values; evaluation similarly uses full-width sections at those widths. Tables retain independent scrolling on narrow screens. No API contract, route, dataset configuration, or model result changed.

Refinement verification: all seven node information panels, orbit/layer/reset controls, actual moving packet offsets, all five real dataset predictions, batch validation/inference/selection/export, evaluation switching, navigation, and footer signature placement were tested in the browser. Responsive bounds passed at 1920, 1440, 1366, 768, 390, 360, and 320px. The 71 API assertions and five route tests passed again. Final production build and compiled-browser evidence are recorded in previews/.
