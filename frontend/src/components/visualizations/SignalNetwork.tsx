import { useCallback, useEffect, useLayoutEffect, useId, useRef, useState, type PointerEvent } from "react";
import { motion, useInView, useReducedMotion } from "motion/react";
import { ArrowUpRight, Play, RotateCcw, Layers3 } from "lucide-react";
import { advanceAngle, canAutoRotate, connectionPath, connectionPoint, graticule, project, surfacePoint } from "./networkGeometry";

const nodes = [
  { id: "UE", name: "User Equipment", full: "User Equipment", plane: "User plane + access signaling", role: "Endpoint", lon: -72, lat: -12, detail: "A handset or endpoint exchanges traffic with the gNodeB over the radio link. This illustration does not discover or monitor devices." },
  { id: "gNB", name: "gNodeB", full: "Next-generation Node B", plane: "User plane + control signaling", role: "Radio access network", lon: -38, lat: -32, detail: "The 5G base station connects user equipment to the core. N3 carries user traffic to the UPF; N2 carries control signaling to the AMF." },
  { id: "UPF", name: "User Plane Function", full: "User Plane Function", plane: "User plane", role: "5G core · Forwarding", lon: 22, lat: -15, detail: "The UPF forwards traffic between the radio access network over N3 and the data network over N6. The SMF controls forwarding over N4." },
  { id: "DN", name: "Data Network", full: "Data Network", plane: "User-plane destination", role: "External network", lon: 76, lat: -2, detail: "The destination network is reached through the UPF over N6. Placement on this logical globe has no geographical or operational meaning." },
  { id: "AMF", name: "Access and Mobility Management Function", full: "Access and Mobility Management Function", plane: "Control plane", role: "5G core · Access & mobility", lon: -32, lat: 28, detail: "The AMF handles access and mobility, with N2 signaling from the gNodeB, N11 coordination with the SMF, and N8 subscriber-data interactions with the UDM." },
  { id: "SMF", name: "Session Management Function", full: "Session Management Function", plane: "Control plane", role: "5G core · Session management", lon: 16, lat: 35, detail: "The SMF manages sessions, coordinates with the AMF over N11, controls the UPF over N4, and interacts with the UDM over N10. User traffic does not pass through it." },
  { id: "UDM", name: "Unified Data Management", full: "Unified Data Management", plane: "Control plane", role: "5G core · Subscriber information", lon: 58, lat: 53, detail: "The UDM manages subscriber information. N8 links its interactions with the AMF; N10 links its interactions with the SMF, separately from user-plane traffic." },
] as const;
type NodeId = (typeof nodes)[number]["id"];
const links: { from: NodeId; to: NodeId; label: string; control?: boolean }[] = [
  { from: "UE", to: "gNB", label: "Radio" },
  { from: "gNB", to: "UPF", label: "N3" },
  { from: "UPF", to: "DN", label: "N6" },
  { from: "gNB", to: "AMF", label: "N2", control: true },
  { from: "AMF", to: "SMF", label: "N11", control: true },
  { from: "SMF", to: "UPF", label: "N4", control: true },
  { from: "SMF", to: "UDM", label: "N10", control: true },
  { from: "AMF", to: "UDM", label: "N8", control: true },
];
const grid = [
  ...[-60, -40, -20, 0, 20, 40, 60].map(lat => ({ lat, lon: null as number | null })),
  ...Array.from({ length: 12 }, (_, i) => ({ lat: null as number | null, lon: i * 30 })),
];
const vectors = Object.fromEntries(nodes.map(n => [n.id, surfacePoint(n.lon, n.lat)]));

export function SignalNetwork() {
  const [selected, setSelected] = useState<NodeId>("UPF");
  const [controls, setControls] = useState(true);
  const [interacting, setInteracting] = useState(false);
  const [tracing, setTracing] = useState(false);
  const [reduceOverride, setReduceOverride] = useState(false);
  const [hidden, setHidden] = useState(document.hidden);
  const systemReduced = useReducedMotion();
  const reduced = !!systemReduced || reduceOverride;
  const svg = useRef<SVGSVGElement>(null);
  const visible = useInView(svg, { amount: 0.08 });
  const angle = useRef(0);
  const clock = useRef(0);
  const traceTime = useRef(-1);
  const frames = useRef(0);
  const drag = useRef<{ x: number; angle: number; pointer: number } | null>(null);
  const dragged = useRef(false);
  const gridPaths = useRef<(SVGPathElement | null)[]>([]);
  const edgePaths = useRef<(SVGPathElement | null)[]>([]);
  const edgeLabels = useRef<(SVGTextElement | null)[]>([]);
  const packets = useRef<(SVGCircleElement | null)[]>([]);
  const nodeGroups = useRef<(SVGGElement | null)[]>([]);
  const uid = useId().replaceAll(":", "");
  const node = nodes.find(n => n.id === selected)!;
  const autoRunning = canAutoRotate({ visible, hidden, reduced, paused: false, interacting });
  const sceneActive = visible && !hidden && !reduced;
  const rotationState = reduced ? "reduced" : interacting ? "interacting" : !visible || hidden ? "offscreen" : "rotating";

  // The SVG owns its animated attributes. React only renders on interaction;
  // this avoids reconciling the inspector and controls on every animation frame.
  const paint = useCallback(() => {
    if (!svg.current) return;
    svg.current.setAttribute("data-angle", angle.current.toFixed(3));
    svg.current.setAttribute("data-frame", String(++frames.current));
    grid.forEach((line, i) => gridPaths.current[i]?.setAttribute("d", graticule(line.lat, line.lon, angle.current)));
    nodes.forEach((n, i) => {
      const p = project(n.lon, n.lat, angle.current);
      const depthScale = 0.88 + (p.z + 1) * 0.06;
      const g = nodeGroups.current[i];
      if (!g) return;
      g.setAttribute("transform", `translate(${p.x.toFixed(3)} ${p.y.toFixed(3)}) scale(${depthScale.toFixed(3)})`);
      g.setAttribute("opacity", String(p.z >= 0 ? 1 : n.id === selected ? 0.78 : 0.4));
      g.style.display = controls || ["UE", "gNB", "UPF", "DN"].includes(n.id) ? "" : "none";
    });
    links.forEach((l, i) => {
      const a = vectors[l.from], b = vectors[l.to];
      const p = connectionPoint(a, b, 0.5, angle.current);
      const active = l.from === selected || l.to === selected;
      const show = controls || !l.control;
      const opacity = !show ? 0 : p.z < 0 ? active ? 0.5 : 0.17 : active ? 0.95 : 0.48;
      edgePaths.current[i]?.setAttribute("d", connectionPath(a, b, angle.current));
      edgePaths.current[i]?.setAttribute("opacity", String(opacity));
      const label = edgeLabels.current[i];
      if (label) {
        label.setAttribute("x", p.x.toFixed(3));
        label.setAttribute("y", (p.y - 12 - (i === 7 ? 12 : 0)).toFixed(3));
        label.setAttribute("opacity", String(!show ? 0 : Math.max(opacity, active ? 0.75 : 0.3)));
      }
      const packet = packets.current[i];
      if (!packet) return;
      const explicitTrace = traceTime.current >= 0 && !l.control;
      const progress = explicitTrace ? (traceTime.current - i * 1600) / 1800 : ((clock.current + i * 630) % 5200) / 2800;
      const packetVisible = show && !reduced && (explicitTrace || autoRunning && active) && progress >= 0 && progress <= 1;
      if (packetVisible) {
        const at = connectionPoint(a, b, progress, angle.current);
        packet.setAttribute("cx", at.x.toFixed(3));
        packet.setAttribute("cy", at.y.toFixed(3));
        packet.setAttribute("opacity", String((at.z < 0 ? 0.28 : 0.95) * Math.min(1, progress * 12, (1 - progress) * 12)));
      } else packet.setAttribute("opacity", "0");
    });
  }, [autoRunning, controls, reduced, selected]);

  useLayoutEffect(() => { paint(); }, [paint]);
  useEffect(() => {
    const handle = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", handle);
    return () => document.removeEventListener("visibilitychange", handle);
  }, []);
  useEffect(() => {
    if (!sceneActive || !autoRunning && !tracing) return;
    let raf = 0;
    let last: number | undefined;
    const frame = (now: number) => {
      const delta = last === undefined ? 0 : Math.max(0, now - last);
      last = now;
      if (autoRunning) {
        angle.current = advanceAngle(angle.current, delta, 1);
        clock.current += delta;
      }
      if (traceTime.current >= 0) {
        traceTime.current += delta;
        if (traceTime.current > 5900) { traceTime.current = -1; setTracing(false); }
      }
      paint();
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [autoRunning, paint, sceneActive, tracing]);
  useEffect(() => {
    if (reduced) { traceTime.current = -1; setTracing(false); }
  }, [reduced]);
  const selectNode = (id: NodeId) => { setSelected(id); };
  const startDrag = (e: PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, angle: angle.current, pointer: e.pointerId };
    dragged.current = false;
    setInteracting(true);
  };
  const moveDrag = (e: PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    if (!start) return;
    if (start.pointer !== e.pointerId) return;
    if (Math.abs(e.clientX - start.x) > 4) {
      dragged.current = true;
      if (!e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.setPointerCapture(e.pointerId);
      angle.current = ((start.angle + (e.clientX - start.x) * 0.3) % 360 + 360) % 360;
      paint();
    }
  };
  const endDrag = (e: PointerEvent<SVGSVGElement>) => {
    if (!drag.current) return;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    setInteracting(false);
  };
  const reset = () => {
    angle.current = 0; clock.current = 0; traceTime.current = -1;
    setInteracting(false); setControls(true); setSelected("UPF"); setTracing(false);
    paint();
  };

  return (
    <section className="signal-network" aria-label="Interactive illustrative 5G topology">
      <div className="network-heading"><span><i /> Spatial network</span><button className="reset-network" aria-label="Reset network view" onClick={reset}><RotateCcw size={16}/></button></div>
      <div className="network-canvas">
        <svg ref={svg} className="signal-globe" data-rotation-state={rotationState} viewBox="0 0 680 540" role="group" tabIndex={0} aria-label="Rotating logical 5G globe. Drag to orbit or use left and right arrow keys."
          onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={endDrag} onPointerCancel={endDrag}
          onPointerLeave={e => { if (!e.currentTarget.hasPointerCapture(e.pointerId)) endDrag(e); }}
          onKeyDown={e => { if (e.target === e.currentTarget && ["ArrowLeft", "ArrowRight"].includes(e.key)) { e.preventDefault(); angle.current += e.key === "ArrowLeft" ? -8 : 8; paint(); } }}>
          <desc>Logical architecture, not live monitoring. Rear nodes and links are faint; labels stay upright. All nodes remain available in the inspector selector.</desc>
          <defs>
            <radialGradient id={`${uid}-sphere`} cx="35%" cy="27%" r="74%"><stop offset="0" stopColor="#252421" /><stop offset=".55" stopColor="#121211" /><stop offset="1" stopColor="#050505" /></radialGradient>
            <radialGradient id={`${uid}-atmosphere`}><stop offset=".88" stopColor="#c6b9aa" stopOpacity="0"/><stop offset="1" stopColor="#c6b9aa" stopOpacity=".07"/></radialGradient>
            <marker id={`${uid}-arrow`} viewBox="0 0 10 10" refX="36" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 1L8 5L1 9" fill="none" stroke="#D16A52" strokeWidth="1.5" /></marker>
            <clipPath id={`${uid}-clip`}><ellipse cx="340" cy="268" rx="232" ry="218" transform="rotate(-12 340 268)" /></clipPath>
          </defs>
          <ellipse cx="340" cy="268" rx="242" ry="228" transform="rotate(-12 340 268)" fill={`url(#${uid}-atmosphere)`}/>
          <ellipse cx="340" cy="268" rx="232" ry="218" transform="rotate(-12 340 268)" fill={`url(#${uid}-sphere)`} stroke="#45443f" />
          <g clipPath={`url(#${uid}-clip)`} fill="none" stroke="#646057" strokeOpacity=".35" strokeWidth=".65">
            {grid.map((line, i) => <path key={i} ref={el => { gridPaths.current[i] = el; }} d={graticule(line.lat, line.lon, 0)} />)}
          </g>
          <ellipse cx="340" cy="273" rx="300" ry="95" transform="rotate(-24 340 273)" fill="none" stroke="#514c44" strokeOpacity=".55" strokeWidth=".8" />
          <text x="36" y="38" className="globe-note">5G / LOGICAL INFRASTRUCTURE</text>
          {links.map((l, i) => <g key={l.from + l.to}>
            <path ref={el => { edgePaths.current[i] = el; }} className="network-edge" data-from={l.from} data-to={l.to} d={connectionPath(vectors[l.from], vectors[l.to], 0)} fill="none" stroke={l.control ? "#aaa49a" : "#D16A52"} strokeWidth={l.from === selected || l.to === selected ? 2 : 1.2} strokeDasharray={l.control ? "3 5" : undefined} markerEnd={!l.control ? `url(#${uid}-arrow)` : undefined}/>
            <text ref={el => { edgeLabels.current[i] = el; }} className={`path-label ${l.control ? "" : "signal-label"}`} textAnchor="middle">{l.label}</text>
            <circle ref={el => { packets.current[i] = el; }} className="packet-indicator" r="3" fill={l.control ? "#c9c3b8" : "#efb39f"} opacity="0" pointerEvents="none"/>
          </g>)}
          {nodes.map((n, i) => {
            const active = selected === n.id;
            return <g ref={el => { nodeGroups.current[i] = el; }} key={n.id} data-node={n.id} role="button" tabIndex={0} aria-label={`Inspect ${n.id}: ${n.name}`} aria-pressed={active} className="globe-node"
              onClick={() => { if (!dragged.current) selectNode(n.id); }}
              onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectNode(n.id); } }}>
              <circle className="node-hit" r="34" fill="transparent"/>
              <motion.circle animate={{ r: active ? 24 : 19 }} whileHover={reduced ? undefined : { r: 23 }} transition={{ duration: reduced ? 0 : .2 }} fill="#101010" stroke={active ? "#D16A52" : "#aaa49a"} strokeWidth="1.3"/>
              <g transform="translate(-8.4 -8.4) scale(1.4)" fill="none" stroke={active ? "#E18A70" : "#e2e2e0"} strokeWidth=".9" aria-hidden="true">
                {n.id === "UE" ? <><rect x="3" y="0" width="6" height="12" rx="1"/><path d="M5 10h2"/></> : n.id === "gNB" ? <path d="M6 2v10M2 12h8M3 2Q6 -1 9 2M1 4Q6 -2 11 4M6 6l-3 6m3-6 3 6"/> : n.id === "UDM" ? <><ellipse cx="6" cy="2" rx="5" ry="2"/><path d="M1 2v8c0 3 10 3 10 0V2M1 6c0 3 10 3 10 0"/></> : <><rect x="0" y="1" width="12" height="10" rx="1"/><path d="M0 5h12M2 3h1m2 0h1M2 8h1m2 0h1"/></>}
              </g>
              {active && <motion.circle fill="none" stroke="#D16A52" animate={autoRunning ? { r: [29, 34, 29], opacity: [.12, .3, .12] } : { r: 30, opacity: .25 }} transition={autoRunning ? { duration: 3.5, repeat: Infinity, ease: "easeInOut" } : { duration: .15 }}/>} 
              <text y="46" textAnchor="middle" className={active ? "node-label selected" : "node-label"}>{n.id}</text>
            </g>;
          })}
        </svg>
        <div className="node-inspector" aria-live="polite">
          <div className="node-picker" aria-label="Explore network nodes">{nodes.map(n => <button key={n.id} aria-pressed={selected === n.id} onClick={() => selectNode(n.id)}>{n.id}</button>)}</div>
          <motion.div className="inspector-content" key={selected} initial={{ opacity: 0, y: reduced ? 0 : 4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: reduced ? .1 : .22 }}>
            <span className="inspector-label">Node inspector</span>
            <div className="inspector-id">{node.id}<ArrowUpRight size={22}/></div>
            <h3>{node.full}</h3><span className="node-role">{node.role}<span className="node-plane">{node.plane}</span></span><p>{node.detail}</p>
            <div className="node-connections"><span>Relevant interfaces</span>{links.filter(l => l.from === selected || l.to === selected).map(l => <button key={l.from + l.to} onClick={() => { setControls(true); selectNode(l.from === selected ? l.to : l.from); }}>{l.label}<ArrowUpRight size={12}/>{l.from === selected ? l.to : l.from}</button>)}</div>
          </motion.div>
        </div>
      </div>
      <div className="network-controls">
        <motion.button className="btn btn-secondary btn-sm" disabled={tracing || reduced} onClick={() => { traceTime.current = 0; setTracing(true); svg.current?.scrollIntoView({ behavior: "smooth", block: "center" }); }} whileHover={reduced ? undefined : { y: -1 }} whileTap={reduced ? undefined : { scale: .98 }}><Play size={13}/>{tracing ? "Tracing illustrative path…" : "Trace user plane"}</motion.button>
        <button className={`layer-toggle ${controls ? "enabled" : ""}`} aria-pressed={controls} onClick={() => { setControls(v => !v); if (controls) setSelected("UPF"); }}><Layers3 size={15}/>Control plane</button>
        <label className="motion-preference"><input type="checkbox" aria-label="Reduce globe motion" checked={reduced} disabled={!!systemReduced} onChange={e => setReduceOverride(e.target.checked)}/>Reduced motion</label>
      </div>
    </section>
  );
}

