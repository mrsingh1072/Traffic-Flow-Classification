// One projection is shared by the globe, nodes, links, labels, and packets.
export type Vector3 = { x: number; y: number; z: number };
export type ProjectedPoint = { x: number; y: number; z: number };
const rad = Math.PI / 180;
const tilt = -12 * rad;
export const REVOLUTION_MS = 25_000;

export function advanceAngle(angle: number, elapsedMs: number, speed: number) {
  return ((angle + elapsedMs * 360 * speed / REVOLUTION_MS) % 360 + 360) % 360;
}

export function surfacePoint(lon: number, lat: number): Vector3 {
  return {
    x: Math.cos(lat * rad) * Math.sin(lon * rad),
    y: Math.sin(lat * rad),
    z: Math.cos(lat * rad) * Math.cos(lon * rad),
  };
}

export function projectVector(v: Vector3, angle: number, lift = 1): ProjectedPoint {
  const a = angle * rad;
  const x = (v.x * Math.cos(a) + v.z * Math.sin(a)) * 232 * lift;
  const y = -v.y * 218 * lift;
  return {
    x: 340 + x * Math.cos(tilt) - y * Math.sin(tilt),
    y: 268 + x * Math.sin(tilt) + y * Math.cos(tilt),
    z: v.z * Math.cos(a) - v.x * Math.sin(a),
  };
}

export function project(lon: number, lat: number, angle: number) {
  return projectVector(surfacePoint(lon, lat), angle);
}

export function connectionPoint(a: Vector3, b: Vector3, t: number, angle: number) {
  const dot = Math.max(-1, Math.min(1, a.x * b.x + a.y * b.y + a.z * b.z));
  const omega = Math.acos(dot);
  const denominator = Math.sin(omega);
  const wa = denominator > 0.0001 ? Math.sin((1 - t) * omega) / denominator : 1 - t;
  const wb = denominator > 0.0001 ? Math.sin(t * omega) / denominator : t;
  return projectVector({ x: a.x * wa + b.x * wb, y: a.y * wa + b.y * wb, z: a.z * wa + b.z * wb }, angle, 1 + Math.sin(t * Math.PI) * 0.09);
}

export function connectionPath(a: Vector3, b: Vector3, angle: number) {
  return Array.from({ length: 25 }, (_, i) => {
    const p = connectionPoint(a, b, i / 24, angle);
    return `${i ? "L" : "M"}${p.x.toFixed(3)},${p.y.toFixed(3)}`;
  }).join(" ");
}

export function graticule(latitude: number | null, longitude: number | null, angle: number, front = true) {
  let drawing = false;
  return Array.from({ length: 65 }, (_, i) => {
    const p = project(longitude ?? -180 + i * 360 / 64, latitude ?? -90 + i * 180 / 64, angle);
    if ((p.z >= 0) !== front) { drawing = false; return ""; }
    const command = drawing ? "L" : "M";
    drawing = true;
    return `${command}${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  }).join(" ");
}

export function canAutoRotate(state: { visible: boolean; hidden: boolean; reduced: boolean; paused: boolean; interacting: boolean }) {
  return state.visible && !state.hidden && !state.reduced && !state.paused && !state.interacting;
}
