import { test } from 'node:test';
import assert from 'node:assert/strict';
import { advanceAngle, canAutoRotate, connectionPath, connectionPoint, graticule, project, REVOLUTION_MS, surfacePoint } from '../src/components/visualizations/networkGeometry.ts';
const close = (a,b) => assert.ok(Math.abs(a-b)<0.00001, `${a} != ${b}`);

test('Natural rotation completes exactly one revolution in 25 seconds',()=>{
  assert.equal(REVOLUTION_MS,25000);
  close(advanceAngle(17,25000,1),17);
  close(advanceAngle(0,12500,1),180);
  close(advanceAngle(0,5000,1.25),90);
  close(advanceAngle(0,5000,.85),61.2);
});
test('Rotation is independent of frame subdivision and wraps through zero',()=>{
  let split=359;
  for(let i=0;i<100;i++)split=advanceAngle(split,10,1);
  close(split,advanceAngle(359,1000,1));
  assert.ok(split>=0&&split<360);
});
test('Hidden, offscreen, paused, reduced, and interacting scenes never auto-rotate',()=>{
  const active={visible:true,hidden:false,reduced:false,paused:false,interacting:false};
  assert.equal(canAutoRotate(active),true);
  for(const [key,value] of Object.entries({visible:false,hidden:true,reduced:true,paused:true,interacting:true}))
    assert.equal(canAutoRotate({...active,[key]:value}),false,key);
});
test('Links stay attached to the same projected node centers for a complete turn',()=>{
  const cases=[[-72,-12,-38,-32],[-38,-32,22,-15],[16,35,58,53]];
  for(const [alon,alat,blon,blat]of cases)for(let angle=0;angle<=360;angle+=15){
    const a=surfacePoint(alon,alat),b=surfacePoint(blon,blat);
    const start=connectionPoint(a,b,0,angle),end=connectionPoint(a,b,1,angle);
    const ap=project(alon,alat,angle),bp=project(blon,blat,angle);
    close(start.x,ap.x);close(start.y,ap.y);close(end.x,bp.x);close(end.y,bp.y);
    assert.ok(!connectionPath(a,b,angle).includes('NaN'));
  }
});
test('Packets follow finite spatial arcs, including coincident endpoint cases',()=>{
  const a=surfacePoint(-38,-32),b=surfacePoint(22,-15);
  for(let angle=0;angle<360;angle+=30)for(let i=0;i<=20;i++){
    const p=connectionPoint(a,b,i/20,angle);
    assert.ok([p.x,p.y,p.z].every(Number.isFinite));
    assert.ok(p.x>25&&p.x<655&&p.y>20&&p.y<520);
  }
  const same=connectionPoint(a,a,.5,0);assert.ok(Number.isFinite(same.x));
});
test('Graticules remain valid during a complete turn and depth changes continuously',()=>{
  for(let angle=0;angle<360;angle+=15)for(const [lat,lon]of [[0,null],[40,null],[null,90],[null,180]]){
    const d=graticule(lat,lon,angle);assert.ok(!/NaN|Infinity/.test(d));assert.ok(!d.trim() || d.trim().startsWith('M'));
  }
  const front=project(0,0,0),rear=project(0,0,180);assert.ok(front.z>0&&rear.z<0);
});

