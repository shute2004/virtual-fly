#!/usr/bin/env python3
"""Read-only P0 wall-clock profiler for the current Flyppy v3 hot loop."""
from __future__ import annotations

import argparse, json, math, statistics, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "scripts" / "embodiment"
sys.path.insert(0, str(EMB))

from virtual_fly.physics import FLYPPY_GEOMETRY_V3
from flybody_v3_adapter import FlyBodyV3NeuromuscularAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_telemetry import LiveTelemetryPublisher
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient, NeuralBridgeError
from whole_body_periphery import WholeBodyPeriphery


class Prof:
    def __init__(self): self.on=False; self.s=defaultdict(list)
    def call(self, key, fn, *a, **kw):
        if not self.on: return fn(*a, **kw)
        t=time.perf_counter()
        try: return fn(*a, **kw)
        finally: self.s[key].append(time.perf_counter()-t)
    def summary(self):
        out={}
        for k,v in sorted(self.s.items()):
            if not v: continue
            q=sorted(v); p95=q[min(len(q)-1, max(0, math.ceil(.95*len(q))-1))]
            out[k]={"calls":len(q),"total_s":sum(q),"mean_ms":statistics.fmean(q)*1e3,
                    "p50_ms":statistics.median(q)*1e3,"p95_ms":p95*1e3,"max_ms":q[-1]*1e3}
        return out


class Bridge(NeuralBridgeClient):
    def __init__(self,*a,prof,**kw):
        self.prof=prof; self.count=False; self.req=defaultdict(int); self.req_n=defaultdict(int); self.resp=0
        super().__init__(*a,**kw)
    def _read_response(self):
        line=self._stdout.readline()
        if not line: raise NeuralBridgeError(f"neural bridge exited unexpectedly (code={self._proc.poll()})")
        if self.count: self.resp += len(line.encode())
        r=json.loads(line)
        if not r.get("ok",False): raise NeuralBridgeError(str(r.get("error",r)))
        return r
    def _request(self,payload):
        if self._proc.poll() is not None: raise NeuralBridgeError(f"neural bridge is not running (code={self._proc.returncode})")
        typ=str(payload.get("type","unknown")); line=json.dumps(payload,separators=(",", ":"))+"\n"
        if self.count: self.req[typ]+=len(line.encode()); self.req_n[typ]+=1
        t=time.perf_counter() if self.count else None
        self._stdin.write(line); self._stdin.flush(); r=self._read_response()
        if t is not None: self.prof.s[f"bridge_request:{typ}"].append(time.perf_counter()-t)
        return r
    def protocol(self):
        return {"request_bytes_total":sum(self.req.values()),"response_bytes_total":self.resp,
                "request_bytes_by_type":dict(self.req),"request_count_by_type":dict(self.req_n)}


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--experiment",type=Path,default=Path("artifacts/experiments/flyppy-v3"))
    p.add_argument("--snapshot",type=Path,default=Path("artifacts/malecns-v1.0"))
    p.add_argument("--groups",type=Path,default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"))
    p.add_argument("--retinotopic-map",type=Path,default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"))
    p.add_argument("--wing-motor-map",type=Path,default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"))
    p.add_argument("--body-motor-map",type=Path,default=Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json"))
    p.add_argument("--viewer-graph",type=Path,default=Path("artifacts/embodiment/neural-viewer-graph-v1.json"))
    p.add_argument("--warmup",type=int,default=32); p.add_argument("--steps",type=int,default=256)
    p.add_argument("--telemetry",choices=("on","off"),default="on")
    p.add_argument("--report",type=Path,default=Path("reports/flyppy/profile_latest.md"))
    p.add_argument("--json",type=Path,default=Path("reports/flyppy/profile_latest.json"))
    return p.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def viewer_ids(path):
    if not path.exists(): return ()
    d=json.loads(path.read_text()); return tuple(sorted({int(n["body_id"]) for n in d.get("nodes",[])}))


def main():
    a=args()
    experiment=absolute(a.experiment); snapshot=absolute(a.snapshot); groups=absolute(a.groups)
    retinotopic_map=absolute(a.retinotopic_map); wing_motor_map=absolute(a.wing_motor_map)
    body_motor_map=absolute(a.body_motor_map); viewer_graph=absolute(a.viewer_graph)
    checkpoint=experiment/"checkpoint"
    required=[snapshot/"manifest.json", groups, retinotopic_map, wing_motor_map, body_motor_map, checkpoint/"manifest.json", experiment/"curriculum-state.json"]
    missing=[str(path) for path in required if not path.exists()]
    if missing: raise SystemExit("missing required profile inputs:\n  " + "\n  ".join(missing))
    st=json.loads((experiment/"curriculum-state.json").read_text())
    spawn=(float(st["spawn_x_mm"]),float(st["spawn_z_mm"]),float(st["initial_speed_mm_s"]))
    course=FlyppyCourse(seed=0,gate_count=6,environment_version="v3"); world=FlyppyWorld(course)
    body=FlyBodyV3NeuromuscularAdapter(tethered=False,world=world,
        spawn_position_mm=(0,0,FLYPPY_GEOMETRY_V3.corridor_high_z_mm/2),
        initial_linear_velocity_mm_s=(0,0,0),enable_vision=True,enable_observer_camera=False)
    per=WholeBodyPeriphery(wing_motor_map, body_motor_map)
    vis=MaleCNSRetina(retinotopic_map,current_gain=2.0)
    ids=viewer_ids(viewer_graph) if a.telemetry=="on" else ()
    pub=LiveTelemetryPublisher(ROOT/"artifacts/profiles/flyppy-v3-p0",enabled=a.telemetry=="on")
    prof=Prof(); orig_eye=vis._eye_readouts
    vis._eye_readouts=lambda sim,fly: prof.call("retina_eye_readout",orig_eye,sim,fly)
    measured=0; ep=0; step_ep=0; pairs=reads=reinf=resets=0

    def reset(brain):
        nonlocal ep,step_ep
        course.reset(); body.reset(); body.set_root_position_mm((spawn[0],0,spawn[1])); body.set_root_linear_velocity_mm_s((spawn[2],0,0))
        per.reset(); vis.reset_adaptation(); brain.reset_dynamics(); ep+=1; step_ep=0

    with Bridge(snapshot=snapshot,groups=groups,backend="gpu",repo_root=ROOT,prof=prof) as brain:
        brain.ping(); brain.load_checkpoint(checkpoint); reset(brain)
        warm_left=a.warmup; wall_start=None
        while measured<a.steps:
            if warm_left==0 and not prof.on: prof.on=True; brain.count=True; wall_start=time.perf_counter()
            active=prof.on; ts=time.perf_counter() if active else None
            nt=(step_ep%5)==0; bt=True
            read_body=tuple(dict.fromkeys((*per.body_ids,*ids))) if nt and ids else per.body_ids
            retinal=prof.call("retina_total",vis.encode,body.sim,body.fly)
            _,sp=prof.call("neural_control_total",brain.step_with_body_readout,stimulate_body=retinal.body_currents,read=(),read_body=read_body,plasticity=True)
            pv=prof.call("periphery",per.step,sp,dt_s=body.timestep*10)
            prof.call("physics",body.step_muscles,pv,physics_steps=10)
            pos=body.thorax_position_mm(); coll=prof.call("collision_query",world.physical_collision_reason,body.sim)
            ev=prof.call("course_update",course.update,float(pos[0]),float(pos[2]),physical_collision_reason=coll,analytic_body_collision=False)
            reward=aversive=False
            if ev.passed_gate:
                prof.call("reinforcement_total",brain.step,stimulate={"reward_dan":2.0},read=(),plasticity=True,steps=4); reward=True; reinf+=int(active)
            if ev.collision:
                prof.call("reinforcement_total",brain.step,stimulate={"aversive_dan":2.0},read=(),plasticity=True,steps=4); aversive=True; reinf+=int(active)
            if a.telemetry=="on":
                if nt or reward or aversive or ev.finished:
                    av=[i for i in ids if sp.get(i,False)]
                    prof.call("telemetry_neural",pub.publish_neural,episode=ep,control_step=step_ep,neural_step=brain.last_step,depolarizing_body_ids=av,hyperpolarizing_body_ids=(),reward=reward,aversive=aversive)
                if bt or reward or aversive or ev.finished:
                    prof.call("telemetry_body",pub.publish_body,episode=ep,control_step=step_ep,sim=body.sim,next_gate=getattr(course,"absolute_next_gate_index",course.next_gate_index),passed_gate=ev.passed_gate,collision=ev.collision,collision_reason=ev.collision_reason,reward=reward,aversive=aversive,motor=pv.compact_diagnostics(),retinal={"active_columns":retinal.active_columns,"active_photoreceptors":retinal.active_photoreceptors,"mean_current":retinal.mean_current,"max_current":retinal.max_current})
            if active:
                pairs+=len(retinal.body_currents); reads+=len(read_body); measured+=1; prof.s["control_step_total"].append(time.perf_counter()-ts)
            else: warm_left-=1
            step_ep+=1
            if ev.collision or ev.finished:
                if active: resets+=1
                prof.call("episode_reset_total",reset,brain)
        wall=time.perf_counter()-wall_start
        payload={"schema_version":1,"backend":brain.ready.get("backend"),"neurons":brain.ready.get("neurons"),"edges":brain.ready.get("edges"),
                 "warmup_control_steps":a.warmup,"measured_control_steps":measured,"measured_wall_seconds":wall,"control_steps_per_second":measured/wall,
                 "telemetry":a.telemetry,"spawn_x_mm":spawn[0],"spawn_z_mm":spawn[1],"initial_speed_mm_s":spawn[2],"mean_retinal_stimulus_pairs":pairs/measured,
                 "mean_motor_read_ids":reads/measured,"reinforcement_events":reinf,"episode_resets":resets,"timing":prof.summary(),"protocol":brain.protocol(),
                 "checkpoint":str(checkpoint),"checkpoint_modified":False,
                 "paths":{"snapshot":str(snapshot),"groups":str(groups),"retinotopic_map":str(retinotopic_map),"wing_motor_map":str(wing_motor_map),"body_motor_map":str(body_motor_map)}}
    json_path=absolute(a.json); report_path=absolute(a.report)
    json_path.parent.mkdir(parents=True,exist_ok=True); report_path.parent.mkdir(parents=True,exist_ok=True)
    json_path.write_text(json.dumps(payload,indent=2)+"\n")
    rows=sorted(payload["timing"].items(),key=lambda kv:kv[1]["total_s"],reverse=True)
    md=["# Flyppy P0 performance profile","",f"- backend: `{payload['backend']}`",f"- measured control steps: {measured}",f"- telemetry: `{a.telemetry}`",f"- wall-clock: {wall:.6f} s",f"- control steps/s: {measured/wall:.3f}",f"- neurons / edges: {payload['neurons']} / {payload['edges']}","","| stage | calls | total s | mean ms | p50 ms | p95 ms |","|---|---:|---:|---:|---:|---:|"]
    for k,v in rows: md.append(f"| {k} | {v['calls']} | {v['total_s']:.6f} | {v['mean_ms']:.3f} | {v['p50_ms']:.3f} | {v['p95_ms']:.3f} |")
    md += ["","## Protocol", "",f"- request bytes: {payload['protocol']['request_bytes_total']}",f"- response bytes: {payload['protocol']['response_bytes_total']}",f"- request count: `{json.dumps(payload['protocol']['request_count_by_type'],sort_keys=True)}`",f"- mean retinal stimulus pairs/step: {payload['mean_retinal_stimulus_pairs']:.1f}",f"- mean read_body IDs/step: {payload['mean_motor_read_ids']:.1f}","","`retina_total` includes `retina_eye_readout`. `bridge_request:step` overlaps conceptually with neural/reinforcement totals. This pass intentionally does not alter the training checkpoint or learning semantics. GPU kernel-level splitting is deferred until this wall-clock profile shows the neural bridge is dominant.",""]
    report_path.write_text("\n".join(md)); print(f"profile_report={report_path}"); print(f"profile_json={json_path}")
    return 0

if __name__=="__main__": raise SystemExit(main())
