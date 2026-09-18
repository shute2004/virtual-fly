#!/usr/bin/env python3
"""Replay pre-rendered Flyppy preview frames with synchronized telemetry."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from live_telemetry import LiveTelemetryPublisher
ROOT=Path(__file__).resolve().parents[2]

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--playback',type=Path,default=Path('artifacts/experiments/flyppy-best-playback/playback.json')); p.add_argument('--loop',action='store_true',default=True); p.add_argument('--start-hold-s',type=float,default=0.6); p.add_argument('--end-hold-s',type=float,default=1.0); return p.parse_args()
def absolute(p): return p if p.is_absolute() else (ROOT/p).resolve()
def atomic_frame(target:Path,payload:bytes):
    target.parent.mkdir(parents=True,exist_ok=True); tmp=target.with_name('.'+target.name+f'.{os.getpid()}.tmp'); tmp.write_bytes(payload); os.replace(tmp,target)

def main():
    a=parse_args(); path=absolute(a.playback); payload=json.loads(path.read_text()); frames=payload['frames']; fps=float(payload['playback_fps']); render=payload.get('render') or {}
    if not frames or fps<=0: raise SystemExit('invalid playback')
    pattern=str(render.get('frame_pattern','frames/frame-%05d.jpg')); root=path.parent; files=[root/(pattern%i) for i in range(len(frames))]
    missing=[str(p) for p in files if not p.exists()]
    if missing: raise SystemExit(f'missing rendered playback frames: {missing[0]}')
    encoded=[p.read_bytes() for p in files]
    live=root/'live'; current=live/'fly.jpg'; publisher=LiveTelemetryPublisher(root,enabled=True,on_demand=True)
    period=1.0/fps; loop_index=0
    try:
        print(f'playback_ready frames={len(frames)} fps={fps:.1f} duration_s={len(frames)/fps:.3f} passed_gates={payload.get("passed_gates")} checkpoint_step={payload.get("checkpoint_neural_step")} global_v={payload.get("global_weight_version")}',flush=True)
        while True:
            if a.start_hold_s>0: time.sleep(a.start_hold_s)
            deadline=time.perf_counter(); last_control_step=None
            for index,(frame,jpg) in enumerate(zip(frames,encoded)):
                deadline+=period; atomic_frame(current,jpg)
                control_step=int(frame['control_step'])
                telemetry_due=(control_step!=last_control_step) or bool(frame['passed_gate']) or bool(frame['collision'])
                if telemetry_due:
                    publisher.publish_body_state(episode=loop_index,control_step=control_step,sim_time_s=float(frame['sim_time_s']),qpos=frame['qpos'],qvel=frame['qvel'],next_gate=int(frame['next_gate']),passed_gate=bool(frame['passed_gate']),collision=bool(frame['collision']),collision_reason=frame.get('collision_reason'),reward=bool(frame['reward']),aversive=bool(frame['aversive']),motor=frame.get('motor') or {},retinal=frame.get('retinal') or {})
                    publisher.publish_neural(episode=loop_index,control_step=control_step,neural_step=int(frame['neural_step']),depolarizing_body_ids=frame.get('active_neural_body_ids') or [],hyperpolarizing_body_ids=(),reward=bool(frame['reward']),aversive=bool(frame['aversive']))
                    publisher.publish_status(running=True,backend='offline-physics-playback',episode=loop_index,control_step=control_step,curriculum={'mode':'offline-playback','environment_version':payload['environment_version'],'plasticity':False,'course_seed':payload['course_seed'],'passed_gates':payload['passed_gates'],'frame_index':index,'frame_count':len(frames)})
                    last_control_step=control_step
                remaining=deadline-time.perf_counter()
                if remaining>0: time.sleep(remaining)
                else: deadline=time.perf_counter()
            if a.end_hold_s>0: time.sleep(a.end_hold_s)
            loop_index+=1
            if not a.loop: break
    except KeyboardInterrupt: return 0
    finally:
        publisher.publish_status(running=False,backend='offline-physics-playback',episode=loop_index,control_step=None,curriculum={'mode':'offline-playback'}); publisher.close()
    return 0
if __name__=='__main__': raise SystemExit(main())
