#!/usr/bin/env python3
"""Capture one frozen Flyppy episode for deterministic offline playback."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from flyppy_packed_body_worker import spawn_packed_body_processes
from population_neural_bridge_client import PopulationNeuralBridgeClient
from preview_flyppy_best import (
    DEFAULT_CALIBRATION, DEFAULT_COURSE_SEED, DEFAULT_PRODUCTION, DEFAULT_SNAPSHOT,
    DEFAULT_SPAWN_X_MM, DEFAULT_SPAWN_Z_MM, DEFAULT_SPEED_MM_S, DEFAULT_VIEWER_GRAPH,
    absolute, ensure_runtime_environment, load_json, load_viewer_ids,
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('artifacts/experiments/flyppy-best-playback/playback.json')

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--production',type=Path,default=DEFAULT_PRODUCTION); p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT); p.add_argument('--fresh-brain',action='store_true',help='use the MaleCNS snapshot initial weights without loading any learned checkpoint')
    p.add_argument('--snapshot',type=Path,default=DEFAULT_SNAPSHOT); p.add_argument('--viewer-graph',type=Path,default=DEFAULT_VIEWER_GRAPH); p.add_argument('--calibration',type=Path,default=DEFAULT_CALIBRATION)
    p.add_argument('--haltere-sensory-map',type=Path,default=Path('artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json')); p.add_argument('--haltere-current-gain',type=float,default=0.0); p.add_argument('--haltere-transduction',choices=('angular-acceleration-v1','interaction-load-v2'),default='angular-acceleration-v1')
    p.add_argument('--environment-version',choices=('v3','v4','v5','v6','v7'),default='v7'); p.add_argument('--flight-body-version',choices=('v3','v4','v5','v6','v7','v8'),default='v3'); p.add_argument('--vertical-steering-gain',type=float,default=1.0); p.add_argument('--measured-steering-gain',type=float,default=1.0); p.add_argument('--neutral-trim-strength',type=float,default=1.0); p.add_argument('--steering-tau-ms',type=float,default=12.0); p.add_argument('--steering-spike-increment',type=float,default=0.85); p.add_argument('--course-seed',type=int,default=DEFAULT_COURSE_SEED); p.add_argument('--gate-count',type=int,default=6)
    p.add_argument('--spawn-x-mm',type=float,default=DEFAULT_SPAWN_X_MM); p.add_argument('--spawn-z-mm',type=float,default=DEFAULT_SPAWN_Z_MM); p.add_argument('--initial-speed-mm-s',type=float,default=DEFAULT_SPEED_MM_S); p.add_argument('--initial-vz-mm-s',type=float,default=0.0); p.add_argument('--gate2-center-z-mm',type=float,default=None,help='override gate 2 opening center for curriculum/probe playback')
    p.add_argument('--physics-steps',type=int,default=10); p.add_argument('--physics-substep-stride',type=int,default=3); p.add_argument('--max-control-steps',type=int,default=1800)
    p.add_argument('--photoreceptor-current-gain',type=float,default=2.0); p.add_argument('--reward-current',type=float,default=2.0); p.add_argument('--aversive-current',type=float,default=2.0)
    p.add_argument('--reinforcement-steps',type=int,default=4); p.add_argument('--neural-telemetry-stride',type=int,default=5); p.add_argument('--playback-fps',type=float,default=60.0); p.add_argument('--timeout-s',type=float,default=120.0)
    p.add_argument('--plasticity',action='store_true',help='enable episode-local CNS plasticity without committing it')
    p.add_argument('--reward-from-gate',type=int,default=1,help='first 1-based gate pass that receives PAM when plasticity diagnostics are enabled')
    return p.parse_args()

def keep_substep(i,n,stride): return (i+1)%stride==0 or i+1==n

def main():
    a=parse_args(); production=absolute(a.production); output=absolute(a.output); snapshot=absolute(a.snapshot); viewer_graph=absolute(a.viewer_graph); calibration=absolute(a.calibration); haltere_map=absolute(a.haltere_sensory_map)
    if min(a.physics_steps,a.physics_substep_stride,a.neural_telemetry_stride)<=0 or a.playback_fps<=0: raise SystemExit('positive step/fps values required')
    output.parent.mkdir(parents=True,exist_ok=True); ensure_runtime_environment(calibration)
    viewer_ids=load_viewer_ids(viewer_graph)
    if a.fresh_brain:
        gv=0
    else:
        required=[production/'population-state.json',production/'checkpoint'/'manifest.json']
        missing=[str(p) for p in required if not p.exists()]
        if missing: raise SystemExit('missing learned preview inputs:\n  '+'\n  '.join(missing))
        gv=int(load_json(production/'population-state.json').get('global_weight_version',0))
    config={'seed':a.course_seed,'gate_count':a.gate_count,'environment_version':a.environment_version,'flight_body_version':a.flight_body_version,'vertical_steering_gain':a.vertical_steering_gain,'measured_steering_gain':a.measured_steering_gain,'neutral_trim_strength':a.neutral_trim_strength,'steering_tau_ms':a.steering_tau_ms,'steering_spike_increment':a.steering_spike_increment,'wing_motor_map':str(snapshot/'wing-motor-neurons-v0.json'),'body_motor_map':str(snapshot/'body-motor-neurons-v0.json'),'retinotopic_map':str(snapshot/'retinotopic-vision-v1.json'),'haltere_sensory_map':str(haltere_map),'photoreceptor_current_gain':a.photoreceptor_current_gain,'haltere_current_gain':a.haltere_current_gain,'haltere_transduction':a.haltere_transduction,'capture_physics_trace':True}
    workers=[]; frames=[]; passed=0; terminal_reason=None
    try:
        workers,slotmap=spawn_packed_body_processes(population=1,process_count=1,physics_steps=a.physics_steps,timeout_s=a.timeout_s,config=config); worker=slotmap[0]
        motor_ids=tuple(int(x) for x in worker.body_ids[0]); read_ids=tuple(sorted(set(motor_ids)|set(viewer_ids)))
        with PopulationNeuralBridgeClient(snapshot=snapshot,groups=snapshot/'embodiment-groups-v0.json',slots=1) as brain:
            brain.ping()
            if a.fresh_brain:
                loaded={'step':0}
                if int(brain.global_weight_version)!=0:
                    raise RuntimeError(f'fresh MaleCNS bridge unexpectedly started at global weight version {brain.global_weight_version}')
            else:
                loaded=brain.load_checkpoint(production/'checkpoint',global_weight_version=gv)
            brain.restart_slot(0)
            reset_payload={'episode':0,'source_weight_version':brain.global_weight_version,'spawn_x_mm':a.spawn_x_mm,'spawn_z_mm':a.spawn_z_mm,'initial_speed_mm_s':a.initial_speed_mm_s,'initial_vz_mm_s':a.initial_vz_mm_s}
            if a.gate2_center_z_mm is not None: reset_payload['gate_center_overrides']={1:float(a.gate2_center_z_mm)}
            worker.request('reset',{0:reset_payload}); worker.receive('reset')
            initial=worker.call('snapshot',(0,))[0]
            frames.append({'control_step':0,'physics_substep':0,'sim_time_s':float(initial['sim_time_s']),'qpos':initial['qpos'],'qvel':initial['qvel'],'next_gate':0,'passed_gate':False,'collision':False,'collision_reason':None,'reward':False,'aversive':False,'neural_step':int(brain.last_step),'active_neural_body_ids':[],'active_motor_body_ids':[],'motor':{},'retinal':{}})
            control=0; terminal=False; last_active=[]
            while not terminal and control<a.max_control_steps:
                obs=worker.call('observe',(0,))[0]; neural_sample=control%a.neural_telemetry_stride==0
                batch=brain.step_batch([{'slot':0,'stimulate_body':obs['body_currents'],'read_body':read_ids if neural_sample else motor_ids}],plasticity=bool(a.plasticity)); spikes=batch.get(0,{})
                if neural_sample: last_active=[bid for bid in viewer_ids if bool(spikes.get(bid,False))]
                active_motor=tuple(bid for bid in motor_ids if bool(spikes.get(bid,False))); act=worker.call('act',{0:active_motor})[0]; control+=1
                reward=bool(act['passed_gate']); aversive=bool(act['collision'])
                if reward:
                    passed+=1
                    if passed >= int(a.reward_from_gate):
                        brain.step_slot(0,stimulate={'reward_dan':a.reward_current},plasticity=bool(a.plasticity),steps=a.reinforcement_steps)
                if aversive:
                    brain.step_slot(0,stimulate={'aversive_dan':a.aversive_current},plasticity=bool(a.plasticity),steps=a.reinforcement_steps)
                    terminal_reason=str(act.get('collision_reason') or 'collision')
                trace=list(act.get('physics_trace') or []); n=len(trace)
                if not trace: raise RuntimeError('missing physics_trace')
                for i,pose in enumerate(trace):
                    if not keep_substep(i,n,a.physics_substep_stride): continue
                    final=i+1==n
                    frames.append({'control_step':control,'physics_substep':i+1,'sim_time_s':float(pose['sim_time_s']),'qpos':pose['qpos'],'qvel':pose['qvel'],'next_gate':int(act['next_gate'] if final else max(0,int(act['next_gate'])-(1 if reward else 0))),'passed_gate':bool(reward and final),'collision':bool(aversive and final),'collision_reason':act.get('collision_reason') if aversive and final else None,'reward':bool(reward and final),'aversive':bool(aversive and final),'neural_step':int(brain.last_step),'active_neural_body_ids':last_active,'active_motor_body_ids':list(active_motor) if final else [],'motor':act.get('motor') or {},'physics':pose.get('diagnostics') or {},'retinal':{'active_photoreceptors':int(obs['active_photoreceptors']),'active_columns':int(obs['active_columns']),'mean_current':float(obs['mean_current']),'max_current':float(obs['max_current'])}})
                terminal=bool(act['collision'] or act['finished'])
            payload={'schema_version':1,'kind':'flyppy-offline-preview-playback','brain_state':'fresh-snapshot' if a.fresh_brain else 'learned-checkpoint','source_experiment':None if a.fresh_brain else str(production),'checkpoint_neural_step':int(loaded.get('step',0 if a.fresh_brain else -1)),'global_weight_version':int(brain.global_weight_version),'environment_version':a.environment_version,'flight_body_version':a.flight_body_version,'vertical_steering_gain':a.vertical_steering_gain,'neutral_trim_strength':a.neutral_trim_strength,'steering_tau_ms':a.steering_tau_ms,'steering_spike_increment':a.steering_spike_increment,'haltere_current_gain':a.haltere_current_gain,'haltere_transduction':a.haltere_transduction,'course_seed':a.course_seed,'gate_count':a.gate_count,'gate2_center_z_mm':a.gate2_center_z_mm,'spawn_x_mm':a.spawn_x_mm,'spawn_z_mm':a.spawn_z_mm,'initial_speed_mm_s':a.initial_speed_mm_s,'initial_vz_mm_s':a.initial_vz_mm_s,'physics_steps_per_control':a.physics_steps,'physics_substep_stride':a.physics_substep_stride,'playback_fps':a.playback_fps,'passed_gates':passed,'terminal_reason':terminal_reason,'control_steps':control,'plasticity':bool(a.plasticity),'reward_from_gate':int(a.reward_from_gate),'frame_count':len(frames),'frames':frames}
            tmp=output.with_name('.'+output.name+'.tmp'); tmp.write_text(json.dumps(payload,separators=(',',':'))+'\n'); os.replace(tmp,output)
            print(f"playback_capture=PASS checkpoint_step={payload['checkpoint_neural_step']} global_v={payload['global_weight_version']} frames={len(frames)} controls={control} passed_gates={passed} terminal={terminal_reason}"); print(f'playback_json={output}')
    finally:
        for w in workers:
            try:w.close()
            except Exception:pass
    return 0
if __name__=='__main__': raise SystemExit(main())
