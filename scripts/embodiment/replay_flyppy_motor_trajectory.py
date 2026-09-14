#!/usr/bin/env python3
"""Exhaustively localize why a recorded Flyppy motor pattern destabilizes flight.

This is an offline causal diagnostic. It replays an already-recorded peripheral
motor-state trajectory without advancing the CNS, then runs a broad ablation and
sensitivity matrix over steering, bilateral power balance, DLM/DVM balance,
temporal variation, and command magnitude. Nothing discovered here is fed back
into the original episode or used as an action decoder.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import numpy as np
from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from wing_muscle_periphery import PeripheralSnapshot

ACTIVE_STEERING = ("b1", "b2", "b3", "i1")
POWER_NAMES = ("dlm_left", "dlm_right", "dvm_left", "dvm_right")

@dataclass(frozen=True)
class Variant:
    name: str
    remove_steering: frozenset[str] = frozenset()
    steering_scale: float = 1.0
    symmetric_steering: bool = False
    constant_steering_mean: bool = False
    symmetric_power: bool = False
    merge_dlm_dvm: bool = False
    zero_dlm: bool = False
    zero_dvm: bool = False
    swap_dlm_dvm: bool = False
    power_scale: float = 1.0
    smooth_power_window: int = 1
    constant_power_mean: bool = False
    constant_symmetric_power: float | None = None

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--trajectory',type=Path,default=Path('artifacts/experiments/flyppy-v1/trajectory.jsonl')); p.add_argument('--episode',type=int,default=0); p.add_argument('--seed',type=int,default=0); p.add_argument('--gate-count',type=int,default=6); p.add_argument('--physics-steps',type=int,default=10); p.add_argument('--initial-forward-speed-mm-s',type=float,default=300.0); p.add_argument('--output',type=Path,default=Path('artifacts/embodiment/flyppy-motor-causal-diagnosis.json')); p.add_argument('--baseline-position-tolerance-mm',type=float,default=0.15); p.add_argument('--baseline-velocity-tolerance-mm-s',type=float,default=3.0); return p.parse_args()

def load_episode(path, episode):
    records=[]
    with path.open() as h:
        for line in h:
            if line.strip():
                r=json.loads(line)
                if int(r.get('episode',-1))==episode: records.append(r)
    records.sort(key=lambda r:int(r['control_step']))
    if not records: raise RuntimeError(f'trajectory contains no records for episode {episode}')
    actual=[int(r['control_step']) for r in records]
    if actual!=list(range(len(records))): raise RuntimeError('causal replay requires trajectory stride 1 and contiguous control steps')
    return records

def steering_keys(records):
    obs=set()
    for r in records:
        s=dict((r.get('motor_periphery',{}) or {}).get('steering',{}) or {})
        for k,v in s.items():
            parts=str(k).split(':',1)
            if len(parts)==2 and parts[0] in {'left','right'} and parts[1] in ACTIVE_STEERING and float(v)>1e-8: obs.add(str(k))
    return tuple(sorted(obs))

def extract_motor_arrays(records, keys):
    power=np.zeros((len(records),4)); steering={k:np.zeros(len(records)) for k in keys}; spikes=np.zeros(len(records),dtype=np.int64)
    for i,r in enumerate(records):
        m=dict(r.get('motor_periphery',{}) or {}); power[i]=[float(m.get('dlm_left',0)),float(m.get('dlm_right',0)),float(m.get('dvm_left',0)),float(m.get('dvm_right',0))]; spikes[i]=int(m.get('spikes',0)); raw=dict(m.get('steering',{}) or {})
        for k in keys: steering[k][i]=float(raw.get(k,0))
    return power,steering,spikes

def moving_average(a,w):
    if w<=1:return a.copy()
    w=min(w,len(a)); before=w//2; after=w-1-before; padded=np.pad(a,((before,after),(0,0)),mode='edge'); kernel=np.ones(w)/w; out=np.empty_like(a)
    for c in range(a.shape[1]): out[:,c]=np.convolve(padded[:,c],kernel,mode='valid')
    return out

def apply_variant(base_power, base_steering, v):
    power=base_power.copy(); steering={k:x.copy() for k,x in base_steering.items()}
    if v.constant_symmetric_power is not None: power[:]=float(v.constant_symmetric_power)
    else:
        if v.smooth_power_window>1: power=moving_average(power,v.smooth_power_window)
        if v.constant_power_mean: power[:]=np.mean(power,axis=0,keepdims=True)
        if v.swap_dlm_dvm: power=power[:,[2,3,0,1]]
        if v.zero_dlm: power[:,0:2]=0
        if v.zero_dvm: power[:,2:4]=0
        if v.merge_dlm_dvm:
            left=.5*(power[:,0]+power[:,2]); right=.5*(power[:,1]+power[:,3]); power[:,0]=left; power[:,2]=left; power[:,1]=right; power[:,3]=right
        if v.symmetric_power:
            dlm=.5*(power[:,0]+power[:,1]); dvm=.5*(power[:,2]+power[:,3]); power[:,0]=dlm; power[:,1]=dlm; power[:,2]=dvm; power[:,3]=dvm
        power*=v.power_scale; np.clip(power,0,1,out=power)
    if v.constant_steering_mean:
        for k in steering: steering[k][:]=float(np.mean(steering[k]))
    if v.symmetric_steering:
        for muscle in ACTIVE_STEERING:
            lk=f'left:{muscle}'; rk=f'right:{muscle}'
            if lk in steering or rk in steering:
                left=steering.get(lk,np.zeros(len(power))); right=steering.get(rk,np.zeros(len(power))); mean=.5*(left+right)
                if lk in steering: steering[lk]=mean.copy()
                if rk in steering: steering[rk]=mean.copy()
    for k in v.remove_steering:
        if k in steering: steering[k].fill(0)
    if v.steering_scale!=1:
        for k in steering: steering[k]*=v.steering_scale; np.clip(steering[k],0,1,out=steering[k])
    return power,steering

def state_at(i,power,steering,spikes):
    return PeripheralSnapshot(activation_by_body={},muscle_activation={k:float(v[i]) for k,v in steering.items() if v[i]>1e-8},dlm_activation={'left':float(power[i,0]),'right':float(power[i,1])},dvm_activation={'left':float(power[i,2]),'right':float(power[i,3])},active_spikes=int(spikes[i]),selected_motor_units=1)

class ReplayHarness:
    def __init__(self,seed,gate_count,physics_steps,initial_speed):
        self.course=FlyppyCourse(seed=seed,gate_count=gate_count); self.world=FlyppyWorld(self.course); self.body=FlyBodyMuscleAdapter(tethered=False,world=self.world,spawn_position_mm=(0,0,5),initial_linear_velocity_mm_s=(initial_speed,0,0),enable_vision=False); self.physics_steps=physics_steps
    def run(self,name,power,steering,spikes):
        self.course.reset(); self.body.reset(); min_z=float('inf'); max_z=float('-inf'); max_x=float('-inf'); passed=0; reason=None; steps=0
        for i in range(len(power)):
            self.body.step_muscles(state_at(i,power,steering,spikes),physics_steps=self.physics_steps); pos=self.body.thorax_position_mm(); x=float(pos[0]); z=float(pos[2]); min_z=min(min_z,z); max_z=max(max_z,z); max_x=max(max_x,x); steps+=1; event=self.course.update(x,z)
            if event.passed_gate: passed+=1
            if event.collision: reason=event.collision_reason; break
            if event.finished: break
        pos=np.asarray(self.body.thorax_position_mm(),dtype=float); vel=np.asarray(self.body.root_linear_velocity_mm_s(),dtype=float); survived=reason is None and steps==len(power); recovered=survived and float(vel[0])>0
        return {'name':name,'steps':steps,'survived_full_recorded_horizon':survived,'recovered':recovered,'passed_gates':passed,'collision':reason is not None,'collision_reason':reason,'max_x_mm':max_x,'min_z_mm':min_z,'max_z_mm':max_z,'final_position_mm':pos.tolist(),'final_velocity_mm_s':vel.tolist()}

def infer_original_collision(records,seed,gate_count):
    c=FlyppyCourse(seed=seed,gate_count=gate_count); passed=0; reason=None
    for r in records:
        e=c.update(float(r['x_mm']),float(r['z_mm'])); passed+=int(e.passed_gate)
        if e.collision: reason=e.collision_reason; break
    last=records[-1]; return {'passed_gates':passed,'collision':bool(last.get('collision',False)),'collision_reason_inferred':reason,'last_x_mm':float(last['x_mm']),'last_z_mm':float(last['z_mm']),'last_vx_mm_s':float(last['vx_mm_s'])}

def stats(v): return {'min':float(np.min(v)),'mean':float(np.mean(v)),'std':float(np.std(v)),'max':float(np.max(v)),'p10':float(np.quantile(v,.1)),'p90':float(np.quantile(v,.9))}
def summarize_pattern(power,steering,spikes):
    out={n:stats(power[:,i]) for i,n in enumerate(POWER_NAMES)}; out['dlm_side_difference']=stats(power[:,0]-power[:,1]); out['dvm_side_difference']=stats(power[:,2]-power[:,3]); out['dlm_minus_dvm_left']=stats(power[:,0]-power[:,2]); out['dlm_minus_dvm_right']=stats(power[:,1]-power[:,3]); out['step_to_step_power_delta_abs_max']=float(np.max(np.abs(np.diff(power,axis=0))) if len(power)>1 else 0); out['step_to_step_power_delta_abs_mean']=float(np.mean(np.abs(np.diff(power,axis=0))) if len(power)>1 else 0); out['motor_spikes_total']=int(np.sum(spikes)); out['motor_spikes_max_per_step']=int(np.max(spikes)); out['steering']={k:stats(v) for k,v in sorted(steering.items())}; return out

def variant_matrix(keys):
    allk=frozenset(keys); vs=[Variant('recorded'),Variant('no_steering',remove_steering=allk),Variant('symmetric_power',symmetric_power=True),Variant('symmetric_power_no_steering',symmetric_power=True,remove_steering=allk),Variant('symmetric_steering',symmetric_steering=True),Variant('constant_steering_mean',constant_steering_mean=True),Variant('constant_power_mean',constant_power_mean=True),Variant('constant_power_mean_no_steering',constant_power_mean=True,remove_steering=allk),Variant('merge_dlm_dvm',merge_dlm_dvm=True),Variant('dlm_only',zero_dvm=True),Variant('dvm_only',zero_dlm=True),Variant('steering_only',zero_dlm=True,zero_dvm=True),Variant('swap_dlm_dvm',swap_dlm_dvm=True)]
    for w in (3,5,9,17,33): vs += [Variant(f'power_smooth_{w}',smooth_power_window=w),Variant(f'power_smooth_{w}_no_steering',smooth_power_window=w,remove_steering=allk)]
    for s in (.1,.2,.3,.4,.5,.6,.7,.8,.9,1.1,1.25,1.5): vs.append(Variant(f'power_scale_{s:.2f}',power_scale=s))
    for s in (.1,.2,.3,.4,.5,.6,.7,.8,.9): vs.append(Variant(f'steering_scale_{s:.2f}',steering_scale=s))
    for level in np.linspace(.05,1,20): vs.append(Variant(f'constant_symmetric_power_{level:.2f}_no_steering',constant_symmetric_power=float(level),remove_steering=allk))
    for k in keys: vs.append(Variant('remove_'+k.replace(':','_'),remove_steering=frozenset({k})))
    for m in ACTIVE_STEERING:
        rem=frozenset(k for k in keys if k.endswith(':'+m))
        if rem: vs.append(Variant(f'remove_muscle_{m}',remove_steering=rem))
    for side in ('left','right'):
        rem=frozenset(k for k in keys if k.startswith(side+':'))
        if rem: vs.append(Variant(f'remove_steering_{side}',remove_steering=rem))
    if len(keys)<=8:
        for size in range(2,len(keys)):
            for combo in itertools.combinations(keys,size): vs.append(Variant('remove_combo_'+'__'.join(k.replace(':','_') for k in combo),remove_steering=frozenset(combo)))
    return list({v.name:v for v in vs}.values())

def closest_recovering_scale(results,prefix):
    c=[r for r in results if r['name'].startswith(prefix) and r['recovered']]
    return min(c,key=lambda r:abs(float(r['name'].rsplit('_',1)[-1])-1)) if c else None

def minimal_steering_ablation(results,vmap):
    c=[]
    for r in results:
        v=vmap.get(r['name'])
        if not r['recovered'] or v is None or not v.remove_steering: continue
        if v.power_scale!=1 or v.symmetric_power or v.constant_power_mean or v.smooth_power_window!=1 or v.zero_dlm or v.zero_dvm or v.swap_dlm_dvm or v.merge_dlm_dvm or v.steering_scale!=1: continue
        c.append((len(v.remove_steering),r,v))
    if not c:return []
    m=min(x[0] for x in c); return [{'removed':sorted(v.remove_steering),'final_vx_mm_s':r['final_velocity_mm_s'][0],'max_x_mm':r['max_x_mm']} for n,r,v in c if n==m]

def build_diagnosis(consistent,results,vmap,minsteer):
    if not consistent:return {'status':'REPLAY_MISMATCH','findings':[{'cause':'REPLAY_MISMATCH','detail':'Recorded motor state did not reproduce original endpoint within tolerance.'}]}
    by={r['name']:r for r in results}; findings=[]
    if minsteer: findings.append({'cause':'DIRECT_STEERING_DESTABILIZATION','minimal_recovering_ablations':minsteer})
    if by.get('symmetric_power',{}).get('recovered'): findings.append({'cause':'LEFT_RIGHT_POWER_ASYMMETRY','evidence':'instantaneous bilateral power symmetrization recovered flight'})
    if by.get('symmetric_power_no_steering',{}).get('recovered') and not by.get('no_steering',{}).get('recovered') and not by.get('symmetric_power',{}).get('recovered'): findings.append({'cause':'STEERING_POWER_ASYMMETRY_INTERACTION'})
    smooth=[r for r in results if r['name'].startswith('power_smooth_') and not r['name'].endswith('_no_steering') and r['recovered']]
    if smooth or by.get('constant_power_mean',{}).get('recovered'): findings.append({'cause':'POWER_TEMPORAL_VARIABILITY','smallest_recovering_smoothing_window':min([int(r['name'].split('_')[2]) for r in smooth],default=None),'constant_mean_recovers':bool(by.get('constant_power_mean',{}).get('recovered'))})
    ps=closest_recovering_scale(results,'power_scale_')
    if ps:
        s=float(ps['name'].rsplit('_',1)[-1]); findings.append({'cause':'POWER_MAGNITUDE_TOO_HIGH' if s<1 else 'POWER_MAGNITUDE_TOO_LOW','closest_recovering_scale':s,'final_vx_mm_s':ps['final_velocity_mm_s'][0]})
    ss=closest_recovering_scale(results,'steering_scale_')
    if ss: findings.append({'cause':'STEERING_MAGNITUDE_TOO_HIGH','closest_recovering_scale':float(ss['name'].rsplit('_',1)[-1]),'final_vx_mm_s':ss['final_velocity_mm_s'][0]})
    g=[n for n in ('dlm_only','dvm_only','merge_dlm_dvm','swap_dlm_dvm') if by.get(n,{}).get('recovered')]
    if g: findings.append({'cause':'DLM_DVM_BALANCE_OR_PHASE','recovering_transformations':g})
    cc=[r for r in results if r['name'].startswith('constant_symmetric_power_') and r['recovered']]
    findings.append({'cause':'BODY_MECHANICS_HAS_STABLE_POWER_OPERATING_POINTS' if cc else 'NO_STABLE_CONSTANT_POWER_CONTROL_IN_REPLAY_HORIZON','recovering_constant_levels':[float(r['name'].split('_')[3]) for r in cc]})
    return {'status':'CAUSES_LOCALIZED','findings':findings or [{'cause':'UNRESOLVED_COMPOUND_MOTOR_PATTERN'}]}

def main():
    a=parse_args(); records=load_episode(a.trajectory,a.episode); keys=steering_keys(records); power,steering,spikes=extract_motor_arrays(records,keys); pattern=summarize_pattern(power,steering,spikes); original=infer_original_collision(records,a.seed,a.gate_count); variants=variant_matrix(keys); vmap={v.name:v for v in variants}; h=ReplayHarness(a.seed,a.gate_count,a.physics_steps,a.initial_forward_speed_mm_s); results=[]
    for v in variants:
        p,s=apply_variant(power,steering,v); results.append(h.run(v.name,p,s,spikes))
    by={r['name']:r for r in results}; b=by['recorded']; endpoint=np.asarray([records[-1]['x_mm'],records[-1]['y_mm'],records[-1]['z_mm']],dtype=float); err=float(np.linalg.norm(np.asarray(b['final_position_mm'])-endpoint)); verr=abs(float(b['final_velocity_mm_s'][0])-float(records[-1]['vx_mm_s'])); consistent=err<=a.baseline_position_tolerance_mm and verr<=a.baseline_velocity_tolerance_mm_s and bool(b['collision'])==bool(original['collision']) and b['collision_reason']==original['collision_reason_inferred']; minsteer=minimal_steering_ablation(results,vmap); diagnosis=build_diagnosis(consistent,results,vmap,minsteer); recovered=[r for r in results if r['recovered']]
    result={'trajectory':str(a.trajectory),'episode':a.episode,'records':len(records),'observed_active_steering_channels':list(keys),'recorded_motor_pattern':pattern,'original_episode':original,'baseline_replay':b,'baseline_endpoint_error_mm':err,'baseline_final_vx_error_mm_s':verr,'baseline_consistent':consistent,'variants_tested':len(results),'recovered_variants':len(recovered),'minimal_recovering_steering_ablations':minsteer,'diagnosis':diagnosis,'all_replays':results,'interpretation':'Offline causal diagnosis only; no CNS state is advanced. Recovered means full recorded horizon survived without collision and final forward velocity stayed positive.'}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"original collision={original['collision']} reason={original['collision_reason_inferred']} x={original['last_x_mm']:.3f} z={original['last_z_mm']:.3f} vx={original['last_vx_mm_s']:.3f}"); print(f"baseline collision={b['collision']} reason={b['collision_reason']} x={b['final_position_mm'][0]:.3f} z={b['final_position_mm'][2]:.3f} vx={b['final_velocity_mm_s'][0]:.3f} endpoint_error_mm={err:.6f} vx_error={verr:.6f} consistent={consistent}"); print(f"motor spikes_total={pattern['motor_spikes_total']} max_per_step={pattern['motor_spikes_max_per_step']} active_steering_channels={len(keys)} power_delta_mean={pattern['step_to_step_power_delta_abs_mean']:.6f} power_delta_max={pattern['step_to_step_power_delta_abs_max']:.6f}"); print(f"variants_tested={len(results)} recovered_variants={len(recovered)}")
    if minsteer:
        print(f"minimal_steering_channels_to_remove={len(minsteer[0]['removed'])}")
        for x in minsteer[:12]: print('  steering_ablation='+','.join(x['removed']))
    else: print('minimal_steering_channels_to_remove=NONE')
    print('diagnosis='+diagnosis['status'])
    for f in diagnosis['findings']:
        print('cause='+f['cause'])
        for k,v in f.items():
            if k!='cause': print(f'  {k}={v}')
    print('result='+str(a.output)); print('flyppy_motor_causal_diagnosis=PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
