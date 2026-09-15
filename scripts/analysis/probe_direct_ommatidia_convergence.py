#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; EMB=ROOT/'scripts'/'embodiment'; ANA=ROOT/'scripts'/'analysis'
for p in (EMB,ANA):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from probe_direct_ommatidia_adaptation_gap import tree_digest,current_condition,make_body_args,body_id_axis,current_vector
from direct_ommatidia_sensor_bodyexclude import BodyExcludedDirectOmmatidialSensor
from malecns_retina import MaleCNSRetina
import train_flyppy_population as trainer

def corr(a,b):
    if len(a)<2 or np.std(a)<=1e-15 or np.std(b)<=1e-15: return 0.0
    return float(np.corrcoef(a,b)[0,1])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--production',type=Path,default=Path('artifacts/experiments/flyppy-v3')); p.add_argument('--report',type=Path,default=Path('reports/flyppy/direct_ommatidia_convergence.md')); p.add_argument('--samples',type=int,default=12); p.add_argument('--physics-steps',type=int,default=10); p.add_argument('--rays',type=int,nargs='+',default=(1,3,7,13,25,49)); a=p.parse_args()
    prod=(ROOT/a.production).resolve() if not a.production.is_absolute() else a.production; report=(ROOT/a.report).resolve() if not a.report.is_absolute() else a.report; snap=ROOT/'artifacts/malecns-v1.0'
    cal=json.loads((ROOT/'artifacts/embodiment/neural-runtime-calibration-v1.json').read_text()); os.environ['VF_NEURAL_SYNAPSE_SCALE']=str(float(cal['synapse_scale'])); before=tree_digest(prod/'checkpoint')
    slot=trainer.make_slot(make_body_args(snap,2.0),0); trainer.begin_episode(slot,episode=0,source_weight_version=0,condition=current_condition(prod)); mapping=snap/'retinotopic-vision-v1.json'
    ks=tuple(dict.fromkeys(int(k) for k in a.rays)); full_key='all'; ret={k:MaleCNSRetina(mapping,current_gain=2.0) for k in ks}; ret[full_key]=MaleCNSRetina(mapping,current_gain=2.0)
    sensors={k:BodyExcludedDirectOmmatidialSensor(ret[k],rays_per_ommatidium=k) for k in ks}; sensors[full_key]=BodyExcludedDirectOmmatidialSensor(ret[full_key],rays_per_ommatidium=1000000)
    axis=body_id_axis(ret[full_key]); [s.read_eye_readouts(slot.body.sim,slot.body.fly) for s in sensors.values()]; [r.reset_adaptation() for r in ret.values()]
    data={k:{'t':0.0,'light':[],'current':[]} for k in sensors}; quiet={int(x):False for x in slot.periphery.body_ids}; dt=float(slot.body.timestep)*a.physics_steps
    for _ in range(a.samples):
        for k,s in sensors.items():
            t=time.perf_counter(); eyes=s.read_eye_readouts(slot.body.sim,slot.body.fly); data[k]['t']+=time.perf_counter()-t
            lv=np.concatenate([ret[k]._all_local_achromatic(eyes[side])[np.asarray(s.required_by_side[side],dtype=np.int32)] for side in ('L','R')]); cv=current_vector(axis,ret[k].encode_from_eye_readouts(eyes)); data[k]['light'].append(lv); data[k]['current'].append(cv)
        slot.body.step_muscles(slot.periphery.step(quiet,dt_s=dt),physics_steps=a.physics_steps)
    ok=before==tree_digest(prod/'checkpoint'); ref_l=np.concatenate(data[full_key]['light']); ref_c=np.concatenate(data[full_key]['current'])
    lines=['# Flyppy direct ommatidia convergence','',f'- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec="seconds")}',f'- overall: {"PASS" if ok else "FAIL"}',f'- samples: {a.samples}','- reference: every valid FlyGym receptive-field source pixel queried directly with mj_multiRay','- RGB framebuffer used: no',f'- full-reference rays/readout: {sensors[full_key].last_stats.total_rays}',f'- full-reference sensor ms: {1000*data[full_key]["t"]/a.samples:.3f}',f'- production checkpoint modified: {"no" if ok else "YES"}','','| rays/ommatidium | total rays | sensor ms | speedup vs full | light MAE vs full | light r | current MAE vs full | current r |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    full_ms=1000*data[full_key]['t']/a.samples
    for k in ks:
        l=np.concatenate(data[k]['light']); c=np.concatenate(data[k]['current']); ms=1000*data[k]['t']/a.samples
        lines.append(f'| {k} | {sensors[k].last_stats.total_rays} | {ms:.3f} | {full_ms/max(ms,1e-12):.3f}x | {np.mean(np.abs(l-ref_l)):.6f} | {corr(l,ref_l):.6f} | {np.mean(np.abs(c-ref_c)):.6f} | {corr(c,ref_c):.6f} |')
    lines += ['','This measures convergence of the image-free sensor to its own uncompressed receptive-field integral. Raster rendering is not used as the reference.','']
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text('\n'.join(lines),encoding='utf-8'); print(f'direct_ommatidia_convergence={report}'); print(f'full_rays={sensors[full_key].last_stats.total_rays} full_ms={full_ms:.6f}'); return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
