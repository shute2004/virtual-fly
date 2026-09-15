#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "scripts" / "embodiment"
if str(EMB) not in sys.path: sys.path.insert(0, str(EMB))
if str(ROOT / "scripts" / "analysis") not in sys.path: sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from probe_direct_ommatidia_adaptation_gap import tree_digest, current_condition, make_body_args, body_id_axis, current_vector
from direct_ommatidia_sensor_bodyexclude import BodyExcludedDirectOmmatidialSensor
from malecns_retina import MaleCNSRetina
import train_flyppy_population as trainer


def corr(a, b):
    if len(a) < 2 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--production',type=Path,default=Path('artifacts/experiments/flyppy-v3')); p.add_argument('--report',type=Path,default=Path('reports/flyppy/direct_ommatidia_temporal_semantics.md')); p.add_argument('--samples',type=int,default=64); p.add_argument('--physics-steps',type=int,default=10); p.add_argument('--rays',type=int,nargs='+',default=(3,7,13)); a=p.parse_args()
    production=(ROOT/a.production).resolve() if not a.production.is_absolute() else a.production
    report=(ROOT/a.report).resolve() if not a.report.is_absolute() else a.report
    snapshot=ROOT/'artifacts/malecns-v1.0'
    cal=json.loads((ROOT/'artifacts/embodiment/neural-runtime-calibration-v1.json').read_text()); os.environ['VF_NEURAL_SYNAPSE_SCALE']=str(float(cal['synapse_scale']))
    before=tree_digest(production/'checkpoint')
    slot=trainer.make_slot(make_body_args(snapshot,2.0),0); trainer.begin_episode(slot,episode=0,source_weight_version=0,condition=current_condition(production))
    mapping=snapshot/'retinotopic-vision-v1.json'; ref=MaleCNSRetina(mapping,current_gain=2.0); axis=body_id_axis(ref)
    dr={k:MaleCNSRetina(mapping,current_gain=2.0) for k in a.rays}; sensors={k:BodyExcludedDirectOmmatidialSensor(dr[k],rays_per_ommatidium=k) for k in a.rays}
    _=ref._eye_readouts(slot.body.sim,slot.body.fly); [s.read_eye_readouts(slot.body.sim,slot.body.fly) for s in sensors.values()]; ref.reset_adaptation(); [r.reset_adaptation() for r in dr.values()]
    d={k:{'t':0.0,'rc':[],'dc':[],'rd':[],'dd':[]} for k in a.rays}; prev_ref=None; prev_dir={k:None for k in a.rays}; quiet={int(x):False for x in slot.periphery.body_ids}; dt=float(slot.body.timestep)*a.physics_steps
    for _ in range(a.samples):
        re=ref._eye_readouts(slot.body.sim,slot.body.fly); rl={side:ref._all_local_achromatic(re[side]) for side in ('L','R')}; rc=current_vector(axis,ref.encode_from_eye_readouts(re)); rvec=np.concatenate([rl[side][np.asarray(sensors[a.rays[0]].required_by_side[side],dtype=np.int32)] for side in ('L','R')])
        for k in a.rays:
            t=time.perf_counter(); de=sensors[k].read_eye_readouts(slot.body.sim,slot.body.fly); d[k]['t']+=time.perf_counter()-t
            dvec=np.concatenate([dr[k]._all_local_achromatic(de[side])[np.asarray(sensors[k].required_by_side[side],dtype=np.int32)] for side in ('L','R')]); dc=current_vector(axis,dr[k].encode_from_eye_readouts(de)); d[k]['rc'].append(rc.copy()); d[k]['dc'].append(dc.copy())
            if prev_ref is not None: d[k]['rd'].append(rvec-prev_ref); d[k]['dd'].append(dvec-prev_dir[k])
            prev_dir[k]=dvec
        prev_ref=rvec; slot.body.step_muscles(slot.periphery.step(quiet,dt_s=dt),physics_steps=a.physics_steps)
    after=tree_digest(production/'checkpoint'); ok=before==after
    lines=['# Flyppy direct ommatidia temporal semantics','',f'- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec="seconds")}',f'- overall: {"PASS" if ok else "FAIL"}',f'- samples: {a.samples}','- direct path: body-excluded mj_multiRay; no RGB framebuffer',f'- production checkpoint modified: {"no" if ok else "YES"}','', '## Temporal comparison','', '| rays/ommatidium | sensor ms | current Pearson r | current MAE | light-delta Pearson r | light-delta MAE |','|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for k in a.rays:
        rc=np.concatenate(d[k]['rc']); dc=np.concatenate(d[k]['dc']); rd=np.concatenate(d[k]['rd']); dd=np.concatenate(d[k]['dd']); lines.append(f'| {k} | {1000*d[k]["t"]/a.samples:.3f} | {corr(rc,dc):.6f} | {np.mean(np.abs(rc-dc)):.6f} | {corr(rd,dd):.6f} | {np.mean(np.abs(rd-dd)):.6f} |')
        for th in (1e-3,1e-2,5e-2,1e-1):
            active=(np.abs(rc)>=th)|(np.abs(dc)>=th); rows.append((k,th,int(active.sum()),float(np.mean(np.sign(rc[active])==np.sign(dc[active]))) if active.any() else 1.0))
    lines += ['', '## Current sign agreement by magnitude','', '| rays/ommatidium | threshold | compared | sign agreement |','|---:|---:|---:|---:|']
    lines += [f'| {k} | {th:.3g} | {n} | {100*agree:.3f}% |' for k,th,n,agree in rows]
    lines += ['', 'Near-zero sign flips are excluded by the thresholded table. Raster is a temporal compatibility reference, not biological ground truth.', '']
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text('\n'.join(lines),encoding='utf-8'); print(f'direct_ommatidia_temporal_semantics={report}'); return 0 if ok else 1

if __name__=='__main__': raise SystemExit(main())
