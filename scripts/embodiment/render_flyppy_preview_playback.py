#!/usr/bin/env python3
"""Render captured Flyppy physics poses to a JPEG frame sequence."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path
import mujoco as mj
import numpy as np
from flybody_v3_adapter import FlyBodyV3MuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_body_viewer import DEFAULT_CAMERA, CAMERA_FORWARD_LOOK_MM
ROOT=Path(__file__).resolve().parents[2]

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--playback',type=Path,default=Path('artifacts/experiments/flyppy-best-playback/playback.json')); p.add_argument('--width',type=int,default=960); p.add_argument('--height',type=int,default=540); p.add_argument('--jpeg-quality',type=int,default=5); return p.parse_args()
def absolute(p): return p if p.is_absolute() else (ROOT/p).resolve()

def main():
    a=parse_args(); path=absolute(a.playback); payload=json.loads(path.read_text()); frames=payload['frames']; out=path.parent/'frames'
    if not frames: raise SystemExit('playback has no frames')
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    os.environ['VF_COURSE_START_GATE']='0'
    course=FlyppyCourse(seed=int(payload['course_seed']),gate_count=int(payload['gate_count']),environment_version=str(payload['environment_version']))
    world=FlyppyWorld(course); body=FlyBodyV3MuscleAdapter(tethered=False,world=world,spawn_position_mm=(0.0,0.0,(course.floor_z_mm+course.ceiling_z_mm)/2.0),initial_linear_velocity_mm_s=(0,0,0),enable_vision=False,enable_observer_camera=False)
    renderer=mj.Renderer(body.sim.mj_model,height=a.height,width=a.width); camera=mj.MjvCamera(); mj.mjv_defaultCamera(camera); camera.type=mj.mjtCamera.mjCAMERA_FREE; camera.fixedcamid=-1
    ffmpeg=shutil.which('ffmpeg');
    if not ffmpeg: raise SystemExit('ffmpeg is required to render playback frames')
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{a.height}','-r',str(float(payload['playback_fps'])),'-i','pipe:0','-an','-q:v',str(a.jpeg_quality),'-start_number','0',str(out/'frame-%05d.jpg')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        for frame in frames:
            qpos=np.asarray(frame['qpos'],dtype=np.float64); qvel=np.asarray(frame['qvel'],dtype=np.float64)
            body.sim.mj_data.qpos[:]=qpos; body.sim.mj_data.qvel[:]=qvel; body.sim.mj_data.time=float(frame['sim_time_s']); mj.mj_forward(body.sim.mj_model,body.sim.mj_data)
            camera.azimuth=float(DEFAULT_CAMERA['azimuth']); camera.elevation=float(DEFAULT_CAMERA['elevation']); camera.distance=float(DEFAULT_CAMERA['distance'])
            thorax=np.asarray(body.thorax_position_mm(),dtype=np.float64); camera.lookat[:]=np.asarray((thorax[0]+CAMERA_FORWARD_LOOK_MM,0.0,(course.floor_z_mm+course.ceiling_z_mm)/2.0),dtype=np.float64)
            renderer.update_scene(body.sim.mj_data,camera=camera); image=np.ascontiguousarray(renderer.render(),dtype=np.uint8)
            assert proc.stdin is not None; proc.stdin.write(image.tobytes())
        assert proc.stdin is not None; proc.stdin.close(); stderr=proc.stderr.read() if proc.stderr else b''; code=proc.wait()
        if code!=0: raise RuntimeError('ffmpeg failed: '+stderr.decode('utf-8','replace'))
    finally:
        renderer.close()
        if proc.poll() is None: proc.kill(); proc.wait()
    files=sorted(out.glob('frame-*.jpg'))
    if len(files)!=len(frames): raise RuntimeError(f'frame count mismatch: expected {len(frames)}, got {len(files)}')
    payload['render']={'width':a.width,'height':a.height,'format':'jpeg','frame_pattern':'frames/frame-%05d.jpg','frame_count':len(files),'camera':DEFAULT_CAMERA,'forward_look_mm':CAMERA_FORWARD_LOOK_MM}
    tmp=path.with_name('.'+path.name+'.render.tmp'); tmp.write_text(json.dumps(payload,separators=(',',':'))+'\n'); os.replace(tmp,path)
    print(f'playback_render=PASS frames={len(files)} fps={payload["playback_fps"]} size={a.width}x{a.height}'); print(f'frames_dir={out}')
    return 0
if __name__=='__main__': raise SystemExit(main())
