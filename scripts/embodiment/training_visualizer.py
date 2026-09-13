#!/usr/bin/env python3
"""Optional 3D visualization for Flyppy training.

No renderer/viewer is created unless explicitly requested. Live viewing uses
MuJoCo's passive viewer and recording uses FlyGym's offscreen Renderer with the
FlyBody tracking camera. Keeping this separate from the experiment loop makes
headless training pay essentially no visualization cost.
"""

from __future__ import annotations

from pathlib import Path
import time

from flygym.rendering import Renderer
import mujoco.viewer as mjviewer


class TrainingVisualizer:
    def __init__(
        self,
        *,
        sim,
        camera_name: str,
        live: bool = False,
        record_dir: Path | None = None,
        playback_speed: float = 0.2,
        output_fps: int = 30,
        camera_res: tuple[int, int] = (480, 640),
    ) -> None:
        if playback_speed <= 0.0:
            raise ValueError("playback_speed must be positive")
        if output_fps <= 0:
            raise ValueError("output_fps must be positive")
        self.sim = sim
        self.camera_name = camera_name
        self.live = bool(live)
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.playback_speed = float(playback_speed)
        self.output_fps = int(output_fps)
        self.camera_res = camera_res
        self._viewer = None
        self._recorder: Renderer | None = None
        self._episode = -1
        self._episode_wall_start = 0.0
        self._episode_sim_start = 0.0

        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        if self.live:
            # Passive mode lets the training loop remain in control of mj_step.
            # On macOS this must be launched through `mjpython`; train_flyppy.sh
            # handles that automatically when --render is present.
            self._viewer = mjviewer.launch_passive(
                self.sim.mj_model,
                self.sim.mj_data,
                show_left_ui=False,
                show_right_ui=False,
            )

    @property
    def enabled(self) -> bool:
        return self.live or self.record_dir is not None

    def begin_episode(self, episode: int) -> None:
        self.end_episode(save=True)
        self._episode = int(episode)
        self._episode_wall_start = time.perf_counter()
        self._episode_sim_start = float(self.sim.mj_data.time)
        if self.record_dir is not None:
            self._recorder = Renderer(
                self.sim.mj_model,
                self.camera_name,
                camera_res=self.camera_res,
                playback_speed=self.playback_speed,
                output_fps=self.output_fps,
                buffer_frames=True,
            )
            self._recorder.render_as_needed(self.sim.mj_data)
        self.sync()

    def sync(self) -> None:
        if self._recorder is not None:
            self._recorder.render_as_needed(self.sim.mj_data)

        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.sync()
            else:
                # Closing the viewer should not abort a long training run.
                self._viewer.close()
                self._viewer = None
                self.live = False

        if self.live:
            # Pace only the explicitly requested live view. Headless and video-only
            # runs never sleep. `playback_speed=0.2` means 5x slow motion.
            simulated = float(self.sim.mj_data.time) - self._episode_sim_start
            target_wall = simulated / self.playback_speed
            elapsed_wall = time.perf_counter() - self._episode_wall_start
            remaining = target_wall - elapsed_wall
            if remaining > 0.0:
                time.sleep(min(remaining, 0.05))

    def end_episode(self, *, save: bool = True) -> Path | None:
        if self._recorder is None:
            return None
        output: Path | None = None
        try:
            if save and self.record_dir is not None:
                output = self.record_dir / f"episode-{self._episode:04d}.mp4"
                self._recorder.save_video(output)
        finally:
            self._recorder.close()
            self._recorder = None
        return output

    def close(self) -> None:
        self.end_episode(save=True)
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self) -> "TrainingVisualizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
