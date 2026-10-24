"""Front camera of our drone on a 2-axis gimbal (simulator side): renders what the camera sees at the TRUE pose.

Gimbal: pan (about the vertical, relative to the drone's heading) and tilt (up from the horizon), stabilized: the
horizon stays level whatever the drone's roll / pitch. Commands come from the onboard tracker (GimbalCmd on
`camera/gimbal/cmd`); the gimbal slews toward them at `gimbal_rate_deg` per axis and its encoder readings travel with
every frame (GimbalFrame on `camera/gimbal`). The onboard side turns a pixel into a world ray with ITS heading
estimate + those readings (sfhack/perception/drone_detector.py `camera_ray`, the same model as `forward` here).

Rendering: a MuJoCo model camera named "gimbal" (demos/enemy_world.py adds it), moved per frame; our own X2 is hidden
(geom groups 2, 3), the enemy and birds are group 1. `SegRenderer` also gives the class segmentation (dataset labels,
evaluation truth). Onboard code never imports this module.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .. import bus as topics
from ..bus import Bus
from ..config import TrackerConfig
from ..messages import GimbalCmd, GimbalFrame
from .enemy import geom_classes
from .mujoco_sim import yaw_from_R

MOUNT = np.array([0.18, 0.0, -0.06])  # camera centre in the body frame (FLU): under the nose


def gimbal_forward(yaw: float, pan: float, tilt: float) -> np.ndarray:
    """World direction of the optical axis: heading `yaw` (ENU rad) + `pan`, `tilt` up from the horizon (rad)."""
    a = yaw + pan
    return np.array([math.cos(a) * math.cos(tilt), math.sin(a) * math.cos(tilt), math.sin(tilt)])


def cam_axes(forward: np.ndarray, roll: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        right, up = c * right + s * up, -s * right + c * up
    return f, right, up


def cam_quat(forward: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """MuJoCo camera quaternion looking along `forward` (world), level horizon, then rolled by `roll` rad."""
    f, right, up = cam_axes(forward, roll)
    R = np.column_stack([right, up, -f])  # camera x right, y up, looks along -z
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def project(points: np.ndarray, cam_pos: np.ndarray, forward: np.ndarray, roll: float = 0.0,
            w: int = 640, h: int = 480, vfov_deg: float = 46.83) -> np.ndarray:
    """(N, 3) world points -> (N, 3) pixel u, v and depth along the axis (depth <= 0: behind)."""
    f, right, up = cam_axes(forward, roll)
    fy = (h / 2) / math.tan(math.radians(vfov_deg) / 2)
    d = np.atleast_2d(points) - cam_pos
    z = d @ f
    zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return np.column_stack([w / 2 + fy * (d @ right) / zz, h / 2 - fy * (d @ up) / zz, z])


class SegRenderer:
    """RGB + class segmentation of the model camera "gimbal", our own X2 hidden (groups 2, 3)."""

    def __init__(self, model: mujoco.MjModel, classes: dict[str, int] | None = None, w: int = 640,
                 h: int = 480) -> None:
        self.model = model
        self.r = mujoco.Renderer(model, h, w)
        self.cam = model.camera("gimbal").id
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[2] = self.opt.geomgroup[3] = 0
        self.geom_cls = geom_classes(model, classes or {"enemy": 0, "drone": 0, "bird": 1})
        # instance per geom: the body (birds: torso + wings share the name up to "_wing")
        names = [model.body(int(model.geom_bodyid[g])).name.split("_wing")[0] for g in range(model.ngeom)]
        uniq = {n: i for i, n in enumerate(dict.fromkeys(names))}
        self.geom_inst = np.array([uniq[n] if self.geom_cls[g] >= 0 else -1 for g, n in enumerate(names)], np.int32)
        self.inst_names = list(uniq)
        self._inst_cls = {int(i): int(self.geom_cls[g]) for g, i in enumerate(self.geom_inst) if i >= 0}
        self.fog = True

    def place(self, data: mujoco.MjData, pos, forward, roll: float = 0.0) -> None:
        m = self.model
        m.cam_pos[self.cam] = pos
        m.cam_quat[self.cam] = cam_quat(forward, roll)
        mujoco.mj_camlight(m, data)

    def _scene(self, data: mujoco.MjData) -> None:
        self.r.update_scene(data, camera=self.cam, scene_option=self.opt)
        f = self.r.scene.flags
        f[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        f[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        f[mujoco.mjtRndFlag.mjRND_FOG] = int(self.fog)

    def rgb(self, data: mujoco.MjData) -> np.ndarray:
        self._scene(data)
        return self.r.render().copy()

    def seg(self, data: mujoco.MjData) -> np.ndarray:
        """(H, W) instance index (-1 = background); its class: inst_cls(instance), its name: inst_names[instance]."""
        self.r.enable_segmentation_rendering()
        self._scene(data)
        s = self.r.render()
        self.r.disable_segmentation_rendering()
        ids, typ = s[..., 0], s[..., 1]
        ok = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        out = np.full(ids.shape, -1, np.int32)
        out[ok] = self.geom_inst[ids[ok]]
        return out

    def inst_cls(self, inst: int) -> int:
        return self._inst_cls[int(inst)]

    def close(self) -> None:
        self.r.close()


def boxes_from_seg(seg: np.ndarray, sr: SegRenderer, min_side: float = 4.0,
                   min_px: int = 4) -> tuple[list[tuple], list[tuple]]:
    """[(cls, inst, (x0, y0, x1, y1), pixels)] kept and dropped (too small)."""
    kept, dropped = [], []
    for inst in np.unique(seg):
        if inst < 0:
            continue
        ys, xs = np.nonzero(seg == inst)
        box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        row = (sr.inst_cls(int(inst)), int(inst), box, int(len(xs)))
        big = max(box[2] - box[0], box[3] - box[1]) >= min_side and len(xs) >= min_px
        (kept if big else dropped).append(row)
    return kept, dropped


class GimbalCamera:
    """The simulated gimbal + camera on the X2: slews toward the latest GimbalCmd, renders at `cam_hz` and publishes
    GimbalFrame (RGB + encoder readings). `capture` also returns the segmentation when asked (harness truth)."""

    def __init__(self, model: mujoco.MjModel, bus: Bus, cfg: TrackerConfig | None = None, body: str = "x2") -> None:
        self.cfg = cfg or TrackerConfig()
        self.model, self.bus = model, bus
        self.body = model.body(body).id
        self.sr = SegRenderer(model, w=self.cfg.cam_w, h=self.cfg.cam_h)
        self.pan = self.tilt = 0.0  # actual angles, rad
        self.cmd = (0.0, 0.0)
        self.seq = 0
        self.t_next: float | None = None
        self.last: GimbalFrame | None = None
        self.last_seg: np.ndarray | None = None
        self.last_pose: tuple[np.ndarray, np.ndarray] | None = None  # camera position + axis (truth, display)
        self._t: float | None = None
        bus.subscribe(topics.GIMBAL_CMD, self._on_cmd)

    def _on_cmd(self, msg: GimbalCmd) -> None:
        c = self.cfg
        self.cmd = (math.radians(float(np.clip(msg.pan_deg, -c.pan_limit_deg, c.pan_limit_deg))),
                    math.radians(float(np.clip(msg.tilt_deg, c.tilt_min_deg, c.tilt_max_deg))))

    def _slew(self, dt: float) -> None:
        step = math.radians(self.cfg.gimbal_rate_deg) * dt
        self.pan += float(np.clip(self.cmd[0] - self.pan, -step, step))  # pan has end stops: no wrap-around
        self.tilt += float(np.clip(self.cmd[1] - self.tilt, -step, step))

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        R = data.xmat[self.body].reshape(3, 3)
        pos = data.xpos[self.body] + R @ MOUNT
        return pos.copy(), gimbal_forward(yaw_from_R(R), self.pan, self.tilt)

    def step(self, data: mujoco.MjData, seg: bool = False) -> GimbalFrame | None:
        """Call every sim step: slews the gimbal; at the frame rate renders and publishes a frame (returned)."""
        t = float(data.time)
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        self._slew(dt)
        if self.t_next is None:
            self.t_next = t
        if t + 1e-9 < self.t_next:
            return None
        self.t_next += 1.0 / self.cfg.cam_hz
        return self.capture(data, seg)

    def capture(self, data: mujoco.MjData, seg: bool = False) -> GimbalFrame:
        pos, fwd = self.pose(data)
        self.sr.place(data, pos, fwd)
        img = self.sr.rgb(data)
        self.last_seg = self.sr.seg(data) if seg else None
        self.last_pose = (pos, fwd)
        self.seq += 1
        fr = GimbalFrame(float(data.time), self.seq, img, self.cfg.hfov_deg, math.degrees(self.pan),
                         math.degrees(self.tilt))
        self.last = fr
        self.bus.publish(topics.CAMERA_GIMBAL, fr)
        return fr

    def close(self) -> None:
        self.sr.close()
