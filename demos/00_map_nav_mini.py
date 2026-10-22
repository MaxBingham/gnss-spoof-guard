"""
Mini demo: GNSS-free navigation by matching the down camera against an onboard map.

The drone flies from A (green) to B (red) WITHOUT GPS.
- Before takeoff it gets an onboard map (a top-down image of the area).
- In flight it only "knows" its position from dead reckoning (drifts!) plus
  visual fixes: the down-camera image is matched against the onboard map.
- Cyan sphere = where the drone THINKS it is. The drone itself = truth.

Keys in the MuJoCo window:
    V  toggle vision on/off  (off -> watch the cyan estimate drift away)
    R  restart the mission

Usage (uses the repo's third_party/skydio_x2, or a mujoco_menagerie checkout):
    uv run python demos/00_map_nav_mini.py
"""
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# ---------------------------------------------------------------- settings
HALF = 75.0                 # world is 150 m x 150 m
A = np.array([-45.0, -40.0])
B = np.array([45.0, 40.0])
CRUISE_ALT = 35.0
SPEED = 4.0                 # m/s cruise
DOWN_FOVY = 60.0            # degrees
CAM_PX = 200                # down camera resolution
MAP_MPP = 1.0               # onboard map resolution, m per pixel
TPL_HALF = 18               # template half size in map pixels (36 m)
SEARCH = 20                 # search radius around prior, map pixels
FIX_HZ = 4.0
NCC_MIN = 0.55
DRIFT_BIAS = 0.35           # m/s velocity bias of the dead reckoning


def find_scene() -> Path:
    here = Path(__file__).resolve().parent
    repo_x2 = here.parent / "third_party" / "skydio_x2" / "scene.xml"
    if repo_x2.exists():
        return repo_x2
    for base in [here, Path.home(), Path(r"C:\Users\jean-")]:
        p = base / "mujoco_menagerie" / "skydio_x2" / "scene.xml"
        if p.exists():
            return p
    sys.exit("mujoco_menagerie/skydio_x2/scene.xml not found. Put this file next to mujoco_menagerie.")


# ---------------------------------------------------------------- ground texture
def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def chunk(t, data):
        c = struct.pack(">I", len(data)) + t + data
        return c + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def make_ground(rng: np.random.Generator, n: int = 768) -> np.ndarray:
    """Non-repeating 'aerial photo': grass noise, fields, paths, plazas."""
    y, x = np.mgrid[0:n, 0:n] / n
    img = np.zeros((n, n, 3))
    noise = sum(np.sin((x * fx + y * fy) * 2 * np.pi + ph)
                for fx, fy, ph in rng.uniform([-9, -9, 0], [9, 9, 6.28], (12, 3))) / 12
    img[:] = np.stack([0.30 + 0.08 * noise, 0.45 + 0.10 * noise, 0.22 + 0.05 * noise], -1)
    for _ in range(40):  # fields / plazas
        cx, cy, rx, ry = rng.uniform([0, 0, 0.02, 0.02], [1, 1, 0.09, 0.09])
        col = rng.choice([[0.55, 0.52, 0.45], [0.62, 0.60, 0.40], [0.25, 0.35, 0.15], [0.45, 0.42, 0.40]])
        img[(abs(x - cx) < rx) & (abs(y - cy) < ry)] = col
    for _ in range(14):  # paths
        if rng.random() < 0.5:
            c = rng.uniform(0, 1); img[abs(y - c) < 0.006] = [0.75, 0.73, 0.70]
        else:
            c = rng.uniform(0, 1); img[abs(x - c) < 0.006] = [0.75, 0.73, 0.70]
    img += rng.normal(0, 0.02, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- model
def build_model(rng):
    spec = mujoco.MjSpec.from_file(str(find_scene()))
    spec.option.timestep = 0.002

    data = Path(os.environ.get("SFHACK_DATA") or Path.home() / "sf-hack-data")  # generated data stays out of the repo
    data.mkdir(parents=True, exist_ok=True)
    tex_path = data / "map_nav_ground.png"
    write_png(tex_path, make_ground(rng))
    spec.add_texture(name="groundmap", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex_path))
    mat = spec.add_material(name="groundmat", texrepeat=[1, 1], texuniform=False)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundmap"
    floor = spec.geom("floor")
    floor.material = "groundmat"
    floor.size = [HALF, HALF, 0.1]

    world = spec.worldbody
    for i in range(45):  # low buildings (low to limit parallax)
        x, y = rng.uniform(-HALF + 5, HALF - 5, 2)
        if min(np.linalg.norm([x, y] - A), np.linalg.norm([x, y] - B)) < 8:
            continue
        h = rng.uniform(1.5, 5.0)
        shade = rng.uniform(0.55, 0.9)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, h / 2],
                       size=[rng.uniform(2, 6), rng.uniform(2, 6), h / 2],
                       rgba=[shade, shade * 0.95, shade * 0.9, 1])
    for i in range(80):  # trees
        x, y = rng.uniform(-HALF + 3, HALF - 3, 2)
        r = rng.uniform(1.0, 2.2)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[x, y, r], size=[r, 0, 0],
                       rgba=[0.1, rng.uniform(0.35, 0.55), 0.12, 1], contype=0, conaffinity=0)

    drone = spec.body("x2")
    drone.pos = [A[0], A[1], 0.3]
    drone.add_camera(name="down", pos=[0, 0, -0.03], fovy=DOWN_FOVY)
    world.add_camera(name="mapcam", pos=[0, 0, 300], fovy=2 * HALF,
                     proj=mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)

    m = spec.compile()
    m.vis.global_.offwidth = 1024
    m.vis.global_.offheight = 1024
    for i in range(m.nlight):  # drop the spotlight that follows the drone
        if m.light(i).name == "spotlight":
            m.light_active[i] = 0
    m.key_qpos[0][:3] = [A[0], A[1], 0.3]
    return m


# ---------------------------------------------------------------- control
class Controller:
    def __init__(self, m):
        self.mass = float(m.body_subtreemass[m.body("x2").id])
        mix = np.zeros((4, 4))
        for i in range(4):
            x, y = m.site_pos[m.site(f"thrust{i + 1}").id][:2]
            mix[:, i] = [1.0, y, -x, m.actuator_gear[i, 5]]
        self.mix_inv = np.linalg.inv(mix)
        self.kp, self.kd = np.array([1.5, 1.5, 4.0]), np.array([2.2, 2.2, 3.5])
        self.kR, self.kw = np.array([4.0, 4.0, 1.0]), np.array([0.6, 0.6, 0.3])

    def __call__(self, pos, vel, R, omega, pref, vref):
        acc = np.clip(self.kp * (pref - pos) + self.kd * (vref - vel), -4, 4)
        f = self.mass * (acc + [0, 0, 9.81])
        zd = f / np.linalg.norm(f)
        yd = np.cross(zd, [1.0, 0, 0]); yd /= np.linalg.norm(yd)
        Rd = np.column_stack([np.cross(yd, zd), yd, zd])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        tau = -self.kR * np.array([E[2, 1], E[0, 2], E[1, 0]]) - self.kw * omega
        return np.clip(self.mix_inv @ np.array([f @ R[:, 2], *tau]), 0, 13)


# ---------------------------------------------------------------- vision
class MapLocalizer:
    """Matches the ortho-rectified down-camera image against the onboard map (NCC)."""

    def __init__(self, m, d):
        self.m, self.d = m, d
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[2] = 0          # hide the drone itself
        self.cam_id = m.camera("down").id
        self.f = (CAM_PX / 2) / np.tan(np.radians(DOWN_FOVY) / 2)
        # Onboard map: rendered once before takeoff, then the drone only uses this image.
        r = mujoco.Renderer(m, 600, 600)
        r.update_scene(d, camera="mapcam", scene_option=self.opt)
        big = r.render().astype(float)
        r.close()
        k = int(round(600 / (2 * HALF / MAP_MPP)))
        n = big.shape[0] // k
        self.map_rgb = big[:n * k, :n * k].reshape(n, k, n, k, 3).mean((1, 3))
        self.map = self.map_rgb.mean(-1)
        self.n = n
        self.renderer = mujoco.Renderer(m, CAM_PX, CAM_PX)
        g = np.arange(-TPL_HALF, TPL_HALF) + 0.5
        self.gx, self.gy = np.meshgrid(g * MAP_MPP, -g * MAP_MPP)  # row 0 = north

    def world_to_px(self, xy):
        return (HALF - xy[1]) / MAP_MPP - 0.5, (xy[0] + HALF) / MAP_MPP - 0.5  # row, col

    def update(self, prior_xy, altitude):
        self.renderer.update_scene(self.d, camera="down", scene_option=self.opt)
        img = self.renderer.render()
        gray = img.mean(-1)
        # ortho-rectify: for each ground cell, find the camera pixel looking at it
        Rc = self.d.cam_xmat[self.cam_id].reshape(3, 3)
        p = np.stack([self.gx, self.gy, np.full_like(self.gx, -altitude)], -1) @ Rc
        depth = -p[..., 2]
        col = CAM_PX / 2 + self.f * p[..., 0] / depth
        row = CAM_PX / 2 - self.f * p[..., 1] / depth
        valid = (col >= 0) & (col < CAM_PX) & (row >= 0) & (row < CAM_PX)
        if valid.mean() < 0.9:
            return None, img, 0.0
        tpl = gray[np.clip(row, 0, CAM_PX - 1).astype(int), np.clip(col, 0, CAM_PX - 1).astype(int)]
        # search window in the map around the prior
        r0, c0 = (int(round(v)) for v in self.world_to_px(prior_xy))
        t = 2 * TPL_HALF
        top, left = r0 - TPL_HALF - SEARCH, c0 - TPL_HALF - SEARCH
        if top < 0 or left < 0 or top + t + 2 * SEARCH > self.n or left + t + 2 * SEARCH > self.n:
            return None, img, 0.0
        region = self.map[top:top + t + 2 * SEARCH, left:left + t + 2 * SEARCH]
        win = np.lib.stride_tricks.sliding_window_view(region, (t, t))
        tz = (tpl - tpl.mean()) / (tpl.std() + 1e-6)
        wm = win.mean((-1, -2), keepdims=True)
        ws = win.std((-1, -2), keepdims=True) + 1e-6
        ncc = ((win - wm) / ws * tz).mean((-1, -2))
        i, j = np.unravel_index(np.argmax(ncc), ncc.shape)
        score = float(ncc[i, j])
        if score < NCC_MIN:
            return None, img, score
        rc, cc = top + i + TPL_HALF - 0.5, left + j + TPL_HALF - 0.5
        fix = np.array([(cc + 0.5) * MAP_MPP - HALF, HALF - (rc + 0.5) * MAP_MPP])
        return fix, img, score


# ---------------------------------------------------------------- panels
def map_panel(loc, est, truth, size=220):
    idx = (np.arange(size) * loc.n / size).astype(int)
    img = loc.map_rgb[idx][:, idx].copy()
    def dot(xy, color, r=3):
        rr, cc = loc.world_to_px(xy)
        rr, cc = int(rr * size / loc.n), int(cc * size / loc.n)
        img[max(rr - r, 0):rr + r + 1, max(cc - r, 0):cc + r + 1] = color
    dot(A, [40, 220, 40]); dot(B, [230, 40, 40]); dot(truth, [255, 255, 255]); dot(est, [0, 230, 255])
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(7)
    m = build_model(rng)
    d = mujoco.MjData(m)
    ctrl = Controller(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    try:
        loc = MapLocalizer(m, d)   # renders the onboard map at A, before takeoff
    except Exception as e:
        print("Offscreen rendering failed, running without vision:", e)
        loc = None
    state = {"vision": True, "reset": True}

    def key(code):
        if code == ord("V"):
            state["vision"] = not state["vision"]
        if code == ord("R"):
            state["reset"] = True

    with mujoco.viewer.launch_passive(m, d, key_callback=key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = m.body("x2").id
        viewer.cam.distance, viewer.cam.elevation = 22.0, -30
        while viewer.is_running():
            if state["reset"]:
                state["reset"] = False
                mujoco.mj_resetDataKeyframe(m, d, 0)
                mujoco.mj_forward(m, d)
                est = d.qpos[:2].copy()          # start position A is known
                bias = rng.normal(0, 1, 2); bias *= DRIFT_BIAS / np.linalg.norm(bias)
                phase, sp = "TAKEOFF", np.array([*A, CRUISE_ALT])
                next_fix, score, last_fix, cam_img = 0.0, 0.0, "-", None
            frame_start = time.time()
            for _ in range(8):  # 16 ms of sim per frame
                truth_vel = d.qvel[:3]
                bias += rng.normal(0, 0.02, 2) * np.sqrt(m.opt.timestep)
                est += (truth_vel[:2] + bias) * m.opt.timestep      # dead reckoning (IMU-like, drifts)
                alt = d.qpos[2] + rng.normal(0, 0.1)                  # barometer
                vref = np.zeros(3)
                if phase == "TAKEOFF" and alt > CRUISE_ALT - 1:
                    phase = "CRUISE"
                if phase == "CRUISE":
                    delta = B - sp[:2]
                    dist = np.linalg.norm(delta)
                    if dist > SPEED * m.opt.timestep:
                        vref[:2] = delta / dist * SPEED
                        sp[:2] += vref[:2] * m.opt.timestep
                    elif np.linalg.norm(B - est) < 1.5:
                        phase = "DESCEND"
                if phase == "DESCEND":
                    vz = min(3.0, max(0.5, 0.25 * sp[2]))   # fast high up, gentle near ground
                    sp[2] = max(0.25, sp[2] - vz * m.opt.timestep)
                    vref[2] = -vz if sp[2] > 0.25 else 0
                    if d.qpos[2] < 0.35:
                        phase = "LANDED"
                pos = np.array([est[0], est[1], alt])
                vel = np.array([*(truth_vel[:2] + bias), d.qvel[2]])
                R = d.xmat[m.body("x2").id].reshape(3, 3)
                d.ctrl[:] = 0 if phase == "LANDED" else ctrl(pos, vel, R, d.qvel[3:6], sp, vref)
                mujoco.mj_step(m, d)

                if loc is not None and d.time >= next_fix and alt > 10:
                    next_fix = d.time + 1 / FIX_HZ
                    try:
                        fix, cam_img, score = loc.update(est, alt)
                    except Exception as e:  # never crash the demo
                        fix, score = None, 0.0
                        print("vision error:", e)
                    if fix is not None and state["vision"]:
                        est += 0.7 * (fix - est)
                        last_fix = f"{d.time:5.1f}s"

            err = np.linalg.norm(est - d.qpos[:2])
            # overlays
            viewer.user_scn.ngeom = 0
            for p, rgba, size in [((*est, d.qpos[2]), [0, 0.9, 1, 0.5], 1.2),
                                  ((*A, 0.2), [0.1, 0.9, 0.1, 0.8], 2.0),
                                  ((*B, 0.2), [0.95, 0.1, 0.1, 0.8], 2.0)]:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [size, 0, 0],
                                    np.array(p, float), np.eye(3).flatten(), np.array(rgba, np.float32))
                viewer.user_scn.ngeom += 1
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "Phase\nVision (V)\nPosition error\nMatch score\nLast visual fix\nGPS",
                              f"{phase}\n{'ON' if state['vision'] else 'OFF'}\n{err:5.1f} m\n"
                              f"{score:4.2f}\n{last_fix}\nDISABLED"))
            imgs = [] if loc is None else [(mujoco.MjrRect(10, 10, 220, 220), map_panel(loc, est, d.qpos[:2]))]
            if cam_img is not None:
                idx = (np.arange(220) * CAM_PX / 220).astype(int)
                imgs.append((mujoco.MjrRect(240, 10, 220, 220), cam_img[idx][:, idx]))
            if imgs:
                viewer.set_images(imgs)
            viewer.sync()
            time.sleep(max(0.0, 8 * m.opt.timestep - (time.time() - frame_start)))


if __name__ == "__main__":
    main()
