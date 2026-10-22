"""
Autonomous waypoint flight for the Skydio X2 in MuJoCo.

Usage (from the folder that contains mujoco_menagerie):
    py fly_x2.py                 # opens the 3D viewer and flies the mission
    py fly_x2.py --headless      # no window, prints the flight log (quick test)

The controller uses the TRUE state from the simulator for now.
Later this gets replaced by the EKF estimate (camera + IMU, no GPS).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
SCENE = HERE / "mujoco_menagerie" / "skydio_x2" / "scene.xml"

# Mission: list of (x, y, z) waypoints in meters. A = start, B = last point.
CRUISE_ALT = 5.0
WAYPOINTS = [
    (0.0, 0.0, CRUISE_ALT),    # take off above A
    (15.0, 0.0, CRUISE_ALT),   # fly east
    (15.0, 15.0, CRUISE_ALT),  # fly north, arrive above B
    (15.0, 15.0, 0.25),        # land at B
]
MAX_SPEED = 3.0      # m/s, how fast the setpoint moves along the route
LAND_SPEED = 0.7     # m/s, slower descent for the final waypoint
WP_RADIUS = 0.4      # m, waypoint counts as reached inside this radius


def build_model() -> mujoco.MjModel:
    """Load the X2 scene and add a down camera plus a few 'buildings' and 'trees'."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.option.timestep = 0.002  # smaller step = more stable controller

    # Down-looking camera on the drone (MuJoCo cameras look along their -z axis).
    drone = spec.body("x2")
    drone.add_camera(name="down", pos=[0, 0, -0.02], fovy=70)

    # Simple landmarks so the cameras see something.
    world = spec.worldbody
    rng = np.random.default_rng(1)
    for i in range(12):
        x, y = rng.uniform(-5, 22, size=2)
        if min(np.hypot(x - wx, y - wy) for wx, wy, _ in [(0, 0, 0)] + WAYPOINTS) < 3:
            continue  # keep start, route corners and landing spot free
        h = rng.uniform(1.0, 3.5)
        world.add_geom(name=f"building{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[rng.uniform(0.6, 1.5), rng.uniform(0.6, 1.5), h / 2],
                       pos=[x, y, h / 2], rgba=[0.75, 0.75, 0.78, 1])
    for i in range(25):
        x, y = rng.uniform(-5, 22, size=2)
        world.add_geom(name=f"tree{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[0.5, 0, 0], pos=[x, y, 0.5], rgba=[0.15, 0.55, 0.2, 1],
                       contype=0, conaffinity=0)
    return spec.compile()


class Controller:
    """Cascaded geometric controller: position -> attitude -> motor thrusts."""

    def __init__(self, m: mujoco.MjModel):
        body = m.body("x2").id
        self.mass = float(m.body_subtreemass[body])
        self.g = 9.81
        self.ctrl_max = float(m.actuator_ctrlrange[0, 1])

        # Mixer: motor thrusts -> [total force, roll, pitch, yaw torque].
        mix = np.zeros((4, 4))
        for i in range(4):
            site = m.site(f"thrust{i + 1}")
            x, y = m.site_pos[site.id][:2]
            yaw_coeff = m.actuator_gear[i, 5]
            mix[:, i] = [1.0, y, -x, yaw_coeff]
        self.mix_inv = np.linalg.inv(mix)

        # Gains (tuned for this model).
        self.kp_pos = np.array([2.0, 2.0, 4.0])
        self.kd_pos = np.array([2.5, 2.5, 3.5])
        self.kR = np.array([4.0, 4.0, 1.0])
        self.kw = np.array([0.6, 0.6, 0.3])

    def __call__(self, pos, vel, R, omega, pos_ref, vel_ref, yaw_ref=0.0):
        # 1) position loop -> desired force vector in world frame
        acc = self.kp_pos * (pos_ref - pos) + self.kd_pos * (vel_ref - vel)
        acc = np.clip(acc, -4.0, 4.0)
        f_world = self.mass * (acc + np.array([0, 0, self.g]))

        # 2) desired attitude from force direction and yaw
        z_d = f_world / np.linalg.norm(f_world)
        x_c = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        x_d = np.cross(y_d, z_d)
        R_d = np.column_stack([x_d, y_d, z_d])

        # 3) attitude loop -> body torques
        e_R_mat = 0.5 * (R_d.T @ R - R.T @ R_d)
        e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])
        torque = -self.kR * e_R - self.kw * omega
        thrust = float(f_world @ R[:, 2])

        # 4) mixer -> per-motor thrust, clipped to motor limits
        motors = self.mix_inv @ np.array([thrust, *torque])
        return np.clip(motors, 0.0, self.ctrl_max)


class Mission:
    """Moves a smooth setpoint along the waypoints at MAX_SPEED."""

    def __init__(self, start):
        self.points = [np.array(start, dtype=float)] + [np.array(w) for w in WAYPOINTS]
        self.idx = 1
        self.setpoint = self.points[0].copy()
        self.done = False

    def update(self, pos, dt):
        target = self.points[self.idx]
        delta = target - self.setpoint
        dist = np.linalg.norm(delta)
        speed = LAND_SPEED if self.idx == len(self.points) - 1 else MAX_SPEED
        step = speed * dt
        vel_ref = np.zeros(3)
        if dist > step:
            vel_ref = delta / dist * speed
            self.setpoint += vel_ref * dt
        else:
            self.setpoint = target.copy()
            if np.linalg.norm(target - pos) < WP_RADIUS:
                if self.idx < len(self.points) - 1:
                    self.idx += 1
                else:
                    self.done = True
        return self.setpoint, vel_ref


def read_state(d: mujoco.MjData):
    pos = d.qpos[0:3].copy()
    vel = d.qvel[0:3].copy()          # world frame
    R = d.xmat[1].reshape(3, 3).copy()  # body 1 = x2
    omega = d.qvel[3:6].copy()        # body frame
    return pos, vel, R, omega


def run(headless: bool):
    m = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)  # 'hover' keyframe
    mujoco.mj_forward(m, d)

    ctrl = Controller(m)
    mission = Mission(start=d.qpos[0:3])
    print(f"Drone mass: {ctrl.mass:.2f} kg | waypoints: {len(WAYPOINTS)}")

    def step():
        pos, vel, R, omega = read_state(d)
        sp, vref = mission.update(pos, m.opt.timestep)
        d.ctrl[:] = ctrl(pos, vel, R, omega, sp, vref)
        mujoco.mj_step(m, d)
        return pos

    if headless:
        last_print = -1.0
        while d.time < 60 and not mission.done:
            pos = step()
            if d.time - last_print >= 1.0:
                last_print = d.time
                print(f"t={d.time:5.1f}s  pos=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:5.2f})"
                      f"  waypoint {mission.idx}/{len(WAYPOINTS)}")
        pos = d.qpos[0:3]
        goal = np.array(WAYPOINTS[-1])
        print(f"Mission {'COMPLETE' if mission.done else 'NOT finished'} at t={d.time:.1f}s,"
              f" final error to B: {np.linalg.norm(pos[:2] - goal[:2]):.2f} m")
        return

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = m.body("x2").id
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -25
        while viewer.is_running():
            frame_start = time.time()
            for _ in range(int(1 / 60 / m.opt.timestep)):  # ~1/60 s of sim per frame
                if not mission.done:
                    step()
                else:
                    d.ctrl[:] = 0.0  # landed: motors off
                    mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - frame_start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without a window")
    run(parser.parse_args().headless)
