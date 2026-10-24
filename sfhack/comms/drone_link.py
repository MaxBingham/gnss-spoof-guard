"""Drone side of the MAVLink 2 link to the base (onboard: never imports the simulator, never reads the truth).

Sends (docs/MAVLINK.md): HEARTBEAT (1 Hz and on every mission-state change; custom_mode = state), GLOBAL_POSITION_INT
+ LOCAL_POSITION_NED (4 Hz), ATTITUDE (10 Hz), SYS_STATUS + GPS_RAW_INT + EXTENDED_SYS_STATE (1 Hz), HOME_POSITION,
STATUSTEXT (onboard decisions), COMMAND_ACK, and each enemy report from the tracker's ReportOutbox as TRACK_REPORT_INFO
+ CAMERA_TRACKING_GEO_STATUS + its JPEG snapshot (DATA_TRANSMISSION_HANDSHAKE + ENCAPSULATED_DATA).
Receives: HEARTBEAT (link state: none for `link_timeout_s` = down), ACOUSTIC_BEARING (-> bus topic link/bearing, as
the in-process path), COMMAND_LONG / COMMAND_INT (launch = MAV_CMD_NAV_TAKEOFF, recall = MAV_CMD_NAV_RETURN_TO_LAUNCH,
look there = MAV_CMD_DO_SET_ROI_LOCATION), TRACK_REPORT_ACK.

Everything comes from the pilot's estimate (EKF + AHRS), its mission state and the onboard sensors on the bus.
Reports leave the outbox only while the link looks up (base heartbeat fresh); until the base acknowledges one it is
resent every `report_resend_s`, and when the link drops the unacknowledged ones go back to the front of the outbox.
"""

from __future__ import annotations

import math

import numpy as np

from .. import bus as topics
from ..bus import Bus
from ..config import CommsConfig
from ..messages import BearingUpdate, Decision, GnssMsg
from ..mission.enemy_report import EnemyReport
from .mavlink_io import FLIGHT_STATES, ON_GROUND, TRACKER_STATES, Frame, MavEndpoint, enu_to_ned, usec

GNSS_FRESH_S = 2.0


class DroneLink(MavEndpoint):
    def __init__(self, pilot, bus: Bus, cfg: CommsConfig | None = None, gate=None, port: int | None = None,
                 peer: tuple[str, int] | None = None) -> None:
        cfg = self.cfg = cfg or CommsConfig()
        super().__init__(cfg.drone_sysid, cfg.drone_compid, (cfg.host, cfg.drone_port if port is None else port),
                         peer or (cfg.host, cfg.base_port), cfg.signing_key() if cfg.signing else None, link_id=1,
                         gate=gate, name="drone")
        self.pilot, self.bus = pilot, bus
        self.frame = Frame(cfg.origin_lat, cfg.origin_lon, cfg.origin_alt_m)
        m = self.mod
        self.m = m
        self.base_hb_t: float | None = None
        self._next = dict.fromkeys(("hb", "tel", "att", "status", "home"), -math.inf)
        self._state: str | None = None
        self.unacked: dict[int, list] = {}  # report seq -> [report, last sent]
        self.reports_sent = self.reports_resent = self.reports_acked = 0
        self.bearings_rx = 0
        self.own_bearings: list[tuple[float, float, float, float]] = []  # OWN_DRONE bearings (for a future fix)
        self.first_heard_t: float | None = None
        self.roi: np.ndarray | None = None
        self.gnss: GnssMsg | None = None
        self._texts: list[str] = []
        self._text_id = 0
        bus.subscribe(topics.GNSS, self._on_gnss)
        bus.subscribe(topics.DECISION, self._on_decision)

    def _on_gnss(self, msg: GnssMsg) -> None:
        self.gnss = msg

    def _on_decision(self, msg: Decision) -> None:
        self._texts.append(msg.text)

    def link_alive(self, t: float) -> bool:
        return self.base_hb_t is not None and t - self.base_hb_t <= self.cfg.link_timeout_s

    # ------------------------------------------------------------------ periodic

    def update(self, t: float) -> None:
        self.now = t
        cfg, nx = self.cfg, self._next
        st = self.pilot.state
        changed = st != self._state
        self._state = st
        if changed or t >= nx["hb"]:
            nx["hb"] = t + 1.0 / cfg.heartbeat_hz
            self._heartbeat(st)
        if changed or t >= nx["status"]:
            nx["status"] = t + 1.0 / cfg.status_hz
            self._status(t, st)
        if t >= nx["tel"]:
            nx["tel"] = t + 1.0 / cfg.telemetry_hz
            self._position(t)
        if t >= nx["att"]:
            nx["att"] = t + 1.0 / cfg.attitude_hz
            self._attitude(t)
        if t >= nx["home"]:
            nx["home"] = t + 1.0 / cfg.home_hz
            self._home(t)
        while self._texts:
            self._statustext(self._texts.pop(0))
        self._reports(t)

    def _heartbeat(self, st: str) -> None:
        m = self.m
        on_ground = st in ON_GROUND
        mode = m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | m.MAV_MODE_FLAG_AUTO_ENABLED
        if not on_ground:
            mode |= m.MAV_MODE_FLAG_SAFETY_ARMED
        self.mav.heartbeat_send(m.MAV_TYPE_QUADROTOR, m.MAV_AUTOPILOT_GENERIC, mode,
                                FLIGHT_STATES.index(st) if st in FLIGHT_STATES else 255,
                                m.MAV_STATE_STANDBY if on_ground else m.MAV_STATE_ACTIVE)

    def _status(self, t: float, st: str) -> None:
        m, mav = self.m, self.mav
        S = (m.MAV_SYS_STATUS_SENSOR_3D_GYRO | m.MAV_SYS_STATUS_SENSOR_3D_ACCEL | m.MAV_SYS_STATUS_SENSOR_3D_MAG
             | m.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE | m.MAV_SYS_STATUS_SENSOR_GPS)
        fix = self.gnss is not None and t - self.gnss.t <= GNSS_FRESH_S
        health = S if fix else S & ~m.MAV_SYS_STATUS_SENSOR_GPS
        batt = -1
        tc = getattr(self.pilot, "tcfg", None)
        if tc is not None and hasattr(self.pilot, "battery_left"):
            batt = int(np.clip(100.0 * self.pilot.battery_left(t) / tc.battery_s, 0, 100))
        got = self.rx_peer + self.peer_seq_lost
        drop = int(10000 * self.peer_seq_lost / got) if got else 0
        mav.sys_status_send(S, S, health, 0, 0xFFFF, -1, batt, min(drop, 10000), 0, 0, 0, 0, 0)
        if fix:
            lat, lon, alt = self.frame.to_global(self.gnss.pos)
            mav.gps_raw_int_send(usec(t), m.GPS_FIX_TYPE_3D_FIX, lat, lon, int(alt * 1000), 0xFFFF, 0xFFFF, 0xFFFF,
                                 0xFFFF, 255)
        else:
            mav.gps_raw_int_send(usec(t), m.GPS_FIX_TYPE_NO_FIX, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 255)
        landed = {"WAIT": m.MAV_LANDED_STATE_ON_GROUND, "LANDED": m.MAV_LANDED_STATE_ON_GROUND,
                  "TAKEOFF": m.MAV_LANDED_STATE_TAKEOFF, "DESCEND": m.MAV_LANDED_STATE_LANDING}
        mav.extended_sys_state_send(m.MAV_VTOL_STATE_UNDEFINED, landed.get(st, m.MAV_LANDED_STATE_IN_AIR))

    def _position(self, t: float) -> None:
        est = self.pilot.pose(report=False)
        lat, lon, alt = self.frame.to_global(est.pos)
        vn, ve, vd = enu_to_ned(est.vel)
        hdg = (90.0 - math.degrees(est.yaw)) % 360.0
        ms = int(t * 1000) & 0xFFFFFFFF
        c = lambda v: int(np.clip(round(v * 100), -32767, 32767))  # noqa: E731 - cm/s, int16
        self.mav.global_position_int_send(ms, lat, lon, int(alt * 1000), int(est.pos[2] * 1000), c(vn), c(ve), c(vd),
                                          int(round(hdg * 100)) % 36000)
        x, y, z = enu_to_ned(est.pos)
        self.mav.local_position_ned_send(ms, x, y, z, vn, ve, vd)

    def _attitude(self, t: float) -> None:
        est = self.pilot.pose(report=False)
        A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # ENU -> NED
        R = A @ est.R @ np.diag([1.0, -1.0, -1.0])  # body FLU -> FRD
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = -math.asin(float(np.clip(R[2, 0], -1.0, 1.0)))
        yaw = math.atan2(R[1, 0], R[0, 0])
        w = est.omega
        self.mav.attitude_send(int(t * 1000) & 0xFFFFFFFF, roll, pitch, yaw, float(w[0]), -float(w[1]), -float(w[2]))

    def _home(self, t: float) -> None:
        pad = np.array([self.pilot.pad[0], self.pilot.pad[1], 0.0])
        lat, lon, alt = self.frame.to_global(pad)
        x, y, z = enu_to_ned(pad)
        self.mav.home_position_send(lat, lon, int(alt * 1000), x, y, z, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, usec(t))

    def _statustext(self, text: str) -> None:
        raw = text.encode("ascii", "replace")
        chunks = [raw[i:i + 50] for i in range(0, len(raw), 50)] or [b""]
        if len(chunks) > 1 and len(chunks[-1]) == 50:
            chunks.append(b"")  # a chunk shorter than 50 ends the text
        if len(chunks) > 1:
            self._text_id = self._text_id % 65535 + 1
        for k, c in enumerate(chunks):
            self.mav.statustext_send(self.m.MAV_SEVERITY_INFO, c, self._text_id if len(chunks) > 1 else 0, k)

    # ------------------------------------------------------------------ enemy reports

    def _reports(self, t: float) -> None:
        outbox = getattr(self.pilot, "outbox", None)
        if outbox is None:
            return
        if not self.link_alive(t):
            if self.unacked:  # back to the front of the outbox: buffered, resent when the link returns
                outbox.queue[:0] = [v[0] for _, v in sorted(self.unacked.items())]
                self.unacked.clear()
            return
        for seq, v in sorted(self.unacked.items()):
            if t - v[1] >= self.cfg.report_resend_s:
                v[1] = t
                self.reports_resent += 1
                self.send_report(v[0])
        for r in outbox.drain():
            self.unacked[r.seq] = [r, t]
            self.reports_sent += 1
            self.send_report(r)

    def send_report(self, r: EnemyReport) -> None:
        m, mav = self.m, self.mav
        lat, lon, alt = self.frame.to_global(r.pos)
        olat, olon, oalt = self.frame.to_global(r.own_pos)
        jpg = r.snapshot_jpg or b""
        mav.track_report_info_send(usec(r.t), r.seq, olat, olon, oalt, r.agl, r.confidence, len(jpg),
                                   TRACKER_STATES.get(r.state, 0))
        h = math.radians(r.heading_deg)
        active = r.state in ("TRACKING", "DETECTED")
        mav.camera_tracking_geo_status_send(
            m.CAMERA_TRACKING_STATUS_FLAGS_ACTIVE if active else m.CAMERA_TRACKING_STATUS_FLAGS_IDLE,
            lat, lon, alt, r.sigma_m, math.nan, r.speed * math.cos(h), r.speed * math.sin(h), -r.climb, math.nan,
            r.range_m, math.atan2(math.sin(h), math.cos(h)), math.nan, 1)
        if jpg:
            n = self.cfg.image_payload
            parts = [jpg[i:i + n] for i in range(0, len(jpg), n)]
            hgt, wid = (r.snapshot.shape[:2] if r.snapshot is not None else (0, 0))
            mav.data_transmission_handshake_send(m.MAVLINK_DATA_STREAM_IMG_JPEG, len(jpg), wid, hgt, len(parts), n, 80)
            for k, p in enumerate(parts):
                mav.encapsulated_data_send(k, list(p.ljust(253, b"\0")))

    # ------------------------------------------------------------------ receiving

    def handle(self, msg, t: float, from_peer: bool) -> None:
        typ = msg.get_type()
        if typ == "HEARTBEAT":
            if msg.get_srcSystem() == self.cfg.base_sysid:
                self.base_hb_t = t
        elif typ == "ACOUSTIC_BEARING":
            self._bearing(msg, t)
        elif typ in ("COMMAND_LONG", "COMMAND_INT"):
            if msg.target_system in (0, self.cfg.drone_sysid):
                res = self._command(msg, t)
                self.mav.command_ack_send(msg.command, res, 0, 0, msg.get_srcSystem(), msg.get_srcComponent())
        elif typ == "TRACK_REPORT_ACK":
            if self.unacked.pop(msg.report_seq, None) is not None:
                self.reports_acked += 1

    def _bearing(self, msg, t: float) -> None:
        m = self.m
        if msg.source == m.SFHACK_ACOUSTIC_SOURCE_OWN_DRONE:
            self.own_bearings.append((t, msg.azimuth, msg.elevation, msg.azimuth_sigma))
            return
        if msg.source != m.SFHACK_ACOUSTIC_SOURCE_ENEMY:
            return
        self.bearings_rx += 1
        tb = msg.time_usec * 1e-6
        fh = getattr(self.pilot, "first_heard_t", None)
        self.bus.publish(topics.UPLINK_BEARING, BearingUpdate(tb, float(msg.azimuth), tb if fh is None else fh,
                                                              float(msg.azimuth_sigma)))

    def _command(self, msg, t: float) -> int:
        m, p = self.m, self.pilot
        cmd = msg.command
        if cmd == m.MAV_CMD_NAV_TAKEOFF:  # launch: param4 = yaw = the bearing to fly out on, param2 = s since heard
            az = float(msg.param4)
            if not (math.isfinite(az) and p.on_ground and p.state == "WAIT"):
                return m.MAV_RESULT_TEMPORARILY_REJECTED
            if self.first_heard_t is None:  # the base's first-heard time is fixed for the mission
                age = float(msg.param2)
                self.first_heard_t = t - (age if math.isfinite(age) else 0.0)
            if hasattr(p, "first_heard_t"):
                ok = p.launch(az, t, self.first_heard_t)
            else:
                ok = p.launch(az, t)
            return m.MAV_RESULT_ACCEPTED if ok else m.MAV_RESULT_DENIED
        if cmd == m.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            if hasattr(p, "recall"):
                p.recall(t)
                return m.MAV_RESULT_ACCEPTED
            return m.MAV_RESULT_UNSUPPORTED
        if cmd == m.MAV_CMD_DO_SET_ROI_LOCATION:
            if msg.get_type() == "COMMAND_INT":
                self.roi = self.frame.to_world(msg.x, msg.y, msg.z)
            else:
                self.roi = self.frame.to_world(int(msg.param5 * 1e7), int(msg.param6 * 1e7), msg.param7)
            if hasattr(p, "look_at"):  # the tracker has no "look there" behaviour yet
                p.look_at(self.roi, t)
                return m.MAV_RESULT_ACCEPTED
            return m.MAV_RESULT_UNSUPPORTED
        return m.MAV_RESULT_UNSUPPORTED
