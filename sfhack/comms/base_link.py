"""Base side of the MAVLink 2 link to our drone (the UMA-16 PC; no simulator imports, so it can run on the real base).

Sends (docs/MAVLINK.md): HEARTBEAT (1 Hz, MAV_TYPE_GCS), ACOUSTIC_BEARING (the UMA-16 bearing), COMMAND_LONG launch
(MAV_CMD_NAV_TAKEOFF) / recall (MAV_CMD_NAV_RETURN_TO_LAUNCH), COMMAND_INT "look there" (MAV_CMD_DO_SET_ROI_LOCATION),
TRACK_REPORT_ACK. Commands are resent every `cmd_retry_s` until the drone's COMMAND_ACK.
Receives the drone's telemetry (latest of each type in `telemetry`, mission state from HEARTBEAT.custom_mode),
STATUSTEXT, COMMAND_ACK and the enemy reports, reassembled into `EnemyReport`s in `received` (sorted by report
number, the receive time in `rx_t`) and acknowledged.
QGroundControl: every packet received from the drone is forwarded as is (still signed) to UDP `qgc_port` (14550).
"""

from __future__ import annotations

import bisect
import math

import cv2
import numpy as np

from ..config import CommsConfig
from ..mission.enemy_report import EnemyReport
from .mavlink_io import FLIGHT_STATES, ON_GROUND, TRACKER_STATES, Frame, MavEndpoint, ned_to_enu, usec

STATE_OF = {v: k for k, v in TRACKER_STATES.items()}


class BaseLink(MavEndpoint):
    def __init__(self, cfg: CommsConfig | None = None, gate=None, port: int | None = None,
                 peer: tuple[str, int] | None = None, qgc: bool | None = None) -> None:
        cfg = self.cfg = cfg or CommsConfig()
        super().__init__(cfg.base_sysid, cfg.base_compid, (cfg.host, cfg.base_port if port is None else port),
                         peer or (cfg.host, cfg.drone_port), cfg.signing_key() if cfg.signing else None, link_id=2,
                         gate=gate, name="base")
        if cfg.qgc_forward if qgc is None else qgc:
            self.taps.append((cfg.host, cfg.qgc_port))
        self.frame = Frame(cfg.origin_lat, cfg.origin_lon, cfg.origin_alt_m)
        self.m = self.mod
        self.drone_hb_t: float | None = None
        self.drone_state: str | None = None
        self.telemetry: dict[str, object] = {}
        self.texts: list[tuple[float, str]] = []
        self._text_parts: dict[int, list[bytes]] = {}
        self.received: list[EnemyReport] = []
        self.rx_t: list[float] = []
        self._have: set[int] = set()
        self._rx: dict | None = None  # report being reassembled
        self.pending: dict[int, dict] = {}  # command id -> {msg, next}
        self.acks: dict[int, tuple[float, int]] = {}  # command id -> (t, MAV_RESULT) of the latest COMMAND_ACK
        self.bearings_sent = 0
        self._next_hb = -math.inf

    # ------------------------------------------------------------------ state

    def link_alive(self, t: float) -> bool:
        return self.drone_hb_t is not None and t - self.drone_hb_t <= self.cfg.link_timeout_s

    @property
    def drone_airborne(self) -> bool:
        return self.drone_state is not None and self.drone_state not in ON_GROUND

    def drone_pos(self) -> np.ndarray | None:
        """The drone's own position estimate (world ENU, onboard datum) from GLOBAL_POSITION_INT."""
        g = self.telemetry.get("GLOBAL_POSITION_INT")
        return None if g is None else self.frame.to_world(g.lat, g.lon, g.alt / 1000.0)

    # ------------------------------------------------------------------ sending

    def update(self, t: float) -> None:
        self.now = t
        m = self.m
        if t >= self._next_hb:
            self._next_hb = t + 1.0 / self.cfg.heartbeat_hz
            self.mav.heartbeat_send(m.MAV_TYPE_GCS, m.MAV_AUTOPILOT_INVALID, 0, 0, m.MAV_STATE_ACTIVE)
        for cmd, p in list(self.pending.items()):
            if t >= p["next"]:
                p["next"] = t + self.cfg.cmd_retry_s
                if hasattr(p["msg"], "confirmation"):  # COMMAND_LONG: 0 first, +1 per retransmission
                    p["msg"].confirmation = min(p["tries"], 255)
                p["tries"] += 1
                self.send(p["msg"])

    def _command(self, t: float, msg) -> None:
        self.acks.pop(msg.command, None)
        self.pending[msg.command] = {"msg": msg, "next": -math.inf, "tries": 0}
        self.update(t)

    def launch(self, t: float, az_deg: float, first_heard_t: float) -> None:
        """Enemy heard: take off and fly out on `az_deg` (MAV_CMD_NAV_TAKEOFF, param4 = yaw = the bearing, param2 =
        seconds since the enemy was first heard). The result arrives as COMMAND_ACK (`acks`)."""
        m = self.m
        self._command(t, m.MAVLink_command_long_message(
            self.cfg.drone_sysid, self.cfg.drone_compid, m.MAV_CMD_NAV_TAKEOFF, 0,
            math.nan, max(t - first_heard_t, 0.0), math.nan, az_deg % 360.0, math.nan, math.nan, math.nan))

    def recall(self, t: float) -> None:
        m = self.m
        self._command(t, m.MAVLink_command_long_message(
            self.cfg.drone_sysid, self.cfg.drone_compid, m.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, 0, 0, 0, 0, 0))

    def look_at(self, t: float, xyz) -> None:
        """'Look there' (MAV_CMD_DO_SET_ROI_LOCATION). COMMAND_INT: degE7 integers keep the position to ~1 cm,
        COMMAND_LONG's float32 degrees only to ~0.5 m."""
        m = self.m
        lat, lon, alt = self.frame.to_global(xyz)
        self._command(t, m.MAVLink_command_int_message(
            self.cfg.drone_sysid, self.cfg.drone_compid, m.MAV_FRAME_GLOBAL, m.MAV_CMD_DO_SET_ROI_LOCATION, 0, 0,
            0, 0, 0, 0, lat, lon, alt))

    def cancel(self, cmd: int) -> None:
        self.pending.pop(cmd, None)

    def send_bearing(self, t: float, az_deg: float, sigma_deg: float, source: int | None = None,
                     elevation: float = math.nan, el_sigma: float = math.nan, confidence: float = math.nan,
                     array_id: int = 0) -> bool:
        """ACOUSTIC_BEARING to the drone. True if it went out on the air (the link was up)."""
        m = self.m
        n = self.tx_peer
        self.mav.acoustic_bearing_send(usec(t), az_deg % 360.0, elevation, sigma_deg, el_sigma, confidence,
                                       m.SFHACK_ACOUSTIC_SOURCE_ENEMY if source is None else source, array_id)
        sent = self.tx_peer > n
        self.bearings_sent += sent
        return sent

    # ------------------------------------------------------------------ receiving

    def handle(self, msg, t: float, from_peer: bool) -> None:
        if msg.get_srcSystem() != self.cfg.drone_sysid:
            return
        typ = msg.get_type()
        self.telemetry[typ] = msg
        if typ == "HEARTBEAT":
            self.drone_hb_t = t
            self.drone_state = FLIGHT_STATES[msg.custom_mode] if msg.custom_mode < len(FLIGHT_STATES) else "?"
        elif typ == "COMMAND_ACK":
            self.acks[msg.command] = (t, msg.result)
            if msg.result != self.m.MAV_RESULT_IN_PROGRESS:
                self.pending.pop(msg.command, None)
        elif typ == "STATUSTEXT":
            self._statustext(msg, t)
        elif typ == "TRACK_REPORT_INFO":
            self._rx = {"info": msg, "geo": None, "img": bytearray(), "next": 0}
        elif typ == "CAMERA_TRACKING_GEO_STATUS":
            if self._rx is not None and self._rx["geo"] is None:
                self._rx["geo"] = msg
                self._report_done(t)
        elif typ == "ENCAPSULATED_DATA":
            rx = self._rx
            if rx is not None and rx["geo"] is not None and msg.seqnr == rx["next"]:
                rx["next"] += 1
                rx["img"] += bytes(msg.data)
                self._report_done(t)

    def _statustext(self, msg, t: float) -> None:
        text = msg.text if isinstance(msg.text, str) else msg.text.decode("ascii", "replace")
        if not msg.id:
            self.texts.append((t, text))
            return
        parts = self._text_parts.setdefault(msg.id, [])
        parts.append(text)
        if len(text) < 50:
            self.texts.append((t, "".join(self._text_parts.pop(msg.id))))

    def _report_done(self, t: float) -> None:
        rx = self._rx
        info, geo = rx["info"], rx["geo"]
        if len(rx["img"]) < info.image_size:
            return
        self._rx = None
        seq = int(info.report_seq)
        self.mav.track_report_ack_send(seq)
        if seq in self._have:  # a resend whose first ACK was lost
            return
        jpg = bytes(rx["img"][:info.image_size])
        snap = None
        if jpg:
            bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            snap = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        v = ned_to_enu(geo.vel_n, geo.vel_e, geo.vel_d)
        r = EnemyReport(
            t=info.time_usec * 1e-6, seq=seq, state=STATE_OF.get(info.tracker_state, "OTHER"),
            pos=self.frame.to_world(geo.lat, geo.lon, geo.alt), agl=float(info.target_agl),
            speed=float(math.hypot(v[0], v[1])), heading_deg=math.degrees(math.atan2(v[0], v[1])) % 360.0,
            climb=float(v[2]), confidence=float(info.confidence), sigma_m=float(geo.h_acc), range_m=float(geo.dist),
            own_pos=self.frame.to_world(info.own_lat, info.own_lon, info.own_alt), snapshot=snap, snapshot_jpg=jpg)
        self._have.add(seq)
        k = bisect.bisect([x.seq for x in self.received], seq)
        self.received.insert(k, r)
        self.rx_t.insert(k, t)
