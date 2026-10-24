"""Base <-> drone MAVLink 2 link (sfhack/comms): dialect, signing, telemetry, commands, buffered enemy reports."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pymavlink")

from sfhack import bus as topics  # noqa: E402
from sfhack.bus import Bus  # noqa: E402
from sfhack.comms.dialect import load  # noqa: E402
from sfhack.comms.loopback import LoopbackPair  # noqa: E402
from sfhack.comms.mavlink_io import Frame  # noqa: E402
from sfhack.comms.signing_check import run as signing_run  # noqa: E402
from sfhack.config import TrackerConfig  # noqa: E402
from sfhack.messages import PoseEstimate  # noqa: E402
from sfhack.mission.enemy_report import EnemyReport, ReportOutbox, snapshot  # noqa: E402


class FakePilot:
    """The parts of TrackerPilot the drone link uses."""

    def __init__(self) -> None:
        self.state = "WAIT"
        self.pad = np.array([3.0, -4.0])
        self.tcfg = TrackerConfig()
        self.outbox = ReportOutbox()
        self.first_heard_t: float | None = None
        self.launches: list[tuple] = []
        self.recalls: list[float] = []

    @property
    def on_ground(self) -> bool:
        return self.state in ("WAIT", "LANDED")

    def pose(self, *, report: bool = True) -> PoseEstimate:
        c, s = math.cos(0.3), math.sin(0.3)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return PoseEstimate(0.0, np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, -0.5]), 0.3, np.eye(6), R,
                            np.zeros(3))

    def battery_left(self, t: float) -> float:
        return self.tcfg.battery_s / 2

    def launch(self, az: float, t: float, first_heard_t: float | None = None) -> bool:
        self.launches.append((az, t, first_heard_t))
        self.first_heard_t = first_heard_t
        self.state = "TAKEOFF"
        return True

    def recall(self, t: float) -> None:
        self.recalls.append(t)
        self.state = "RETURN"


@pytest.fixture
def link():
    up = {"v": True}
    pilot, bus = FakePilot(), Bus()
    pair_ = LoopbackPair(pilot, bus, gate=lambda: up["v"], fixed_ports=False, qgc=False)
    pair_.up, pair_.pilot, pair_.bus = up, pilot, bus
    yield pair_
    pair_.close()


def report(seq: int, t: float, rng) -> EnemyReport:
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    crop, jpg = snapshot(img, (50, 40, 70, 60))
    return EnemyReport(t=t, seq=seq, state="TRACKING", pos=np.array([120.5, -33.25, 41.0]), agl=38.5, speed=4.2,
                       heading_deg=250.0, climb=-0.3, confidence=0.8, sigma_m=2.5, range_m=16.0,
                       own_pos=np.array([110.0, -20.0, 44.0]), snapshot=crop, snapshot_jpg=jpg)


def test_dialect_has_the_custom_messages() -> None:
    m = load()
    assert m.MAVLINK_MSG_ID_ACOUSTIC_BEARING == 54400
    assert m.MAVLINK_MSG_ID_TRACK_REPORT_INFO == 54401 and m.MAVLINK_MSG_ID_TRACK_REPORT_ACK == 54402
    assert m.MAVLINK_MSG_ID_CAMERA_TRACKING_GEO_STATUS == 276  # common.xml is included


def test_frame_round_trip() -> None:
    f = Frame(49.25, 31.9, 150.0)
    p = np.array([-812.3, 433.7, 42.0])
    assert np.allclose(f.to_world(*f.to_global(p)), p, atol=0.02)


def test_signing_drops_forged_commands() -> None:
    r = signing_run(verbose=False)
    assert r["signed (shared key)"] == {"executed": True, "acked": True, "rejected": 0}
    for k in ("unsigned", "signed, wrong key", "replayed (captured)"):
        assert r[k] == {"executed": False, "acked": False, "rejected": 1}, k


def test_launch_bearing_recall_and_telemetry(link) -> None:
    base, drone, pilot = link.base, link.drone, link.pilot
    got = []
    link.bus.subscribe(topics.UPLINK_BEARING, got.append)
    link.step(1.0)
    assert base.drone_state == "WAIT" and not base.drone_airborne
    base.launch(1.0, 60.0, 1.0)
    link.exchange(1.0)
    m = base.m
    assert base.acks[m.MAV_CMD_NAV_TAKEOFF][1] == m.MAV_RESULT_ACCEPTED
    assert pilot.launches == [(60.0, 1.0, 1.0)]
    link.step(1.01)
    assert base.drone_state == "TAKEOFF" and base.drone_airborne  # HEARTBEAT on the state change
    assert base.send_bearing(1.02, 61.5, 4.0)
    link.exchange(1.02)
    assert len(got) == 1 and got[0].az_deg == pytest.approx(61.5) and got[0].sigma_deg == pytest.approx(4.0)
    assert got[0].first_heard_t == 1.0
    pos = base.drone_pos()
    assert np.allclose(pos, [10.0, 20.0, 30.0], atol=0.02)
    att = base.telemetry["ATTITUDE"]
    assert att.yaw == pytest.approx(math.pi / 2 - 0.3, abs=1e-6) and abs(att.roll) < 1e-6  # ENU yaw -> NED
    assert base.telemetry["SYS_STATUS"].battery_remaining == 50
    link.up["v"] = False  # recall while the link is down: retried until it gets through
    base.recall(2.0)
    for k in range(10):
        link.step(2.0 + 0.1 * k)
    assert pilot.recalls == []
    link.up["v"] = True
    for k in range(10):
        link.step(3.0 + 0.1 * k)
    assert len(pilot.recalls) == 1 and m.MAV_CMD_NAV_RETURN_TO_LAUNCH not in base.pending


def test_reports_buffered_while_down_and_delivered_once(link) -> None:
    rng = np.random.default_rng(0)
    base, drone, pilot = link.base, link.drone, link.pilot
    pilot.state = "TRACKING"
    t = 0.0
    for _ in range(3):
        link.step(t)
        t += 0.1
    r1 = report(1, t, rng)
    pilot.outbox.put(r1)
    link.step(t)
    assert [r.seq for r in base.received] == [1] and not drone.unacked
    got = base.received[0]
    assert np.allclose(got.pos, r1.pos, atol=0.02) and np.allclose(got.own_pos, r1.own_pos, atol=0.02)
    assert got.speed == pytest.approx(4.2, abs=1e-4) and got.heading_deg == pytest.approx(250.0, abs=1e-3)
    assert got.climb == pytest.approx(-0.3, abs=1e-5) and got.state == "TRACKING" and got.t == pytest.approx(r1.t)
    assert got.snapshot_jpg == r1.snapshot_jpg and got.snapshot.shape == r1.snapshot.shape
    # link down: the drone keeps sending until the base heartbeat times out, then buffers in the outbox
    link.up["v"] = False
    for seq in (2, 3):
        pilot.outbox.put(report(seq, t, rng))
        link.step(t)
        t += 0.1
    while t < 6.0:
        link.step(t)
        t += 0.1
    assert [r.seq for r in base.received] == [1]
    assert [r.seq for r in pilot.outbox.queue] == [2, 3] and not drone.unacked  # back in the outbox
    pilot.outbox.put(report(4, t, rng))
    link.up["v"] = True
    for _ in range(30):
        link.step(t)
        t += 0.1
    assert [r.seq for r in base.received] == [1, 2, 3, 4] and len(base.rx_t) == 4
    assert not drone.unacked and not len(pilot.outbox)
    assert drone.rejected == {} and base.rejected == {}


def test_decisions_arrive_as_statustext(link) -> None:
    from sfhack.messages import Decision

    texts = ["TAKEOFF: bearing 060.0 deg", "x" * 100, "LOST: no detection for 2 s; predicting on the base bearing"]
    for k, text in enumerate(texts):
        link.bus.publish(topics.DECISION, Decision(1.0 + k, text))
    link.step(1.0)
    assert [t for _, t in link.base.texts] == texts
