"""demos/51_counter_drone.py: listener contract parsing, audio gating / silence, intruder limits, headless missions."""
import importlib
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))
cd = importlib.import_module("51_counter_drone")

WALK = "0:0,10:0,40:30,55:30"   # hold, walk 1 deg/s, hold; silent after 55 s


def flat(x, y):
    return 0.0 * np.asarray(x, float)


def contract(az, seq, **kw):
    return json.dumps({"t": 100.0 + seq * 0.2, "az_deg": az, "el_deg": 10.0, "sigma_az_deg": 5.0, "sigma_el_deg": 15.0,
                       "label": "drone", "confidence": 0.8, "array_enu": [0, 0, 3], "az_rate_dps": 0.5,
                       "level_db": -31.5, "detected": True, "seq": seq, **kw})


# ------------------------------------------------------------------------------------------------ parsing
def test_parse_contract_and_legacy():
    m = cd.parse_message(contract(123.4, 7))
    assert m["detected"] and m["az"] == pytest.approx(123.4) and m["sigma"] == 5.0
    assert m["rate"] == 0.5 and m["level_db"] == -31.5 and m["confidence"] == 0.8 and m["seq"] == 7
    lost = cd.parse_message('{"detected": false, "seq": 8, "t": 101.8}')
    assert lost == {"detected": False, "seq": 8, "t": 101.8}
    clap = cd.parse_message(json.dumps({"t": 1.0, "az_deg": 370.0, "el_deg": 30, "sigma_az_deg": 4.0,
                                        "label": "clap", "confidence": 0.9, "array_enu": None}))
    assert clap["az"] == pytest.approx(10.0) and clap["seq"] is None and clap["rate"] is None
    assert cd.parse_message('{"t": 23.0, "compass_az": 124.9, "el": 34.9}')["az"] == pytest.approx(124.9)
    assert cd.parse_message("45.5")["az"] == 45.5
    for bad in ("", "{", "[1, 2]", '{"az_deg": "x"}', '{"az_deg": NaN}', '{"t": 1}', "abc"):
        assert cd.parse_message(bad) is None, bad


def test_udp_track_drops_out_of_order_seq():
    u = cd.UdpTrack(0)
    for s in (1, 2, 4, 3, 4, 5):
        u.push(contract(10.0, s))
    assert [m["seq"] for m in u.pop_all()] == [1, 2, 4, 5] and u.n_out_of_order == 2
    u.push(contract(10.0, 1000))
    u.push(contract(10.0, 1))   # far below: the listener restarted
    assert [m["seq"] for m in u.pop_all()] == [1000, 1]
    assert not u.push("garbage") and u.n_bad == 1


def test_detected_false_silences_immediately():
    a, u = cd.Audio(), cd.UdpTrack(0)
    for k in range(10):
        u.push(contract(50.0, k))
    u.push('{"detected": false, "seq": 10, "t": 102.0}')
    t = 0.0
    for m in u.pop_all():
        t += 0.2
        if m["detected"]:
            a.update(m["az"], m["sigma"], t, m["rate"], m["level_db"], m["confidence"], "udp")
            assert a.hearing(t)
        else:
            a.lose(t, "udp")
    assert not a.hearing(t) and a.status(t).startswith("SILENT")
    a.update(50.0, 5.0, t + 0.2)   # sound back right away
    assert a.hearing(t + 0.2)


def test_audio_gates_single_outliers_but_follows_a_persistent_jump():
    a = cd.Audio()
    t = 0.0
    for k in range(50):
        t += 0.2
        st = a.update(200.0 if k % 10 == 5 else 100.0 + np.random.default_rng(k).normal(0, 3), 5.0, t)
        if k % 10 == 5:
            assert st == "gated"
    assert abs(cd.adiff(a.az, 100.0)) < 3 and a.n_gated == 5 and a.hearing(t)
    states = []
    for _ in range(10):   # the source really moved to 220
        t += 0.2
        states.append(a.update(220.0, 5.0, t))
    assert "jump" in states and states.index("jump") <= 6
    assert abs(cd.adiff(a.az, 220.0)) < 2


def test_audio_rate_and_dropout_extrapolation():
    a = cd.Audio()
    t = 0.0
    for k in range(60):   # 2 deg/s, the listener also sends its rate
        t += 0.2
        a.update(10.0 + 2.0 * t, 5.0, t, rate=2.0)
    assert a.rate == pytest.approx(2.0, abs=0.3)
    a.lose(t)
    assert a.predict(t + 2.0) == pytest.approx((a.az + 2 * a.rate) % 360, abs=1e-6)
    assert a.predict(t + 10.0) == pytest.approx((a.az + cd.EXTRAP_S * a.rate) % 360, abs=1e-6)
    assert a.update(10.0 + 2.0 * (t + 2.0), 5.0, t + 2.0) == "resume"
    a.lose(t + 2.0)
    assert a.update(250.0, 5.0, t + 4.0) == "gated"   # back somewhere else: wait for it to persist
    assert not a.hearing(t + 4.0)


def test_intruder_speed_limit_hover_and_return_elsewhere():
    a, it = cd.Audio(), cd.Intruder(500.0, 80.0, 20.0, flat)
    t, dt, nxt = 0.0, 0.01, 0.0
    v_max, pos_silent = 0.0, None

    def az_at(t):
        return None if 30 <= t < 40 else 0.0 if t < 30 else 180.0   # silence, then back on the other side

    while t < 120:
        if t >= nxt:
            nxt += 0.2
            az = az_at(t)
            if az is None:
                if a.hearing(t):
                    a.lose(t)
            else:
                st = a.update(az + np.random.default_rng(int(t * 5)).normal(0, 6), 6.0, t)
                if st != "gated":
                    it.hear(a.az, t, st in ("new", "jump"))
        p0 = it.pos[:2].copy() if it.pos is not None else None
        it.update(t, dt, a.hearing(t))
        if p0 is not None:
            v_max = max(v_max, float(np.linalg.norm(it.pos[:2] - p0)) / dt)
        if 34 <= t < 34 + dt:
            pos_silent = it.pos[:2].copy()
        if 39 <= t < 39 + dt:
            assert np.linalg.norm(it.pos[:2] - pos_silent) < 1.0   # hovering
        t += dt
    assert v_max <= 20.0 + 1e-6
    assert it.v_max >= 19.0   # it flew to the new bearing at its max speed
    assert np.linalg.norm(it.pos[:2] - cd.az_to_dir(180.0) * 500.0) < 50.0


def test_hunter_never_beyond_max_out_and_lands():
    """No camera hits: fly the audio ray out to MAX_OUT, never further, come back, land (kinematic, no sim)."""
    a, h = cd.Audio(), cd.Hunter(flat)
    t, dt, r_max = 0.0, 0.02, 0.0
    while t < 600 and h.state != "LANDED":
        if abs(t / 0.2 - round(t / 0.2)) < 1e-6:
            a.update(45.0, 5.0, t)
        h.update(h.sp.copy(), t, dt, a)
        r_max = max(r_max, float(np.linalg.norm(h.sp[:2])))
        t += dt
    assert h.state == "LANDED" and cd.MAX_OUT - 20 < r_max <= cd.MAX_OUT


# ------------------------------------------------------------------------------------------------ missions
@pytest.fixture(scope="module")
def farm():
    m, info = cd.build("farm")
    return m, info


def headless(farm, *argv):
    m, info = farm
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    args = cd.parser().parse_args(["--headless", "--range", "250", "--walk", WALK, "--max-t", "200", *argv])
    return cd.run_headless(m, d, info, args, quiet=True)


def test_clean_walk_found_and_landed(farm):
    r = headless(farm)
    assert r["found"] and r["state"] == "LANDED" and r["landing_err_m"] < 1.0, r
    assert r["max_out_m"] <= cd.MAX_OUT


def test_intruder_on_the_sound_line():
    a, it = cd.Audio(), cd.Intruder(500.0, 80.0, 20.0, flat)
    t, dt = 0.0, 0.01
    while t < 60:
        if abs(t / 0.2 - round(t / 0.2)) < 1e-6:
            a.update(10.0 + 1.0 * t + np.random.default_rng(int(t * 5)).normal(0, 4), 5.0, t)
            it.hear(a.az, t)
        it.update(t, dt, a.hearing(t), a.az)
        assert np.linalg.norm(it.pos[:2] - cd.az_to_dir(a.az) * 500.0) < 1e-6
        t += dt


def test_messy_walk_found_and_landed(farm):
    r = headless(farm, "--walk-noise", "6", "--outliers", "0.05", "--dropouts", "2:20", "--dropout-loss-msg",
                 "--intruder-free")
    assert r["found"] and r["state"] == "LANDED" and r["landing_err_m"] < 1.0, r
    assert r["gated"] > 0 and r["loss_msgs"] >= 2
    assert r["intruder_max_speed_mps"] <= 20.0 + 1e-6


def test_silence_never_takes_off(farm):
    r = headless(farm, "--walk", "none")
    assert not r["launched"] and r["state"] == "WAIT" and r["max_out_m"] < 25.0, r


def test_replay_clap_log(farm, tmp_path):
    lines = [{"t": 23.0 + 0.4 * k, "compass_az": 120.0 + 0.1 * k, "el": 30.0} for k in range(20)]
    p = tmp_path / "claps.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    replay = cd.load_replay(p)
    assert replay[0][0] == 0.0 and replay[-1][0] == pytest.approx(7.6) and math.isclose(replay[0][1]["az"], 120.0)
    m, info = farm
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    args = cd.parser().parse_args(["--headless", "--range", "250", "--replay", str(p), "--max-t", "12"])
    r = cd.run_headless(m, d, info, args, quiet=True)
    assert r["launched"] and r["msgs"] == 20
