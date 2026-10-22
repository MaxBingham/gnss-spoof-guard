"""Degradation levers (sfhack.degradation): the Conditions API, each lever's model, and its simulator hook."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from sfhack import bus as topics
from sfhack.config import default_config
from sfhack.degradation import Conditions, ConditionsMsg, weather_visuals
from sfhack.degradation import models as lever
from sfhack.degradation.cli import add_cli_args, conditions_from_args, parse_schedule
from sfhack.degradation.conditions import GnssConditions, WeatherConditions, WindConditions
from sfhack.runner import FlightSystem

# ---------------------------------------------------------------------------------------- Conditions API


def test_conditions_defaults_are_nominal() -> None:
    c = Conditions()
    assert c.gnss.mode == "off" and c.weather.kind == "clear" and c.wind.speed == 0.0
    assert not c.camera.lost and not c.lidar.lost
    assert c.summary() == "nominal"


def test_conditions_set_update_shorthand_roundtrip() -> None:
    c = Conditions().update("gnss=drag:2@45,weather=fog:0.6,wind=6@270,camera=0.3,lidar=lost")
    c.update(wind__gust=2, **{"camera.lost": "true"})
    assert (c.gnss.mode, c.gnss.drag_rate, c.gnss.drag_heading_deg) == ("drag", 2.0, 45.0)
    assert (c.weather.kind, c.weather.intensity) == ("fog", 0.6)
    assert (c.wind.speed, c.wind.direction_deg, c.wind.gust) == (6.0, 270.0, 2.0)
    assert c.camera.lost and c.camera.drop_rate == 0.3 and c.lidar.lost
    assert c.get("gnss.mode") == "drag"
    assert Conditions.from_dict(c.to_dict()) == c
    assert c.copy() == c and c.copy() is not c
    jump = Conditions().update("gnss=jump:80@180")
    assert (jump.gnss.jump_m, jump.gnss.jump_heading_deg) == (80.0, 180.0)


def test_conditions_validation_rejects_bad_values_and_keeps_old() -> None:
    c = Conditions()
    for path, value in (("gnss.mode", "spoof"), ("weather.kind", "snow"), ("weather.intensity", 1.5),
                        ("camera.drop_rate", -0.1), ("wind.speed", -1)):
        with pytest.raises(ValueError):
            c.set(path, value)
    assert c == Conditions()
    with pytest.raises(KeyError):
        c.set("gnss.bogus", 1)
    with pytest.raises(KeyError):
        c.set("radar.lost", True)


def test_cli_and_schedule() -> None:
    ap = argparse.ArgumentParser()
    add_cli_args(ap)
    args = ap.parse_args(["--gnss", "jump:30@90", "--weather", "rain:0.4", "--wind", "5@180", "--camera-drop", "0.2",
                          "--lidar-dropout", "0.5", "--cond", "wind.rotation_deg_s=3", "--cond-at", "30:gnss=jam"])
    c = conditions_from_args(args)
    assert c.gnss.mode == "jump" and c.gnss.jump_m == 30.0
    assert c.weather.kind == "rain" and c.weather.intensity == 0.4
    assert c.wind.speed == 5.0 and c.wind.direction_deg == 180.0 and c.wind.rotation_deg_s == 3.0
    assert c.camera.drop_rate == 0.2 and c.lidar.dropout == 0.5
    sched = parse_schedule(args.cond_at + ["90:gnss=ok;60:wind=0"])
    assert [t for t, _ in sched] == [30.0, 60.0, 90.0]
    with pytest.raises(ValueError):
        parse_schedule("10:gnss=bogus")


def test_config_carries_initial_conditions() -> None:
    cfg = default_config()
    cfg.conditions.update("gnss=ok")
    assert default_config().conditions.gnss.mode == "off"  # no shared mutable default
    assert cfg.conditions.gnss.mode == "ok"


# ---------------------------------------------------------------------------------------- lever models


def _gnss_run(lever_cfg: GnssConditions, n: int = 500, seed: int = 0) -> np.ndarray:
    g = lever.GnssModel(np.random.default_rng(seed))
    out = []
    for k in range(n):
        fix = g.sample(k * 0.2, np.zeros(3), lever_cfg)
        out.append(fix.pos[:2] if fix is not None else [np.nan, np.nan])
    return np.array(out)


def test_gnss_ok_noise_is_realistic() -> None:
    err = _gnss_run(GnssConditions(mode="ok"), n=3000)
    assert np.all(np.isfinite(err))
    assert np.abs(err.mean(axis=0)).max() < 0.5
    assert 1.3 < err.std(axis=0).mean() < 2.0  # 1.5 m white + 0.7 m slow


def test_gnss_jam_and_off_publish_nothing() -> None:
    assert np.all(np.isnan(_gnss_run(GnssConditions(mode="jam"), n=20)))
    assert np.all(np.isnan(_gnss_run(GnssConditions(mode="off"), n=20)))


def test_gnss_drag_slides_at_rate_and_heading() -> None:
    pos = _gnss_run(GnssConditions(mode="drag", drag_rate=2.0, drag_heading_deg=90.0), n=251)  # 50 s
    assert pos[-1, 0] == pytest.approx(100.0, abs=6.0)  # east
    assert abs(pos[-1, 1]) < 6.0
    north = _gnss_run(GnssConditions(mode="drag", drag_rate=1.0, drag_heading_deg=0.0), n=101)
    assert north[-1, 1] == pytest.approx(20.0, abs=6.0)


def test_gnss_jump_offsets_at_once_and_ok_clears() -> None:
    g = lever.GnssModel(np.random.default_rng(1))
    jump = GnssConditions(mode="jump", jump_m=50.0, jump_heading_deg=180.0)
    first = g.sample(0.0, np.zeros(3), jump)
    assert first.pos[1] == pytest.approx(-50.0, abs=6.0)  # south, on the very first fix
    back = g.sample(0.2, np.zeros(3), GnssConditions(mode="ok"))
    assert np.linalg.norm(back.pos[:2]) < 8.0


def test_weather_effect_scales_with_intensity_and_kind() -> None:
    assert lever.weather_effect(WeatherConditions("clear", 1.0)).pixel_sigma == 0.0
    assert lever.weather_effect(WeatherConditions("fog", 0.0)).range_dropout == 0.0
    lo, hi = (lever.weather_effect(WeatherConditions("rain", i)) for i in (0.3, 0.9))
    assert hi.pixel_sigma > lo.pixel_sigma and hi.range_dropout > lo.range_dropout
    assert hi.visibility_m < lo.visibility_m and hi.streaks > lo.streaks
    fog = lever.weather_effect(WeatherConditions("fog", 0.9))
    assert fog.visibility_m < hi.visibility_m  # fog hides more than rain
    v = weather_visuals(WeatherConditions("dust", 0.7))
    assert v.kind == "dust" and 0 < v.fog_start_m < v.fog_end_m == pytest.approx(v.visibility_m) and 0 < v.haze <= 1


def test_degrade_image_fog_kills_contrast_rain_adds_noise() -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    fog = lever.degrade_image(img, lever.weather_effect(WeatherConditions("fog", 0.8)), 40.0, rng)
    assert fog.std() < 0.3 * img.std()
    flat = np.full((120, 160, 3), 100, np.uint8)
    rain = lever.degrade_image(flat, lever.weather_effect(WeatherConditions("rain", 0.8)), 40.0, rng)
    assert rain.std() > 10
    clear = lever.weather_effect(WeatherConditions())
    assert lever.degrade_image(img, clear, 40.0, rng) is img


def test_degrade_range_dropout_noise_and_loss() -> None:
    rng = np.random.default_rng(0)
    clear = lever.weather_effect(WeatherConditions())
    assert lever.degrade_range(10.0, clear, 0.0, False, rng) == 10.0
    assert math.isnan(lever.degrade_range(10.0, clear, 0.0, True, rng))
    r = np.array([lever.degrade_range(10.0, clear, 0.4, False, rng) for _ in range(4000)])
    assert np.isnan(r).mean() == pytest.approx(0.4, abs=0.03)
    e = lever.weather_effect(WeatherConditions("rain", 1.0))
    r = np.array([lever.degrade_range(10.0, e, 0.0, False, rng) for _ in range(4000)])
    assert np.isnan(r).mean() == pytest.approx(e.range_dropout, abs=0.03)
    assert np.nanstd(r) == pytest.approx(e.range_sigma, rel=0.1)


def test_wind_force_direction_rotation_and_gusts() -> None:
    wm = lever.WindModel(np.random.default_rng(0))
    w = WindConditions(speed=10.0, direction_deg=90.0)
    air = wm.air_velocity(w, 0.002)
    f = wm.force(w, air, np.zeros(3), 0.15)
    assert f[0] == pytest.approx(1.5) and abs(f[1]) < 1e-9  # toward east
    assert np.allclose(wm.force(WindConditions(), np.zeros(3), np.array([5.0, 0, 0]), 0.15), 0.0)  # calm
    rot = lever.WindModel(np.random.default_rng(0))
    wr = WindConditions(speed=5.0, direction_deg=0.0, rotation_deg_s=10.0)
    for _ in range(4500):  # 9 s
        a = rot.air_velocity(wr, 0.002)
    assert math.degrees(math.atan2(a[0], a[1])) == pytest.approx(90.0, abs=1.0)
    gm = lever.WindModel(np.random.default_rng(1))
    wg = WindConditions(speed=0.0, gust=2.0)
    g = np.array([gm.air_velocity(wg, 0.02)[:2] for _ in range(20000)])
    assert g.std(axis=0).mean() == pytest.approx(2.0, rel=0.25)


def test_camera_frame_drop_rate() -> None:
    rng = np.random.default_rng(0)
    c = Conditions().update("camera=0.25")
    e = lever.weather_effect(c.weather)
    kept = np.mean([lever.camera_frame_kept(c, e, rng) for _ in range(4000)])
    assert kept == pytest.approx(0.75, abs=0.03)


# ---------------------------------------------------------------------------------------- simulator hooks


def _collect(fs: FlightSystem, topic: str) -> list:
    out: list = []
    fs.bus.subscribe(topic, out.append)
    return out


def _run_to(fs: FlightSystem, t: float) -> None:
    while fs.t < t:
        fs.step()


def test_sim_gnss_5hz_jam_and_runtime_switch() -> None:
    fs = FlightSystem(localizer="off", conditions=Conditions().update("gnss=ok"),
                      schedule=parse_schedule("2:gnss=jam;3:gnss=jump:40@90"))
    fixes = _collect(fs, topics.GNSS)
    _run_to(fs, 4.0)
    true_x = fs.truth.state().pos[0]  # still on the ground at A
    fs.close()
    ts = np.array([m.t for m in fixes])
    assert np.sum(ts < 2.0) == pytest.approx(10, abs=1)  # 5 Hz
    assert not np.any((ts > 2.01) & (ts < 2.99))  # jammed
    late = np.array([m.pos[:2] for m in fixes if m.t > 3.0])
    assert len(late) >= 4 and late[:, 0].mean() - true_x == pytest.approx(40.0, abs=4.0)  # jump east


def test_sim_lidar_and_rangefinder_levers() -> None:
    fs = FlightSystem(localizer="off")
    scans, ranges = _collect(fs, topics.LIDAR), _collect(fs, topics.RANGE)
    _run_to(fs, 1.0)
    assert len(scans) == pytest.approx(10, abs=1) and scans[-1].ranges.shape == (8,)
    assert np.all(np.isfinite(scans[-1].ranges)) and np.all(scans[-1].ranges < 0.5)  # on the ground
    fs.update_conditions(lidar__lost=True)
    n0, r0 = len(scans), len(ranges)
    _run_to(fs, 2.0)
    assert all(np.all(np.isnan(s.ranges)) for s in scans[n0:])
    assert all(math.isnan(m.range) for m in ranges[r0:])
    fs.update_conditions("lidar=0.5")
    n0 = len(scans)
    _run_to(fs, 6.0)
    frac = np.mean([np.isnan(s.ranges).mean() for s in scans[n0:]])
    fs.close()
    assert frac == pytest.approx(0.5, abs=0.12)


def test_sim_wind_is_a_real_force() -> None:
    fs = FlightSystem(localizer="off", conditions=Conditions().update("wind=8@90"))
    fs.step()
    assert fs.sim.wind_force[0] > 1.0 and abs(fs.sim.wind_force[1]) < 1e-6
    assert fs.sim.data.xfrc_applied[fs.sim.body_id, 0] == pytest.approx(fs.sim.wind_force[0])
    fs.disturb("wind", (0.0, -5.0))  # legacy event = the same lever
    assert fs.conditions.wind.speed == pytest.approx(5.0) and fs.conditions.wind.direction_deg == pytest.approx(180.0)
    fs.close()


def test_sim_camera_loss_drop_and_weather() -> None:
    fs = FlightSystem(localizer="off", conditions=Conditions().update("camera=lost"))
    frames = _collect(fs, topics.CAMERA_DOWN)
    _run_to(fs, 0.5)
    assert frames and not any(f.image.any() for f in frames)
    fs.update_conditions("camera=1.0")
    n0 = len(frames)
    _run_to(fs, 1.5)
    assert len(frames) == n0 and fs.sim.frames_dropped >= 4
    fs.update_conditions("camera=ok")
    _run_to(fs, 2.0)
    clear = frames[-1].image.astype(float)
    fs.update_conditions("weather=fog:0.9")
    _run_to(fs, 2.5)
    foggy = frames[-1].image.astype(float)
    fs.close()
    assert clear.std() > 0
    assert foggy.mean() > clear.mean() or foggy.std() < clear.std()
    assert np.abs(foggy - np.array([0.78, 0.80, 0.82]) * 255).mean() < np.abs(clear - np.array([0.78, 0.80, 0.82]) * 255).mean()


def test_conditions_bus_topic_reaches_the_sim() -> None:
    fs = FlightSystem(localizer="off")
    fs.bus.publish(topics.CONDITIONS, ConditionsMsg(fs.t, Conditions().update("weather=dust:0.5")))
    assert fs.sim.conditions.weather.kind == "dust"
    assert fs.sim.weather_visuals().visibility_m < 1000
    fs.close()


def test_drag_spoof_pulls_naive_baseline_off_course_while_confident() -> None:
    """The naive baseline fuses every GNSS fix: a 2 m/s eastward drag moves the REAL drone west while the
    estimate stays on the planned track with a small 2-sigma."""
    fs = FlightSystem(localizer="vision", seed=0, conditions=Conditions().update("gnss=ok"),
                      schedule=parse_schedule("15:gnss=drag:2@90"))
    _run_to(fs, 40.0)
    tr, est = fs.truth.state().pos, fs.autopilot.pose()
    fs.close()
    track_x = np.interp(tr[1], [-283.86, 283.86], [-21.97, 21.97])  # the A -> B line at this northing
    assert est.pos[0] - track_x == pytest.approx(0.0, abs=5.0)  # the estimate thinks it is on track
    assert tr[0] - track_x < -35.0  # the drone is ~50 m west of it
    assert est.sigma2_horizontal() < 5.0  # ... and the estimate looks confident
