"""Moving-source bearing tracker (sfhack/acoustic/track.py) on synthetic 16-ch audio at 44.1 kHz."""
import numpy as np

from sfhack.acoustic.array import Mounting
from sfhack.acoustic.doa import angle_diff_deg
from sfhack.acoustic.synth import engine_signal, plane_wave
from sfhack.acoustic.track import BearingTracker, TrackConfig, TrackPipeline

FS = 44100
BLOCK = int(0.1 * FS)
LEARN_S = 5.0
MOUNT = Mounting(30.0, True)


def _blocks(duration_s, sources, noise=0.05, seed=0, el=15.0):
    """Yield (t_end, (BLOCK, 16)) blocks. sources: list of (mono, compass_az(t) or None when silent). Each block
    is a plane wave computed on a padded slice of the mono signal, so fractional delays stay exact."""
    rng = np.random.default_rng(seed)
    pad = 1024
    for k in range(int(duration_s * FS) // BLOCK):
        i0 = k * BLOCK
        x = noise * rng.standard_normal((BLOCK, 16))
        for mono, az_fn in sources:
            az = az_fn((i0 + BLOCK / 2) / FS)
            if az is None:
                continue
            lo, hi = max(i0 - pad, 0), min(i0 + BLOCK + pad, len(mono))
            pw = plane_wave(mono[lo:hi], MOUNT.from_compass(az), el, fs=FS)
            x += pw[i0 - lo:i0 - lo + BLOCK]
        yield (i0 + BLOCK) / FS, x


def _drone(duration_s, seed=0):
    return 0.3 * engine_signal(duration_s, FS, rpm=5400, seed=seed)   # f0 180 Hz, harmonics + hiss to 5 kHz


def _run(blocks):
    pipe = TrackPipeline(TrackConfig(fs=FS, learn_s=LEARN_S), MOUNT)
    steps = [s for t, x in blocks if (s := pipe.process(x, t)) is not None]
    return pipe, steps


def _truth_sweep(t):
    return None if t < LEARN_S + 1.0 else 2.0 * (t - LEARN_S - 1.0)    # 0 -> 90 deg at 2 deg/s


def test_follows_moving_source():
    dur = LEARN_S + 1.0 + 45.0
    _, steps = _run(_blocks(dur, [(_drone(dur), _truth_sweep)]))
    msgs = [s.message for s in steps if s.message is not None]
    assert msgs and all(m["detected"] for m in msgs)
    assert [m["seq"] for m in msgs] == list(range(1, len(msgs) + 1))
    for key in ("t", "az_deg", "el_deg", "sigma_az_deg", "sigma_el_deg", "label", "confidence", "array_enu",
                "az_rate_dps", "level_db"):
        assert key in msgs[0]
    assert msgs[0]["label"] == "drone"
    late = [m for m in msgs if m["t"] > LEARN_S + 1.0 + 5.0]
    err = np.array([angle_diff_deg(m["az_deg"], _truth_sweep(m["t"])) for m in late])
    rms = float(np.sqrt(np.mean(err ** 2)))
    assert len(late) > 150 and rms < 8.0, rms
    rates = np.array([m["az_rate_dps"] for m in late])
    assert np.median(rates) > 0.5, np.median(rates)          # compass az increasing (clockwise)


def test_silence_sends_nothing():
    _, steps = _run(_blocks(LEARN_S + 10.0, []))
    assert steps and not any(s.message for s in steps)
    assert sum(s.hit for s in steps) <= 1


def test_burst_does_not_yank_bearing_and_lost_message():
    dur = LEARN_S + 12.0
    on_until = LEARN_S + 10.0
    burst_t = LEARN_S + 6.05
    rng = np.random.default_rng(3)
    burst = np.zeros(int(dur * FS))
    k = int(burst_t * FS)
    burst[k:k + int(0.2 * FS)] = 1.0 * rng.standard_normal(int(0.2 * FS))     # ~+10 dB over the drone
    src = [(_drone(dur), lambda t: 45.0 if LEARN_S + 1.0 <= t < on_until else None),
           (burst, lambda t: 200.0)]
    _, steps = _run(_blocks(dur, src))
    est = [s.message for s in steps if s.message and s.message["detected"]]
    after = [m for m in est if m["t"] > LEARN_S + 3.0]
    assert after
    worst = max(abs(angle_diff_deg(m["az_deg"], 45.0)) for m in after)
    assert worst < 10.0, worst
    assert any(s.state is not None and s.state.accepted is False for s in steps)   # the burst was gated out
    lost = [s.message for s in steps if s.message and not s.message["detected"]]
    assert len(lost) == 1 and lost[0]["t"] > on_until and set(lost[0]) == {"detected", "seq", "t"}
    assert lost[0]["seq"] == est[-1]["seq"] + 1


def test_tracker_reinit_after_consistent_outliers():
    trk = BearingTracker(TrackConfig())
    for i in range(10):
        trk.update(0.2 * i, 10.0, 20.0)
    states = [trk.update(2.0 + 0.2 * i, 120.0, 20.0) for i in range(4)]
    assert [s.accepted for s in states[:3]] == [False, False, False]
    assert states[3].accepted and abs(angle_diff_deg(states[3].az_deg, 120.0)) < 1.0
