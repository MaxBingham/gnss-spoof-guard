"""Clap -> BearingEstimate on synthetic 16-ch audio at the Linux UMA-16 rate (44.1 kHz)."""
import numpy as np

from sfhack.acoustic.array import Mounting
from sfhack.acoustic.clap import ClapBearing, ClapConfig
from sfhack.acoustic.doa import angle_diff_deg
from sfhack.acoustic.synth import plane_wave

FS = 44100


def _claps(compass_az, times, mount, duration_s=6.0, el=10.0, noise=2e-4, seed=0):
    rng = np.random.default_rng(seed)
    mono = np.zeros(int(duration_s * FS))
    for t in times:
        k = int(t * FS)
        n = int(0.03 * FS)
        mono[k:k + n] += 0.3 * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.004 * FS))
    x = plane_wave(mono, mount.from_compass(compass_az), el, fs=FS)
    return x + noise * rng.standard_normal(x.shape)


def _run(x, det, block_s=0.1):
    n = int(block_s * FS)
    claps = []
    for i in range(0, len(x) - n + 1, n):
        c = det.process(x[i:i + n], (i + n) / FS)
        if c is not None:
            claps.append(c)
    return claps, det.bearing(len(x) / FS)


def test_three_claps_give_the_bearing():
    mount = Mounting(30.0, True)
    det = ClapBearing(ClapConfig(fs=FS), mount, array_enu=(0.0, 0.0, 1.2))
    claps, b = _run(_claps(135.0, [2.0, 3.0, 4.0], mount), det)
    assert len(claps) == 3
    for c in claps:
        assert abs(angle_diff_deg(c.compass_az_deg, 135.0)) < 3.0
    assert b.label == "clap" and abs(angle_diff_deg(b.az_deg, 135.0)) < 3.0
    assert 5.0 <= b.sigma_az_deg < 10.0 and b.array_enu == (0.0, 0.0, 1.2)


def test_mics_down_mirrors():
    mount = Mounting(0.0, False)
    _, b = _run(_claps(300.0, [2.0], mount), ClapBearing(ClapConfig(fs=FS), mount))
    assert abs(angle_diff_deg(b.az_deg, 300.0)) < 3.0


def test_noise_alone_gives_nothing():
    mount = Mounting(0.0, True)
    claps, b = _run(_claps(0.0, [], mount), ClapBearing(ClapConfig(fs=FS), mount))
    assert claps == [] and b is None


def test_one_stray_clap_does_not_pull_the_bearing():
    mount = Mounting(0.0, True)
    det = ClapBearing(ClapConfig(fs=FS), mount)
    x = _claps(135.0, [1.5, 2.5, 3.5], mount, duration_s=6.0) + _claps(250.0, [4.5], mount, duration_s=6.0, seed=1)
    claps, b = _run(x, det)
    assert len(claps) == 4
    assert abs(angle_diff_deg(b.az_deg, 135.0)) < 3.0


def test_echo_half_a_second_later_is_one_clap():
    mount = Mounting(0.0, True)
    claps, _ = _run(_claps(90.0, [2.0, 2.5], mount), ClapBearing(ClapConfig(fs=FS), mount))
    assert len(claps) == 1


def test_old_claps_are_forgotten():
    mount = Mounting(0.0, True)
    det = ClapBearing(ClapConfig(fs=FS, memory_s=2.0), mount)
    _run(_claps(90.0, [1.5], mount, duration_s=5.0), det)
    assert det.bearing(5.0) is None
