"""Synthetic speaker walk around the UMA-16 (sfhack/acoustic/walk_sim.py): SRP-PHAT recovers the walked bearing."""
import numpy as np
import pytest

from sfhack.acoustic.walk_sim import WalkConfig, doa_check, drone_signal, simulate_walk, to_int16


@pytest.mark.parametrize("north, mics_up, reverb", [(0.0, True, False), (92.7, False, True)])
def test_short_walk_bearings(north, mics_up, reverb):
    cfg = WalkConfig(az_from=20.0, az_to=35.0, rate=3.0, pre_s=1.5, hold_s=1.0, hold_end_s=0.5, snr_db=10.0,
                     north=north, mics_up=mics_up, reverb=reverb)
    assert cfg.duration_s == pytest.approx(8.0)
    x, truth = simulate_walk(cfg, drone_signal(4.0, cfg.fs, seed=1))
    assert x.shape == (int(8.0 * cfg.fs), 16)
    assert truth[0]["source_on"] is False and truth[-1]["compass_az"] == pytest.approx(35.0)
    r = doa_check(to_int16(x).astype(float) / 32768.0, cfg)
    assert len(r) >= 14
    assert np.ptp(r[:, 1]) > 10.0                     # the windows really cover the walk
    assert np.max(np.abs(r[:, 3])) < 6.0


def test_drone_sound_loops_and_band():
    fs = 44100
    x = drone_signal(4.0, fs, seed=0, hf_frac=0.55)
    assert np.max(np.abs(x)) == pytest.approx(1.0)
    step = np.mean(np.abs(np.diff(x)))
    assert abs(x[0] - x[-1]) < 6 * step               # periodic: no click at the loop seam
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    assert P[(f >= 1000) & (f <= 4000)].sum() / P[1:].sum() == pytest.approx(0.55, abs=0.03)
