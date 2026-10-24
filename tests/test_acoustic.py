"""Base UMA-16 path: geometry, DOA on synthetic plane waves (known truth), detection -> BearingEstimate."""
import numpy as np
import pytest

from sfhack.acoustic.array import Mounting, mic_positions
from sfhack.acoustic.bearing import BearingEstimate
from sfhack.acoustic.detect import AcousticPipeline
from sfhack.acoustic.doa import SrpPhat, angle_diff_deg
from sfhack.acoustic.synth import array_recording
from sfhack.config import AcousticConfig


def test_geometry_channel_map():
    pos = mic_positions()
    assert pos.shape == (16, 3)
    # Not row by row: channel 0 and 1 share a row (y = -0.063), channel 8 is the far corner (-x, +y).
    np.testing.assert_allclose(pos[0], [0.021, -0.063, 0.0])
    np.testing.assert_allclose(pos[8], [-0.063, 0.063, 0.0])
    assert len({tuple(p) for p in pos}) == 16
    np.testing.assert_allclose(np.ptp(pos[:, :2], axis=0), [0.126, 0.126])


def test_bearing_json_roundtrip(tmp_path):
    b = BearingEstimate(t=1.0, az_deg=-13.0, el_deg=2.0, sigma_az_deg=6.0, sigma_el_deg=10.0, label="engine",
                        confidence=0.8, array_enu=(1.0, 2.0, 3.0))
    assert b.az_deg == pytest.approx(347.0)
    b2 = BearingEstimate.load(b.save(tmp_path / "bearing.json"))
    assert b2 == b


def test_mounting_compass():
    up = Mounting(x_axis_bearing_deg=90.0, mics_up=True)       # +x east, +y north (ENU)
    assert up.to_compass(0.0) == pytest.approx(90.0)
    assert up.to_compass(90.0) == pytest.approx(0.0)           # array +y = north
    down = Mounting(x_axis_bearing_deg=90.0, mics_up=False)
    assert down.to_compass(90.0) == pytest.approx(180.0)
    for m in (up, down):
        assert m.from_compass(m.to_compass(33.0)) == pytest.approx(33.0)


@pytest.mark.parametrize("az,el,snr", [(30, 5, 10), (200, 0, 0), (300, 40, 0), (123, 2, -5)])
def test_srp_phat_synthetic(az, el, snr):
    s = SrpPhat()
    x = array_recording(az, el, duration_s=2.5, snr_db=snr, onset_s=1.0, seed=az)
    s.learn_noise(x[:48000])
    r = s.estimate(x[48000:72000])
    assert abs(angle_diff_deg(r.array_az_deg, az)) < 3.0
    if el >= 30:
        assert abs(r.array_el_deg - el) < 5.0


def test_pipeline_emits_compass_bearing():
    cfg = AcousticConfig(x_axis_bearing_deg=90.0, learn_s=4.0)
    p = AcousticPipeline(cfg, array_enu=(0.0, 0.0, 3.0))
    x = array_recording(40.0, 3.0, duration_s=9.0, snr_db=5.0, onset_s=4.0, seed=4)
    n = int(cfg.window_s * cfg.fs)
    out = None
    for k in range(0, len(x) - n + 1, n):
        r, b = p.process(x[k:k + n], t=k / cfg.fs)
        if r.learning:
            assert b is None
        if b is not None and out is None:
            out = b
    assert out is not None
    assert abs(angle_diff_deg(out.az_deg, 50.0)) < 3.0       # 90 - 40
    assert cfg.sigma_az_floor_deg <= out.sigma_az_deg < 10.0
    assert out.label == "engine"
    assert out.array_enu == (0.0, 0.0, 3.0)


def test_pipeline_quiet_no_bearing():
    cfg = AcousticConfig(learn_s=3.0)
    p = AcousticPipeline(cfg)
    x = np.random.default_rng(0).standard_normal((48000 * 8, 16))
    n = int(cfg.window_s * cfg.fs)
    assert all(p.process(x[k:k + n], t=k / cfg.fs)[1] is None for k in range(0, len(x) - n + 1, n))
