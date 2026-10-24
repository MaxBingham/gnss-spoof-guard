"""1-D surface energy balance: diurnal surface temperature curves per material.

Each variant (material x direct-sun exposure x sky view factor) is a layered slab, discretized into nodes
from 2 mm at the surface growing to the bottom of the stack, integrated implicitly (backward Euler, batched
Thomas solve over all variants at once) for several clear, windless days until the cycle repeats. The
surface balance per unit area:

    C0 dT0/dt = (1 - albedo) (exposure S_dir + svf S_diff) + eps L_in - eps sigma T0^4 - h (T0 - T_air) - LE + G
    L_in      = svf eps_sky sigma T_air^4 + (1 - svf) sigma (T_air + 2 K)^4     (walls / trees fill the rest)
    LE        = evap * max(available energy, 0)                                    (daytime evaporation)

Apparent (radiometric, emissivity-1) temperature: T_app = ((eps sigma T0^4 + (1 - eps) L_in) / sigma)^(1/4):
what a thermal camera reports. Low-emissivity metal looks cold because it reflects the cold sky.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .materials import INDOOR_K, MATERIALS, Climate, Material

SIGMA = 5.670374e-8
S0 = 1361.0


def solar(hour_solar: np.ndarray, lat_deg: float, doy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(direct beam on a horizontal surface W/m2, diffuse W/m2, sun elevation rad, sun azimuth rad from north cw)."""
    h = np.asarray(hour_solar, float)
    dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365.0)
    lat = math.radians(lat_deg)
    ha = np.radians(15.0 * (h - 12.0))
    sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * np.cos(ha)
    el = np.arcsin(np.clip(sin_el, -1, 1))
    cos_az = (math.sin(dec) - np.sin(el) * math.sin(lat)) / np.maximum(np.cos(el) * math.cos(lat), 1e-9)
    az = np.arccos(np.clip(cos_az, -1, 1))
    az = np.where(ha > 0, 2 * math.pi - az, az)
    up = sin_el > 0.01
    am = 1.0 / np.maximum(sin_el, 0.01)
    dni = np.where(up, S0 * 0.7 ** (am ** 0.678), 0.0)  # Meinel clear-sky beam
    direct = dni * np.maximum(sin_el, 0.0)
    diffuse = np.where(up, 0.12 * direct + 15.0 * np.maximum(sin_el, 0.0), 0.0)
    return direct, diffuse, el, az


def air_temp(hour_solar: np.ndarray, c: Climate) -> np.ndarray:
    """Tmin at t_min_h, Tmax at t_max_h, half-cosine ramps in between (K)."""
    h = np.mod(np.asarray(hour_solar, float), 24.0)
    tmin, tmax = c.tmin_c + 273.15, c.tmax_c + 273.15
    rise = c.t_max_h - c.t_min_h
    fall = 24.0 - rise
    t_up = (h - c.t_min_h) % 24.0
    up = t_up <= rise
    f_up = (1 - np.cos(math.pi * t_up / rise)) / 2
    f_dn = (1 + np.cos(math.pi * (t_up - rise) / fall)) / 2
    return np.where(up, tmin + (tmax - tmin) * f_up, tmin + (tmax - tmin) * f_dn)


def sky_emissivity(dew_c: float) -> float:
    """Clear sky (Berdahl & Fromberg 1982)."""
    t = dew_c / 100.0
    return 0.711 + 0.56 * t + 0.73 * t * t


@dataclass
class Curves:
    """Surface temperature over one solar day for every variant: hours (T,), t_surf / t_app (V, T)."""

    hours: np.ndarray
    t_surf: np.ndarray
    t_app: np.ndarray
    t_air: np.ndarray
    exposures: np.ndarray
    svfs: np.ndarray
    n_mat: int

    def at(self, hour_solar: float) -> np.ndarray:
        """(n_mat, n_exp, n_svf) apparent temperature at that solar hour (linear in time)."""
        h = hour_solar % 24.0
        v = np.array([np.interp(h, self.hours, row, period=24.0) for row in self.t_app])
        return v.reshape(self.n_mat, len(self.exposures), len(self.svfs))

    def air_at(self, hour_solar: float) -> float:
        return float(np.interp(hour_solar % 24.0, self.hours, self.t_air, period=24.0))


def _discretize(m: Material, n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node thickness, conductivity, heat capacity: 2 mm at the top, x1.3 per node, layer edges respected."""
    dz, k, c = [], [], []
    for thick, kk, rc in m.layers:
        z, d = 0.0, 0.002 if not dz else max(dz[-1] * 1.3, 0.002)
        d = min(d, thick)
        while z < thick - 1e-9:
            d = min(d, thick - z)
            if thick - z - d < 0.3 * d:
                d = thick - z
            dz.append(d), k.append(kk), c.append(rc)
            z += d
            d *= 1.3
    while len(dz) < n_max:  # pad with negligible extra nodes so all variants share one array shape
        dz.append(1e-4), k.append(k[-1]), c.append(c[-1])
    return np.array(dz[:n_max]), np.array(k[:n_max]), np.array(c[:n_max])


def simulate(lat_deg: float, climate: Climate, materials: list[Material] = MATERIALS,
             exposures: tuple[float, ...] = (0.0, 0.5, 1.0), svfs: tuple[float, ...] = (0.3, 0.65, 1.0),
             days: int = 5, dt: float = 120.0) -> Curves:
    nm, ne, ns = len(materials), len(exposures), len(svfs)
    # node counts differ per material; use the max and pad
    n_nodes = max(len(_discretize(m, 999)[0][_discretize(m, 999)[0] > 2e-4]) for m in materials)
    N = n_nodes + 1
    props = [_discretize(m, N) for m in materials]
    V = nm * ne * ns
    mi = np.repeat(np.arange(nm), ne * ns)
    ex = np.tile(np.repeat(np.asarray(exposures, float), ns), nm)
    sv = np.tile(np.asarray(svfs, float), nm * ne)
    dz = np.stack([props[i][0] for i in mi])
    kk = np.stack([props[i][1] for i in mi])
    cc = np.stack([props[i][2] for i in mi])
    alb = np.array([materials[i].albedo for i in mi])
    eps = np.array([materials[i].emissivity for i in mi])
    evap = np.array([materials[i].evap for i in mi])
    hmul = np.array([materials[i].h_mult for i in mi])
    bottom = [materials[i].bottom for i in mi]

    cap = cc * dz  # J/m2/K per node
    cond = 1.0 / (dz[:, :-1] / (2 * kk[:, :-1]) + dz[:, 1:] / (2 * kk[:, 1:]))  # W/m2/K between nodes
    t_mean = (climate.tmax_c + climate.tmin_c) / 2 + 273.15
    t_deep = np.array([INDOOR_K if b == "indoor" else t_mean + 1.5 for b in bottom])
    k_bot = np.array([0.0 if b == "adiabatic" else 1.0 for b in bottom]) * kk[:, -1] / (dz[:, -1] / 2)
    eps_sky = sky_emissivity(climate.dew_c)

    T = np.repeat(t_deep[:, None], N, axis=1).astype(float)
    T[:, :] = t_mean
    steps_day = int(round(86400 / dt))
    hours = np.arange(steps_day) * dt / 3600.0
    d_dir, d_dif, _, _ = solar(hours, lat_deg, climate.doy)
    t_air_day = air_temp(hours, climate)
    out_s = np.zeros((V, steps_day))
    out_a = np.zeros((V, steps_day))
    for day in range(days):
        for s in range(steps_day):
            ta = t_air_day[s]
            l_in = sv * eps_sky * SIGMA * ta ** 4 + (1 - sv) * SIGMA * (ta + 2.0) ** 4
            tp = T[:, 0]
            q_sw = (1 - alb) * (ex * d_dir[s] + sv * d_dif[s])
            avail = q_sw + eps * (l_in - SIGMA * tp ** 4)
            le = evap * np.maximum(avail, 0.0)
            dtemp = np.maximum(tp - ta, 0.0)
            h = hmul * (2.5 + 1.7 * np.cbrt(dtemp))
            lin = 4 * eps * SIGMA * tp ** 3
            # tridiagonal system  a T[i-1] + b T[i] + c T[i+1] = d
            a = np.zeros((V, N))
            b = cap / dt
            c = np.zeros((V, N))
            d = cap / dt * T
            b = b.copy()
            b[:, :-1] += cond
            b[:, 1:] += cond
            a[:, 1:] = -cond
            c[:, :-1] = -cond
            b[:, 0] += lin + h
            d[:, 0] += q_sw + eps * l_in - eps * SIGMA * tp ** 4 + lin * tp + h * ta - le
            b[:, -1] += k_bot
            d[:, -1] += k_bot * t_deep
            # batched Thomas
            cp = np.zeros((V, N))
            dp = np.zeros((V, N))
            cp[:, 0] = c[:, 0] / b[:, 0]
            dp[:, 0] = d[:, 0] / b[:, 0]
            for i in range(1, N):
                den = b[:, i] - a[:, i] * cp[:, i - 1]
                cp[:, i] = c[:, i] / den
                dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / den
            T[:, -1] = dp[:, -1]
            for i in range(N - 2, -1, -1):
                T[:, i] = dp[:, i] - cp[:, i] * T[:, i + 1]
            if day == days - 1:
                out_s[:, s] = T[:, 0]
                out_a[:, s] = ((eps * SIGMA * T[:, 0] ** 4 + (1 - eps) * l_in) / SIGMA) ** 0.25
    return Curves(hours, out_s, out_a, t_air_day, np.asarray(exposures, float), np.asarray(svfs, float), nm)
