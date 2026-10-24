"""1-D surface energy balance: surface temperature of common materials over a clear, calm summer day/night.

Research tool for docs/NIGHT_RESEARCH.md (not part of the flight stack, imports nothing from sfhack).
Each material is a column of layers (explicit finite differences) driven by clear-sky sun, clear-sky
longwave from the air (Brutsaert emissivity), convection to the air and, for vegetation, evaporation.
Run 6 days to a periodic state, report day 6. Prints a table and writes figures + CSV next to this file.

    uv run --no-project --with numpy --with matplotlib python docs/night_research/surface_energy_balance.py

Clock time: both Berkeley (PDT, 122.3 W) and central Ukraine (EEST, ~28 E) have solar noon ~13:15 in July,
so `clock = solar + 1.25 h` is used for both.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIGMA = 5.670e-8
OUT = Path(__file__).resolve().parent
CLOCK_OFFSET_H = 1.25  # clock time - local solar time (both sites, mid July)


@dataclass
class Site:
    name: str
    lat_deg: float
    doy: int
    t_max_c: float  # air temperature (2 m) of the hot day
    t_min_c: float  # air temperature minimum at sunrise
    dewpoint_c: float
    cloud_night: float = 0.0  # 0 clear .. 1 overcast after sunset
    h_day: float = 10.0  # convective coefficient W/m^2K, light breeze, unstable day
    h_night: float = 5.0  # calm night (free convection + residual turbulence)
    deep_c: float | None = None  # ground temperature at the bottom of the column


@dataclass
class Layer:
    thick_m: float
    k: float  # W/mK
    rho_c: float  # J/m^3K
    dz_m: float = 0.005


@dataclass
class Material:
    name: str
    albedo: float
    emissivity: float
    layers: list[Layer]
    evap_frac_day: float = 0.0  # share of positive net radiation that goes into evaporation
    bottom: str = "ground"  # "ground" (deep soil temperature) or "indoor" (fixed 24 C)
    skin_c: float = 0.0  # extra areal heat capacity of the surface node (J/m^2K)
    note: str = ""


@dataclass
class Result:
    t_clock_h: np.ndarray
    ts_c: np.ndarray
    t_app_c: np.ndarray  # radiometric (what a broadband LWIR camera reads, emissivity 1 assumed)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- forcing

def solar_geometry(site: Site, t_solar_h: np.ndarray):
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + site.doy) / 365)
    lat = math.radians(site.lat_deg)
    ha = np.radians(15.0 * (t_solar_h - 12.0))
    cosz = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * np.cos(ha)
    ws = math.degrees(math.acos(-math.tan(lat) * math.tan(decl)))  # sunset hour angle
    return np.clip(cosz, 0, None), 12 - ws / 15, 12 + ws / 15


def shortwave(site: Site, t_solar_h):
    cosz, _, _ = solar_geometry(site, t_solar_h)
    am = 1.0 / np.maximum(cosz, 0.05)
    direct = 1361.0 * 0.75 ** am * cosz
    diffuse = 0.1 * 1361.0 * cosz * 0.75 ** am  # crude clear-sky diffuse
    return np.where(cosz > 0, direct + diffuse, 0.0)


def air_temperature(site: Site, t_solar_h):
    """Parton & Logan (1981) style: sine by day (max 1.5 h after noon), exponential decay at night."""
    _, rise, sset = solar_geometry(site, np.array([12.0]))
    t = np.mod(t_solar_h, 24.0)
    day_len = sset - rise
    peak = 13.5
    tx, tn = site.t_max_c, site.t_min_c
    out = np.empty_like(t)
    day = (t >= rise) & (t <= sset)
    # day: sine from tn at sunrise to tx at peak, continuing to sunset
    out[day] = tn + (tx - tn) * np.sin(np.pi * (t[day] - rise) / (2 * (peak - rise)))
    out[day & (t > peak)] = tn + (tx - tn) * np.sin(
        np.pi / 2 + np.pi / 2 * (t[day & (t > peak)] - peak) / (day_len - (peak - rise)) * 0.55)
    t_set = tn + (tx - tn) * math.sin(np.pi / 2 + np.pi / 2 * 0.55)
    night_len = 24 - day_len
    since = np.mod(t[~day] - sset, 24.0)
    out[~day] = tn + (t_set - tn) * (np.exp(-2.2 * since / night_len) - math.exp(-2.2)) / (1 - math.exp(-2.2))
    return out


def sky_longwave(site: Site, ta_k, is_night):
    e_hpa = 6.112 * math.exp(17.67 * site.dewpoint_c / (site.dewpoint_c + 243.5))
    eps_clear = 1.24 * (e_hpa / ta_k) ** (1 / 7)  # Brutsaert (1975)
    c = np.where(is_night, site.cloud_night, 0.0)
    eps = eps_clear * (1 - c) + 0.98 * c
    return eps * SIGMA * ta_k ** 4, eps


# --------------------------------------------------------------------------- materials

def materials() -> list[Material]:
    soil_moist = Layer(1.0, 1.0, 2.5e6, 0.02)
    soil_dry_deep = Layer(1.0, 0.5, 1.6e6, 0.02)
    return [
        Material("asphalt road", 0.10, 0.95,
                 [Layer(0.10, 1.2, 2.1e6), Layer(0.30, 1.0, 1.9e6, 0.01), soil_dry_deep],
                 note="aged asphalt, albedo 0.10 (new 0.05, old 0.15-0.2)"),
        Material("concrete pavement", 0.30, 0.92,
                 [Layer(0.20, 1.6, 2.1e6), Layer(0.20, 1.0, 1.9e6, 0.01), soil_dry_deep]),
        Material("gravel/dirt road", 0.25, 0.92,
                 [Layer(0.10, 0.6, 1.5e6), soil_dry_deep], evap_frac_day=0.0,
                 note="compacted dry earth/gravel, lower conductivity than asphalt"),
        Material("bare soil (dry)", 0.25, 0.92,
                 [Layer(0.10, 0.25, 1.2e6), soil_dry_deep]),
        Material("grass (irrigated lawn)", 0.23, 0.98,
                 [Layer(0.02, 0.15, 0.8e6), soil_moist], evap_frac_day=0.7, skin_c=4000,
                 note="blades + thatch insulate the surface from the soil"),
        Material("grass (dry, summer)", 0.25, 0.97,
                 [Layer(0.02, 0.10, 0.5e6), soil_dry_deep], evap_frac_day=0.15, skin_c=3000),
        Material("roof dark, light-weight", 0.10, 0.92,
                 [Layer(0.01, 0.2, 1.5e6, 0.0025), Layer(0.10, 0.04, 3e4, 0.01)], bottom="indoor",
                 note="bitumen / shingle over insulation"),
        Material("roof white membrane", 0.70, 0.90,
                 [Layer(0.01, 0.2, 1.5e6, 0.0025), Layer(0.10, 0.04, 3e4, 0.01)], bottom="indoor"),
        Material("roof heavy (gravel on concrete)", 0.20, 0.92,
                 [Layer(0.05, 0.7, 1.7e6), Layer(0.15, 1.6, 2.1e6, 0.01), Layer(0.08, 0.04, 3e4, 0.01)],
                 bottom="indoor"),
    ]


# --------------------------------------------------------------------------- solver

def build_grid(mat: Material):
    dz, k, rc = [], [], []
    for lay in mat.layers:
        n = max(1, int(round(lay.thick_m / lay.dz_m)))
        dz += [lay.thick_m / n] * n
        k += [lay.k] * n
        rc += [lay.rho_c] * n
    return np.array(dz), np.array(k), np.array(rc)


def simulate(site: Site, mat: Material, days: int = 6) -> Result:
    dz, k, rc = build_grid(mat)
    n = len(dz)
    # conductance between neighbouring node centres (W/m^2K), heat capacity per node (J/m^2K)
    gint = 1.0 / (0.5 * dz[:-1] / k[:-1] + 0.5 * dz[1:] / k[1:])
    cap = rc * dz
    cap[0] += mat.skin_c
    g_bot = 2 * k[-1] / dz[-1]
    g_node = np.zeros(n)
    g_node[:-1] += gint
    g_node[1:] += gint
    g_node[0] += 25.0  # surface radiation + convection, linearised
    g_node[-1] += g_bot
    dt = min(0.4 * float(np.min(cap / g_node)), 30.0)
    steps_per_day = 144 * int(math.ceil(86400 / dt / 144))
    dt = 86400 / steps_per_day
    mean_air = 0.5 * (site.t_max_c + site.t_min_c)
    deep = (site.deep_c if site.deep_c is not None else mean_air + 2.0) + 273.15
    t_bottom = (24.0 + 273.15) if mat.bottom == "indoor" else deep
    temp = np.full(n, mean_air + 273.15)
    rec_t, rec_ts, rec_app = [], [], []
    rec_every = steps_per_day // 144
    t_solar = 0.0
    for d in range(days):
        for s in range(steps_per_day):
            t_solar = s * dt / 3600.0
            tsol = np.array([t_solar])
            ta = air_temperature(site, tsol)[0] + 273.15
            sw = shortwave(site, tsol)[0]
            night = sw <= 0
            ldn, eps_sky = sky_longwave(site, ta, night)
            ts = temp[0]
            rn = (1 - mat.albedo) * sw + mat.emissivity * (ldn - SIGMA * ts ** 4)
            h = site.h_night if night else site.h_day
            le = mat.evap_frac_day * max(rn, 0.0) if not night else 0.0
            q_surf = rn - h * (ts - ta) - le
            flux = gint * (temp[:-1] - temp[1:])  # positive downward
            dT = np.zeros(n)
            dT[0] += q_surf
            dT[:-1] -= flux
            dT[1:] += flux
            dT[-1] += g_bot * (t_bottom - temp[-1])
            temp = temp + dt * dT / cap
            if d == days - 1 and s % rec_every == 0:
                ts = temp[0]
                # radiometric temperature: emitted + reflected sky, as seen by a broadband 8-14 um camera
                t_app = ((mat.emissivity * SIGMA * ts ** 4 + (1 - mat.emissivity) * ldn) / SIGMA) ** 0.25
                rec_t.append(t_solar)
                rec_ts.append(ts - 273.15)
                rec_app.append(t_app - 273.15)
    t_clock = np.mod(np.array(rec_t) + CLOCK_OFFSET_H, 24.0)
    order = np.argsort(t_clock)
    return Result(t_clock[order], np.array(rec_ts)[order], np.array(rec_app)[order])


def leaf_canopy(site: Site, t_clock: np.ndarray) -> Result:
    """Tree canopy: no storage. Top leaves lose longwave to the sky on one side, convect on both;
    the camera sees a mix of sunlit/sky-facing and inner leaves, so take half the sky-facing deficit."""
    t_solar = np.mod(t_clock - CLOCK_OFFSET_H, 24.0)
    ta = air_temperature(site, t_solar) + 273.15
    sw = shortwave(site, t_solar)
    night = sw <= 0
    ldn, _ = sky_longwave(site, ta, night)
    eps, h2 = 0.97, 2 * 12.0
    rn_iso = 0.15 * 0.5 * sw + eps * (ldn - SIGMA * ta ** 4)  # absorbed SW after ~70 % goes to transpiration
    dt = rn_iso / (h2 + 4 * eps * SIGMA * ta ** 3)
    tc = ta + 0.5 * dt
    t_app = ((eps * SIGMA * tc ** 4 + (1 - eps) * ldn) / SIGMA) ** 0.25
    return Result(t_clock, tc - 273.15, t_app - 273.15)


def water_body(site: Site, t_clock: np.ndarray, bulk_c: float) -> Result:
    t_solar = np.mod(t_clock - CLOCK_OFFSET_H, 24.0)
    ta = air_temperature(site, t_solar) + 273.15
    sw = shortwave(site, t_solar)
    ldn, _ = sky_longwave(site, ta, sw <= 0)
    # skin: ~0.3 K below bulk at night (cool skin), ~+1 K warm layer in calm afternoon sun
    skin = bulk_c + np.where(sw > 0, 1.0 * sw / 900.0, -0.3) + 273.15
    eps = 0.985
    t_app = ((eps * SIGMA * skin ** 4 + (1 - eps) * ldn) / SIGMA) ** 0.25
    return Result(t_clock, skin - 273.15, t_app - 273.15)


def metal_roof_apparent(site: Site, t_clock, eps=0.25) -> Result:
    """Unpainted metal: thin, tracks a radiative/convective balance with no storage, low emissivity.
    The camera mostly sees reflected sky -> it looks COLD although its true temperature is near air."""
    t_solar = np.mod(t_clock - CLOCK_OFFSET_H, 24.0)
    ta = air_temperature(site, t_solar) + 273.15
    sw = shortwave(site, t_solar)
    night = sw <= 0
    ldn, _ = sky_longwave(site, ta, night)
    h = np.where(night, site.h_night, site.h_day)
    absorb = 0.6
    t = ta.copy()
    for _ in range(50):  # steady state (no mass): absorb*sw + eps*(ldn - sigma t^4) = h (t - ta)
        f = absorb * sw + eps * (ldn - SIGMA * t ** 4) - h * (t - ta)
        dfdt = -4 * eps * SIGMA * t ** 3 - h
        t = t - f / dfdt
    t_app = ((eps * SIGMA * t ** 4 + (1 - eps) * ldn) / SIGMA) ** 0.25
    return Result(t_clock, t - 273.15, t_app - 273.15)


SITES = {
    # Berkeley after a hot July day (normals: ~22/13 C; hot day 28 C), dew point ~11 C.
    "berkeley": Site("Berkeley, hot July day", 37.87, 196, 28.0, 14.0, 11.0),
    # Central Ukraine (Vinnytsia normals ~25/14 C), hot day 30/16 C, dew point ~14 C.
    "ukraine": Site("Central Ukraine, hot July day", 49.2, 196, 30.0, 16.0, 14.0),
}

WATER_BULK = {"berkeley": 17.0, "ukraine": 24.0}  # SF Bay ~16-18 C in summer; Ukrainian ponds/rivers ~22-25 C


def run_site(key: str, site: Site | None = None):
    site = site or SITES[key]
    results = {}
    for mat in materials():
        results[mat.name] = simulate(site, mat)
    tc = next(iter(results.values())).t_clock_h
    results["trees (canopy top)"] = leaf_canopy(site, tc)
    results["water"] = water_body(site, tc, WATER_BULK[key])
    results["metal roof (unpainted)"] = metal_roof_apparent(site, tc)
    air = air_temperature(site, np.mod(tc - CLOCK_OFFSET_H, 24.0))
    results["air (2 m)"] = Result(tc, air, air)
    return site, results


def at(res: Result, hour: float, which="app"):
    arr = res.t_app_c if which == "app" else res.ts_c
    i = int(np.argmin(np.abs(((res.t_clock_h - hour + 12) % 24) - 12)))
    return float(arr[i])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = [13, 21, 23, 3]
    rows = []
    for key in SITES:
        site, res = run_site(key)
        _, rise, sset = solar_geometry(site, np.array([12.0]))
        print(f"\n== {site.name}: sunrise {rise + CLOCK_OFFSET_H:.2f}, sunset {sset + CLOCK_OFFSET_H:.2f} (clock)")
        print(f"{'material':34s}" + "".join(f"{h:>8d}h" for h in hours) + "   (apparent LWIR temperature, C)")
        for name, r in res.items():
            vals = [at(r, h) for h in hours]
            print(f"{name:34s}" + "".join(f"{v:9.1f}" for v in vals))
            rows.append([key, name] + [round(v, 2) for v in vals])
        # crossover: hours where asphalt - grass(lawn) and asphalt - trees change sign
        a = res["asphalt road"].t_app_c
        for other in ("grass (irrigated lawn)", "trees (canopy top)", "water", "gravel/dirt road"):
            d = a - res[other].t_app_c
            sgn = np.sign(d)
            idx = np.where(np.diff(sgn) != 0)[0]
            print(f"  asphalt - {other}: min {d.min():5.1f} K, at 23h {d[int(np.argmin(np.abs(res['asphalt road'].t_clock_h - 23)))]:5.1f} K,"
                  f" crossovers at {[round(float(res['asphalt road'].t_clock_h[i]), 1) for i in idx]} h")

        fig, ax = plt.subplots(figsize=(10, 5.5))
        palette = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, (name, r) in enumerate(res.items()):
            if name.startswith("air"):
                ax.plot(r.t_clock_h, r.t_app_c, "k--", label=name, lw=1.5)
            else:
                ax.plot(r.t_clock_h, r.t_app_c, color=palette[(i * 3) % 20] if "asphalt" not in name else "k",
                        label=name, lw=2.6 if "asphalt" in name else 1.3)
        for h in (21, 23, 27 - 24):
            ax.axvline(h, color="gray", lw=0.6, ls=":")
        ax.axvspan(sset + CLOCK_OFFSET_H, 24, color="0.9", zorder=0)
        ax.axvspan(0, rise + CLOCK_OFFSET_H, color="0.9", zorder=0)
        ax.set_xlabel("clock time (h)")
        ax.set_ylabel("apparent LWIR temperature (deg C)")
        ax.set_title(f"1-D surface energy balance, clear calm night: {site.name}")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"seb_diurnal_{key}.png", dpi=120)
        plt.close(fig)

    # sensitivity at 23 h, Berkeley: wind, clouds
    print("\n== sensitivity, Berkeley 23h: asphalt - lawn / asphalt - trees (apparent, K)")
    base = SITES["berkeley"]
    variants = {
        "clear, calm (h_night 5)": base,
        "clear, breeze ~3 m/s (h_night 14)": Site(**{**base.__dict__, "h_night": 14.0}),
        "overcast after sunset": Site(**{**base.__dict__, "cloud_night": 1.0}),
        "half cloud after sunset": Site(**{**base.__dict__, "cloud_night": 0.5}),
        "cool day 22/13 C": Site(**{**base.__dict__, "t_max_c": 22.0, "t_min_c": 13.0}),
    }
    sens = []
    for label, s in variants.items():
        _, res = run_site("berkeley", s)
        a = at(res["asphalt road"], 23)
        g = at(res["grass (irrigated lawn)"], 23)
        tr = at(res["trees (canopy top)"], 23)
        rd = at(res["roof dark, light-weight"], 23)
        print(f"  {label:36s} asphalt {a:5.1f}  lawn {g:5.1f}  trees {tr:5.1f}  dark roof {rd:5.1f}"
              f"  | a-lawn {a - g:5.1f}  a-trees {a - tr:5.1f}")
        sens.append([label, round(a, 2), round(g, 2), round(tr, 2), round(rd, 2)])

    with open(OUT / "seb_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "material"] + [f"T_app_{h}h_C" for h in hours])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["sensitivity_berkeley_23h", "asphalt", "lawn", "trees", "dark_roof"])
        w.writerows(sens)


if __name__ == "__main__":
    main()
