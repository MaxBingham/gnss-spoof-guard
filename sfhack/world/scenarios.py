"""V3 scenarios: where to fly (A, B by lat/lon), the world to build there, and the sensors / mission
parameters that go with it. Everything is plain data; `build_v3.py` turns a scenario into a world
package under $SFHACK_DATA/v3/worlds/<name>.

  s1     urban hills, San Francisco: Aquatic Park -> over Russian Hill -> Washington Square (~1.6 km)
  s2     natural terrain, Carpathian foothills (western Ukraine, south of Boryslav), ~8.4 km
  s2us   natural terrain, US alternative: Berkeley Hills (Tilden) -> Briones, ~6 km

Override A/B on the command line of the builder (`--a LAT,LON --b LAT,LON --name my_world`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass
class AltimeterConfig:
    kind: str = "laser"  # laser | radar (only changes the defaults below)
    max_range: float = 60.0  # m; beyond this the sensor returns no reading
    sigma: float = 0.05  # m, white range noise
    hz: float = 20.0
    dropout_prob: float = 0.02  # per sample, random missing reading
    outlier_prob: float = 0.005  # per sample, spurious short return (birds, wires, rain)


@dataclass
class MapMismatch:
    """How the onboard map differs from the simulated reality (the world is never the onboard array)."""

    building_height_frac: float = 0.20  # onboard building heights perturbed by +- this (uniform)
    building_drop_frac: float = 0.10  # this fraction of buildings is missing in the onboard map
    dem_coarsen_m: float = 30.0  # onboard terrain = DEM averaged to this cell size (loses small relief)
    small_relief_rms: float = 0.5  # m, small-scale relief added to the TRUE terrain only
    small_relief_scales_m: tuple[float, ...] = (8.0, 25.0)
    vertical_offset: float = 0.8  # m, onboard datum offset (absorbed by the TRN bias state)
    ortho_blur_px: float = 1.0  # onboard ortho vs world texture: blur, gamma, contrast
    ortho_gamma: float = 1.15
    ortho_contrast: float = 0.9
    ortho_change_frac: float = 0.0  # fraction of the onboard ortho area repainted ("changed since imaging")


@dataclass
class Scenario:
    name: str
    title: str
    a: tuple[float, float]  # (lat, lon)
    b: tuple[float, float]
    margin_m: float  # around the A-B box
    kind: str  # urban | natural
    dsm_mpp: float  # truth + onboard surface raster resolution
    hfield_mpp: float  # MuJoCo terrain heightfield resolution
    imagery: str  # usgs (real orthophoto) | synthetic (hillshade + land cover)
    ortho_mpp: float
    cruise_agl: float  # m above the surface (terrain + buildings) along the route
    cruise_speed: float  # m/s
    altimeter: AltimeterConfig = field(default_factory=AltimeterConfig)
    mismatch: MapMismatch = field(default_factory=MapMismatch)
    landing_patch_m: float = 0.0  # natural scenes: flat, high-res textured landing site around B (IBN there)
    landing_patch_mpp: float = 0.25
    no_fly: list[list[tuple[float, float]]] = field(default_factory=list)  # polygons of (lat, lon)
    success_radius: float = 3.0  # m, landed within this of B = success (S2: arrival accuracy, see eval)
    planner_cell_m: float = 10.0
    ibn_everywhere: bool = True  # False: IBN only inside the landing patch (natural scenes)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Scenario":
        d = dict(d)
        d["altimeter"] = AltimeterConfig(**d["altimeter"])
        mm = dict(d["mismatch"])
        mm["small_relief_scales_m"] = tuple(mm["small_relief_scales_m"])
        d["mismatch"] = MapMismatch(**mm)
        d["a"], d["b"] = tuple(d["a"]), tuple(d["b"])
        d["no_fly"] = [[tuple(p) for p in poly] for poly in d.get("no_fly", [])]
        return cls(**d)


LASER = AltimeterConfig(kind="laser", max_range=60.0, sigma=0.05, hz=20.0)
RADAR = AltimeterConfig(kind="radar", max_range=400.0, sigma=0.5, hz=20.0, dropout_prob=0.02, outlier_prob=0.002)

NATURAL_MISMATCH = MapMismatch(
    building_height_frac=0.0, building_drop_frac=0.0, dem_coarsen_m=30.0,
    small_relief_rms=2.0, small_relief_scales_m=(30.0, 90.0), vertical_offset=3.0,
)

SCENARIOS: dict[str, Scenario] = {
    "s1": Scenario(
        name="s1", title="S1 urban hills - San Francisco (Aquatic Park -> Russian Hill -> Washington Square)",
        a=(37.8058, -122.4262), b=(37.80148, -122.40982), margin_m=300.0, kind="urban",
        dsm_mpp=1.0, hfield_mpp=2.5, imagery="usgs", ortho_mpp=0.5,
        cruise_agl=35.0, cruise_speed=10.0, altimeter=LASER,
        # Lombard Street's crooked block (tourist crowds): no overflight
        no_fly=[[(37.80240, -122.41930), (37.80240, -122.41760), (37.80150, -122.41760), (37.80150, -122.41930)]],
        success_radius=3.0, planner_cell_m=10.0,
    ),
    "s2": Scenario(
        name="s2", title="S2 natural terrain - Carpathian foothills, western Ukraine (south of Boryslav)",
        a=(49.1520, 23.4000), b=(49.1180, 23.5000), margin_m=1500.0, kind="natural",
        dsm_mpp=15.0, hfield_mpp=15.0, imagery="synthetic", ortho_mpp=5.0,
        cruise_agl=120.0, cruise_speed=18.0, altimeter=RADAR, mismatch=NATURAL_MISMATCH,
        landing_patch_m=240.0, landing_patch_mpp=0.25,
        # a village between A and B: no overflight
        no_fly=[[(49.1420, 23.4380), (49.1420, 23.4520), (49.1330, 23.4520), (49.1330, 23.4380)]],
        success_radius=10.0, planner_cell_m=60.0, ibn_everywhere=False,
    ),
    "s2us": Scenario(
        name="s2us", title="S2-US natural terrain - Berkeley Hills (Tilden) -> Briones",
        a=(37.8985, -122.2475), b=(37.9150, -122.1800), margin_m=1200.0, kind="natural",
        dsm_mpp=15.0, hfield_mpp=15.0, imagery="synthetic", ortho_mpp=5.0,
        cruise_agl=120.0, cruise_speed=18.0, altimeter=RADAR, mismatch=NATURAL_MISMATCH,
        landing_patch_m=240.0, landing_patch_mpp=0.25,
        success_radius=10.0, planner_cell_m=60.0, ibn_everywhere=False,
    ),
}

HEAVY_MISMATCH_URBAN = MapMismatch(
    building_height_frac=0.40, building_drop_frac=0.25, dem_coarsen_m=60.0, small_relief_rms=0.5,
    small_relief_scales_m=(8.0, 25.0), vertical_offset=2.5, ortho_blur_px=2.0, ortho_gamma=1.4,
    ortho_contrast=0.7, ortho_change_frac=0.15,
)
HEAVY_MISMATCH_NATURAL = replace(NATURAL_MISMATCH, dem_coarsen_m=90.0, vertical_offset=8.0, ortho_blur_px=2.0,
                                 ortho_gamma=1.4, ortho_contrast=0.7, ortho_change_frac=0.15)


# Steady wind of the simulated weather (harness): speed uniform in [0.4, 1] x this, random direction per seed,
# plus turbulence (config.DisturbanceConfig). Natural scenes fly higher and in the open.
WIND_MAX: dict[str, float] = {"urban": 3.0, "natural": 5.0}


def heavy_mismatch(sc: Scenario) -> MapMismatch:
    """The 'map mismatch heavy' onboard variant (truth stays the same; only the onboard map gets worse)."""
    base = HEAVY_MISMATCH_URBAN if sc.kind == "urban" else HEAVY_MISMATCH_NATURAL
    return replace(base, small_relief_rms=sc.mismatch.small_relief_rms, small_relief_scales_m=sc.mismatch.small_relief_scales_m)


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r} (have {list(SCENARIOS)})")
    return SCENARIOS[name]
