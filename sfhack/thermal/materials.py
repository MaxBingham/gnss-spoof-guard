"""Surface material classes of the thermal world model and their physical parameters.

Each class is a 1-D layered slab (top layer first) with a surface albedo, a broadband LWIR emissivity, an
evaporative fraction (share of the available daytime energy that leaves as latent heat) and a multiplier on
the convective exchange coefficient (rough canopies exchange heat with the air much faster than a flat road).
Values are textbook / literature numbers (Oke, "Boundary Layer Climates"; Hendel et al. 2016 for asphalt;
ASHRAE for roofs; see docs/THERMAL_MODEL.md for sources and how they compare with measured night contrasts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

INDOOR_K = 297.0  # roofs: the bottom of the stack is the (air-conditioned) building interior


@dataclass(frozen=True)
class Material:
    name: str
    layers: tuple[tuple[float, float, float], ...]  # (thickness m, conductivity W/m/K, volumetric heat J/m3/K)
    albedo: float
    emissivity: float
    evap: float = 0.0  # evaporative fraction of the positive available energy (daytime)
    h_mult: float = 1.0  # convective exchange multiplier
    bottom: str = "deep"  # "deep": fixed deep-soil temperature, "indoor": INDOOR_K, "adiabatic"
    prior_sigma_K: float = 1.5  # how well the onboard prior knows this class at night (1 std)


SOIL = (0.6, 0.9, 1.7e6)
MATERIALS: list[Material] = [
    Material("soil", ((0.6, 0.4, 1.4e6),), 0.22, 0.93, evap=0.05, prior_sigma_K=2.0),
    Material("asphalt", ((0.08, 1.2, 2.0e6), (0.30, 1.0, 1.8e6), SOIL), 0.09, 0.95, prior_sigma_K=1.5),
    Material("asphalt_minor", ((0.05, 1.1, 2.0e6), (0.25, 0.9, 1.7e6), SOIL), 0.13, 0.95, prior_sigma_K=1.8),
    Material("concrete", ((0.12, 1.2, 1.9e6), (0.20, 0.8, 1.6e6), SOIL), 0.35, 0.92, prior_sigma_K=1.8),
    Material("pavers", ((0.08, 1.0, 1.7e6), (0.10, 0.5, 1.4e6), SOIL), 0.25, 0.92, prior_sigma_K=2.0),
    Material("grass", ((0.02, 0.15, 0.8e6), (0.6, 1.0, 2.4e6)), 0.23, 0.98, evap=0.6, h_mult=1.3, prior_sigma_K=1.5),
    Material("dry_grass", ((0.03, 0.07, 0.3e6), (0.6, 0.35, 1.3e6)), 0.26, 0.96, evap=0.08, h_mult=1.3,
             prior_sigma_K=2.0),
    Material("tree", ((0.01, 0.5, 1.5e6),), 0.15, 0.97, evap=0.7, h_mult=20.0, bottom="adiabatic", prior_sigma_K=1.2),
    Material("water", ((1.0, 30.0, 4.18e6),), 0.06, 0.98, evap=0.5, prior_sigma_K=1.5),
    Material("dirt_road", ((0.6, 0.6, 1.6e6),), 0.24, 0.93, evap=0.03, prior_sigma_K=2.0),
    # flat roofs: membrane over insulation over the deck (most commercial roofs: little mass above the
    # insulation, so they cool below air at night), ballasted / concrete-topped roofs keep their heat
    Material("roof_dark", ((0.01, 0.17, 1.0e6), (0.03, 0.5, 1.5e6), (0.08, 0.04, 4e4), (0.15, 1.4, 2.0e6)), 0.10, 0.93,
             bottom="indoor", prior_sigma_K=3.5),
    Material("roof_light", ((0.002, 0.2, 1.5e6), (0.03, 0.5, 1.5e6), (0.08, 0.04, 4e4), (0.15, 1.4, 2.0e6)), 0.55, 0.90,
             bottom="indoor", prior_sigma_K=3.5),
    Material("roof_gravel", ((0.05, 0.7, 1.6e6), (0.15, 1.4, 2.0e6), (0.08, 0.04, 4e4)), 0.20, 0.93,
             bottom="indoor", prior_sigma_K=3.5),
    Material("roof_metal", ((0.001, 45.0, 3.6e6), (0.10, 0.035, 4e4)), 0.40, 0.25, bottom="indoor",
             prior_sigma_K=6.0),
    Material("turf", ((0.03, 0.15, 1.0e6), (0.15, 0.6, 1.6e6), SOIL), 0.10, 0.93, prior_sigma_K=2.5),
    Material("parking", ((0.05, 1.1, 2.0e6), (0.25, 0.9, 1.7e6), SOIL), 0.13, 0.95, prior_sigma_K=1.8),
]
CLASS_NAMES: list[str] = [m.name for m in MATERIALS]
CLS: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}
ROOF_CLASSES = tuple(CLS[n] for n in ("roof_dark", "roof_light", "roof_gravel", "roof_metal"))
PAVED_CLASSES = tuple(CLS[n] for n in ("asphalt", "asphalt_minor", "concrete", "pavers", "parking"))


@dataclass(frozen=True)
class Climate:
    """Clear, windless day -> night. Hours are local SOLAR time inside the model."""

    doy: int = 196  # mid July
    tmax_c: float = 30.0  # "after a hot day"
    tmin_c: float = 16.0
    t_max_h: float = 15.0
    t_min_h: float = 5.5
    dew_c: float = 12.0
    extra: dict[str, float] = field(default_factory=dict)


SEASONS = {
    "summer": Climate(),
    "spring": Climate(doy=105, tmax_c=20.0, tmin_c=8.0, t_min_h=6.0, dew_c=6.0),
    "autumn": Climate(doy=288, tmax_c=21.0, tmin_c=9.0, t_min_h=6.5, dew_c=7.0),
    "winter": Climate(doy=15, tmax_c=9.0, tmin_c=1.0, t_max_h=14.0, t_min_h=7.0, dew_c=-2.0),
}
