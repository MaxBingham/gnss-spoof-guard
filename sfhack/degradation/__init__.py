"""Degradation levers: GNSS (ok / jam / drag / jump), weather, wind, camera loss, lidar loss.

`Conditions` is the single source of the levers; `models` holds the lever physics the simulator applies.
Onboard code never imports this package (tests/test_truth_separation.py).
"""

from .conditions import (
    GNSS_MODES, WEATHER_KINDS, CameraConditions, Conditions, ConditionsMsg, GnssConditions, LidarConditions,
    WeatherConditions, WindConditions, compass_unit, parse_assignments,
)
from .models import WeatherVisuals, weather_visuals

__all__ = [
    "GNSS_MODES", "WEATHER_KINDS", "CameraConditions", "Conditions", "ConditionsMsg", "GnssConditions",
    "LidarConditions", "WeatherConditions", "WindConditions", "WeatherVisuals", "compass_unit",
    "parse_assignments", "weather_visuals",
]
