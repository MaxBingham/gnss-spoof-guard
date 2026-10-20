"""Tiny synchronous in-process pub/sub.

Every message carries its own sim timestamp (`msg.t`); the bus adds a global sequence number.
Callbacks run immediately in the publisher's thread, so the closed loop stays deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

Callback = Callable[[str, Any], None]

# Topic names (one place, so typos fail loudly at import time rather than silently at runtime).
IMU = "imu"
BARO = "baro"
MAG = "mag"
GNSS = "gnss"
CAMERA_DOWN = "camera/down"
CAMERA_FRONT = "camera/front"
FIX = "fix"
POSE = "pose"
HEALTH = "health"
SETPOINT = "setpoint"
MOTOR = "motor"
DISTURBANCE = "disturbance"


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callback]] = defaultdict(list)
        self._all: list[Callback] = []
        self._latest: dict[str, Any] = {}
        self.seq = 0

    def subscribe(self, topic: str, cb: Callable[[Any], None]) -> None:
        self._subs[topic].append(lambda _topic, msg: cb(msg))

    def subscribe_all(self, cb: Callback) -> None:
        """Receive (topic, msg) for every topic, e.g. for the recorder."""
        self._all.append(cb)

    def publish(self, topic: str, msg: Any) -> None:
        if not hasattr(msg, "t"):
            raise TypeError(f"message on {topic!r} has no timestamp field 't'")
        self.seq += 1
        self._latest[topic] = msg
        for cb in self._subs.get(topic, ()):
            cb(topic, msg)
        for cb in self._all:
            cb(topic, msg)

    def latest(self, topic: str) -> Any | None:
        return self._latest.get(topic)
