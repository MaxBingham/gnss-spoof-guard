# Base <-> drone link: MAVLink 2

Everything the base (UMA-16 PC) and our drone say to each other goes over **MAVLink 2** (pymavlink), **signed**,
over **UDP** in the simulation, one packet per datagram, so the same endpoints can later sit on a serial radio.

| file | side | what |
|---|---|---|
| `sfhack/comms/sfhack.xml` | both | the dialect: `common.xml` + 3 messages (below) |
| `sfhack/comms/dialect.py` | both | generates the Python bindings with mavgen on first use (cached in `$SFHACK_DATA/cache/mavlink`) |
| `sfhack/comms/mavlink_io.py` | both | `MavEndpoint`: UDP socket, encoder / parser, signing, counters; ENU <-> WGS84 `Frame` |
| `sfhack/comms/base_link.py` | base | `BaseLink`: bearings, commands, telemetry in, report reassembly + ACK, forward to QGC |
| `sfhack/comms/drone_link.py` | drone (onboard) | `DroneLink`: telemetry, reports out (buffered, resent), commands / bearings in |
| `sfhack/comms/loopback.py` | harness | both endpoints in one process, in lockstep with the sim |
| `sfhack/comms/signing_check.py` | tool | shows that forged commands are dropped |
| `sfhack/comms/gcs_listen.py` | tool | a ground-station stand-in on UDP 14550 (what QGC sees) |

```
uv run python demos/52_track_enemy.py --bearing 60                  # viewer, MAVLink (default)
uv run python demos/52_track_enemy.py --headless --bearing 60 --comms inproc    # the old in-process path
uv run python -m sfhack.comms.signing_check                        # forged / replayed commands are dropped
uv run python -m sfhack.comms.gcs_listen                           # QGC stand-in on UDP 14550
uv run python -m pytest tests/test_mavlink.py -q
```

`--comms inproc | mavlink` (default `mavlink`) picks the transport in `demos/52_track_enemy.py` /
`demos/track_scenario.py`. The tracker and the dashboard see the same objects either way (the pilot's outbox, the
base's `received` / `rx_t`); only the transport changes.

## Ports

| UDP port | who listens | traffic |
|---|---|---|
| 14560 | base (`CommsConfig.base_port`) | drone -> base |
| 14561 | drone (`CommsConfig.drone_port`) | base -> drone |
| 14550 | QGroundControl (its default) | the base forwards every packet it receives from the drone, as is |

All on `127.0.0.1` in the sim (`CommsConfig.host`). If a port is taken (several sims at once) the endpoint falls
back to an ephemeral port and says so; the harness pairs the two endpoints by their real addresses.
System ids: drone 1 / component 1 (autopilot), base 250 / component 190 (not 255: QGC uses 255).

## Messages

**Base -> drone**

| message | when | content |
|---|---|---|
| `HEARTBEAT` | 1 Hz | `MAV_TYPE_GCS`. The drone's link state: no base heartbeat for `link_timeout_s` = 2.5 s -> link down |
| `ACOUSTIC_BEARING` (sfhack, 54400) | ~1 Hz while the enemy is heard and the drone is airborne | UMA-16 azimuth, elevation (NaN: not measured in the sim), sigmas, confidence (NaN), source `ENEMY` / `OWN_DRONE`, array id. On the drone: `BearingUpdate` on bus topic `link/bearing`, as before |
| `COMMAND_LONG` `MAV_CMD_NAV_TAKEOFF` | enemy heard, drone waiting | launch: **param4 (yaw) = the bearing to fly out on**, param2 (empty in the spec) = seconds since the enemy was first heard |
| `COMMAND_LONG` `MAV_CMD_NAV_RETURN_TO_LAUNCH` | operator (key R) | recall |
| `COMMAND_INT` `MAV_CMD_DO_SET_ROI_LOCATION` | `BaseLink.look_at` | "look there": x / y degE7, z AMSL. COMMAND_INT, not COMMAND_LONG: float32 degrees only resolve ~0.5 m. The drone acknowledges `MAV_RESULT_UNSUPPORTED` until the pilot has a `look_at` behaviour (the tracker has none yet) |
| `TRACK_REPORT_ACK` (sfhack, 54402) | per complete report | report number received |

Commands are resent every `cmd_retry_s` = 0.5 s (`confirmation` + 1) until the `COMMAND_ACK`.

**Drone -> base** (all from the onboard estimate and the mission state, never the truth)

| message | rate | content |
|---|---|---|
| `HEARTBEAT` | 1 Hz + on every state change | `MAV_TYPE_QUADROTOR`, `MAV_AUTOPILOT_GENERIC`; `custom_mode` = mission state (index into `FLIGHT_STATES`: WAIT, TAKEOFF, ..., SEARCH, DETECTED, TRACKING, LOST, RETURN, DESCEND, LANDED); armed flag in flight; `MAV_STATE_STANDBY` / `ACTIVE` |
| `GLOBAL_POSITION_INT` | 4 Hz | EKF position (WGS84 + AMSL, relative alt above the pad), velocity, heading |
| `LOCAL_POSITION_NED` | 4 Hz | the same, NED from the base mast |
| `ATTITUDE` | 10 Hz | AHRS roll / pitch / yaw (NED, FRD), body rates |
| `SYS_STATUS` | 1 Hz | sensors present / healthy (GPS unhealthy without fixes), battery = flight-time budget left in %, drop rate of the base's packets |
| `GPS_RAW_INT` | 1 Hz | `3D_FIX` with the receiver's fix, `NO_FIX` while GNSS is off / jammed |
| `EXTENDED_SYS_STATE` | 1 Hz + on change | landed state (on ground / takeoff / in air / landing) |
| `HOME_POSITION` | 0.2 Hz | the pad |
| `STATUSTEXT` | per onboard decision | the decision log (`decision` topic), chunked when > 50 characters |
| `COMMAND_ACK` | per command | |
| `TRACK_REPORT_INFO` (sfhack, 54401) + `CAMERA_TRACKING_GEO_STATUS` + `DATA_TRANSMISSION_HANDSHAKE` + n x `ENCAPSULATED_DATA` | per enemy report (1 Hz while tracking) | one `EnemyReport` (see below) |

### The custom messages and why

- **`ACOUSTIC_BEARING`**: no standard message carries a sound direction with its uncertainty and source.
- **`TRACK_REPORT_INFO`, `TRACK_REPORT_ACK`**: the track itself is the standard `CAMERA_TRACKING_GEO_STATUS`
  (position, velocity, horizontal accuracy, distance, heading). But that message is a live status stream: no
  timestamp, no sequence number, no tracker state, no delivery guarantee. A report buffered through a link outage
  needs all four (the base plots and scores each report at the time it was made, the dashboard shows the tracker
  state and confidence). So each report is preceded by `TRACK_REPORT_INFO` (time made, report number, tracker state,
  confidence, target AGL, our own position then, snapshot size) and the base answers `TRACK_REPORT_ACK`.

### Enemy reports: buffering and resend

The drone takes reports out of the tracker's `ReportOutbox` only while it believes the link is up (base heartbeat
younger than 2.5 s). A report stays in an "unacknowledged" list until `TRACK_REPORT_ACK` and is resent every
`report_resend_s` = 1 s. When the heartbeat times out, the unacknowledged ones go back to the front of the outbox
(the dashboard's "buffered" count includes them) and go out again when the base's heartbeat returns. The base
reassembles a report from its packets in order (info, geo status, image), acknowledges it, ignores duplicates
and keeps `received` sorted by report number.

### Snapshot path: `DATA_TRANSMISSION_HANDSHAKE` + `ENCAPSULATED_DATA`

The snapshot (96 x 96 JPEG, quality 80, `sfhack/mission/enemy_report.py`) goes in-band: one handshake (type
`MAVLINK_DATA_STREAM_IMG_JPEG`, size, width, height, packet count) and the JPEG in 253-byte `ENCAPSULATED_DATA`
packets (`seqnr` from 0). Measured: 1.6 kB per snapshot on average (~7 packets); the whole drone -> base stream incl. telemetry was 135 kB in 70 s of flight (~2 kB/s, signed), the base -> drone stream 6 kB. Light enough for one image per report, and it shares the link state, the
signing and the resend logic of the report it belongs to, so no second channel. For full frames or video a
separate channel (RTP / a second radio) would be the way; not needed for a 96 px crop.

## Signing

MAVLink 2 signing is on for both endpoints (`CommsConfig.signing`). Key = SHA-256 of `CommsConfig.signing_passphrase`
(env `SFHACK_MAVLINK_KEY` overrides; the default is a demo key, change it for anything real). Every outgoing packet
carries the 13-byte signature (link id, 48-bit timestamp in 10 us since 2015, first 6 bytes of SHA-256(key, packet)).
Incoming packets are dropped by the parser, before any handler, when they are **unsigned**, **signed with another
key**, or **replayed** (timestamp not newer than the last one seen from that sender). The endpoint counts them
(`rejected`, printed in the stats line at the end of a run).

```
$ uv run python -m sfhack.comms.signing_check
signed (shared key)    -> EXECUTED + COMMAND_ACK  (rejected 0, drone rejected so far: {})
unsigned               -> DROPPED by the drone  (rejected 1, drone rejected so far: {'unsigned': 1})
signed, wrong key      -> DROPPED by the drone  (rejected 1, drone rejected so far: {'unsigned': 1, 'bad signature': 1})
replayed (captured)    -> DROPPED by the drone  (rejected 1, drone rejected so far: {'unsigned': 1, 'bad signature': 2})
```

The check runs a drone endpoint, the real base and an attacker socket on loopback: the base's signed RETURN is
executed and acknowledged; the same command unsigned (with the base's system id), signed with a guessed key, or a
byte-for-byte copy of the base's own signed packet replayed later is dropped: no recall, no `COMMAND_ACK`.
`--target 127.0.0.1:14561` fires an unsigned and a wrong-key RETURN at a running demo. Tried while the drone was
tracking: it kept TRACKING, and the run's stats line showed `drone: ... rx 163 from the peer + 2 other, rejected: bad
signature 1, unsigned 1`. `tests/test_mavlink.py::test_signing_drops_forged_commands` asserts the self-contained check.

## QGroundControl: our drone on the map

1. Install QGroundControl (https://qgroundcontrol.com, Windows / Linux / macOS) and start it. By default it listens
   on UDP 14550 (Application Settings -> Comm Links / AutoConnect: UDP on).
2. Start a MAVLink run, e.g. `uv run python demos/52_track_enemy.py --bearing 60` (or `--headless`). The base forwards
   every packet it receives from the drone to `127.0.0.1:14550` (`--no-qgc` / `CommsConfig.qgc_forward` = False turns
   it off; `CommsConfig.qgc_port` to change the port; for QGC on another PC set `CommsConfig.host` / forward
   address accordingly).
3. QGC connects to vehicle 1 (generic quadrotor) and shows it on the map at the demo origin (49.25 N, 31.90 E, a
   made-up rural spot: `CommsConfig.origin_lat/lon/alt`), with heading, altitude, battery (flight-time budget),
   GPS state and the onboard decisions as messages. What it shows is the drone's **estimate**, as a real vehicle
   would report it. Key N (GNSS off) -> GPS "no fix" while the drone keeps flying; the link model dropping the link
   -> QGC "communication lost" until it returns.

QGC needs no signing key: the forwarded packets are still signed, and a receiver without a key accepts signed
packets (it cannot check them). QGC's own requests (parameters, mission, `AUTOPILOT_VERSION`) are not forwarded to
the drone (they would be unsigned and dropped anyway), so QGC reports that the parameter download did not finish;
the map, attitude and telemetry work regardless. QGC itself was not available when this was written; the stream
was checked with `python -m sfhack.comms.gcs_listen`, which decodes it with pymavlink's plain `common.xml` and no
key, the way QGC does:

```
$ uv run python -m sfhack.comms.gcs_listen        # while demos/52_track_enemy.py --bearing 60 --link 1 runs
   STATUSTEXT: LAUNCH 060.0 deg received, waiting for the IMU calibration
sys 1 WAIT      49.249910, 31.900248  alt  150.0 m AMSL (  0.0 m rel)  hdg  88.7  battery 100 %  GPS 3D fix
   STATUSTEXT: PREFLIGHT done: IMU calibrated on the pad, baro zeroed
   STATUSTEXT: TAKEOFF: bearing 060.0 deg
sys 1 TAKEOFF   49.249910, 31.900247  alt  153.3 m AMSL (  3.3 m rel)  hdg  89.9  battery 99 %  GPS 3D fix
...
   STATUSTEXT: SEARCH
sys 1 SEARCH    49.250149, 31.900710  alt  187.7 m AMSL ( 37.7 m rel)  hdg  44.6  battery 97 %  GPS 3D fix
...
   STATUSTEXT: DETECTED: drone 0.90 at ~165 m
   STATUSTEXT: TRACKING: confirmed (3/5 frames), 174 m away
sys 1 TRACKING  49.251049, 31.902704  alt  188.9 m AMSL ( 38.9 m rel)  hdg  51.4  battery 95 %  GPS 3D fix
message types: {'ATTITUDE': 685, 'CAMERA_TRACKING_GEO_STATUS': 29, 'COMMAND_ACK': 1, 'DATA_TRANSMISSION_HANDSHAKE': 29,
 'ENCAPSULATED_DATA': 204, 'EXTENDED_SYS_STATE': 71, 'GLOBAL_POSITION_INT': 276, 'GPS_RAW_INT': 71, 'HEARTBEAT': 71,
 'HOME_POSITION': 14, 'LOCAL_POSITION_NED': 276, 'STATUSTEXT': 8, 'SYS_STATUS': 71}  unknown ids: {54401: 29}
```

## Simulation: link model and lockstep

The link model (`sfhack/sim/link.py`, ~80 % up, simulated reality) never reaches the endpoints: the harness passes a
`gate()` callable, and a packet sent while it returns False is lost (the MAVLink sequence number still advances,
so the receiver's drop rate is real). The gate is open before takeoff (the drone sits on the pad next to the base
antenna; the launch command always gets through, as in the in-process path), afterwards it is the link model.

Both endpoints run in the sim process; `LoopbackPair.step(t)` (once per physics step) does the periodic sends and
then delivers every datagram in flight, including the replies it causes, before the step ends (`exchange`: read
until each side has as many datagrams as the other put on the air). So a packet is handled in the step it was
sent in, the way the in-process path calls the pilot directly, and runs are repeatable. A real deployment calls
`update(t)` and `poll()` in its own loop instead.

## Same mission over both transports

`demos/52_track_enemy.py --headless --bearing 60 --link L --comms C` (seed 0, 400 s, farm world), 2026-10-24:

| detector | link | inproc | mavlink |
|---|---|---|---|
| oracle | 0 % | TRACKING; detect 36.1 s, track 39.6 s; reports 354 made / 0 received; closest 11.7 m; bearings 0 | identical (whole log line by line) |
| oracle | 80 % (89 % measured) | TRACKING; detect 37.3 s, track 43.6 s; 350 / 350; report error 1.9 m; closest 11.7 m; bearings 349 | identical (whole log line by line) |
| oracle | 100 % | TRACKING; detect 38.6 s, track 40.0 s; 354 / 354; report error 1.9 m; closest 11.9 m; bearings 394 | identical (whole log line by line) |
| YOLO | 0 % | TRACKING; detect 31.8 s, track 45.0 s; 349 / 0; closest 11.8 m | identical RESULT line |
| YOLO | 80 % | TRACKING; detect 21.2 s, track 46.9 s; 347 / 347; report error 1.6 m (median 1.3); closest 12.0 m | same, report error median 1.4 m |
| YOLO | 100 % | TRACKING; detect 44.0 s, track 44.3 s; 349 / 349; report error 1.5 m; closest 11.1 m | same, closest approach 11.6 m |

Own position error 1.6 m in all runs. At 80 % the drone resent 4 reports whose packets hit an outage (heartbeat
not yet timed out), all 347 arrived once. Traffic at 100 %: drone -> base 12 052 packets / 1046 kB in 395 s
(~2.6 kB/s, most of it snapshots, 1.4 kB each), base -> drone 1143 packets / 43 kB. Speed: the MAVLink runs were
within ~5 % of the in-process ones (2.5-2.6x real time with YOLO on the GPU).

The only numeric difference the transport itself introduces is float32 quantization (MAVLink `float` fields:
bearing ~4e-6 deg, positions in degE7 ~1 cm). The closed loop can amplify it (an earlier 120 s oracle pair differed
by 0.1 m in the track after 90 s, same RESULT line); the 400 s oracle pairs above came out identical.
With the YOLO detector on the GPU, runs are not bit-repeatable anyway: ultralytics turns on `cudnn.benchmark`, so a
borderline detection can come out differently from run to run of the same transport, more so when the GPU is busy
with other jobs. `--detector oracle` (truth boxes, no network) is the exact comparison.
