"""Simulated upload of the onboard package to the drone: a throttled link, per-layer progress, SHA-256 check.

The package is streamed in chunks at `mbps` (MB/s, default 2) from <mission>/onboard to <mission>/drone
(the drone's storage), hashed on the receiving side and compared with manifest.json. Only a fully
verified copy gets drone/upload_receipt.json; FlightSystem(mission=...) flies that copy when it exists.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .build import mission_dir

CHUNK = 64 * 1024
FLY_COMMAND = "uv run python demos/demo_live.py --mission {id}"


def upload(mission_id: str, mbps: float = 2.0, on_event: Callable[[dict[str, Any]], None] | None = None,
           corrupt: str | None = None) -> bool:
    """Stream every layer over the simulated link. `corrupt` = layer key to flip one byte in (test hook)."""
    emit = on_event or (lambda e: None)
    mdir = mission_dir(mission_id)
    src, dst = mdir / "onboard", mdir / "drone"
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    layers = [(k, v["file"], int(v["bytes"]), v["sha256"]) for k, v in manifest["layers"].items()]
    man_bytes = (src / "manifest.json").stat().st_size
    total = sum(b for _, _, b, _ in layers) + man_bytes
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    rate = max(mbps, 0.01) * 1e6
    emit({"type": "start", "mission_id": mission_id, "mbps": mbps, "total_bytes": total,
          "eta_s": round(total / rate, 1), "layers": [{"key": k, "file": f, "bytes": b} for k, f, b, _ in layers],
          "text": f"uploading {total / 1e6:.2f} MB to the drone at {mbps:g} MB/s (~{total / rate:.0f} s)"})
    t0 = time.time()
    sent = 0
    ok_all = True
    results = {}
    for key, fn, size, want in layers + [("manifest", "manifest.json", man_bytes, None)]:
        h = hashlib.sha256()
        done = 0
        last = 0.0
        with open(src / fn, "rb") as fi, open(dst / fn, "wb") as fo:
            while chunk := fi.read(CHUNK):
                if corrupt == key and done == 0:
                    chunk = bytes([chunk[0] ^ 0xFF]) + chunk[1:]  # a bit error on the link
                fo.write(chunk)
                h.update(chunk)
                done += len(chunk)
                sent += len(chunk)
                lag = sent / rate - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
                if time.time() - last > 0.1 or done == size:
                    last = time.time()
                    emit({"type": "progress", "layer": key, "layer_bytes": size, "layer_sent": done,
                          "total_bytes": total, "total_sent": sent})
        got = h.hexdigest()
        ok = want is None or got == want
        ok_all &= ok
        results[key] = {"file": fn, "bytes": size, "sha256": got, "verified": ok if want else "n/a (index)"}
        emit({"type": "layer_done", "layer": key, "file": fn, "bytes": size, "sha256": got, "ok": ok,
              "text": f"  {fn:16s} {size / 1e6:7.2f} MB  sha256 {got[:16]}...  "
                      f"{'verified' if ok else 'MISMATCH'}" if want else f"  {fn:16s} {size / 1e6:7.2f} MB  (index)"})
    secs = time.time() - t0
    if ok_all:
        receipt = {"mission_id": mission_id, "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "link_mbps": mbps, "seconds": round(secs, 1), "total_bytes": total, "layers": results}
        (dst / "upload_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    cmd = FLY_COMMAND.format(id=mission_id)
    emit({"type": "done", "ok": ok_all, "seconds": round(secs, 1), "command": cmd,
          "text": (f"READY TO FLY ({total / 1e6:.2f} MB in {secs:.1f} s, all checksums verified)\n  {cmd}" if ok_all
                   else "UPLOAD FAILED: checksum mismatch, the drone keeps no package (re-upload)")})
    if not ok_all:
        shutil.rmtree(dst, ignore_errors=True)
    return ok_all


def drone_package(mdir: Path) -> Path | None:
    """The verified on-drone copy, if the package was uploaded."""
    d = mdir / "drone"
    return d if (d / "upload_receipt.json").exists() and (d / "manifest.json").exists() else None
