"""Convert the V1 prototype's Berkeley map package (gnss-free-nav/backend/data/sample/berkeley) into
the V2 package layout at sfhack/world/sample_berkeley (manifest key "ground" -> "ortho", ortho.jpg).

    uv run python scripts/import_v1_sample.py [path/to/v1/package]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "gnss-free-nav" / "backend" / "data" / "sample" / "berkeley"
DST = ROOT / "sfhack" / "world" / "sample_berkeley"

manifest = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
ground = manifest.pop("ground")
DST.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SRC / ground["file"], DST / "ortho.jpg")
shutil.copyfile(SRC / "landmarks.json", DST / "landmarks.json")
manifest["ortho"] = {**ground, "file": "ortho.jpg"}
(DST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"wrote {DST}")
