"""The sfhack MAVLink 2 dialect (sfhack/comms/sfhack.xml = common.xml + ACOUSTIC_BEARING + two report messages).

The Python module is generated with pymavlink's mavgen on first use and cached under $SFHACK_DATA/cache/mavlink,
keyed by the XML and the pymavlink version (generated code stays out of the repo). Any other MAVLink stack (C on a
flight controller, QGroundControl, MAVSDK) generates its bindings from the same XML.

    uv run python -m sfhack.comms.dialect      # (re)generate and print the module path
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

XML = Path(__file__).with_name("sfhack.xml")


def _cache_dir() -> Path:
    from ..config import data_dir

    return data_dir() / "cache" / "mavlink"


def module_path() -> Path:
    import pymavlink

    key = hashlib.sha256(XML.read_bytes() + pymavlink.__version__.encode()).hexdigest()[:12]
    return _cache_dir() / f"sfhack_mav_{key}.py"


def generate(out: Path) -> Path:
    import pymavlink
    from pymavlink.generator import mavgen

    common = Path(pymavlink.__file__).parent / "message_definitions" / "v1.0" / "common.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = out.with_suffix(".xml")  # mavgen resolves <include> next to the dialect: point it at pymavlink's common.xml
    src.write_text(XML.read_text(encoding="utf-8").replace("<include>common.xml</include>",
                                                          f"<include>{common.as_posix()}</include>"), encoding="utf-8")
    tmp = out.with_name(out.stem + f"_tmp{os.getpid()}")
    opts = mavgen.Opts(str(tmp), wire_protocol="2.0", language="Python3", validate=False)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        ok = mavgen.mavgen(opts, [str(src)])
    if not ok or not tmp.with_suffix(".py").exists():
        raise RuntimeError(f"mavgen failed for {XML}:\n{log.getvalue()}")
    os.replace(tmp.with_suffix(".py"), out)  # atomic: parallel runs may generate at the same time
    return out


@lru_cache(maxsize=1)
def load() -> ModuleType:
    """The generated dialect module (pymavlink API: MAVLink, MAVLink_*_message, enums, constants)."""
    path = module_path()
    if not path.exists():
        generate(path)
    spec = importlib.util.spec_from_file_location("sfhack_mav", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sfhack_mav", mod)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    p = generate(module_path())
    print(p)
