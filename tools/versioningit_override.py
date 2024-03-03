from os import getenv
from typing import Any, Dict

from versioningit import VCSDescription
from versioningit.basics import DEFAULT_FORMATS

_ENVVARNAME = "VERSIONINGIT_FOR_PACKAGE_INDEX"


def pymolresponse_format(
    *, description: VCSDescription, base_version: str, next_version: str, params: Dict[str, Any]
) -> str:
    state = description.state
    assert state in {"distance", "dirty", "distance-dirty"}

    if getenv(_ENVVARNAME, "False").lower() in ("true", "1", "t"):
        fmt_distance = "{base_version}.post{distance}"
        if state != "distance":
            raise RuntimeError("dirty state doesn't make sense when building for a package index")
    else:
        # Default but missing {vcs} before {rev}
        fmt_distance = "{base_version}.post{distance}+{rev}"
        # Default
        fmt_dirty = DEFAULT_FORMATS["dirty"]
        # Default but missing {vcs} before {rev}
        fmt_distance_dirty = "{base_version}.post{distance}+{rev}.d{build_date:%Y%m%d}"

    if state == "distance":
        fmt = fmt_distance
    elif state == "dirty":
        fmt = fmt_dirty
    elif state == "distance-dirty":
        fmt = fmt_distance_dirty

    return fmt.format_map({**description.fields, "base_version": base_version})
