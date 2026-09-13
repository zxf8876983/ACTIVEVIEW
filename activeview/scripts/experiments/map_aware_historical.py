"""Load historical selector metrics for comparison tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def historical_methods(structured_path: Path, shared_path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if structured_path.is_file():
        payload = json.loads(structured_path.read_text(encoding="utf-8"))
        if "ScalarVisibility+Geometry" in payload.get("methods", {}):
            output["ScalarVisibility+Geometry"] = dict(payload["methods"]["ScalarVisibility+Geometry"])
    if shared_path.is_file():
        payload = json.loads(shared_path.read_text(encoding="utf-8"))
        if "Frame0 deployable best" in payload.get("methods", {}):
            output["Frame0 deployable best"] = dict(payload["methods"]["Frame0 deployable best"])
    return output
