#!/usr/bin/env python3
"""Re-run the direct ommatidia compatibility probe with camera-body exclusion.

This preserves the original probe contract and only swaps the direct sensor
implementation so the small eye-camera marker geom cannot occlude every ray.
"""

from __future__ import annotations

from pathlib import Path

import probe_direct_ommatidia_rays as probe
from direct_ommatidia_sensor_bodyexclude import BodyExcludedDirectOmmatidialSensor


if __name__ == "__main__":
    probe.DirectOmmatidialSensor = BodyExcludedDirectOmmatidialSensor
    probe.DEFAULT_REPORT = Path("reports/flyppy/direct_ommatidia_bodyexclude_probe.md")
    raise SystemExit(probe.main())
