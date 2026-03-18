from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pado_hologram


def test_pado_hologram_metadata() -> None:
    assert pado_hologram.PROJECT_NAME == "PADO Hologram"
    assert pado_hologram.PACKAGE_NAME == "pado_hologram"
    assert "holography" in pado_hologram.DESCRIPTION.lower()
