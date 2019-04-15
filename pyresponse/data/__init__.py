from pathlib import Path

_datadir = Path(__file__).resolve().parent
COORDDIR = _datadir / "coords"
REFDIR = _datadir / "reference"

del Path
del _datadir
