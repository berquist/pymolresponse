try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

_datadir = files("pymolresponse.data")
COORDDIR = _datadir / "coords"
REFDIR = _datadir / "reference"

del _datadir
del files
