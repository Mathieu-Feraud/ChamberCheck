"""Lowercase import compatibility shim for ChamberCheck."""

import sys

import ChamberCheck as _legacy
from ChamberCheck import Config, __author__, __license__, __version__
from ChamberCheck import analysis as _analysis
from ChamberCheck import CC_derived_metrics as _cc_derived_metrics
from ChamberCheck import model_analysis as _model_analysis
from ChamberCheck import models as _models
from ChamberCheck import preprocessing as _preprocessing
from ChamberCheck import reporting as _reporting
from ChamberCheck import scoring as _scoring
from ChamberCheck import scrapers as _scrapers
from ChamberCheck import utils as _utils

__path__ = _legacy.__path__

sys.modules[__name__] = sys.modules["ChamberCheck"]
sys.modules[__name__ + ".analysis"] = _analysis
sys.modules[__name__ + ".CC_derived_metrics"] = _cc_derived_metrics
sys.modules[__name__ + ".model_analysis"] = _model_analysis
sys.modules[__name__ + ".models"] = _models
sys.modules[__name__ + ".preprocessing"] = _preprocessing
sys.modules[__name__ + ".reporting"] = _reporting
sys.modules[__name__ + ".scoring"] = _scoring
sys.modules[__name__ + ".scrapers"] = _scrapers
sys.modules[__name__ + ".utils"] = _utils

__all__ = [
    "Config",
    "__version__",
    "__author__",
    "__license__",
]
