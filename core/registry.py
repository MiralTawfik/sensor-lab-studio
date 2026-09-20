"""Catalogue of every available sensor model, grouped by transducer family."""
from __future__ import annotations

from . import glossary
from .base import SensorModel
from .electrical import PiezoelectricForce, PiezoresistivePressure
from .mechanical import MassSpringDamper
from .optical import (FabryPerotPressure, FiberBraggGrating, SPRKretschmann, WGMMicrobubble,
                      WGMMicroringSiN, WGMMicroringSOI, WGMMicrosphere, WGMMicrotoroid)

_ALL = [
    MassSpringDamper(),
    PiezoresistivePressure(),
    PiezoelectricForce(),
    WGMMicrosphere(),
    WGMMicrotoroid(),
    WGMMicroringSOI(),
    WGMMicroringSiN(),
    WGMMicrobubble(),
    FiberBraggGrating(),
    FabryPerotPressure(),
    SPRKretschmann(),
]

MODELS: dict[str, SensorModel] = {m.key: m for m in _ALL}

# every input gets a plain-language description (shown as the "?" tooltip in the sidebar)
for _m in _ALL:
    for _p in _m.params:
        if not _p.help:
            _p.help = glossary.param_help(_p.key, _p.label)

FAMILY_HELP = {
    "Mechanical": "The measurand moves a proof mass on a spring; the displacement is read electrically.",
    "Electrical": "The measurand changes an electrical property (resistance) or generates charge (piezoelectric effect).",
    "Optical": "The measurand shifts an optical resonance or interference fringe (wavelength, angle or intensity).",
}


def families() -> list[str]:
    return list(FAMILY_HELP)


def subfamilies(family: str) -> list[str]:
    seen = []
    for m in _ALL:
        if m.family == family and m.subfamily not in seen:
            seen.append(m.subfamily)
    return seen


def models_of(family: str, subfamily: str | None = None) -> list[SensorModel]:
    return [m for m in _ALL if m.family == family and (subfamily is None or m.subfamily == subfamily)]


def get(key: str) -> SensorModel:
    return MODELS[key]
