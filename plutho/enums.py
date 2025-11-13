"""Module for enums."""


# Python standard library
from enum import Enum


__all__ = [
    "SolverType",
    "NonlinearType",
    "FieldType"
]


class SolverType(Enum):
    """Contains the possible solver types."""
    PiezoFreq = "PiezoFreq",
    PiezoTime = "PiezoTime",
    ThermoPiezoTime = "ThermoPiezoTime",
    ThermoTime = "ThermoTime"
    PiezoHB = "PiezoHarmonicBalance"


class NonlinearType(Enum):
    """Contains the various possible nonlinear simulation types."""
    QuadraticRayleigh = 0,
    QuadraticCustom = 1,
    CubicRayleigh = 2,
    CubicCustom = 3


class FieldType(Enum):
    """Possible field types which are calculated using differnet simulations.
    """
    U_R = 0
    U_Z = 1
    PHI = 2
    THETA = 3
