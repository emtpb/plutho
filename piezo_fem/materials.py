"""Contains different materials which can be used in the simulations."""

# Local libraries
from .simulation.base import MaterialData


pic255 = MaterialData(
    "pic255",
    {
        "c11": 1.19e11,
        "c12": 0.84e11,
        "c13": 0.83e11,
        "c33": 1.17e11,
        "c44": 0.21e11,
        "e15": 12.09,
        "e31": -6.03,
        "e33": 15.49,
        "eps11": 8.15e-9,
        "eps33": 6.58e-9,
        "alpha_m": 0,
        "alpha_k": 6.259e-10,
        "density": 7800,
        "heat_capacity": 350,
        "thermal_conductivity": 1.1,
        "temperatures": 20
    }
)
