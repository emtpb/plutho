"""Contains different materials which can be used in the simulations."""

# Python standard libraries
from typing import Union
import numpy as np
import numpy.typing as npt

# Third party libraries
from scipy import interpolate

# Local libraries
from .simulation.base import MaterialData


class MaterialManager:
    """Contains the data for temperature (in)dependent materials.
    Currently all parameters except density, thermal_conductivity,
    heat_capacity, alpha_m and alpha_k can be set temperature independent.
    For the temperature independent parameters an array must be given where
    the index corresponds to the index of the temperature array.
    For example:
        temperatures = [10, 20, 30]
        c11 = [15, 30, 60]
    In this case the c11 value of 15 is taken at temperature 10.

    Parameters:
        material_name: Name of the material.
        material_data: MaterialData object containing the material parameters.
        number_of_elements: Number of elements of the mesh.
        start_temperature: Either float or np.ndarray. Depending on if the
            start temperature is set constant over the whole domain or not.
    """

    def __init__(
            self,
            material_name: str,
            material_data: MaterialData,
            number_of_elements: int,
            start_temperature: Union[float, np.ndarray]):
        self.name = material_name
        self.is_temperature_dependent = False
        self.material_data = material_data
        self.number_of_elements = number_of_elements

        # Check if its temperature dependent
        if isinstance(material_data.temperatures, np.ndarray):
            needed_size = material_data.temperatures.shape[0]
            self.is_temperature_dependent = True

            # Check if all values that are arrays have the same length
            for attribute, value in material_data.__dict__.items():
                if isinstance(value, np.ndarray):
                    if value.shape[0] != needed_size:
                        raise ValueError(
                            f"The temperature dependent attribute {attribute} "
                            "must have the same size as the temperatures "
                            "array."
                        )

            self.create_interpolations()
            if isinstance(start_temperature, float) or \
                    isinstance(start_temperature, int):
                self.update_temperature(
                    start_temperature*np.ones(number_of_elements)
                )
            else:
                if start_temperature.shape[0] != number_of_elements:
                    raise ValueError(
                        "Start temperature field must be the size of number"
                        "of elements"
                    )
                self.update_temperature(start_temperature)

    def create_interpolations(self):
        """Prepares the interpolations for each material parameter for
        the different temperatures.
        """
        if not self.is_temperature_dependent:
            return

        # The _local variables are initialized with ones to prevent divide
        # by zero
        self.c11_local = np.ones(self.number_of_elements)
        self.c11_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c11,
            fill_value="extrapolate"
        )
        self.c12_local = np.ones(self.number_of_elements)
        self.c12_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c12,
            fill_value="extrapolate"
        )
        self.c13_local = np.ones(self.number_of_elements)
        self.c13_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c13,
            fill_value="extrapolate"
        )
        self.c33_local = np.ones(self.number_of_elements)
        self.c33_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c33,
            fill_value="extrapolate"
        )
        self.c44_local = np.ones(self.number_of_elements)
        self.c44_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c44,
            fill_value="extrapolate"
        )
        self.e15_local = np.ones(self.number_of_elements)
        self.e15_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.e15,
            fill_value="extrapolate"
        )
        self.e31_local = np.ones(self.number_of_elements)
        self.e31_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.e31,
            fill_value="extrapolate"
        )
        self.e33_local = np.ones(self.number_of_elements)
        self.e33_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.e33,
            fill_value="extrapolate"
        )
        self.eps11_local = np.ones(self.number_of_elements)
        self.eps11_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.eps11,
            fill_value="extrapolate"
        )
        self.eps33_local = np.ones(self.number_of_elements)
        self.eps33_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.eps33,
            fill_value="extrapolate"
        )

    def update_temperature(self, local_temperatures: npt.NDArray):
        """Given the local temperature distribution the material
        parameters are updated when the change of any parameter is greater than
        10 %.

        Parameters:
            local_temperatures: Temperatures of each element. Must have the
                size of number of elements.

        Returns:
            True if the material parameters are updated. False otherwise.
        """
        if not self.is_temperature_dependent:
            return None

        # When for any element the material parameter change is greater
        # than 10 % update material parameters
        update = False

        c11_updated = self.c11_interp(local_temperatures)
        if np.any(np.abs(self.c11_local - c11_updated)/self.c11_local > 0.1):
            update = True

        c12_updated = self.c12_interp(local_temperatures)
        if np.any(np.abs(self.c12_local - c12_updated)/self.c12_local > 0.1):
            update = True

        c13_updated = self.c13_interp(local_temperatures)
        if np.any(np.abs(self.c13_local - c13_updated)/self.c13_local > 0.1):
            update = True

        c33_updated = self.c33_interp(local_temperatures)
        if np.any(np.abs(self.c33_local - c33_updated)/self.c33_local > 0.1):
            update = True

        c44_updated = self.c44_interp(local_temperatures)
        if np.any(np.abs(self.c44_local - c44_updated)/self.c44_local > 0.1):
            update = True

        e15_updated = self.e15_interp(local_temperatures)
        if np.any(np.abs(self.e15_local - e15_updated)/self.e15_local > 0.1):
            update = True

        e31_updated = self.e31_interp(local_temperatures)
        if np.any(np.abs(self.e31_local - e31_updated)/self.e31_local > 0.1):
            update = True

        e33_updated = self.e33_interp(local_temperatures)
        if np.any(np.abs(self.e33_local - e33_updated)/self.e33_local > 0.1):
            update = True

        eps11_updated = self.eps11_interp(local_temperatures)
        if np.any(np.abs(self.eps11_local - eps11_updated)/ \
                  self.eps11_local > 0.1):
            update = True

        eps33_updated = self.eps33_interp(local_temperatures)
        if np.any(np.abs(self.eps33_local - eps33_updated)/ \
                  self.eps33_local > 0.1):
            update = True

        # If any value changed enough, all values can be updated
        if update:
            self.c11_local = c11_updated
            self.c12_local = c12_updated
            self.c13_local = c13_updated
            self.c33_local = c33_updated
            self.c44_local = c44_updated
            self.e15_local = e15_updated
            self.e31_local = e31_updated
            self.e33_local = e33_updated
            self.eps11_local = eps11_updated
            self.eps33_local = eps33_updated

        return update

    def get_density(self):
        """Returns the density material parameter."""
        return self.material_data.density

    def get_heat_capacity(self):
        """Returns the heat capacity material parameter."""
        return self.material_data.heat_capacity

    def get_thermal_conductivity(self):
        """Returns the thermal conductivity material parameter."""
        return self.material_data.thermal_conductivity

    def get_alpha_m(self):
        """Returns the alpha m material parameter."""
        return self.material_data.alpha_m

    def get_alpha_k(self):
        """Returns the alpha k material parameter."""
        return self.material_data.alpha_k

    def get_permittivity_matrix(self, element_index: int):
        """Returns the permittivity matrix. Can be temperature dependent."""
        if self.is_temperature_dependent:
            eps11 = self.eps11_local[element_index]
            eps33 = self.eps33_local[element_index]
        else:
            eps11 = self.material_data.eps11
            eps33 = self.material_data.eps33

        return np.diag([eps11, eps33])

    def get_piezo_matrix(self, element_index: int):
        """Returns the piezo matrix. Can be temperature dependent."""
        if self.is_temperature_dependent:
            e15 = self.e15_local[element_index]
            e31 = self.e31_local[element_index]
            e33 = self.e33_local[element_index]
        else:
            e15 = self.material_data.e15
            e31 = self.material_data.e31
            e33 = self.material_data.e33

        return np.array([
            [  0,   0, e15,   0],
            [e31, e33,   0, e31]
        ])

    def get_elasticity_matrix(self, element_index: int):
        """Returns the elasticity matrix. Can be temperature dependent."""
        if self.is_temperature_dependent:
            c11 = self.c11_local[element_index]
            c12 = self.c12_local[element_index]
            c13 = self.c13_local[element_index]
            c33 = self.c33_local[element_index]
            c44 = self.c44_local[element_index]
        else:
            c11 = self.material_data.c11
            c12 = self.material_data.c12
            c13 = self.material_data.c13
            c33 = self.material_data.c33
            c44 = self.material_data.c44

        return np.array([
            [c11, c13,   0, c12],
            [c13, c33,   0, c13],
            [  0,   0, c44,   0],
            [c12, c13,   0, c11]
        ])


pic255 = MaterialData(
    **{
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
