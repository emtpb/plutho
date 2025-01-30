"""Contains different materials which can be used in the simulations."""

# Python standard libraries
from typing import List, Union
import numpy as np
import numpy.typing as npt

# Third party libraries
from scipy import interpolate

# Local libraries
from .simulation.base import MaterialData


class Material:
    """Class to handle a single material. The material can have either constant
    material properties or temperature variant properties.
    The temperature properties are determined using linear interpolation.
Plugin 'tmsvg/pear-tree'

    Attributes:
        material_name: Name of the material.
        material_data: MaterialData object containing the material parameters.
        is_temperature_dependent: Boolean, True if the material is temperature
            dependent.
    """

    material_name: str
    material_data: MaterialData
    is_temperature_dependent: bool
    physical_group_name: str

    def __init__(
            self,
            material_name: str,
            material_data: MaterialData,
            physical_group_name: str):
        self.material_name = material_name
        self.material_data = material_data
        self.is_temperature_dependent = False
        self.physical_group_name = physical_group_name

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

    def to_dict(self):
        return {
            "temperature_dependent": self.is_temperature_dependent,
            "physical_group_name": self.physical_group_name,
            "material_data": self.material_data.to_dict()
        }

    def create_interpolations(self):
        """Prepares the interpolations for each material parameter for
        the different temperatures.
        """
        # The _local variables are initialized with ones to prevent divide
        # by zero
        self.c11_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c11,
            fill_value="extrapolate"
        )
        self.c12_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c12,
            fill_value="extrapolate"
        )
        self.c13_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c13,
            fill_value="extrapolate"
        )
        self.c33_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c33,
            fill_value="extrapolate"
        )
        self.c44_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.c44,
            fill_value="extrapolate"
        )
        self.e15_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.e15,
            fill_value="extrapolate"
        )
        self.e31_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.e31,
            fill_value="extrapolate"
        )
        self.e33_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.e33,
            fill_value="extrapolate"
        )
        self.eps11_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.eps11,
            fill_value="extrapolate"
        )
        self.eps33_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.eps33,
            fill_value="extrapolate"
        )
        self.alpha_k_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.alpha_k,
            fill_value="extrapolate"
        )
        self.alpha_m_interp = interpolate.interp1d(
            self.material_data.temperatures,
            self.material_data.alpha_m,
            fill_value="extrapolate"
        )


class MaterialManager:

    materials: List[Material]
    element_index_to_material_data: npt.NDArray
    is_temperature_dependent: bool

    def __init__(self,
                 number_of_elements: int):
        self.number_of_elements = number_of_elements
        self.element_index_to_material_data = -1 * np.ones(
            number_of_elements, dtype=int
        )
        self.materials = []
        self.is_temperature_dependent = False

    def add_material(
            self,
            material_name,
            material_data,
            physical_group_name,
            element_indices):
        new_material = Material(
                material_name,
                material_data,
                physical_group_name
            )
        self.materials.append(new_material)
        material_index = len(self.materials)-1

        for element_index in element_indices:
            if self.element_index_to_material_data[element_index] != -1:
                raise ValueError(
                    "A material has already been set for"
                    f"element with index {element_index}."
                )
            else:
                self.element_index_to_material_data[element_index] = \
                    material_index

        if new_material.is_temperature_dependent:
            self.is_temperature_dependent = True

    def initialize_materials(
            self,
            starting_temperature: Union[float, npt.NDArray] = None):
        if isinstance(starting_temperature, float):
            starting_temperature = np.ones(self.number_of_elements) * \
                starting_temperature

        self.c11_local = np.zeros(self.number_of_elements)
        self.c12_local = np.zeros(self.number_of_elements)
        self.c13_local = np.zeros(self.number_of_elements)
        self.c33_local = np.zeros(self.number_of_elements)
        self.c44_local = np.zeros(self.number_of_elements)
        self.e15_local = np.zeros(self.number_of_elements)
        self.e31_local = np.zeros(self.number_of_elements)
        self.e33_local = np.zeros(self.number_of_elements)
        self.eps11_local = np.zeros(self.number_of_elements)
        self.eps33_local = np.zeros(self.number_of_elements)
        self.alpha_m_local = np.zeros(self.number_of_elements)
        self.alpha_k_local = np.zeros(self.number_of_elements)
        self.density_local = np.zeros(self.number_of_elements)
        self.heat_capacity_local = np.zeros(self.number_of_elements)
        self.thermal_conductivity_local = np.zeros(self.number_of_elements)

        if starting_temperature is None:
            if self.is_temperature_dependent:
                raise ValueError(
                    "Please give a starting temperature when using temperature"
                    " dependent material parameters."
                )
            # Material is not temperature dependent
            for element_index in range(self.number_of_elements):
                material_data = self.materials[
                    self.element_index_to_material_data[element_index]
                ].material_data
                self.c11_local[element_index] = material_data.c11
                self.c12_local[element_index] = material_data.c12
                self.c13_local[element_index] = material_data.c13
                self.c33_local[element_index] = material_data.c33
                self.c44_local[element_index] = material_data.c44
                self.e15_local[element_index] = material_data.e15
                self.e31_local[element_index] = material_data.e31
                self.e33_local[element_index] = material_data.e33
                self.eps11_local[element_index] = material_data.eps11
                self.eps33_local[element_index] = material_data.eps33
                self.alpha_k_local[element_index] = material_data.alpha_k
                self.alpha_m_local[element_index] = material_data.alpha_m
                self.density_local[element_index] = material_data.density
                self.heat_capacity_local[element_index] = \
                    material_data.heat_capacity
                self.thermal_conductivity_local[element_index] = \
                    material_data.thermal_conductivity
        else:
            # Material is temperature dependent
            for element_index in range(self.number_of_elements):
                material = self.materials[
                    self.element_index_to_material_data[element_index]
                ]
                self.c11_local[element_index] = material.c11_interp(
                    starting_temperature[element_index]
                )
                self.c12_local[element_index] = material.c12_interp(
                    starting_temperature[element_index]
                )
                self.c13_local[element_index] = material.c13_interp(
                    starting_temperature[element_index]
                )
                self.c33_local[element_index] = material.c33_interp(
                    starting_temperature[element_index]
                )
                self.c44_local[element_index] = material.c44_interp(
                    starting_temperature[element_index]
                )
                self.e15_local[element_index] = material.e15_interp(
                    starting_temperature[element_index]
                )
                self.e31_local[element_index] = material.e31_interp(
                    starting_temperature[element_index]
                )
                self.e33_local[element_index] = material.e33_interp(
                    starting_temperature[element_index]
                )
                self.eps11_local[element_index] = material.eps11_interp(
                    starting_temperature[element_index]
                )
                self.eps33_local[element_index] = material.eps33_interp(
                    starting_temperature[element_index]
                )
                self.alpha_k_local[element_index] = material.alpha_k_interp(
                    starting_temperature[element_index]
                )
                self.alpha_m_local[element_index] = material.alpha_m_interp(
                    starting_temperature[element_index]
                )
                self.density_local[element_index] = \
                    material.material_data.density
                self.heat_capacity_local[element_index] = \
                    material.material_data.heat_capacity
                self.thermal_conductivity_local[element_index] = \
                    material.material_data.thermal_conductivity

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

        threshold = 0.01

        # When for any element the material parameter change is greater
        # than 10 % update material parameters
        update = False

        # TODO Check runtime performance and if this can be made faster
        # Get the new values based on given temperature
        c11_updated = np.zeros(self.number_of_elements)
        c12_updated = np.zeros(self.number_of_elements)
        c13_updated = np.zeros(self.number_of_elements)
        c33_updated = np.zeros(self.number_of_elements)
        c44_updated = np.zeros(self.number_of_elements)
        e15_updated = np.zeros(self.number_of_elements)
        e31_updated = np.zeros(self.number_of_elements)
        e33_updated = np.zeros(self.number_of_elements)
        eps11_updated = np.zeros(self.number_of_elements)
        eps33_updated = np.zeros(self.number_of_elements)
        alpha_m_updated = np.zeros(self.number_of_elements)
        alpha_k_updated = np.zeros(self.number_of_elements)

        for element_index in range(self.number_of_elements):
            current_material = self.materials[
                self.element_index_to_material_data[element_index]
            ]
            c11_updated[element_index] = current_material.c11_interp(
                local_temperatures[element_index]
            )
            c12_updated[element_index] = current_material.c12_interp(
                local_temperatures[element_index]
            )
            c13_updated[element_index] = current_material.c13_interp(
                local_temperatures[element_index]
            )
            c33_updated[element_index] = current_material.c33_interp(
                local_temperatures[element_index]
            )
            c44_updated[element_index] = current_material.c44_interp(
                local_temperatures[element_index]
            )
            e15_updated[element_index] = current_material.e15_interp(
                local_temperatures[element_index]
            )
            e31_updated[element_index] = current_material.e31_interp(
                local_temperatures[element_index]
            )
            e33_updated[element_index] = current_material.e33_interp(
                local_temperatures[element_index]
            )
            eps11_updated[element_index] = current_material.eps11_interp(
                local_temperatures[element_index]
            )
            eps33_updated[element_index] = current_material.eps33_interp(
                local_temperatures[element_index]
            )
            alpha_k_updated[element_index] = current_material.alpha_k_interp(
                local_temperatures[element_index]
            )
            alpha_m_updated[element_index] = current_material.alpha_m_interp(
                local_temperatures[element_index]
            )

        if np.any(np.abs(self.c11_local - c11_updated) /
                  self.c11_local > threshold):
            update = True

        if not update:
            # print(np.max(np.abs(self.c12_local - c12_updated)/self.c12_local))
            if np.any(np.abs(self.c12_local - c12_updated) /
                      self.c12_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.c13_local - c13_updated)/self.c13_local))
            if np.any(np.abs(self.c13_local - c13_updated) /
                      self.c13_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.c33_local - c33_updated)/self.c33_local))
            if np.any(np.abs(self.c33_local - c33_updated) /
                      self.c33_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.c44_local - c44_updated)/self.c44_local))
            if np.any(np.abs(self.c44_local - c44_updated) /
                      self.c44_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.e15_local - e15_updated)/self.e15_local))
            if np.any(np.abs(self.e15_local - e15_updated) /
                      self.e15_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.e31_local - e31_updated)/self.e31_local))
            if np.any(np.abs(self.e31_local - e31_updated) /
                      self.e31_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.e33_local - e33_updated)/self.e33_local))
            if np.any(np.abs(self.e33_local - e33_updated) /
                      self.e33_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.eps11_local - eps11_updated)/self.eps11_local))
            if np.any(np.abs(self.eps11_local - eps11_updated) /
                      self.eps11_local > threshold):
                update = True

        if not update:
            # print(np.max(np.abs(self.eps33_local - eps33_updated)/self.eps33_local))
            if np.any(np.abs(self.eps33_local - eps33_updated) /
                      self.eps33_local > threshold):
                update = True

        if not update:
            if np.any(np.abs(self.alpha_k_local - alpha_k_updated) /
                      self.alpha_k_local > threshold):
                update = True

        if not update:
            if np.any(np.abs(self.alpha_m_local - alpha_m_updated) /
                      self.alpha_m_local > threshold):
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
            self.alpha_k_local = alpha_k_updated
            self.alpha_m_local = alpha_m_updated

        return update

    def get_density(self, element_index: int):
        """Returns the density material parameter."""
        return self.density_local[element_index]

    def get_heat_capacity(self, element_index: int):
        """Returns the heat capacity material parameter."""
        return self.heat_capacity_local[element_index]

    def get_thermal_conductivity(self, element_index: int):
        """Returns the thermal conductivity material parameter."""
        return self.thermal_conductivity_local[element_index]

    def get_alpha_m(self, element_index: int):
        """Returns the alpha m material parameter."""
        return self.alpha_m_local[element_index]

    def get_alpha_k(self, element_index: int):
        """Returns the alpha k material parameter."""
        return self.alpha_k_local[element_index]

    def get_permittivity_matrix(self, element_index: int):
        """Returns the permittivity matrix. Can be temperature dependent."""
        eps11 = self.eps11_local[element_index]
        eps33 = self.eps33_local[element_index]

        return np.diag([eps11, eps33])

    def get_piezo_matrix(self, element_index: int):
        """Returns the piezo matrix. Can be temperature dependent."""
        e15 = self.e15_local[element_index]
        e31 = self.e31_local[element_index]
        e33 = self.e33_local[element_index]

        return np.array([
            [  0,   0, e15,   0],
            [e31, e33,   0, e31]
        ])

    def get_elasticity_matrix(self, element_index: int):
        """Returns the elasticity matrix. Can be temperature dependent."""
        c11 = self.c11_local[element_index]
        c12 = self.c12_local[element_index]
        c13 = self.c13_local[element_index]
        c33 = self.c33_local[element_index]
        c44 = self.c44_local[element_index]

        return np.array([
            [c11, c13,   0, c12],
            [c13, c33,   0, c13],
            [  0,   0, c44,   0],
            [c12, c13,   0, c11]
        ])

    def print_material_data(self, element_index: int):
        """Prints the current material data."""
        print(
            "Elasticity matrix:",
            self.get_elasticity_matrix(element_index)
        )
        print(
            "Piezo matrix:",
            self.get_piezo_matrix(element_index)
        )
        print(
            "Permittivity matrix:",
            self.get_permittivity_matrix(element_index)
        )
        print("Alpha_k:", self.get_alpha_k(element_index))
        print("Alpha_m:", self.get_alpha_m(element_index))
        print("Density:", self.get_density(element_index))
        print("Heat capacity:", self.get_heat_capacity(element_index))
        print(
            "Thermal conductivity:",
            self.get_thermal_conductivity(element_index)
        )


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

pic255_alpha_m_nonzero = MaterialData(
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
        "alpha_m": 1.267e5,
        "alpha_k": 6.259e-10,
        "density": 7800,
        "heat_capacity": 350,
        "thermal_conductivity": 1.1,
        "temperatures": 20
    }
)

pic255_olga = MaterialData(
    **{
        "c11": 1.11e11,
        "c12": 0.72e11,
        "c13": 0.79e11,
        "c33": 1.19e11,
        "c44": 0.2e11,
        "e15": 12.09,  # 2.35e-6,
        "e31": -6.57,
        "e33": 15.18,
        "eps11": 8.15e-9,  # 167.8e-9,
        "eps33": 6.59e-9,
        "alpha_m": 1.343e5,
        "alpha_k": 7.278e-10,
        "density": 7800,
        "heat_capacity": 350,
        "thermal_conductivity": 1.1,
        "temperatures": 20
    }
)

pic181_20_extrapolated = MaterialData(
    **{
        "c11": 1.4099e11,
        "c12": 0.8196e11,
        "c13": 0.8036e11,
        "c33": 1.3712e11,
        "c44": 0.2977e11,
        "e15": 12.9483,  # 2.35e-6,
        "e31": -5.0732,
        "e33": 14.073,
        "eps11": 1.3019e-8,  # 167.8e-9,
        "eps33": 5.2963e-9,
        "alpha_m": 0.0,
        "alpha_k": 1.1868e-10,
        "density": 7850,
        "heat_capacity": 350,
        "thermal_conductivity": 1.1,
        "temperatures": 20
    }
)

pic181_90_extrapolated = MaterialData(
    **{
        "c11": 1.41405244e11,
        "c12": 8.34514249e10,
        "c13": 7.99705391e10,
        "c33": 1.37506008e11,
        "c44": 3.02865266e10,
        "e15": 13.90159052,  # 2.35e-6,
        "e31": -5.04276938,
        "e33": 15.71897841,
        "eps11": 1.46031733e-8,  # 167.8e-9,
        "eps33": 6.58773038e-9,
        "alpha_m": 0.0,
        "alpha_k": 5.99305326e-11,
        "density": 7850,
        "heat_capacity": 350,
        "thermal_conductivity": 1.1,
        "temperatures": 20
    }
)

pic184_25 = MaterialData(
    **{
        "c11": 141496343521.27835,
        "c12": 86169430271.64551,
        "c13": 79915318364.24962,
        "c33": 124909327336.97089,
        "c44": 26771638883.39679,
        "e15": 13.815676514507956,
        "e31": -4.004768707540417,
        "e33": 13.168634717101696,
        "eps11": 1.086683329447101e-08,
        "eps33": 5.6431956643221666e-09,
        "alpha_m": 0.0,
        "alpha_k": 5.143833571076851e-10,
        "density": 7850,
        "heat_capacity": 350,
        "thermal_conductivity": 1.1,
        "temperatures": 25
    }
)
