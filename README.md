# piezo_fem - Simulation of thermo-electro-mechanical fields for piezoceramics

This package implements the finite element method (FEM) for piezoelectric
systems in Python. The solvers are written from scratch only by using
numpy/scipy as well as gmsh for generating the used meshes.
Currently the simulation of piezoelectric ceramics are supported. The
simulations are done in an axisymmetric model such that the ceramics
have a rectangular cross section.

## Features
- Simulation of thermal field in time domain
- Simulation of piezoelectric (electro-mechanical) field in time and
  frequency domain
  - Calculation of charges and impedances in time and frequency domain
- Simulation of a thermo-piezoeletric field in time domain
  - Calculation of mechanical losses which are used as a source for the thermal
    simulation
- Coupled simulation of a thermo-piezoelectric and a single thermal simulation
  where the thermo-piezoelectric simulation can be in time or frequency domain
- Nonlinear time domain simulation
  - Material nonlinearity
  - Solver currently is WIP
- Automatic mesh generation with gmsh
- Exporting functions to plot the fields in gmsh or as csv files
- Supporting disc and ring shapes for piezoelectric ceramics

## Installation

The package can be cloned using git and installed using pip install. It is
recommended to install and activate a python virtual environment first.
```python
git clone https://atuin.emt.uni-paderborn.de/git/jonasho/piezo_fem.git
cd piezo_fem
pip install .
```

## Usage

Some example simulation setups can be found [here](scripts/basic_example.py).
The examples in this file are documented in detail.

## Tests

For the solvers some basic tests are implemented. They run simple example
simulations and compare the results with known test data which is created
at a verified solver state.
In order to run the tests it is necessary to install pytest
```python
pip install pytest
```

## For developers

In order to make changes to the local piezo_fem installation make sure to
install it using the -e parameter:
```python
pip install -e .
```

## TODOs
Here are some additional features and optimizations which could be applied to
the code. Some of those feature can be discussed and are not mandatory:
- Add tests for the solvers
  - Thermo simulation DONE (-> Energy check)
  - Piezo time simulation DONE
  - Piezo freq simulation DONE (1 frequency step)
  - Thermo-piezo time simulation DONE (10000 time steps)
  - Test for ring structure?
- Update importing structure?
- Make simulation faster
  - Calculating element nodes and jacobians beforehand and reusing DONE
  - Using lru caches?
  - Using jit from numba?
  - Parallelizing assembly procedure?
  - Updating only changed material matrices in temperature dependent
    simulation?
- Creating post processing views for gmsh takes a very long time
- When creating the simulation results folder the simulation name can be
  removed from the file names
- Is the boundary condition for u_r at the symmetry axis even needed?
- Make explicit time solving scheme in nonlinear simulation faster by using
  matrix decomposition and inversion
  - scipy sparse has no implementation for cholesky decomposition
- Check if multiple materials already works or if continuity and boundary
  conditions are needed
- Add docs (.rst)
- Expand readme (Installation/Usage + More general information)
