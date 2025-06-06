# plutho - Python Library for coUpled THermo-piezOelectric simulations

[![DOI](https://zenodo.org/badge/997346473.svg)](https://zenodo.org/badge/latestdoi/997346473)

This library implements the finite element method (FEM) for various
thermo-piezoelectric systems in Python.
The solvers are written using numpy and scipy as well as gmsh for generating
2D meshes.
The models are always assumed to be rotational symmetric but besides that
arbitray geometies can be simulated.
Excitations and material parameters can be set freely based on the definition
of physical groups in gmsh.

## Features

Right now various simulation types are supported:
- Heat condution simulation in time domain
- Piezoelectric simulation (coupled mechanical and electrical field) in
  time and frequency domain
- Coupled thermo-piezoelectric simulations in time domain
- Nonlinear piezoelectric simulations using mechanical quadratic nonlinearities
  in time and frequency domain

Additionaly there helper classes for specific coupled simulation types:
- Thermo-piezoelectric simulation in time domain with mechanical loss
  calculation for hamonic exictations with a subsequent separate heat
  conduction simulation utilizing the averaged mechanical losses
- Thermo-piezoelectric simulation in frequency domain with a subsequent
  heat conduction simulation in time domain which can utilize temperature
  dependent material parameters

## Installation

The package can be cloned using git and installed using pip install. It is
recommended to install and activate a python virtual environment first.

## Usage

Some example simulation setups can be found [here](scripts/basic_example.py).
The examples in this file are documented in detail.

## Tests

For the solvers some basic tests are implemented. They run simple example
simulations and compare the results with known test data which is created
at a verified solver state.
In order to run the tests it is necessary to install pytest.
```console
pip install pytest
```
Then the tests can be run using
```console
pytest
```

## Documentation

In order to run the documentation sphinx and sphinx_rtd_theme must be
installed.
```console
pip install sphinx sphinx_rtd_theme
```
In order to build the documentation the code *.rst files can be created with
```console
sphinx-apidoc -f -o docs plutho
```
and afterwards, the whole documentation can be built using:
```console
sphinx-build -M html docs docs/build
```

## For developers

In order to make changes to the local plutho installation make sure to
install it using the -e parameter:
```console
pip install -e .
```

### TODOs

Here are some additional features and optimizations which could be applied to
the code. Some of those feature can be discussed and are not mandatory:
- Remove print statements and implement logging
- Add tests for the nonlinear solvers
- Simulation speedups
  - Using lru caches?
  - Using jit from numba?
  - Parallelizing assembly procedure?
- When creating the simulation results folder the simulation name can be
  removed from the file names
- Update importing structure Use __add__ and import *
- Check if multiple materials already works or if continuity and boundary
  conditions are needed

### Issues

- Creating post processing views for gmsh takes a very long time
