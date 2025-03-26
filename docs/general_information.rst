===================
General information
===================

Here general information about the usage or the project structure can be found.

Usage
-----

For examples on how to work with the package have a look at the
``scripts/basic_example.py`` which provides extensive information about how to
setup the different simulations and how to configure them properly.

For developers
--------------

Here are informations for developers working on this project.

Project structure
^^^^^^^^^^^^^^^^^

The folder ``piezo_fem`` provides modules covering the general work with the
project such as different simulation handling classes for single or coupled
simulations, working with the mesh and materials or exporting simulation
results.
The folder ``piezo_fem/simulation`` contains the files for the different
simulation types:
- ``thermo_time.py``: Thermal field simulation in the time domain.
- ``piezo_time.py``: Electro-mechanical (piezo) simulation in the time domain.
- ``piezo_freq.py``: Electro-mechanical (piezo) simulation in the frequency domain.
- ``thermo_piezo_time.py``: Thermo-piezoelectric simulation in the time domain.

The ``base.py`` module contains a lot of different functions and classes which
are used in all the different solvers and are useable in a general way.
Functions which are specific to single solver types can be found in the module
of the specific solver.
Additionaly a ``piezo_fem/simulation/nonlinear`` can be found containing
a solver for a nonlinear simulation but this is still work in progress and not
properly working yet. Also the nonlinear solver is not embed in the
SingleSimulation class and is yet to be discussed if it even belongs there.
