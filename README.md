# piezo_fem

Implements the Finite Element Method for piezoelectric systems.
Gmsh is internally used to create the mesh.

## Features
- Simulation of piezoelectric systems with ro without thermal field
- Frequency and time domain simulation possible
- Calculation of charge and impedance curve as well as mechanical losses
- Nonlinear time domain simulation -> Mechanical nonlinearity
- Automatic mesh generation with gmsh
- Resulting fields can be viewed in gmsh, matplotlib or paraview

## TODO
- Add tests for the solvers
- Fix importing structure
- Test simulation class (integration test)
- Simulation can be made much faster when calculating all the element nodes and
    jacobians beforehand and reusing (DONE)
    - Maybe lru caches can be utilized?
- Creating post processing views for gmsh takes a very long time
- When creating the simulation results folder the simulation name can be
  removed from the file names
- Check if code can made faster using jfit from numba
- Is the boundary condition for u_r at the symmetry axis even needed?
- Assembly in temperature dependent material parameters can be made faster by
  checking which material parameter has changed and then only chaning the
  corresponding matrix (is this really faster?)
- In the rectangular mesh calculation update the x_offset and change it so an
  inner and outer radius can be given
- Make the assembly procedure faster
- Make explicit time solving scheme in nonlinear simulation faster by using
  matrix inversion
- Update example files
- Check if multiple materials already works or if continuity and boundary
  conditions are needed
- Rename simulation_handler to single_sim

