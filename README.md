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
