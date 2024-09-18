# piezo_fem

Implements the Finite Element Method for piezoelectric systems.
Gmsh is internally used to create the mesh.

Currently only validated for 1 single parameter set and compared with OpenCFS. 

## Features

- Automatic mesh generation with gmsh
- Resulting fields automatically can be saved to *.msh files to be viewed in gmsh
- Calculation of charge and impedence curve and mechanical losses

## TODO
- Add tests for the solvers
- Fix importing structure
- Finish refactoring and polishing