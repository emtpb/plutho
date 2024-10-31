# piezo_fem

Implements the Finite Element Method for piezoelectric systems.
Gmsh is internally used to create the mesh.

Currently only validated for 1 single parameter set and compared with OpenCFS. 

## Features
- Automatic mesh generation with gmsh
- Resulting fields automatically saved to *.msh files to be viewed in gmsh
- Calculation of charge and impedence curve and mechanical losses
- Simulation with or without temperature field

## TODO
- Add tests for the solvers
- Fix importing structure
- Test simulation class
- Add possibility to load simulation settings without creating a mesh
- Refactor whole project -> Base classes for thermal (and 
maybe electrical and mechanical sim)
    -> Different api to use
    -> Refactor the dirichlet nodes and calculation of f vector as well
        as the usage of boundary conditions
- Simulation can be made much faster when calculating all the element nodes and
    jacobians beforehand and reusing
    - Maybe lru caches can be utilized?

## Developer

In order to use the scripts from the scripts/ folder out of the box it is
necessary to add a environment variable for the path of the simulation results.
In order to do this only for the current workspace create a .env file
and add the following vscode setting:
```json
"python.envFile": "${workspaceFolder}/.env"
```