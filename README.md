# piezo_fem

Implements the Finite Element Method for piezoelectric systems.
Gmsh is internally used to create the mesh.

## Features
- Automatic mesh generation with gmsh
- Resulting fields automatically saved to *.msh files to be viewed in gmsh
- Calculation of charge and impedence curve and mechanical losses
- Simulation with or without temperature field

## TODO
- Add tests for the solvers
- Fix importing structure
- Test simulation class
- Refactor whole project -> Base classes for thermal (and 
maybe electrical and mechanical sim)
    -> Different api to use
    -> Refactor the dirichlet nodes and calculation of f vector as well
        as the usage of boundary conditions
- Simulation can be made much faster when calculating all the element nodes and
    jacobians beforehand and reusing
    - Maybe lru caches can be utilized?
- Creating post processing views takes a very long time
- When creating the simulation results folder the simulation name can be 
removed from the file names
- Check if code can made faster using jfit from numba
- Implement temperature-dependent material properties

## Developer

In order to use the scripts from the scripts/ folder out of the box it is
necessary to add a environment variable for the path of the simulation results.
In order to do this only for the current workspace create a .env file
and add the following vscode setting:
```json
"python.envFile": "${workspaceFolder}/.env"
```
The .env file needs be in the workspace directory and has to contain the following variables
```
piezo_fem_simulation_path=path-to-file
piezo_fem_plot_path=path-to-file
```