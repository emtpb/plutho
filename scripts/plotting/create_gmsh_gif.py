
# Python standard libraries
import os
import numpy as np

# Third party libraries
from dotenv import load_dotenv
import gmsh


def create_images(
        results_path,
        output_image_folder,
        visible_view_indices,
        time_step_indices):
    gmsh.initialize()
    gmsh.open(results_path)

    # Set all other views to invisible
    number_of_views = int(gmsh.option.getNumber("PostProcessing.NbViews"))
    for i in range(0, number_of_views):
        gmsh.option.setNumber(f"View[{i}].Visible", 0)
    for i in visible_view_indices:
        gmsh.option.setNumber(f"View[{i}].Visible", 1)

    # Iterate over the given time steps
    # Per time step draw a picture
    gmsh.fltk.initialize()
    for image_index, time_step_index in enumerate(time_step_indices):
        for i in visible_view_indices:
            gmsh.option.setNumber(f"View[{i}].TimeStep", time_step_index)

        gmsh.write(os.path.join(output_image_folder, f"{image_index}.png"))
    gmsh.fltk.finalize()

    gmsh.finalize()

def create_gif(images_folder, output_file_path):
    # convert -delay 20 -loop 0 {0..49}.png result.gif
    pass


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    PIEZO_SIM_NAME = "temp_dep_mat_sim_250k"
    sim_folder = os.path.join(CWD, PIEZO_SIM_NAME)
    results_file_path = os.path.join(
        sim_folder,
        f"{PIEZO_SIM_NAME}_ring_results.msh"
    )
    images_folder = os.path.join(sim_folder, "images")

    create_images(
        results_file_path,
        images_folder,
        [0, 2],
        np.arange(250)[-50:]
    )
