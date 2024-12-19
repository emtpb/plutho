
# Python standard libraries
import os

# Third party libraries
import gmsh
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def plot_view(
        input_file_path: str,
        output_file_path: str,
        view_name: str):

    gmsh.initialize()
    gmsh.open(input_file_path)

    # Set colormap to plasma
    plasma_cmap = plt.get_cmap("plasma")
    num_colors = 256
    plasma_colors = [plasma_cmap(i / num_colors)[:3] for i in range(num_colors)]

    gmsh.option.set

    # Find view
    all_tags = gmsh.view.get_tags()
    found_tag = -1
    for tag in all_tags:
        name = gmsh.view.option.get_string(tag, "Name")
        if name == view_name:
            found_tag = tag
            break

    if found_tag == -1:
        raise Exception(f"No view with name {view_name} found")

    # Only make given view visible
    gmsh.fltk.initialize()
    for tag in all_tags:
        if tag != found_tag:
            gmsh.view.option.set_number(tag, "Visible", 0)

    # Dont show mesh
    #_, element_tags, _ = gmsh.model.mesh.get_elements()
    #gmsh.model.mesh.set_visibility(element_tags, 0)

    gmsh.fltk.run()

    # gmsh.write(output_file_path)
    # gmsh.fltk.finalize()
    gmsh.finalize()


if __name__ == "__main__":
    #file_path = os.path.join(
    #    "/home/jonash/uni/Masterarbeit/simulations/real_model_10k",
    #    "real_model_10k_disc_results.msh"
    #)
    MODEL_NAME = "real_model"
    load_dotenv()
    simulation_folder = os.environ["piezo_fem_simulation_path"]
    file_path = os.path.join(
        simulation_folder,
        MODEL_NAME,
        f"{MODEL_NAME}_disc_results.msh"
    )

    plot_view(file_path, "test.png", "Displacement")
