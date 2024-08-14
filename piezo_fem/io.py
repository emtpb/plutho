"""Module for io functions for the resulting fields and values."""

# Python standard libraries
import numpy as np
import os

# TODO Maybe write general opencfs parser
def read_charge_open_cfs(file_path):
    lines = []
    with open(file_path, "r", encoding="UTF-8") as fd:
        lines = fd.readlines()[3:]

    time = []
    charge = []
    for line in lines:
        current_time, current_charge = line.split()
        time.append(float(current_time))
        charge.append(float(current_charge))

    return time, charge

def parse_hist_file(file_path):
    lines = []
    with open(file_path, "r", encoding="UTF-8") as fd:
        lines = fd.readlines()
    time_steps = []
    u_r = []
    u_z = []
    for time_index, line in enumerate(lines[3:]):
        current_time_step, current_u_r, current_u_z = line.split() 
        time_steps.append(float(current_time_step))
        u_r.append(float(current_u_r))
        u_z.append(float(current_u_z))

    return time_steps, u_r, u_z

def create_vector_field_as_csv(u, nodes, folder_path):
    # TODO Add check if folder exists
    number_of_nodes = len(nodes)
    number_of_time_steps = u.shape[1]

    vector_field = np.zeros((number_of_time_steps, number_of_nodes, 5))
    for time_step in range(number_of_time_steps):
        for node_index, node in enumerate(nodes):
            current_u_r = u[2*node_index, time_step]
            current_u_z = u[2*node_index+1, time_step]
            current_v = u[2*number_of_nodes+node_index, time_step]
            vector_field[time_step, node_index, 0] = node[0]
            vector_field[time_step, node_index, 1] = node[1]
            vector_field[time_step, node_index, 2] = current_u_r
            vector_field[time_step, node_index, 3] = current_u_z
            vector_field[time_step, node_index, 4] = current_v
        
    for time_step in range(number_of_time_steps):
        current_file_path = os.path.join(folder_path, f"u_{time_step}.csv")
        field = vector_field[time_step]

        text = f"r,z,u_r,u_z,v\n"
        for node_index in range(number_of_nodes):
            r = field[node_index][0]
            z = field[node_index][1]
            u_r = field[node_index][2]
            u_z = field[node_index][3]
            v = field[node_index][4]
            text += f"{r},{z},{u_r},{u_z},{v}\n"
            
        with open(current_file_path, "w", encoding="UTF-8") as fd:
            fd.write(text)  