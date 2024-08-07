import numpy as np
import numpy.typing as npt
from tkinter import *
from typing import List
import matplotlib
import matplotlib.pyplot as plt

class Fields:
    points: npt.NDArray
    u_r: npt.NDArray
    u_z: npt.NDArray
    v: npt.NDArray

    def __init__(self, time_list, number_of_points, csv_files):
        # Init arrays
        self.number_of_time_steps = len(time_list)
        self.points = np.zeros((number_of_points, 2))
        self.u_r = np.zeros((number_of_points, self.number_of_time_steps))
        self.u_z = np.zeros((number_of_points, self.number_of_time_steps))
        self.v = np.zeros((number_of_points, self.number_of_time_steps))

        for time_index, csv_file in zip(range(len(time_list)), csv_files):
            self.read_file(time_index, csv_file, time_index == 0)

        # Init canvas
        self.top = Tk()
        self.top.bind("<Key>", self.on_key_press)
        canvas_width = 1100
        canvas_height = 300
        self.top.geometry(f"{canvas_width+10}x{canvas_height+10}")
        self.canvas = Canvas(self.top, bg="white", height=str(canvas_height), width=str(canvas_width))

        offset_x = 20
        offset_y = 20

        self.transform_coords = lambda x: [2e5*x[0]+offset_x,
                                    -2e5*x[1]+(canvas_height-2*offset_y)+offset_y]
        self.time_index = 0

    def read_file(self, time_index, file_path, read_points = False):
        lines = []
        with open(file_path, "r", encoding="UTF-8") as fd:
            lines = fd.readlines()

        points = []
        u_r = []
        u_z = []
        v = []
        for line in lines[1:]:
            r, z, current_u_r, current_u_z, current_v = line[:-1].split(",")
            u_r.append(float(current_u_r))
            u_z.append(float(current_u_z))
            v.append(float(current_v))

            if read_points:
                points.append([float(r), float(z)])

        if read_points:
            self.points = points
        self.u_r[:, time_index] = np.array(u_r)
        self.u_z[:, time_index] = np.array(u_z)
        self.v[:, time_index] = np.array(v)

    def get_field(self, time_index, field_identifier):
        if field_identifier == "u_r":
            return self.u_r[:, time_index]
        elif field_identifier == "u_z":
            return self.u_z[:, time_index]
        elif field_identifier == "v":
            return self.v[:, time_index]
        else:
            raise Exception(f"Field identifier {field_identifier} not found.")

    def _from_rgb(self, rgb):
        """translates an rgb tuple of int to a tkinter friendly color code
        """
        r, g, b = rgb
        return f'#{r:02x}{g:02x}{b:02x}'


    def on_key_press(self, event):
        if event.keysym == "Left":
            # Left array
            if self.time_index > 0:
                self.time_index -= 1
                self.update()
                self.top.update()
                print(f"Time step {self.time_index}")
        elif event.keysym == "Right":
            # Right arrow
            if self.time_index < self.number_of_time_steps:
                self.time_index += 1
                self.update()
                self.top.update()
                print(f"Time step {self.time_index}")
        else:
            pass

    def get_value_at(self, field_identifier, r, z):
        # Get point
        point_index = None
        for index, point in enumerate(self.points):
            if point[0] == r and point[1] == r:
                point_index = index
                break

        if field_identifier == "u_r":
            return self.u_r[point_index, :]
        elif field_identifier == "u_z":
            return self.u_z[point_index, :]
        elif field_identifier == "v":
            return self.v[point_index, :]
        else:
            return None

    def update(self):
        point_thickness = 13
        cmap = matplotlib.cm.get_cmap("Reds")

        values = fields.get_field(self.time_index, self.field_identifier)
        min_value = np.min(values)
        max_value = np.max(values)

        value_map = lambda x: 1/(max_value-min_value)*x+min_value/(min_value-max_value)

        for point, value in zip(fields.points, values):
            x, y = self.transform_coords(point)
            rgb = cmap(value_map(value))[:-1]
            rgb = [int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)]
            color = self._from_rgb(rgb)
            self.canvas.configure()
            self.canvas.create_oval(x-point_thickness/2,
                        y-point_thickness/2,
                        x+point_thickness/2,
                        y+point_thickness/2, fill=color)

    def draw_fields(self, field_identifier = "v"):     
        self.field_identifier = field_identifier   
        self.update()

        self.canvas.pack()
        self.top.mainloop()

def plot_field_valuse(time_steps, values_CFS, values_FEM):
    plt.plot(time_steps, values_CFS[:len(time_steps)], label="CFS")
    plt.plot(time_steps, values_FEM[:len(time_steps)], label="MyFEM")
    plt.grid()
    plt.legend()
    plt.show()

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


if __name__ == "__main__":
    NUMBER_TIME_STEPS = 8192
    files_high_delta_t = [f"fields/u_{i}.csv" for i in range(NUMBER_TIME_STEPS)]
    #files_low_delta_t = [f"fields_low_delta_t/u_{i}.csv" for i in range(NUMBER_TIME_STEPS)]
    DELTA_T = 1e-8
    NUMBER_POINTS = 318
    time_list_high_delta_t = np.arange(0, NUMBER_TIME_STEPS)*DELTA_T
    #time_list_low_delta_t = np.arange(0, 10*NUMBER_TIME_STEPS)*1/10*DELTA_T
    fields_high_delta_t = Fields(time_list_high_delta_t, NUMBER_POINTS, files_high_delta_t)
    #fields_low_delta_t = Fields(time_list_low_delta_t, NUMBER_POINTS, files_low_delta_t)

    #fields.draw_fields("v")

    u_z_fem_high_delta_t = fields_high_delta_t.get_value_at("u_z", 0, 0)
    #u_z_fem_low_delta_t = fields_low_delta_t.get_value_at("u_z", 0, 0)

    cfs_file = "piezo-mechDisplacement-node-1-result_node.hist"
    time_steps, u_r_cfs, u_z_cfs = parse_hist_file(cfs_file)


    plt.plot(time_list_high_delta_t, u_z_cfs, "+",label="OpenCFS")
    plt.plot(time_list_high_delta_t, u_z_fem_high_delta_t, label="MyFEM")
    #plt.plot(time_list_low_delta_t, u_z_fem_low_delta_t[:10*NUMBER_TIME_STEPS], label="MyFEM low delta_t")
    plt.grid()
    plt.legend()
    plt.show()

    #plot_field_valuse(time_list_high_delta_t, u_z_cfs, u_z_fem)