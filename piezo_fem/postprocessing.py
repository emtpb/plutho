"""Module for postprocessing functions."""

# Python standard libraries
import numpy as np
import numpy.typing as npt
import tkinter as tk
import matplotlib

def calculate_impedance(q, excitation, delta_t):     
    excitation_fft = np.fft.fft(excitation)[1:]
    q_fft = np.fft.fft(q)[1:]

    # since f=k*sample_frequency/N
    sample_frequency = 1/delta_t
    frequencies = np.arange(1, len(q))*sample_frequency/len(q)
    impedance = excitation_fft/(2*np.pi*1j*frequencies*q_fft)

    return frequencies, impedance

class FieldPlotter:
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
        self.top = tk.Tk()
        self.top.bind("<Key>", self.on_key_press)
        canvas_width = 1100
        canvas_height = 300
        self.top.geometry(f"{canvas_width+10}x{canvas_height+10}")
        self.canvas = tk.Canvas(self.top, bg="white", height=str(canvas_height), width=str(canvas_width))

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

        values = self.get_field(self.time_index, self.field_identifier)
        min_value = np.min(values)
        max_value = np.max(values)

        value_map = lambda x: 1/(max_value-min_value)*x+min_value/(min_value-max_value)

        for point, value in zip(self.points, values):
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