"""Module to draw meshes."""

# Python standard libraries
import tkinter as tk

def draw_mesh(nodes, elements):
    top = tk.Tk()
    canvas_width = 1100
    canvas_height = 300
    top.geometry(f"{canvas_width+10}x{canvas_height+10}")
    c = tk.Canvas(top, bg="white", height=str(canvas_height), width=str(canvas_width))

    point_thickness = 5
    offset_x = 10
    offset_y = 10

    transform_coords = lambda x: [2e5*x[0]+offset_x,
                                  -2e5*x[1]+(canvas_height-2*offset_y)+offset_y]

    for element in elements:
        for index in range(len(element)):
            x1, y1 = transform_coords(nodes[element[index]])
            x2, y2 = transform_coords(nodes[element[(index+1)%len(element)]])
            c.create_line(x1, y1, x2, y2)
        #if len(element) == 3:
        #    x1, y1 = transform_coords(nodes[element[0]])
        #    x2, y2 = transform_coords(nodes[element[1]])
        #    x3, y3 = transform_coords(nodes[element[2]])
        #    c.create_line(x1, y1, x2, y2)
        #    c.create_line(x1, y1, x3, y3)
        #    c.create_line(x2, y2, x3, y3)

    for node in nodes:
        x, y = transform_coords(node)
        c.create_oval(x-point_thickness/2,
                      y-point_thickness/2,
                      x+point_thickness/2,
                      y+point_thickness/2, fill="black")

    c.pack()
    top.mainloop()