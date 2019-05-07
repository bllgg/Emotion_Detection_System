import tkinter as tk


# Define the data points
data = [20, 15, 10, 7, 5, 4]

root = tk.Tk()
root.title("Bar Graph")

c_width = 400  # Define it's width
c_height = 200  # Define it's height
c = tk.Canvas(root, width=c_width, height=c_height, bg='white')
c.pack()

# The variables below size the bar graph
y_stretch = 5  # The highest y = max_data_value * y_stretch
y_gap = 20  # The gap between lower canvas edge and x axis
x_stretch = 30  # Stretch x wide enough to fit the variables
x_width = 30  # The width of the x-axis
x_gap = 30  # The gap between left canvas edge and y axis

# A quick for loop to calculate the rectangle
for x, y in enumerate(data):

    # coordinates of each bar

    # Bottom left coordinate
    x0 = x * x_stretch + x * x_width + x_gap

    # Top left coordinates
    y0 = c_height - (y * y_stretch + y_gap)

    # Bottom right coordinates
    x1 = x * x_stretch + x * x_width + x_width + x_gap

    # Top right coordinates
    y1 = c_height - y_gap

    # Draw the bar
    c.create_rectangle(x0, y0, x1, y1, fill="red")

    # Put the y value above the bar
    c.create_text(x0 + 2, y0, anchor=tk.SW, text=str(y))

root.mainloop()
