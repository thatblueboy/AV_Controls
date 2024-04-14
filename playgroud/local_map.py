import matplotlib.pyplot as plt
import numpy as np

def generate_sector_points(radius, angle_start, angle_end, num_points):
    angles = np.linspace(angle_start, angle_end, num_points)
    x_points = radius * np.cos(np.radians(angles))
    y_points = radius * np.sin(np.radians(angles))
    return x_points, y_points

# Parameters for the circles and sectors
num_circles = 5
radius_start = 5   # Starting radius for the innermost circle
radius_step = 2    # Difference in radius between circles
angle_start = 30   # Start angle in degrees
angle_end = 150    # End angle in degrees
num_points = 100   # Number of points to generate

# Generate points for concentric circles with sectors
plt.figure(figsize=(8, 8))
for i in range(num_circles):
    radius = radius_start + i * radius_step
    x_points, y_points = generate_sector_points(radius, angle_start, angle_end, num_points)
    plt.plot(x_points, y_points, label=f'Radius {radius}')

# Plotting
plt.plot([0], [0], 'ro')  # Plot origin in red
plt.title('Points covering sectors of concentric circles')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio for x and y axes
plt.show()
