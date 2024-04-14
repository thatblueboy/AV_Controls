import numpy as np
import matplotlib.pyplot as plt
import json


def generate_circle_points(radius, start_angle_deg, end_angle_deg, num_points):
    start_angle_rad = np.radians(start_angle_deg)
    end_angle_rad = np.radians(end_angle_deg)

    angles = np.linspace(start_angle_rad, end_angle_rad, num_points, endpoint=True)
    circle_points = np.array([radius * np.cos(angles), radius * np.sin(angles)]).T

    return circle_points

def generate_concentric_circle_points(num_circles, delta_radius, num_points_per_circle, start_angle_deg, end_angle_deg):
    circles = [np.array([[0, 0]])]  # Start with origin
    for i in range(num_circles):
        radius = (i+1) * delta_radius
        circle_points = generate_circle_points(radius, start_angle_deg, end_angle_deg, num_points_per_circle)
        circles.append(circle_points)
    return np.concatenate(circles)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def construct_graph(points, delta_radius):
    graph = {}
    point_indices = {tuple(point): index for index, point in enumerate(points)}
    num_points = len(points)
    for i in range(num_points):
        graph[i] = {}
        for j in range(num_points):
            print(np.linalg.norm(points[i]) - np.linalg.norm(points[j]))
            if i != j and delta_radius - 0.1 <= abs(np.linalg.norm(points[i]) - np.linalg.norm(points[j])) <= delta_radius + 0.1:
                weight = euclidean_distance(points[i], points[j])
                graph[i][j] = weight
    return graph, point_indices

def plot_graph(graph, points):
    plt.figure(figsize=(8, 8))
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            plt.plot([points[node][0], points[neighbor][0]], [points[node][1], points[neighbor][1]], 'b-', linewidth=0.5)
            plt.text((points[node][0] + points[neighbor][0]) / 2, (points[node][1] + points[neighbor][1]) / 2, f'{weight:.2f}', fontsize=8)

    plt.plot(points[:, 0], points[:, 1], 'ro')  # Plot the points
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Graph with Nodes on Sectors of Concentric Circles (Excluding Connections within Same Circle)')
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    plt.show()

# Example parameters
num_circles = 4
delta_radius = 4
num_points_per_circle = 5
start_angle_deg = 60
end_angle_deg = 120

points = generate_concentric_circle_points(num_circles, delta_radius, num_points_per_circle, start_angle_deg, end_angle_deg)
graph, point_indices = construct_graph(points, delta_radius)
print(graph)
# print(points)
plot_graph(graph, points)
try:
    with open('points.json', 'w') as file:
        json.dump(points.tolist(), file)
    print("Points saved successfully!")
except Exception as e:
    print("Error saving points:", e)
# Save graph to a file
try:
    with open('graph.json', 'w') as file:
        json.dump(graph, file)
    print("Graph saved successfully!")
except Exception as e:
    print("Error saving graph:", e)

