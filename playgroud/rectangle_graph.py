import matplotlib.pyplot as plt

class GraphPlotter:
    def __init__(self, n, m, distance_n, distance_m):
        self.n = n
        self.m = m
        self.distance_n = distance_n
        self.distance_m = distance_m
        self.points = self._generate_points()
        self.graph = self._construct_graph()

    def _generate_points(self):
        points = []
        for j in range(self.m):
            for i in range(self.n):
                x_coord = -j * self.distance_m  # Columns along negative x-axis
                y_coord = (i - (self.n - 1) / 2) * self.distance_n  # Rows centered along y = 0
                points.append((x_coord, y_coord))
        return points

    def _construct_graph(self):
        graph = {}
        for idx1, point1 in enumerate(self.points):
            graph[idx1] = {}
            x1, y1 = point1
            for idx2, point2 in enumerate(self.points):
                if idx1 != idx2:
                    x2, y2 = point2
                    distance_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    if distance_squared <= self.distance_n ** 2 + self.distance_m ** 2:
                        graph[idx1][idx2] = distance_squared ** 0.5
        return graph

    def plot_points(self):
        plt.figure(figsize=(8, 6))
        for coord in self.points:
            plt.plot(coord[0], coord[1], 'bo')  # Plotting points as blue circles

        plt.title(f"{self.n}x{self.m} Points with Rows Centered at y = 0")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

    def plot_graph(self):
        self.plot_points()

        for idx1, neighbors in self.graph.items():
            x1, y1 = self.points[idx1]
            for idx2, distance in neighbors.items():
                x2, y2 = self.points[idx2]
                plt.plot([x1, x2], [y1, y2], 'g-')  # Plotting edges as green lines

        plt.gca().invert_xaxis()  # Invert x-axis to rotate by 90 degrees
        plt.gca().invert_yaxis()  # Invert y-axis to rotate by 90 degrees
        plt.show()

    def get_origin_index(self):
        # Find the index of the origin point (0, 0)
        for idx, point in enumerate(self.points):
            if point == (0, 0):
                return idx
        return None

    def get_upmost_column(self):
        # Find the points and keys of the upmost column
        upmost_points = {}
        for idx, point in enumerate(self.points):
            if point[0] == 0:
                upmost_points[idx] = point
        return upmost_points
    
    def get_bottommost_column(self):
        # Find the points and keys of the bottommost column
        min_x = min(point[0] for point in self.points)
        bottommost_column = {idx: point for idx, point in enumerate(self.points) if point[0] == min_x}
        return bottommost_column
    
    def get_upmost_column(self):
        # Find the points and keys of the upmost column
        max_x = max(point[0] for point in self.points)
        upmost_column = {idx: point for idx, point in enumerate(self.points) if point[0] == max_x}
        return upmost_column

# Usage example:
n = 9  # Number of rows
m = 10  # Number of columns
distance_n = 1  # Distance between rows
distance_m = 2  # Distance between columns

graph_plotter = GraphPlotter(n, m, distance_n, distance_m)
print(graph_plotter.points)
print(graph_plotter.graph)
print(graph_plotter.get_origin_index())
print(graph_plotter.get_upmost_column())
graph_plotter.plot_graph()
