import matplotlib.pyplot as plt

class GraphPlotter:
    def __init__(self, m, n, distance_m, distance_n):
        self.n = n
        self.m = m
        self.distance_n = distance_n
        self.distance_m = distance_m
        self.points = self._generate_points()
        self.graph = self._construct_graph()

    def _generate_points(self):
        points = []
        for i in range(self.n):
            for j in range(self.m):
                x_coord = j * self.distance_m
                y_coord = i * self.distance_n - 0.5 * (self.n - 1) * self.distance_n
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

        plt.title(f"{self.n}x{self.m} Points with n Centered at Zero")
        plt.xlabel("n centered at zero")
        plt.ylabel("m")
        plt.grid(True)

    def plot_graph(self):
        self.plot_points()

        for idx1, neighbors in self.graph.items():
            x1, y1 = self.points[idx1]
            for idx2, distance in neighbors.items():
                x2, y2 = self.points[idx2]
                plt.plot([x1, x2], [y1, y2], 'g-')  # Plotting edges as green lines

        plt.show()

    def get_origin_index(self):
        # Find the index of the origin point (0, 0)
        for idx, point in enumerate(self.points):
            if point == (0, 0):
                return idx
        return None  # Return None if the origin point is not found in the points list

# Usage example:
m = 5  # Number of rows
n = 4  # Number of columns
distance_m = 1.5  # Distance between rows
distance_n = 1  # Distance between columns

graph_plotter = GraphPlotter(m, n, distance_m, distance_n)
print("Origin index:", graph_plotter.get_origin_index())
graph_plotter.plot_graph()
