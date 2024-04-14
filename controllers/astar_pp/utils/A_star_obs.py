import heapq
import math
import matplotlib.pyplot as plt

class AStar:
    def __init__(self, points, graph, start_index, goal_index):
        self.points = points
        self.graph = graph
        self.start_index = start_index
        self.goal_index = goal_index

        self.open_list = []  
        heapq.heappush(self.open_list, (0, start_index))  
        self.came_from = {}  
        self.g_scores = {index: math.inf for index in range(len(points))}  
        self.g_scores[start_index] = 0

        self.f_scores = {index: math.inf for index in range(len(points))}  
        self.f_scores[start_index] = self.heuristic(start_index, goal_index)

        self.obstacle_indices = []

    def heuristic(self, node_index, goal_index):
        node = self.points[node_index]
        goal = self.points[goal_index]
        return math.sqrt((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2)

    def add_obstacles(self, obstacle_indices):
        self.obstacle_indices.extend(obstacle_indices)

    def reconstruct_path(self, current_index):
        path = [current_index]
        while current_index in self.came_from:
            current_index = self.came_from[current_index]
            path.append(current_index)
        path.reverse()
        return path

    def update_graph(self, new_graph):
        self.graph = new_graph

    def astar_search(self, new_goal_index=None, new_graph=None, new_start_index=None, new_obstacle_indices=None):
        if new_goal_index is not None:
            self.goal_index = new_goal_index
        if new_graph is not None:
            self.update_graph(new_graph)
        if new_start_index is not None:
            self.start_index = new_start_index
        if new_obstacle_indices is not None:
            self.obstacle_indices = new_obstacle_indices

        start_index = int(self.start_index)  # Convert to integer if it's a string
        goal_index = int(self.goal_index)  # Convert to integer if it's a string

        self.open_list = []
        heapq.heappush(self.open_list, (0, start_index))
        self.came_from = {}
        self.g_scores = {index: math.inf for index in range(len(self.points))}
        self.g_scores[start_index] = 0
        self.f_scores = {index: math.inf for index in range(len(self.points))}
        self.f_scores[start_index] = self.heuristic(start_index, goal_index)

        while self.open_list:
            _, current_index = heapq.heappop(self.open_list)
            if current_index == goal_index:
                return self.reconstruct_path(current_index)

            if current_index not in self.graph:
                continue

            for neighbor_index, cost in self.graph[current_index].items():
                neighbor_index = int(neighbor_index)  # Convert to integer if it's a string
                if neighbor_index in self.obstacle_indices:
                    continue
                tentative_g_score = self.g_scores[current_index] + cost
                if tentative_g_score < self.g_scores[neighbor_index]:
                    self.came_from[neighbor_index] = current_index
                    self.g_scores[neighbor_index] = tentative_g_score
                    self.f_scores[neighbor_index] = tentative_g_score + self.heuristic(neighbor_index, goal_index)
                    heapq.heappush(self.open_list, (self.f_scores[neighbor_index], neighbor_index))

        return None
    
    def plot_graph(self, shortest_path_indices=None):
        plt.figure(figsize=(8, 8))
        for node_index, edges in self.graph.items():
            node = self.points[node_index]
            if shortest_path_indices and node_index in shortest_path_indices:
                plt.plot(node[0], node[1], 'go')  
            else:
                plt.plot(node[0], node[1], 'ro')  

            for neighbor_index, cost in edges.items():
                neighbor = self.points[neighbor_index]
                plt.plot([node[0], neighbor[0]], [node[1], neighbor[1]], 'b-', linewidth=0.5)
                plt.text((node[0] + neighbor[0]) / 2, (node[1] + neighbor[1]) / 2, f'{cost:.2f}', fontsize=8)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Graph with Nodes and Connections')
        plt.grid(True)
        plt.axis('equal')  
        plt.show()

if __name__ == '__main__':
    # Example usage
    points = [(0, -4.0), (0, -3.0), (0, -2.0), (0, -1.0), (0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0), (0, 4.0), (-2, -4.0), (-2, -3.0), (-2, -2.0), (-2, -1.0), (-2, 0.0), (-2, 1.0), (-2, 2.0), (-2, 3.0), (-2, 4.0), (-4, 0.0), (-4, -1.0), (-4, -2.0), (-4, -3.0), (-4, -4.0), (-4, 1.0), (-4, 2.0), (-4, 3.0), (-4, 4.0)]
    graph = {0: {1: 1, 4: 1}, 1: {0: 1, 2: 1, 5: 1}, 2: {1: 1, 3: 1, 6: 1}, 3: {2: 1, 7: 1}, 4: {0: 1, 8: 1}, 5: {1: 1, 9: 1}, 6: {2: 1, 10: 1}, 7: {3: 1, 11: 1}, 8: {4: 1, 12: 1}, 9: {5: 1, 13: 1}, 10: {6: 1, 14: 1}, 11: {7: 1, 15: 1}, 12: {8: 1, 16: 1}, 13: {9: 1, 17: 1}, 14: {10: 1, 18: 1}, 15: {11: 1, 19: 1}, 16: {12: 1, 17: 1}, 17: {13: 1, 16: 1}, 18: {14: 1, 19: 1}, 19: {15: 1, 18: 1}}
    start_index = 0
    goal_index = 19

    astar = AStar(points, graph, start_index, goal_index)
    astar.plot_graph()

    obstacle_indices = [1, 2, 3, 4]  # Example obstacle indices
    astar.add_obstacles(obstacle_indices)
    shortest_path = astar.astar_search()
    if shortest_path:
        print("Shortest Path:", shortest_path)
    else:
        print("No path found!")

    # Refreshing obstacles after search
    new_obstacles = [5, 6, 7, 8]  # New obstacle indices
    shortest_path_with_new_obstacles = astar.astar_search(new_obstacle_indices=new_obstacles)
    if shortest_path_with_new_obstacles:
        print("Shortest Path with New Obstacles:", shortest_path_with_new_obstacles)
    else:
        print("No path found with new obstacles!")

    astar.plot_graph(shortest_path_with_new_obstacles)
