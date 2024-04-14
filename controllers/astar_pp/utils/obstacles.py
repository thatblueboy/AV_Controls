import math
OBS = (-900, 9)
radius = 5

def check_for_obstacles(points):
    obstacles_indices = []
    for index, point in enumerate(points):
        distance = math.sqrt((point[0] - OBS[0])**2 + (point[1] - OBS[1])**2)
        if distance <= radius:
            obstacles_indices.append(index)
    return obstacles_indices


if __name__ == '__main__':
    # Example usage:
    points_to_check = [(0, 0), (-900.5, 9), (-901, 9), (-902, 10), (-899, 8)]
    obstacles = check_for_obstacles(points_to_check)
    print("Obstacles encountered:", obstacles)
