# Weighted A* implementation

# Import Math Libraries
import math
from matplotlib import markers
import numpy as np

import networkx as nx
import queue

import matplotlib.pyplot as plt

class A_Star:
    def __init__(self, map_array, g_weight=1, h_weight=1):
        self.set_map(map_array=map_array)
        self.reset_data()
    
        self.g_weight = float(g_weight)
        self.h_weight = float(h_weight)
        pass

    def set_map(self, map_array):
        self.map_array = map_array
        self.map_row = map_array.shape[0]
        self.map_col = map_array.shape[1]
        self.reset_data()

    def get_neighbors(self, point: tuple):
        right   = self.get_neighbor(point, [0, 1])
        down    = self.get_neighbor(point, [1, 0])
        left    = self.get_neighbor(point, [0, -1])
        up      = self.get_neighbor(point, [-1, 0])

        return (right, down, left, up)

    def get_neighbor(self, point: tuple, direction: list):
        neighbor = [point[0] + direction[0], point[1] + direction[1]]

        if  neighbor[0] in range(0, self.map_col) and \
            neighbor[1] in range(0, self.map_row):
            return tuple(neighbor)
        else:
            return None

    def is_point_occupied(self, point):
        if self.map_array[point[0], point[1]] == 0:
            return True
        return False
    
    def get_euclidean_distance(self, point1: tuple, point2: tuple):
        return math.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]))

    def reset_data(self):
        self.frontier = queue.PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.path = []
 
    def draw_map(self, start=None, goal=None):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw path if found
        x_pts = [x for x, y in self.path]
        y_pts = [y for x, y in self.path]

        ax.plot(y_pts, x_pts, color='blue')

        # Draw start and goal
        if start is not None:
            ax.plot(start[1], start[0], color='red', marker='o')

        if goal is not None:
            ax.plot(goal[1], goal[0], color='green', marker='o')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    def find_path(self, start, goal):

        self.reset_data()
        self.frontier.put((0, start))

        self.came_from[start] = None
        self.cost_so_far[start] = 0

        while not self.frontier.empty():
            weight, current = self.frontier.get()

            if current == goal:
                break

            for next in self.get_neighbors(point=current):
                if next is None or self.is_point_occupied(next):
                    continue
                new_cost = self.g_weight * self.cost_so_far[current] + self.h_weight * self.get_euclidean_distance(current, next)
                if next not in self.cost_so_far.keys() or new_cost < self.cost_so_far.get(next):
                    self.cost_so_far[next] = new_cost
                    priority = new_cost + self.get_euclidean_distance(goal, next)
                    self.frontier.put((priority, next))
                    self.came_from[next] = current
        
        curr_point = goal
        while curr_point != start:
            self.path.append(curr_point)
            curr_point = self.came_from.get(curr_point)

        self.draw_map(start = start, goal = goal)