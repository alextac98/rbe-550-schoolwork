#!/usr/bin/env python3

import math
from priorityqueue import PriorityQueue
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

class map2d:
    def __init__(self, grid: list):
        self.map = grid
        self.width = len(self.map[0])
        self.height = len(self.map)
    
    def get_value(self, point: list):
        return self.map[point[0]][point[1]]

    def get_neighbors(self, point: list):
        
        right = self.get_neighbor(point, [0, 1])
        down = self.get_neighbor(point, [1, 0])
        left = self.get_neighbor(point, [0, -1])
        up = self.get_neighbor(point, [-1, 0])

        return (right, down, left, up)

    def get_neighbor(self, point: list, direction: list):
        neighbor = [point[0] + direction[0], point[1] + direction[1]]

        if  neighbor[0] in range(0, self.width) and \
            neighbor[1] in range(0, self.height):
            return neighbor
        else:
            return None
    
    def get_manhattan_distance(self, point1: list, point2: list):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def get_euclidean_distance(self, point1: list, point2: list):
        return math.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]))

    def draw_path(self, start, goal, path: list = [], title: str = "Map ", came_from: dict = None):
        # Visualization of the found path using matplotlib
        fig, ax = plt.subplots(1)
        ax.margins()
        # Draw map
        row = len(self.map)     # map size
        col = len(self.map[0])  # map size
        for i in range(row):
            for j in range(col):
                if self.map[i][j]: 
                    ax.add_patch(Rectangle((j-0.5, i-0.5),1,1,edgecolor='k',facecolor='k'))  # obstacle
                else:          
                    ax.add_patch(Rectangle((j-0.5, i-0.5),1,1,edgecolor='k',facecolor='w'))  # free space
        # Draw path
        for x, y in path:
            ax.add_patch(Rectangle((y-0.5, x-0.5),1,1,edgecolor='k',facecolor='b'))

        if came_from is not None:
            for (current_node, value) in came_from.items():
                from_node = value.get('from')
                cost = value.get('cost')

                plt.text(current_node[1], current_node[0], str(cost), color='blue', fontsize = 12)


        ax.add_patch(Rectangle((start[1]-0.5, start[0]-0.5),1,1,edgecolor='k',facecolor='g'))# start
        ax.add_patch(Rectangle((goal[1]-0.5, goal[0]-0.5),1,1,edgecolor='k',facecolor='r'))  # goal
        # Graph settings
        plt.title(title)
        plt.axis('scaled')
        plt.gca().invert_yaxis()
        plt.show()