#!/usr/bin/env python3

import math

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