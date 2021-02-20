#!/usr/bin/env python3

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