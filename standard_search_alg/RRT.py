# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*
import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 1000.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node 
        # self.start.parent = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.vertices_new = []
        self.found = False                    # found flag
        

    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    
    def distance(self, p1: Node, p2: Node):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        return math.sqrt((p1.row - p2.row) ** 2 + (p1.col - p2.col) ** 2)

    def midpoint(self, p1: Node, p2: Node):
        ''' Calculate midpoint between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            midpoint - [row column]
        '''
        return Node((p1.row + p2.row)/2, (p1.col + p2.col)/2)

    
    def check_collision(self, node1: Node, node2: Node):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            False if the new node is valid to be connected
        '''
        min_section_lengths = 1
        pts_list = [node1, node2]

        while (pts_list[1].row - pts_list[0].row) ** 2 + (pts_list[1].col - pts_list[0].col) ** 2 >= min_section_lengths ** 2:
            tmp_pts = [pts_list[0]]
            for i in range(len(pts_list) - 1):
                # Find midpoint
                midpoint = self.midpoint(pts_list[i], pts_list[i+1])
                # Check collisions
                test_pts = self.get_pts_in_range(midpoint)
                for pt in test_pts:
                    if self.is_point_occupied(pt):
                        return True
                # Add midpoint to the collision list if no collision
                tmp_pts.append(midpoint)
                tmp_pts.append(pts_list[i+1])
            pts_list = tmp_pts
        return False

    def get_pts_in_range(self, pt:Node):
        """ Get all whole points in a range (max distance)
            Assumes range is 1 for now
        arguments:
            pt - point [row, col] (not ints)
        return:
            list of points (all ints)
        """
        pts_list = set()
        pts_list.add(Node(math.floor(pt.row),    math.floor(pt.col)))
        pts_list.add(Node(math.floor(pt.row),    math.ceil(pt.col)))
        pts_list.add(Node(math.ceil(pt.row),     math.floor(pt.col)))
        pts_list.add(Node(math.ceil(pt.row),     math.ceil(pt.col)))
        return list(pts_list)

    def is_point_occupied(self, point:Node):
        """ Checks if the point is occupied in the map
        arguments:
            point - point [row, col]
        return:
            True if occupied, False if not occupied
        """
        if self.map_array[point.row, point.col] == 0:
            return True
        return False

    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point
                        Number from 0% to 100%
                            0 => Goal will be chosen 0% of the time
                            100 => Goal will be chosen 100% of the time
        return:
            point - the new point
        '''
        
        if np.random.randint(100) < goal_bias:
            return self.goal

        return Node(np.random.randint(self.size_row), np.random.randint(self.size_col))
    
    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        return self.vertices[0]


    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance 
        '''
        
        potential_neighbors = []
        # Get all nodes within neighbor range
        for node in self.vertices:
            if self.distance(new_node, node) < neighbor_size:
                potential_neighbors.append(node)
        return potential_neighbors


    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###

    
    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')
        
        # for node in self.vertices_new:
        #     plt.plot(node.col, node.row, markersize=2, marker='o', color='r')
        
        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col and cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()


    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        
        goal_bias = 2
        neighbor_threshold = 40
        new_point_distance = 20
        # Remove previous result
        self.init_map()

        for i in range(0, n_pts):
            # Pick a random position
            point = self.get_new_point(goal_bias=goal_bias)
            # If occupied, go to next iteration
            if self.is_point_occupied(point):
                continue
            # Connect to nearest neighbor
            neighbors = self.get_neighbors(point, neighbor_threshold)
            for neighbor in neighbors:
                if not self.check_collision(point, neighbor):
                    dist = self.distance(point, neighbor)
                    if dist < point.cost:
                        point.parent = neighbor
                        point.cost = dist

            if point.parent is not None:
                # Check if current point is goal
                # self.vertices.append(point)
                if point == self.goal:
                    self.vertices.append(point)
                    self.found = True
                    break
                
                if self.distance(point, point.parent) < new_point_distance:
                    self.vertices.append(point)
                    continue

                # Find point_new that is new_point_distance away from point
                # 1. Get unit vector between selected neighbor and new_point
                u_vec = [(point.row - point.parent.row) / point.cost, (point.col - point.parent.col) / point.cost] 
                # 2. Calculate point_new using equation [point + new_point_distance * unit_vector]
                point_new = Node(point.parent.row + u_vec[0] * new_point_distance, point.parent.col + u_vec[1] * new_point_distance)
                point_new.parent = point.parent
                # 3. Add new_point to the vertices list
                self.vertices.append(point_new)

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
        # Draw result
        self.draw_map()


    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            neighbor_size - the neighbor distance
        
        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
