# Standard Algorithm Implementation
# Sampling-based Algorithms PRM
import math
from typing import NoReturn
import matplotlib.pyplot as plt
from networkx.drawing.layout import kamada_kawai_layout
import numpy as np
import networkx as nx
from numpy.ma import outerproduct
from scipy import spatial

# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path

        np.random.seed(0)

    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        
        min_section_lengths = 1
        pts_list = [p1, p2]

        while (pts_list[1][0]-pts_list[0][0]) ** 2 + (pts_list[1][1] - pts_list[0][1]) ** 2 >= min_section_lengths ** 2:
            tmp_pts = [pts_list[0]]
            for i in range(len(pts_list) - 1):
                # Find midpoint
                midpoint = ((pts_list[i][0] + pts_list[i+1][0])/2, (pts_list[i][1] + pts_list[i+1][1])/2)
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

    def is_point_occupied(self, point):
        """ Checks if the point is occupied in the map
        arguments:
            point - point [row, col]
        return:
            True if occupied, False if not occupied
        """
        return self.map_array[point[0], point[1]] == 0

    def get_pts_in_range(self, pt):
        """ Get all whole points in a range (max distance)
            Assumes range is 1 for now
        arguments:
            pt - point [row, col] (not ints)
        return:
            list of points (all ints)
        """
        pts_list = set()
        pts_list.add((math.floor(pt[0]),    math.floor(pt[1])))
        pts_list.add((math.floor(pt[0]),    math.ceil(pt[1])))
        pts_list.add((math.ceil(pt[0]),     math.floor(pt[1])))
        pts_list.add((math.ceil(pt[0]),     math.ceil(pt[1])))
        return list(pts_list)

    def distance(self, p1, p2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''

        

    
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''

        for i in range(0, n_pts):
            # Generate random unifrom sample:
            ran_row = round(np.random.uniform(size=1, low=0, high=self.map_array.shape[0] - 1)[0])
            ran_col = round(np.random.uniform(size=1, low=0, high=self.map_array.shape[1] - 1)[0])

            # Add to samples list if doesn't collide
            if not self.is_point_occupied([ran_row, ran_col]):
                self.samples.append([ran_row, ran_col])

        ### YOUR CODE HERE ###
        self.samples.append((0, 0))


    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''

        ### YOUR CODE HERE ###
        self.samples.append((0, 0))


    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''

        ### YOUR CODE HERE ###
        self.samples.append((0, 0))


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02), 
        #          (p_id1, p_id2, weight_12) ...]
        pairs = []
        num_neighbors = 10
        kdtree = spatial.KDTree(np.array(self.samples))
        distances, neighbors = kdtree.query(x=self.samples, k=num_neighbors)

        pairs_raw = []

        for neighbor_list in neighbors:
            node = neighbor_list[0]
            for i in range(1, num_neighbors):
                pairs_raw.append((node, neighbor_list[i], distances[node][i]))

        for pair in pairs_raw:
            if not self.check_collision(tuple(self.samples[pair[0]]), tuple(self.samples[pair[1]])):
                pairs.append(pair)

        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01), 
        #                                     (p_id0, p_id2, weight_02), 
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of 
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from(range(len(self.samples)))
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))


    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])
        # Add connections to start/goal

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1), 
        #                (start_id, p_id2, weight_s2) ...]
        start_pairs = []
        goal_pairs = []

        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        