from map2d import map2d
from priorityqueue import PriorityQueue

# Basic searching algorithms

def bfs(grid, start, goal):
    '''Return a path found by BFS alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> bfs_path, bfs_steps = bfs(grid, start, goal)
    It takes 10 steps to find a path using BFS
    >>> bfs_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    map = map2d(grid)

    frontier = []
    frontier.append(start)
    visited = {}
    visited[tuple(start)] = None

    while not len(frontier) == 0:
        current = frontier[0]
        if tuple(goal) in visited.keys():
            found = True
            break
        for neighbor in map.get_neighbors(current):
            if neighbor is not None and \
               tuple(neighbor) not in visited.keys() and \
               map.get_value(neighbor) == 0:
                frontier.append(neighbor)
                visited[tuple(neighbor)] = tuple(current)
        frontier.pop(0)

    steps = len(visited)
    curr_point = goal
    while curr_point != start:
        path.append(curr_point)
        curr_point = list(visited.get(tuple(curr_point)))
    path.append(start)
    path.reverse()

    if found:
        print(f"It takes {steps} steps to find a path using BFS")
    else:
        print("No path found")
    return path, steps


def dfs(grid, start, goal):
    '''Return a path found by DFS alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dfs_path, dfs_steps = dfs(grid, start, goal)
    It takes 9 steps to find a path using DFS
    >>> dfs_path
    [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    map = map2d(grid)

    frontier = []
    frontier.append(start)
    visited = {}
    current = None

    while len(frontier) != 0 :
        if tuple(goal) in visited.keys():
            found = True
            break
        visited[tuple(frontier[0])] = tuple(current) if current is not None else None
        current = frontier[0]
        frontier.pop(0)
        valid_neighbors = []
        for neighbor in map.get_neighbors(current):
            if neighbor is not None and \
               tuple(neighbor) not in visited.keys() and \
               map.get_value(neighbor) == 0:
                valid_neighbors.append(neighbor)
        frontier = valid_neighbors + frontier
    
    steps = len(visited)
    curr_point = goal
    while curr_point != start:
        path.append(curr_point)
        curr_point = list(visited.get(tuple(curr_point)))
    path.append(start)
    path.reverse()

    if found:
        print(f"It takes {steps} steps to find a path using DFS")
    else:
        print("No path found")
    return path, steps


def dijkstra(grid, start, goal):
    '''Return a path found by Dijkstra alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dij_path, dij_steps = dijkstra(grid, start, goal)
    It takes 10 steps to find a path using Dijkstra
    >>> dij_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    debug_draw = False

    path = []
    steps = 0
    found = False

    map = map2d(grid)

    frontier = PriorityQueue()
    frontier.put(start, 0)

    came_from = {}
    came_from[tuple(start)] = {
        'from': None,
        'cost': 0
    }

    while not frontier.is_empty():
        (curr_cost, current) = frontier.get()
        frontier.remove()
        if tuple(goal) in came_from.keys():
            found = True
            break

        for neighbor in map.get_neighbors(current):
            if neighbor is None or map.get_value(neighbor) == 1:
                continue
            neighbor_cost = curr_cost + map.get_manhattan_distance(current, neighbor) 
            if tuple(neighbor) not in came_from or \
               neighbor_cost < came_from.get(tuple(neighbor)).get('cost'):
                frontier.put(neighbor, neighbor_cost)
                came_from[tuple(neighbor)] = {
                    'from': current,
                    'cost': neighbor_cost
                }
        if debug_draw: map.draw_path(start = start, goal = goal, path = path, came_from = came_from)
        
    # found = True
    steps = len(came_from)
    curr_point = goal
    while curr_point != start:
        path.append(curr_point)
        curr_point = came_from.get(tuple(curr_point)).get('from')
    path.append(start)
    path.reverse()

    if found:
        print(f"It takes {steps} steps to find a path using Dijkstra")
    else:
        print("No path found")
    return path, steps


def astar(grid, start, goal):
    '''Return a path found by A* alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> astar_path, astar_steps = astar(grid, start, goal)
    It takes 7 steps to find a path using A*
    >>> astar_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    
    debug_draw = False

    path = []
    steps = 0
    found = False

    map = map2d(grid)

    frontier = PriorityQueue()
    frontier.put(start, map.get_manhattan_distance(start, goal))

    came_from = {}
    came_from[tuple(start)] = {
        'from': None,
        'cost': 0
    }

    while not frontier.is_empty():
        (curr_cost, current) = frontier.get()
        frontier.remove()
        if tuple(goal) in came_from.keys():
            found = True
            break

        for neighbor in map.get_neighbors(current):
            if neighbor is None or map.get_value(neighbor) == 1:
                continue
            neighbor_cost = curr_cost - map.get_manhattan_distance(current, goal) + \
                map.get_manhattan_distance(current, neighbor) + \
                map.get_manhattan_distance(neighbor, goal)
            if tuple(neighbor) not in came_from or \
               neighbor_cost < came_from.get(tuple(neighbor)).get('cost'):
                frontier.put(neighbor, neighbor_cost)
                came_from[tuple(neighbor)] = {
                    'from': current,
                    'cost': neighbor_cost
                }
        if debug_draw: map.draw_path(start = start, goal = goal, path = path, came_from = came_from)
        
    # found = True
    steps = len(came_from) - 1
    curr_point = goal
    while curr_point != start:
        path.append(curr_point)
        curr_point = came_from.get(tuple(curr_point)).get('from')
    path.append(start)
    path.reverse()

    if found:
        print(f"It takes {steps} steps to find a path using A*")
    else:
        print("No path found")
    return path, steps


# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()
