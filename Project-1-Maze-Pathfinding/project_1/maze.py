import random
from queue import Queue

#Constants for maze array
maze_size = 51
EMPTY = "_"
VISITED = "V"
BLOCKED = "#"

"""
Function to create maze
"""

def make_maze():
    maze = [[BLOCKED for x in range(maze_size)] for y in range(maze_size)]  # defaults all cells to blocked
    for y in range(maze_size):
        for x in range(maze_size):
            if random.uniform(0, 1) > 0.28:  # if random number is greater than threshold for cell it will be unblocked
                maze[y][x] = EMPTY
    # start and end are forced to be unblocked
    maze[0][0] = EMPTY
    maze[maze_size - 1][maze_size - 1] = EMPTY
    check_result = check_maze(maze)
    if check_result == False:
        return make_maze()      #if maze is not solvable try to make a new one
    else:
        return check_result


# Function to print a maze in its current state

def print_maze(maze):
    print("Maze: ")
    for y in range(maze_size):
        for x in range(maze_size):
            print(maze[y][x], end="")
        print("")


""" Adds newly visited cells to fringe and visited (seen cells)"""

def validate_cell(maze, cur, temp_x, temp_y, visited, fringe):
    if 0 <= temp_y < maze_size and 0 <= temp_x < maze_size and (
            maze[temp_y][temp_x] is not BLOCKED):
        if visited.get((temp_x, temp_y)) is None:
            temp_tuple = (temp_x, temp_y)
            fringe.put(temp_tuple)
            visited[(temp_x, temp_y)] = visited[cur] + 1 #adds cell to visited with shortest path list and length
    return visited


""" Implemented using BFS algorithm to check all visitable nodes and their shortest path to goal (needed for creating ghosts + speedup) """


def check_maze(maze, x_pos=maze_size - 1, y_pos=maze_size - 1):
    cur = (x_pos, y_pos)
    fringe = Queue()
    visited = {}

    if maze[y_pos][x_pos] == BLOCKED:
        return False

    fringe.put(cur)
    visited[cur] = (0)

    while not fringe.empty():

        cur = fringe.get()
        x_pos, y_pos = cur

        """Series to check all four directions for BFS"""
        # check_left
        visited = validate_cell(maze, cur, x_pos - 1, y_pos, visited, fringe)
        # check up
        visited = validate_cell(maze, cur, x_pos, y_pos - 1, visited, fringe)
        # check_right
        visited = validate_cell(maze, cur, x_pos + 1, y_pos, visited, fringe)
        # check_down
        visited = validate_cell(maze, cur, x_pos, y_pos + 1, visited, fringe)

    if visited.get((0, 0)) is None: #returns False if there is no viable path (cannot reach start from the end)
        return False
    else:
        return maze, visited

