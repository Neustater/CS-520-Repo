import heapq
import random
import maze
import copy

# constant used for printing debug
GHOST = "G"


# helpers for copying and distances
def deepcopy2D(arr):
    return [x if not isinstance(x, list) else x[:] for x in arr]

def true_distance(x_1, y_1, x_2, y_2):
    return ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) ** 0.5


"""Ghost object to hold a map of ghosts"""


class ghosts:
    def __init__(self, maze_tuple, num_ghosts, nearest_ghost_queue=[], ghost_occupy={}):
        maze, visitable = maze_tuple
        self.maze, self.visitable = maze, visitable
        self.maze_size = len(self.maze)
        self.num_ghosts = num_ghosts
        self.nearest_ghost_queue = []  # uses heapq for fast lookup of closest ghost
        self.ghost_occupy = {}  # uses dict for O(1) lookup of if a ghost occupies a cell (true if in space, false if in wall)

        if not ghost_occupy and not nearest_ghost_queue:
            self.generate_ghosts()
        else:
            self.nearest_ghost_queue = nearest_ghost_queue.copy()
            self.ghost_occupy = ghost_occupy.copy()

    """generates ghosts only in positions initially visitable by a player (uses visitable from maze generation)"""

    def generate_ghosts(self):
        for _ in range(self.num_ghosts):
            position, _ = random.choice(list(self.visitable.items()))
            self.nearest_ghost_queue.append((true_distance(0, 0, position[0], position[1]), position))
            self.ghost_occupy[position] = True
        heapq.heapify(self.nearest_ghost_queue)

    """Used to get ghost distance heapq queue from a given position that is not the current agent position (used in agents 4 and 5"""
    def get_ghost_dist(self, x_pos, y_pos):
        temp_array = [(true_distance(x_pos, y_pos, x[1][0], x[1][1]), x[1]) for x in self.nearest_ghost_queue if true_distance(x_pos, y_pos, x[1][0], x[1][1]) <= 10]
        heapq.heapify(temp_array)

        return temp_array

    """Moves ghosts and creates new queue and dict"""
    def move_ghosts(self, x_player, y_player):
        new_ghost_queue = []
        new_ghost_occupy = {}
        for z in self.nearest_ghost_queue:
            _, (x_pos, y_pos) = z
            directions = []
            if x_pos - 1 == -1:
                directions.append((x_pos + 1, y_pos))
            elif x_pos + 1 == self.maze_size:
                directions.append((x_pos - 1, y_pos))
            else:
                directions.append((x_pos + 1, y_pos))
                directions.append((x_pos - 1, y_pos))

            if y_pos - 1 == -1:
                directions.append((x_pos, y_pos + 1))
            elif y_pos + 1 == self.maze_size:
                directions.append((x_pos, y_pos - 1))
            else:
                directions.append((x_pos, y_pos + 1))
                directions.append((x_pos, y_pos - 1))

            chosen_direction = random.choice(directions)

            if self.maze[chosen_direction[1]][chosen_direction[0]] == maze.BLOCKED:
                if round(random.uniform(0, 1)) == 0:
                    chosen_direction = (x_pos, y_pos)

            new_ghost_queue.append(
                           (true_distance(x_player, y_player, chosen_direction[0], chosen_direction[1]),
                            chosen_direction))
            if self.maze[chosen_direction[1]][chosen_direction[0]] == maze.BLOCKED:
                new_ghost_occupy[chosen_direction] = False
            else:
                new_ghost_occupy[chosen_direction] = True
        heapq.heapify(new_ghost_queue)
        self.nearest_ghost_queue = new_ghost_queue
        self.ghost_occupy = new_ghost_occupy

    # Returns a map with ghosts only for debug/printing
    def return_ghosts(self, maze):
        temp_maze = deepcopy2D(maze)
        temp_ghost_queue = self.nearest_ghost_queue[::]
        while temp_ghost_queue:
            ghost_tuple = heapq.heappop(temp_ghost_queue)
            _, (x_pos, y_pos) = ghost_tuple
            temp_maze[y_pos][x_pos] = GHOST
        return temp_maze
