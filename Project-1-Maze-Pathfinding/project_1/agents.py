import copy
import heapq
from math import inf
from queue import PriorityQueue

import ghosts

# Constants for maze array
BLOCKED = "#"
EMPTY = "_"
VISITED = "V"
PATH = "P"


# helper to copy 2D array faster
def deepcopy2D(arr):
    return [x if not isinstance(x, list) else x[:] for x in arr]


# euclidian distance helper
def true_distance(x_1, y_1, x_2, y_2):
    return ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) ** 0.5


""" Class to represent an agent (all agent types use same class and have corresponding walk algorithms walk_agent_X) """


class agent:
    def __init__(self, input_maze, visitable):
        self.generate_ghosts = None  # stores current ghost pattern object being used by agent
        self.maze_size = len(input_maze)  # stores current maze size being used by agent
        self.path_traveled = []  # stores path_traveled used by walk agent
        self.dist_traveled = 0  # stores dist_traveled used by walk agent
        self.maze_master = input_maze  # stores a copy of current maze in case maze is modified
        self.path = []  # path created by A* algo for planning
        self.dist_path = 0  # current length of path from start to current node used in A*
        self.goal_dist = 0  # distance of node from goal used in A*
        self.maze = deepcopy2D(input_maze)  # copy of maze used by agent (may be edited once passed in)
        self.visitable = copy.deepcopy(
            visitable)  # distance of each visitable cell from the goal (shortest path using bfs from maze creation)
        self.fringe = PriorityQueue()  # fringe used in A*
        self.visited = {}  # closed set used in A*
        self.x_pos = 0  # x postion of current node in A* or Agent in walk
        self.y_pos = 0  # y postion of current node in A* or Agent in walk
        self.heat_map = {}  # last known map of ghosts only used by Agent 5

    """ Fully resets agent allowing for a new agent type walk to be used with same agent object """

    def hard_reset_agent(self):
        self.path_traveled = []
        self.dist_traveled = 0
        self.path = []
        self.goal_dist = 0
        self.dist_path = 0
        self.maze = deepcopy2D(self.maze_master)
        self.fringe = PriorityQueue()
        self.visited = {}
        self.generate_ghosts = None
        self.x_pos = 0
        self.y_pos = 0
        self.heat_map = {}

    """ Resets agent allowing for new search of same agent type (keeps map, path and distance traveled so far) """

    def soft_reset_agent(self):
        self.path = []
        self.goal_dist = 0
        self.dist_path = 0
        self.fringe = PriorityQueue()
        self.visited = {}
        self.x_pos = 0
        self.y_pos = 0

    """Heuristic of distance to goal (shortest path as found when creating the maze)"""

    def get_dist_to_goal(self, temp_x, temp_y):
        if self.visitable.get((temp_x, temp_y)) is not None:
            return self.visitable.get((temp_x, temp_y))
        else:
            return False

    """ Checks if cell exists in visited and updates with shortest otherwise adds it to fringe (used by A*)"""

    def validate_cell(self, temp_x, temp_y, dist_path):
        # if newly added node is the goal then add it to the fringe
        if temp_x == self.x_pos and temp_y == self.y_pos:
            temp_tuple = ((self.get_dist_to_goal(temp_x, temp_y) + 0), 0, temp_x, temp_y,
                          None)
            self.fringe.put(temp_tuple)
            return

        # if this node has been visited already and the current way it is being added is in fewer steps than before update in visited
        if self.visited.get((temp_x, temp_y)) is not None and self.visited[(temp_x, temp_y)][0] > \
                self.visited[(self.x_pos, self.y_pos)][0] + 1:
            self.visited[(temp_x, temp_y)] = (self.visited[(self.x_pos, self.y_pos)][0] + 1, (self.x_pos, self.y_pos))
            return

        # if this node has been not visited already add to fringe
        if self.visitable.get((temp_x, temp_y)) is not None and self.visited.get((temp_x, temp_y)) is None and (
                self.generate_ghosts is None or self.generate_ghosts.ghost_occupy.get(
            (self.x_pos, self.y_pos)) is None):
            temp_tuple = ((self.get_dist_to_goal(temp_x, temp_y) + dist_path), dist_path, temp_x, temp_y,
                          (self.x_pos, self.y_pos))
            self.fringe.put(temp_tuple)

        return False

    """ Implemented using A* algorithm"""

    # version 0 is default, other versions are for certain agents
    def find_path(self, version=0):
        # creates cur from agents current position

        if self.maze[self.y_pos][self.x_pos] == BLOCKED:
            return False

        # starting path is goal return true
        if self.x_pos == self.maze_size - 1 and self.y_pos == self.maze_size - 1:
            self.dist_path = 0
            self.path.append((self.x_pos, self.y_pos))
            return True

        # versioning allows modified checking/ heuristic for agents 4, 4 blind and 5
        if version == 0:
            self.validate_cell(self.x_pos, self.y_pos, 0)
        elif version == 1:
            self.validate_cell_avoid(self.x_pos, self.y_pos, 0)
        elif version == 2:
            self.validate_cell_avoid_blind(self.x_pos, self.y_pos, 0)
        elif version == 3:
            self.validate_cell_avoid_blind_optimized(self.x_pos, self.y_pos, 0)

        # add starting point to fringe
        initial = self.fringe.get()
        self.fringe.put(initial)

        while not self.fringe.empty():

            # gets new current of fringe and sets agents location
            cur = self.fringe.get_nowait()

            # break if node is the goal
            if (cur[2], cur[3]) == (self.maze_size - 1, self.maze_size - 1):
                self.visited[(cur[2], cur[3])] = (self.visited[cur[4]][0] + 1, cur[4])
                break

            self.goal_dist, self.dist_path, self.x_pos, self.y_pos, parent = cur

            # if visited already then don't assess again
            if self.visited.get((self.x_pos, self.y_pos)) is not None:
                continue

            # Adds node to visited
            if cur != initial:
                self.visited[(self.x_pos, self.y_pos)] = (self.visited[cur[4]][0] + 1, cur[4])
            else:
                self.visited[(self.x_pos, self.y_pos)] = (0, None)

            # versioning allows modified checking/ heuristic for agents 4, 4 blind and 5
            if version == 0:
                # check all directions to add to fringe
                # check_left
                self.validate_cell(self.x_pos - 1, self.y_pos, self.dist_path + 1)
                # check up
                self.validate_cell(self.x_pos, self.y_pos - 1, self.dist_path + 1)
                # check_right
                self.validate_cell(self.x_pos + 1, self.y_pos, self.dist_path + 1)
                # check_down
                self.validate_cell(self.x_pos, self.y_pos + 1, self.dist_path + 1)
            elif version == 1:
                # check all directions to add to fringe
                # check_left
                self.validate_cell_avoid(self.x_pos - 1, self.y_pos, self.dist_path + 1)
                # check up
                self.validate_cell_avoid(self.x_pos, self.y_pos - 1, self.dist_path + 1)
                # check_right
                self.validate_cell_avoid(self.x_pos + 1, self.y_pos, self.dist_path + 1)
                # check_down
                self.validate_cell_avoid(self.x_pos, self.y_pos + 1, self.dist_path + 1)
            elif version == 2:
                # check all directions to add to fringe
                # check_left
                self.validate_cell_avoid_blind(self.x_pos - 1, self.y_pos, self.dist_path + 1)
                # check up
                self.validate_cell_avoid_blind(self.x_pos, self.y_pos - 1, self.dist_path + 1)
                # check_right
                self.validate_cell_avoid_blind(self.x_pos + 1, self.y_pos, self.dist_path + 1)
                # check_down
                self.validate_cell_avoid_blind(self.x_pos, self.y_pos + 1, self.dist_path + 1)
            elif version == 3:
                # check all directions to add to fringe
                # check_left
                self.validate_cell_avoid_blind_optimized(self.x_pos - 1, self.y_pos, self.dist_path + 1)
                # check up
                self.validate_cell_avoid_blind_optimized(self.x_pos, self.y_pos - 1, self.dist_path + 1)
                # check_right
                self.validate_cell_avoid_blind_optimized(self.x_pos + 1, self.y_pos, self.dist_path + 1)
                # check_down
                self.validate_cell_avoid_blind_optimized(self.x_pos, self.y_pos + 1, self.dist_path + 1)

        # returns false if the goal could not be reaches
        if self.visited.get((self.maze_size - 1, self.maze_size - 1)) is None:
            return False

        # resets certain global values
        self.goal_dist, self.dist_path, self.x_pos, self.y_pos = (
            0, self.visited.get((self.maze_size - 1, self.maze_size - 1)), self.maze_size - 1, self.maze_size - 1)

        # back traces through visited to get the shortest path
        back_trace_path = []
        pos = (self.maze_size - 1, self.maze_size - 1)
        while pos is not None:
            back_trace_path.append(pos)
            pos = self.visited.get(pos)[1]

        self.path = back_trace_path

        return True

    """Agent 1 function to walk each step and check for collisions with ghosts"""

    def walk_agent_1(self, ghost_map):
        # resets for agent_1 and sets up for walk from 0,0
        self.hard_reset_agent()
        self.dist_traveled = 0
        self.find_path()
        self.x_pos, self.y_pos = self.path.pop()
        self.generate_ghosts = ghost_map

        # walks along path until goal or until death (uses speed up array created with maze recommended by prof for path)
        while True:
            # returns false if collides with ghost
            if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                return False

            # returns true if reaches end
            if self.x_pos == self.maze_size - 1 and self.y_pos == self.maze_size - 1:
                # self.print_traveled()
                return True

            # removes node from path and adds to traveled
            self.x_pos, self.y_pos = self.path.pop()
            self.path_traveled.append((self.x_pos, self.y_pos))

            self.dist_traveled += 1
            # returns false if collides with ghost
            if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                return False
            # calls to move ghosts
            self.generate_ghosts.move_ghosts(self.x_pos, self.y_pos)

    """Attempts to move away from coords provided in direction resulting in furthest distance"""

    # Unlimited flag can be turned off to only move away from ghosts in a radius of 6 units
    def move_away(self, x_pos, y_pos, unlimited=True):
        cur_longest_dist = true_distance(self.x_pos, self.y_pos, x_pos, y_pos)
        cur_longest_move = (self.x_pos, self.y_pos)

        # checks if player can move in all 4 directions and which one results in furthest direction from nearest ghost
        temp_dist = true_distance(x_pos, y_pos, self.x_pos + 1, self.y_pos)
        if self.x_pos + 1 != self.maze_size and (self.maze[self.y_pos][
                                                     self.x_pos + 1] != BLOCKED) and temp_dist > cur_longest_dist:
            cur_longest_move = (self.x_pos + 1, self.y_pos)
            cur_longest_dist = temp_dist
        temp_dist = true_distance(x_pos, y_pos, self.x_pos, self.y_pos + 1)
        if self.y_pos + 1 != self.maze_size and (self.maze[self.y_pos + 1][
                                                     self.x_pos] != BLOCKED) and temp_dist > cur_longest_dist:
            cur_longest_move = (self.x_pos, self.y_pos + 1)
            cur_longest_dist = temp_dist
        temp_dist = true_distance(x_pos, y_pos, self.x_pos - 1, self.y_pos)
        if self.x_pos - 1 != -1 and (self.maze[self.y_pos][self.x_pos - 1] != BLOCKED) and temp_dist > cur_longest_dist:
            cur_longest_move = (self.x_pos - 1, self.y_pos)
            cur_longest_dist = temp_dist
        temp_dist = true_distance(x_pos, y_pos, self.x_pos, self.y_pos - 1)
        if self.y_pos - 1 != -1 and (self.maze[self.y_pos - 1][self.x_pos] != BLOCKED) and temp_dist > cur_longest_dist:
            cur_longest_move = (self.x_pos, self.y_pos - 1)
            cur_longest_dist = temp_dist

        # updates move ment if move away is possible
        if cur_longest_move != (self.x_pos, self.y_pos) and (cur_longest_dist < 6 or unlimited):
            self.x_pos, self.y_pos = cur_longest_move
            self.path_traveled.append(cur_longest_move)
            self.dist_traveled += 1

    """Agent 2 wrapper function to walk from 0,0"""

    def walk_agent_2(self, ghost_map):
        return self.walk_agent_bootstrappable(0, 0, ghost_map)

    """Agent 2 behaving function to walk each step and check for collisions with ghosts 
        + retreats if all are blocked"""

    # version allows this to function slighly differently depending on the agent calling it
    # replan_fail flag used by agent 3 to reduce execution time by considering a simulation failed if a move away occurs

    def walk_agent_bootstrappable(self, x_pos, y_pos, ghost_map, version=0, replan_fail=False):
        # resets special agent_2 for use by other agents
        self.hard_reset_agent()
        self.generate_ghosts = ghost_map
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.dist_traveled = 0
        self.path_traveled.append((self.x_pos, self.y_pos))
        xray = False  # xray is set to true in agents that can "see" in walls (agent 4)
        unlimited = True
        if version == 1:
            xray = True
            unlimited = False
        if version == 2 or version == 3:
            unlimited = False
        while True:
            # returns false if collides with ghost
            if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                return False

            # returns true if reaches goal
            if self.x_pos == self.maze_size - 1 and self.y_pos == self.maze_size - 1:
                return True

            x_temp = self.x_pos
            y_temp = self.y_pos
            # attempts to find a new path
            if self.find_path(version) is False:
                # if no paths exist, finds the nearest ghost and moves away
                temp_ghost_queue = self.generate_ghosts.nearest_ghost_queue[::]
                self.soft_reset_agent()  # resets path data since find path could not find a path
                self.x_pos, self.y_pos = x_temp, y_temp  # sets x and y values that were removed by soft reset
                if replan_fail == True:
                    return False
                # checks through queue of ghosts until a visible ghost is found
                while temp_ghost_queue:
                    z, check = heapq.heappop(temp_ghost_queue)
                    if self.generate_ghosts.ghost_occupy.get(check) is True or xray == True:
                        self.move_away(check[0], check[1], unlimited)
                        break
                # returns false if collides with ghost
                if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                    return False
                self.generate_ghosts.move_ghosts(self.x_pos, self.y_pos)
                continue
            self.x_pos, self.y_pos = x_temp, y_temp

            # gets next position off path
            x_pos, y_pos = self.path.pop()
            # in rare case path contains self
            if (x_pos, y_pos) == (self.x_pos, self.y_pos):
                x_pos, y_pos = self.path.pop()

            # resets agent for next search
            temp_path = self.path
            self.soft_reset_agent()
            self.x_pos, self.y_pos = (x_pos, y_pos)
            self.path = temp_path
            self.path_traveled.append((self.x_pos, self.y_pos))
            self.dist_traveled += 1

            # returns false if collides with ghost
            if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                return False
            self.generate_ghosts.move_ghosts(self.x_pos, self.y_pos)

    """Stands still for one turn then runs a regular agent2 simulation (used by Agent 3)"""

    def stationary(self, x_pos, y_pos, ghost_map):
        self.hard_reset_agent()
        self.generate_ghosts = ghost_map
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.dist_traveled = 0
        self.path_traveled.append((self.x_pos, self.y_pos))

        if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
            return False
        self.generate_ghosts.move_ghosts(self.x_pos, self.y_pos)
        return self.walk_agent_bootstrappable(x_pos, y_pos, ghost_map, 0, True)

    """Agent 3 function to walk each step and check for collisions with ghosts, leverages bootstrappable agent"""

    def walk_agent_3(self, ghost_map, blind=False):
        # setup for agents
        self.hard_reset_agent()
        self.dist_traveled = 0
        self.generate_ghosts = ghost_map
        self.path_traveled.append((self.x_pos, self.y_pos))
        sim_depth = 8  # number of simulations in each direction
        # runs until death or goal
        while True:

            # collision with ghosts
            if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                return False
            # reaches goal
            if self.x_pos == self.maze_size - 1 and self.y_pos == self.maze_size - 1:
                return True

            # INITILIZES SHORTEST PATH TO WORST CASES
            shortest_avg_path_length = float(inf)
            shortest_avg_path = None
            successes = 0

            # IF BLIND IS ENABLED, THE GHOSTMAP REMOVES ALL GHOSTS THAT ARE INSIDE WALLS WHEN RUNNING SIMULATIONS
            if blind == True:
                master_temp_queue = self.generate_ghosts.nearest_ghost_queue.copy()
                master_temp_occupy = self.generate_ghosts.ghost_occupy.copy()
                for k, r in self.generate_ghosts.ghost_occupy.items():
                    if r == False:
                        list(filter(lambda a: a != k, master_temp_queue))
                        del master_temp_occupy[k]
                heapq.heapify(master_temp_queue)
            else:
                master_temp_queue = self.generate_ghosts.nearest_ghost_queue.copy()
                master_temp_occupy = self.generate_ghosts.ghost_occupy.copy()

            """Series of if statements that attempt to simulate each direction and standing 
            still for one move, highest survival rate and shortest path of sim_depth tries 
            is picked in a tie"""
            simulator_agent = agent(self.maze_master, self.visitable)
            # simulate down
            if self.visitable.get((self.x_pos, self.y_pos + 1)) is not None:
                temp_success = 0  # counts successes
                temp_length = 0  # keeps total distances of success runs

                # simulates runs
                for _ in range(sim_depth):
                    temp_ghost = ghosts.ghosts((self.maze_master, self.visitable), self.generate_ghosts.num_ghosts,
                                               master_temp_queue,
                                               master_temp_occupy)
                    x = simulator_agent.walk_agent_bootstrappable(self.x_pos, self.y_pos + 1, temp_ghost, 0, True)
                    if x is not False:
                        # adds to successes if goal reaches
                        temp_success += 1
                        temp_length += len(simulator_agent.path_traveled)
                # calculates avg length of run
                if temp_success != 0:
                    temp_length = temp_length / temp_success

                # if shortest and most successfull then considers direction best possible
                if (temp_success > successes) or (temp_success == successes and temp_length < shortest_avg_path_length):
                    successes = temp_success
                    shortest_avg_path_length = temp_length
                    shortest_avg_path = (self.x_pos, self.y_pos + 1)
            # simulate right
            if self.visitable.get((self.x_pos + 1, self.y_pos)) is not None:
                temp_success = 0
                temp_length = 0
                for _ in range(sim_depth):
                    temp_ghost = ghosts.ghosts((self.maze_master, self.visitable), self.generate_ghosts.num_ghosts,
                                               master_temp_queue,
                                               master_temp_occupy)
                    x = simulator_agent.walk_agent_bootstrappable(self.x_pos + 1, self.y_pos, temp_ghost, 0, True)
                    if x is not False:
                        temp_success += 1
                        temp_length += len(simulator_agent.path_traveled)
                if temp_success != 0:
                    temp_length = temp_length / temp_success
                if (temp_success > successes) or (temp_success == successes and temp_length < shortest_avg_path_length):
                    successes = temp_success
                    shortest_avg_path_length = temp_length
                    shortest_avg_path = (self.x_pos + 1, self.y_pos)
            # simulate up
            if self.visitable.get((self.x_pos, self.y_pos - 1)) is not None:
                temp_success = 0
                temp_length = 0
                for _ in range(sim_depth):
                    temp_ghost = ghosts.ghosts((self.maze_master, self.visitable), self.generate_ghosts.num_ghosts,
                                               master_temp_queue,
                                               master_temp_occupy)
                    x = simulator_agent.walk_agent_bootstrappable(self.x_pos, self.y_pos - 1, temp_ghost, 0, True)
                    if x is not False:
                        temp_success += 1
                        temp_length += len(simulator_agent.path_traveled)
                if temp_success != 0:
                    temp_length = temp_length / temp_success
                if (temp_success > successes) or (temp_success == successes and temp_length < shortest_avg_path_length):
                    successes = temp_success
                    shortest_avg_path_length = temp_length
                    shortest_avg_path = (self.x_pos, self.y_pos - 1)
            # simulate left
            if self.visitable.get((self.x_pos - 1, self.y_pos)) is not None:
                temp_success = 0
                temp_length = 0
                for _ in range(sim_depth):
                    temp_ghost = ghosts.ghosts((self.maze_master, self.visitable), self.generate_ghosts.num_ghosts,
                                               master_temp_queue,
                                               master_temp_occupy)
                    x = simulator_agent.walk_agent_bootstrappable(self.x_pos - 1, self.y_pos, temp_ghost, 0, True)
                    if x is not False:
                        temp_success += 1
                        temp_length += len(simulator_agent.path_traveled)
                if temp_success != 0:
                    temp_length = temp_length / temp_success
                if (temp_success > successes) or (temp_success == successes and temp_length < shortest_avg_path_length):
                    successes = temp_success
                    shortest_avg_path_length = temp_length
                    shortest_avg_path = (self.x_pos - 1, self.y_pos)
            temp_success = 0
            temp_length = 0
            # simulate stationary
            for _ in range(sim_depth):
                temp_ghost = ghosts.ghosts((self.maze_master, self.visitable), self.generate_ghosts.num_ghosts,
                                           master_temp_queue,
                                           master_temp_occupy)
                x = simulator_agent.stationary(self.x_pos, self.y_pos, temp_ghost)
                if x is not False:
                    temp_success += 1
                    temp_length += len(simulator_agent.path_traveled)
            if temp_success != 0:
                temp_length = temp_length / temp_success
            if (temp_success > successes) or (temp_success == successes and temp_length < shortest_avg_path_length):
                successes = temp_success
                shortest_avg_path_length = temp_length
                shortest_avg_path = (self.x_pos, self.y_pos)

            """Attempts to move away from ghosts if all paths result in death"""

            x_temp, y_temp = self.x_pos, self.y_pos
            if temp_success == 0:
                temp_ghost_queue = self.generate_ghosts.nearest_ghost_queue[::]
                self.soft_reset_agent()
                self.x_pos, self.y_pos = x_temp, y_temp
                while temp_ghost_queue:
                    _, check = heapq.heappop(temp_ghost_queue)  # ghosts are stored in a heapq to find nearest quickly
                    if self.generate_ghosts.ghost_occupy.get(
                            check) is True:  # ghosts are stored identically in a dictionary which is true if in open space and false in walls
                        self.move_away(check[0], check[1])
                        break
                # returns false if collide with ghosts
                if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                    return False
                self.generate_ghosts.move_ghosts(self.x_pos, self.y_pos)
                continue

            # updates new distances and path traveled, only if not standing still
            if (self.x_pos, self.y_pos) != shortest_avg_path:
                self.dist_traveled += 1
                self.path_traveled.append(shortest_avg_path)
            self.x_pos, self.y_pos = shortest_avg_path
            # returns false if collide with ghosts
            if self.generate_ghosts.ghost_occupy.get((self.x_pos, self.y_pos)) is not None:
                return False
            self.generate_ghosts.move_ghosts(self.x_pos, self.y_pos)

    """returns a map of traveled positons by agent, only for print/ debug"""

    def return_traveled(self, maze):
        temp_maze = deepcopy2D(maze)
        for i in self.path_traveled:
            temp_x, temp_y = i
            temp_maze[temp_y][temp_x] = "@"

        return temp_maze

    """special version of validate cell used by Agent 4 (version flag 1) which attempts to avoid ghosts using a danger factor (offset)"""

    def validate_cell_avoid(self, temp_x, temp_y, dist_path):
        # adds current node to fringe if goal and includes danger factor in weight
        if temp_x == self.x_pos and temp_y == self.y_pos:
            temp_ghosts = self.generate_ghosts.get_ghost_dist(temp_x, temp_y)
            closest_offset = 0
            while temp_ghosts:
                closest = temp_ghosts.pop()
                # if a ghost is 5 or fewer units away, it is added to the danger weight
                if closest[0] <= 5:
                    if self.generate_ghosts.ghost_occupy.get(closest[1]) is True:
                        closest_offset += (6 - (
                        closest[0]))  # 6 - distance of ghost away gives value in which closer ghosts are more dangerous
                    else:
                        closest_offset += (6 - (closest[0]) * 0.8)

            temp_tuple = ((self.get_dist_to_goal(temp_x, temp_y) + closest_offset), 0, temp_x, temp_y,
                          None)
            self.fringe.put(temp_tuple)
            return

        # updates weight if shorter path is found, keeps danger factor
        if self.visited.get((temp_x, temp_y)) is not None and self.visited[(temp_x, temp_y)][0] > \
                self.visited[(self.x_pos, self.y_pos)][0] + 1:
            self.visited[(temp_x, temp_y)] = (self.visited[(self.x_pos, self.y_pos)][0] + 1, (self.x_pos, self.y_pos))

        # adds current node to fringe and includes danger factor in weight
        if self.visitable.get((temp_x, temp_y)) is not None and self.visited.get((temp_x, temp_y)) is None and (
                self.generate_ghosts is None or self.generate_ghosts.ghost_occupy.get(
            (self.x_pos, self.y_pos)) is None):
            temp_ghosts = self.generate_ghosts.get_ghost_dist(temp_x, temp_y)
            closest_offset = 0  # value to represent danger of a cell
            while temp_ghosts:
                closest = temp_ghosts.pop()
                # if a ghost is 5 or fewer units away, it is added to the danger weight
                if closest[0] <= 5:
                    if self.generate_ghosts.ghost_occupy.get(closest[1]) is True:
                        closest_offset += (6 - (
                        closest[0]))  # 6 - distance of ghost away gives value in which closer ghosts are more dangerous
                    else:
                        closest_offset += (6 - (closest[0]) * 0.8)  # ghosts in walls are weighted as less
            temp_tuple = (
                (self.get_dist_to_goal(temp_x, temp_y) + dist_path + closest_offset), dist_path + closest_offset,
                temp_x,
                temp_y,
                (self.x_pos, self.y_pos))
            self.fringe.put(temp_tuple)

        return False

    """Calls bootstrappable agent with 1 flag (agent 4 behavior)"""

    def walk_agent_4(self, ghost_map):
        return self.walk_agent_bootstrappable(0, 0, ghost_map, 1)

    """special version of validate cell used by Agent 4 blind (version flag 2) which attempts to avoid ghosts using a danger factor (offset) but is blind"""

    def validate_cell_avoid_blind(self, temp_x, temp_y, dist_path):
        # adds current node to fringe if goal and includes danger factor in weight
        if temp_x == self.x_pos and temp_y == self.y_pos:
            temp_ghosts = self.generate_ghosts.get_ghost_dist(temp_x, temp_y)
            closest_offset = 0  # value to represent danger of a cell
            while temp_ghosts:
                closest = temp_ghosts.pop()
                # if a ghost is 5 or fewer units away, it is added to the danger weight
                if closest[0] <= 5:
                    if self.generate_ghosts.ghost_occupy.get(closest[1]) is True:
                        closest_offset += (6 - (
                        closest[0]))  # 6 - distance of ghost away gives value in which closer ghosts are more dangerous
            temp_tuple = ((self.get_dist_to_goal(temp_x, temp_y) + closest_offset), 0, temp_x, temp_y,
                          None)
            self.fringe.put(temp_tuple)
            return

        # updates weight if shorter path is found, keeps danger factor
        if self.visited.get((temp_x, temp_y)) is not None and self.visited[(temp_x, temp_y)][0] > \
                self.visited[(self.x_pos, self.y_pos)][0] + 1:
            self.visited[(temp_x, temp_y)] = (self.visited[(self.x_pos, self.y_pos)][0] + 1, (self.x_pos, self.y_pos))

        # adds current node to fringe if goal and includes danger factor in weight
        if self.visitable.get((temp_x, temp_y)) is not None and self.visited.get((temp_x, temp_y)) is None and (
                self.generate_ghosts is None or self.generate_ghosts.ghost_occupy.get(
            (self.x_pos, self.y_pos)) is None):
            temp_ghosts = self.generate_ghosts.get_ghost_dist(temp_x, temp_y)
            closest_offset = 0  # value to represent danger of a cell
            while temp_ghosts:
                closest = temp_ghosts.pop()
                # if a ghost is 5 or fewer units away, it is added to the danger weight
                if closest[0] <= 5:
                    if self.generate_ghosts.ghost_occupy.get(closest[1]) is True:
                        closest_offset += (6 - (
                        closest[0]))  # 6 - distance of ghost away gives value in which closer ghosts are more dangerous

            temp_tuple = (
                (self.get_dist_to_goal(temp_x, temp_y) + dist_path + closest_offset), dist_path + closest_offset,
                temp_x,
                temp_y,
                (self.x_pos, self.y_pos))
            self.fringe.put(temp_tuple)

        return False

    """special version of validate cell used by Agent 5 (version flag 3) which attempts to avoid ghosts using a danger 
    factor (offset) and last step ghost danger averaging (heatmap) but is blind"""

    def validate_cell_avoid_blind_optimized(self, temp_x, temp_y, dist_path):
        # adds current node to fringe if goal and includes danger factor in weight
        if temp_x == self.x_pos and temp_y == self.y_pos:
            temp_ghosts = self.generate_ghosts.get_ghost_dist(temp_x, temp_y)
            closest_offset = 0  # value to represent danger of a cell
            while temp_ghosts:
                closest = temp_ghosts.pop()
                # if a ghost is 5 or fewer units away, it is added to the danger weight
                if closest[0] <= 5:
                    if self.generate_ghosts.ghost_occupy.get(closest[1]) is True:
                        closest_offset += (6 - (closest[0]))  # 6 - distance of ghost away gives value in which closer ghosts are more dangerous
            # averages current danger with previous danger
            if self.heat_map.get((temp_x, temp_y)) is not None:
                closest_offset = (self.heat_map.get((temp_x, temp_y)) + closest_offset) / 2

            temp_tuple = ((self.get_dist_to_goal(temp_x, temp_y) + closest_offset), 0, temp_x, temp_y,
                          None)
            self.heat_map[(temp_x, temp_y)] = closest_offset
            self.fringe.put(temp_tuple)
            return

        # updates weight if shorter path is found, keeps danger factor
        if self.visited.get((temp_x, temp_y)) is not None and self.visited[(temp_x, temp_y)][0] > \
                self.visited[(self.x_pos, self.y_pos)][0] + 1:
            self.visited[(temp_x, temp_y)] = (self.visited[(self.x_pos, self.y_pos)][0] + 1, (self.x_pos, self.y_pos))

        # adds current node to fringe and includes danger factor in weight
        if self.visitable.get((temp_x, temp_y)) is not None and self.visited.get((temp_x, temp_y)) is None and (
                self.generate_ghosts is None or self.generate_ghosts.ghost_occupy.get(
            (self.x_pos, self.y_pos)) is None):
            temp_ghosts = self.generate_ghosts.get_ghost_dist(temp_x, temp_y)
            closest_offset = 0
            while temp_ghosts:
                closest = temp_ghosts.pop()
                # if a ghost is 5 or fewer units away, it is added to the danger weight
                if closest[0] <= 5:
                    if self.generate_ghosts.ghost_occupy.get(closest[1]) is True:
                        closest_offset += (6 - (
                        closest[0]))  # 6 - distance of ghost away gives value in which closer ghosts are more dangerous
            # averages current danger with previous danger
            if self.heat_map.get((temp_x, temp_y)) is not None:
                closest_offset = (self.heat_map.get((temp_x, temp_y)) + closest_offset) / 2

            temp_tuple = (
                (self.get_dist_to_goal(temp_x, temp_y) + dist_path + closest_offset), dist_path + closest_offset,
                temp_x,
                temp_y,
                (self.x_pos, self.y_pos))
            self.heat_map[(temp_x, temp_y)] = closest_offset
            self.fringe.put(temp_tuple)

        return False

    """Calls bootstrappable agent with 2 flag (agent 4 blind behavior)"""

    def walk_agent_4_blind(self, ghost_map):
        return self.walk_agent_bootstrappable(0, 0, ghost_map, 2)

    """Calls bootstrappable agent with 3 flag (agent 5 behavior)"""

    def walk_agent_5(self, ghost_map):
        return self.walk_agent_bootstrappable(0, 0, ghost_map, 3)
