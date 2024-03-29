import random
import numpy as np

class Predator:

    def __init__(self) -> None:
        self.pos = random.choice(range(0,49))

    """Moves predator towards agent
    Returns True if moved and didn't collide with agent"""
    def move(self, environment, agent_pos):
        if self.pos != agent_pos: #was not properly chcecking for collision, now fixed
            if environment.lis[self.pos].degree == 2:
                options = np.array([environment.lis[self.pos].index, environment.lis[self.pos].left_node_index,  environment.lis[self.pos].right_node_index])
                option_distances = [environment.shortest_paths[environment.lis[self.pos].index][agent_pos], 
                environment.shortest_paths[environment.lis[self.pos].left_node_index][agent_pos],  
                environment.shortest_paths[environment.lis[self.pos].right_node_index][agent_pos]]
            else:
                options = np.array([environment.lis[self.pos].index, environment.lis[self.pos].left_node_index,  environment.lis[self.pos].right_node_index,  environment.lis[self.pos].other_node_index])
                option_distances = np.array([environment.shortest_paths[environment.lis[self.pos].index][agent_pos], 
                environment.shortest_paths[environment.lis[self.pos].left_node_index][agent_pos],  
                environment.shortest_paths[environment.lis[self.pos].right_node_index][agent_pos],  
                environment.shortest_paths[environment.lis[self.pos].other_node_index][agent_pos]])
            index = random.choice(np.where(np.isclose(option_distances, np.amin(option_distances)))[0] )
            self.pos = options[index]
            if self.pos == agent_pos:
                return False
            else:
                return True
        else:
            return False

    def move_distractable(self, environment, agent_pos):
        if random.random() <= 0.6:
            if self.pos != agent_pos: #was not properly chcecking for collision, now fixed
                if environment.lis[self.pos].degree == 2:
                    options = np.array([environment.lis[self.pos].index, environment.lis[self.pos].left_node_index,  environment.lis[self.pos].right_node_index])
                    option_distances = [environment.shortest_paths[environment.lis[self.pos].index][agent_pos], 
                    environment.shortest_paths[environment.lis[self.pos].left_node_index][agent_pos],  
                    environment.shortest_paths[environment.lis[self.pos].right_node_index][agent_pos]]
                else:
                    options = np.array([environment.lis[self.pos].index, environment.lis[self.pos].left_node_index,  environment.lis[self.pos].right_node_index,  environment.lis[self.pos].other_node_index])
                    option_distances = np.array([environment.shortest_paths[environment.lis[self.pos].index][agent_pos], 
                    environment.shortest_paths[environment.lis[self.pos].left_node_index][agent_pos],  
                    environment.shortest_paths[environment.lis[self.pos].right_node_index][agent_pos],  
                    environment.shortest_paths[environment.lis[self.pos].other_node_index][agent_pos]])
                index = random.choice(np.where(np.isclose(option_distances, np.amin(option_distances)))[0] )
                self.pos = options[index]
                if self.pos == agent_pos:
                    return False
                else:
                    return True
            else:
                return False
        else:
            if self.pos != agent_pos: #was not properly chcecking for collision, now fixed
                if environment.lis[self.pos].degree == 2:
                    options = np.array([environment.lis[self.pos].left_node_index,  environment.lis[self.pos].right_node_index])
                else:
                    options = np.array([environment.lis[self.pos].left_node_index,  environment.lis[self.pos].right_node_index,  environment.lis[self.pos].other_node_index])
                node_index = random.choice(options)
                self.pos = node_index
                if self.pos == agent_pos:
                    return False
                else:
                    return True
            else:
                return False