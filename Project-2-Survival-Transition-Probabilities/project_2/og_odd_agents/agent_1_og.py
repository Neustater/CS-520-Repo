import random
import predator
import prey
import environment

class Agent_1:

    def __init__(self, input_predator = None, input_prey = None, input_environment = None, input_pos = None) -> None:
        if input_predator is None:
            self.predator = predator.Predator()
        else: 
            self.predator = input_predator
        
        if input_prey is None:
            self.prey = prey.Prey()
        else:
            self.prey = input_prey

        if input_environment is None:
            self.environment = environment.Env(50)
        else:
            self.environment = input_environment
        
        if input_pos is None:
            self.pos = random.choice(range(0,49))
        else:
            self.pos = input_pos

        self.steps = 0

        #make sure agent doesnt start in occupied node
        while self.prey.pos == self.pos or self.predator.pos == self.pos:
            self.pos = random.choice(range(0,49))

    """Movement function for agent 1
    returns 1 if catches prey, 0 if dies, -1 if timeout"""

    def move(self):
        #runs for 100 steps else returns false
        while self.steps <= 5000:
            self.steps += 1
            predator_pos = self.predator.pos
            prey_pos = self.prey.pos
            current_node = self.environment.lis[self.pos]
            shortest_paths = self.environment.shortest_paths

            #array of possible choices
            adjacent_nodes = [current_node.left_node_index,
            current_node.right_node_index,
            current_node.other_node_index] ## Corrected the adjacent_nodes. Previously there was only left_index, left_index and other_index. 

            #gets distances to predator from each direction
            left_pred_dist = shortest_paths[current_node.left_node_index][predator_pos]
            right_pred_dist = shortest_paths[current_node.right_node_index][predator_pos]
            other_pred_dist = shortest_paths[current_node.other_node_index][predator_pos]
            cur_pred_dist = shortest_paths[self.pos][predator_pos]

            #puts distances from predator in array
            pred_dist_array = [left_pred_dist, right_pred_dist, other_pred_dist]

            #gets distances to prey from each direction
            left_prey_dist = shortest_paths[current_node.left_node_index][prey_pos]
            right_prey_dist = shortest_paths[current_node.right_node_index][prey_pos]
            other_prey_dist = shortest_paths[current_node.other_node_index][prey_pos]
            cur_prey_dist = shortest_paths[self.pos][prey_pos]

            #puts distances from prey in array
            prey_dist_array = [left_prey_dist, right_prey_dist, other_prey_dist]

            #creates array of length 7, each index corresponding to the possible scenarios outlined in writeup
            #please check if this what the writeup meant
            options = [[] for i in range(7)]
            for i in range(len(prey_dist_array)):
                if prey_dist_array[i] < cur_prey_dist and pred_dist_array[i] > cur_pred_dist:  ## Neighbors that are closer to the Prey and farther from the Predator
                    options[0].append(adjacent_nodes[i])
                elif prey_dist_array[i] < cur_prey_dist and not pred_dist_array[i] < cur_pred_dist:  ## Neighbors that are closer to the Prey and not closer to the Predator. # I beleive that we have to check that the chosen node is not closer to the predator here as priority 2
                    options[1].append(adjacent_nodes[i])
                elif prey_dist_array[i] == cur_prey_dist and pred_dist_array[i] > cur_pred_dist:
                    options[2].append(adjacent_nodes[i])
                elif prey_dist_array[i] == cur_prey_dist and not pred_dist_array[i] < cur_pred_dist:
                    options[3].append(adjacent_nodes[i])
                elif pred_dist_array[i] > cur_pred_dist:
                    options[4].append(adjacent_nodes[i])
                elif pred_dist_array[i] == cur_pred_dist:
                    options[5].append(adjacent_nodes[i])
                else:
                    options[6].append(current_node.index)
            
            #randomly picks a choice if multiple good choices (could be optimized instead of picking randomly, but write up says randomly I believe)
            '''
            for w in options:
                if len(w) > 0:
                    result_index = random.choice(w)
                    break
            '''
            for result in options:
                if result:
                    result_index = random.choice(result)
                    break
            self.pos = result_index
            #returns 0 if moves into predator or predator moves into it
            if predator_pos == self.pos: 
                return 0, self.steps
            #returns 1 if moves into prey 
            if prey_pos == self.pos:
                return 1, self.steps
            #returns 1 if prey moves into it
            if not self.prey.move(self.environment,self.pos):
                return 1, self.steps
            #returns 0 if predator moves into it
            if not self.predator.move(self.environment,self.pos):
                return 0, self.steps
        #returns -1 if timeout
        return -1, self.steps


def main():
    count1 = 0
    count0 = 0
    count11 = 0
    cycles = 1000
    for _ in range(cycles):
        ag = Agent_1()
        k = ag.move()
        print(k[0])
        if k[0] == 0:
            count0 += 1
        elif k[0] == 1:
            count1 += 1
        else:
            count11 += 1
    print("\nAgent 2:")
    print(f"Caught (including timeout): {round(((count1)/cycles)*100,3)}% | Died (including timeout): {round(count0)}% | Timed Out %: {round((count11))}%")

if __name__ == '__main__':
    main()
