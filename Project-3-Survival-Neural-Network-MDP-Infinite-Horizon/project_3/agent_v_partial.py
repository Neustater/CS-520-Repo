from itertools import cycle
import random
import predator
import prey
import get_optimal_node
import environment
import numpy as np
import dill

class Agent_v_partial:

    def __init__(self, input_predator = None, input_prey = None, input_environment = None, input_pos = None, verbose = False, env_v_partial = None) -> None:
             #Sets predator position if not specified
        if input_predator is None:
            self.predator = predator.Predator()
        else: 
            self.predator = input_predator

        #Sets prey position if not specified
        if input_prey is None:
            self.prey = prey.Prey()
        else:
            self.prey = input_prey

        #Sets environment if not specified
        if input_environment is None:
            self.environment = environment.Env(50)
        else:
            self.environment = input_environment
        
        #Sets agent pos if not specified
        if input_pos is None:
            self.pos = random.choice(range(0,49))
        else:
            self.pos = input_pos

        if env_v_partial is None:
            raise Exception("v_partial cannot be none")
        else:
            self.env_v_partial = env_v_partial

        #make sure agent doesnt start in occupied node
        while self.prey.pos == self.pos or self.predator.pos == self.pos:
            self.pos = random.choice(range(0,49))

        #Sets initial Belief Values for Prey
        prey_probability_array = [(1/49)] * 50
        prey_probability_array[self.pos] = 0
        self.prey_probability_array = np.array(prey_probability_array) #Belief array (sum of elements is 1)

        self.steps = 0

        #make sure agent doesnt start in occupied node
        while self.prey.pos == self.pos or self.predator.pos == self.pos:
            self.pos = random.choice(range(0,49))

        #keeps track of positions for animations
        self.agent_steps = [self.pos]
        self.prey_steps = []
        self.predator_steps = [self.predator.pos]
        self.actual_prey_steps = [self.prey.pos]
        self.actual_predator_steps = [self.predator.pos]

        #keeps track of when certain of prey pos
        self.certain_prey_pos = 0

    #normalizes probability
    def update_probability(self, num, prob_sum):
        if prob_sum == 0:
            return 0
        return (num) / (prob_sum) 

    """Function to handle surveying of a node and belief updates, returns next highest belief prey pos"""
    
    def survey(self):   #if agent_move is true, use transition matrix to update probability (for when agent moves)
        array = np.where(np.isclose(self.prey_probability_array, np.amax(self.prey_probability_array)))[0] #most likely position is surveyed (random if multiple)
        choice = np.random.choice(array)

        if choice != self.prey.pos:     #if survey is false
            vfunction = np.vectorize(self.update_probability)       #apply update probabilty to the p vector
            self.prey_probability_array[choice] = 0
            self.prey_probability_array = vfunction(self.prey_probability_array, np.sum(self.prey_probability_array))

            #pick highest probability node and return it
            array = np.where(np.isclose(self.prey_probability_array, np.amax(self.prey_probability_array)))[0]    #most likely position after removal of surveyed returned (random if multiple)
            choice = np.random.choice(array)
        else:       #if the survey is true

            #all probabilites become false except the node of the prey and all adjacent to it
            self.prey_probability_array.fill(0)
            self.prey_probability_array[choice] = 1
        return choice

    """Function to handle agent movement belief updates"""
    
    def agent_moved(self):
        vfunction = np.vectorize(self.update_probability)
        self.prey_probability_array[self.pos] = 0
        self.prey_probability_array = vfunction(self.prey_probability_array, np.sum(self.prey_probability_array))
        
    """Function to handle belief updates after actor movement"""

    def transition(self):
        vfunction = np.vectorize(self.update_probability)
        self.prey_probability_array = np.dot(self.prey_probability_array, self.environment.prey_trans_matrix)
        self.prey_probability_array[self.pos] = 0
        self.prey_probability_array = vfunction(self.prey_probability_array, np.sum(self.prey_probability_array)) 

    """Movement function for agent 1
    returns 1 if catches prey, 0 if dies, -1 if timeout"""

    def move(self):
        #runs for 100 steps else returns false
        while self.steps <= 5000:
            self.steps += 1
            actual_predator_pos = self.predator.pos
            actual_prey_pos = self.prey.pos
            prey_pos = self.survey()

            if prey_pos == actual_prey_pos and np.isclose(self.prey_probability_array[prey_pos], 1):
                self.certain_prey_pos += 1

            new_pos = self.get_next_move(self.prey_probability_array)
            self.pos = new_pos

            #Keep track of positions for animations
            self.predator_steps.append(self.predator.pos)
            self.prey_steps.append(self.prey.pos)
            self.agent_steps.append(self.pos)
            self.actual_prey_steps =self.prey_steps
            self.actual_predator_steps = self.predator_steps

            #returns 0 if moves into predator or predator moves into it
            if actual_predator_pos == self.pos: 
                return 0, self.steps, self.agent_steps, self.prey_steps, self.predator_steps, self.actual_prey_steps, self.actual_predator_steps
            #returns 1 if moves into prey 
            if actual_prey_pos == self.pos:
                return 1, self.steps, self.agent_steps, self.prey_steps, self.predator_steps, self.actual_prey_steps, self.actual_predator_steps
            #returns 1 if prey moves into it
            if not self.prey.move(self.environment,self.pos):
                return 1, self.steps, self.agent_steps, self.prey_steps, self.predator_steps, self.actual_prey_steps, self.actual_predator_steps
            #returns 0 if predator moves into it
            if not self.predator.move_distractable(self.environment,self.pos):
                self.prey_steps.append(self.prey.pos)
                self.predator_steps.append(self.predator.pos)
                return 0, self.steps, self.agent_steps, self.prey_steps, self.predator_steps, self.actual_prey_steps, self.actual_predator_steps

        #returns -1 if timeout
        return -1, self.steps, self.agent_steps, self.prey_steps, self.predator_steps, self.actual_prey_steps, self.actual_predator_steps


    def get_next_move(self, vector):
        predator_index = self.predator.pos
        prey_index = self.prey.pos
        agent_index = self.pos

        environment = self.environment

        vector = vector.copy()
        vfunction = np.vectorize(self.update_probability)
        self.prey_probability_array = np.dot(vector, self.environment.prey_trans_matrix)

        #predator movements with 0.4 p of occuring
        if environment.lis[predator_index].degree == 2:
            unfocused_options = [environment.lis[predator_index].left_node_index,  environment.lis[predator_index].right_node_index]
        else:
            unfocused_options = [environment.lis[predator_index].left_node_index,  environment.lis[predator_index].right_node_index,  environment.lis[predator_index].other_node_index]

        if environment.lis[agent_index].degree == 2:
            agent_options = [environment.lis[agent_index].index, environment.lis[agent_index].left_node_index,  environment.lis[agent_index].right_node_index]
        else:
            agent_options = [environment.lis[agent_index].index, environment.lis[agent_index].left_node_index,  environment.lis[agent_index].right_node_index,  environment.lis[agent_index].other_node_index]

        focused_options_given_agent = []

        for itr, agent_next_index in enumerate(agent_options):
            #predator_movements with 0.6 p of occuring
            if environment.lis[predator_index].degree == 2:
                focused_options = np.array([environment.lis[predator_index].index, environment.lis[predator_index].left_node_index,  environment.lis[predator_index].right_node_index])
                option_distances = np.array([environment.shortest_paths[environment.lis[predator_index].index][agent_next_index], 
                environment.shortest_paths[environment.lis[predator_index].left_node_index][agent_next_index],  
                environment.shortest_paths[environment.lis[predator_index].right_node_index][agent_next_index]])
            else:
                focused_options = np.array([environment.lis[predator_index].index, environment.lis[predator_index].left_node_index,  environment.lis[predator_index].right_node_index,  environment.lis[predator_index].other_node_index])
                option_distances = np.array([environment.shortest_paths[environment.lis[predator_index].index][agent_next_index], 
                environment.shortest_paths[environment.lis[predator_index].left_node_index][agent_next_index],  
                environment.shortest_paths[environment.lis[predator_index].right_node_index][agent_next_index],  
                environment.shortest_paths[environment.lis[predator_index].other_node_index][agent_next_index]])

            #gets options to only be shortest distance next nodes
            focused_options = focused_options[np.where(np.isclose(option_distances, np.amin(option_distances)))[0]]

            focused_options_given_agent.append(focused_options)

        agent_options_vals = [0] * len(agent_options)

        for (itr, focused_options), ag_index in zip(enumerate(focused_options_given_agent), agent_options):

            total_value = 0
            focused_options_prob = 1/len(focused_options) * 0.6
            unfocused_options_prob = 1/len(unfocused_options) * 0.4

            if not(predator_index == ag_index):
                for pred_focus_ind in focused_options:
                    if ag_index == pred_focus_ind:
                        total_value = np.Inf
                        break
                    total_value += focused_options_prob * self.env_v_partial.get_value_partial(pred_focus_ind, vector, ag_index)
                    #self.u_part(pred_focus_ind, vector, ag_index)
                    
                for pred_unfocus_ind in unfocused_options:
                    if ag_index == pred_unfocus_ind:
                        total_value = np.Inf
                        break
                    total_value += unfocused_options_prob * self.env_v_partial.get_value_partial(pred_unfocus_ind, vector, ag_index)
                    #self.u_part(pred_unfocus_ind, vector, ag_index)
                        
            else:
                total_value = np.Inf

            agent_options_vals[itr] = total_value + 1

        return agent_options[agent_options_vals.index(min(agent_options_vals))]


def main():
    
    with open("v_partial_env.pkl", 'rb') as file:
        v_partial = dill.load(file)
    with open("u_star_env.pkl", 'rb') as file:
        env, u_star = dill.load(file)

    count = 0
    steps = 0
    itr = 100
    for i in range(itr):
        ag = Agent_v_partial(input_environment=env, env_v_partial=v_partial)
        k = ag.move()
        if k[0] == 1:
            count += 1 
        steps += k[1]
        print(i)

    print('---------------------------')
    print('Success count :' + str(count))
    print('Steps:', steps/itr)
    print('Agent moves:')
    print(ag.agent_steps)
    print('Prey moves:')
    print(ag.prey_steps)
    print('Predator moves:')
    print(ag.predator_steps)
    print('pred, predy and agent last steps')
    print(ag.predator.pos)
    print(ag.prey.pos)
    print(ag.pos)

if __name__ == '__main__':
    main()
