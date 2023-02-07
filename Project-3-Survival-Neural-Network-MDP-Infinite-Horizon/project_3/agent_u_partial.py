import random
import predator
import prey
import environment
import numpy as np
import u_star
import dill

class Agent_u_partial:

    def __init__(self, input_predator = None, input_prey = None, input_environment = None, input_pos = None, env_u_star = None, collect_data = False) -> None:
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

        if env_u_star is None:
            self.env_u_star = u_star.u_star(self.environment)
        else:
            self.env_u_star = env_u_star

        self.options_matrix = self.env_u_star.options_matrix
        self.utility_matrix = self.env_u_star.utility_matrix

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

        self.u_partial_combos = []
        self.u_partial_returns = []
        self.collect_data = collect_data

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
            
            self.pos = self.get_partial_action(self.predator.pos, self.prey_probability_array, self.pos)

            self.predator_steps.append(self.predator.pos)
            self.prey_steps.append(prey_pos)
            self.actual_prey_steps.append(self.prey.pos)
            self.agent_steps.append(self.pos)
            self.actual_predator_steps.append(self.predator.pos)

            self.agent_moved()
            #returns 0 if moves into predator or predator moves into it
            if actual_predator_pos == self.pos: 
                return 0, self.steps
            #returns 1 if moves into prey 
            if actual_prey_pos == self.pos:
                return 1, self.steps
            #returns 1 if prey moves into it
            if not self.prey.move(self.environment,self.pos):
                self.prey_steps.append(self.survey())
                self.actual_prey_steps.append(self.prey.pos)
                return 1, self.steps
            #returns 0 if predator moves into it
            if not self.predator.move_distractable(self.environment,self.pos):
                self.prey_steps.append(self.survey())
                self.actual_prey_steps.append(self.prey.pos)
                self.predator_steps.append(self.predator.pos)
                self.actual_predator_steps = self.predator_steps 
                return 0, self.steps
                
            #update probabilites after movement (will only survey agents current pos not highest probability since True flag)
            self.transition()

        #returns -1 if timeout
        return -1, self.steps

    def get_partial_action(self, i, vector, k):
        predator_index = i
        agent_index = k

        agent_options, _, _, _ = self.options_matrix[i][0][k]
        agent_options_vals = [0] * len(agent_options)
        for prey_index, belief in enumerate(vector):
            _, focused_options_given_agent, unfocused_options, prey_options = self.options_matrix[i][prey_index][k]
            
            for (itr, focused_options), ag_index in zip(enumerate(focused_options_given_agent), agent_options):
                total_value = 0
                prey_options_prob = 1/len(prey_options)
                focused_options_prob = 1/len(focused_options) * 0.6
                unfocused_options_prob = 1/len(unfocused_options) * 0.4

                if not (ag_index == prey_index and predator_index != ag_index):
                    if not(predator_index == ag_index):
                        for pred_focus_ind in focused_options:
                            #if prey_ind == ag_index and agent_index != pred_focus_ind:
                                #continue
                            total_value += focused_options_prob * prey_options_prob * self.utility_matrix[pred_focus_ind][prey_index][ag_index]
                        for pred_unfocus_ind in unfocused_options:
                            #if prey_ind == ag_index and agent_index != pred_unfocus_ind:
                                #continue
                            total_value += unfocused_options_prob * prey_options_prob * self.utility_matrix[pred_unfocus_ind][prey_index][ag_index]
                    else:
                        total_value = np.Inf

                    if belief != 0:
                        total_value = total_value * belief
                    else:
                        total_value = 0
                    agent_options_vals[itr] += total_value
        if self.collect_data == True:
            if min(agent_options_vals) != np.Inf:
                self.u_partial_returns.append([[min(agent_options_vals) + 1]])
                i_env = np.zeros(50)
                i_env[i] = 1
                j_env = vector.copy()
                k_env = np.zeros(50)
                k_env[k] = 1
                input = np.concatenate((i_env, j_env))
                input = np.concatenate((input,k_env))
                self.u_partial_combos = self.u_partial_combos + [[input]]

        return agent_options[agent_options_vals.index(min(agent_options_vals))]


def generate_train_data():
    with open("u_star_env.pkl", 'rb') as file:
            loaded_tuple = dill.load(file)
	
    env, u_star_obj = loaded_tuple

    u_partial_combos = []
    u_partial_returns = []

    count = 0
    for i in range(10000):
        print(f"{i} ", end="", flush=True)
        ag = Agent_u_partial(input_environment=env, env_u_star=u_star_obj, collect_data=True)
        k = ag.move()
        if k[0] == 1:
            count += 1
        u_partial_combos += ag.u_partial_combos
        u_partial_returns += ag.u_partial_returns

    np.save('v_partial_training_combos.npy', u_partial_combos)
    np.save('v_partial_training_returns.npy', u_partial_returns)

def main():

    with open("u_star_env.pkl", 'rb') as file:
            loaded_tuple = dill.load(file)
	
    env, u_star_obj = loaded_tuple

    u_partial_combos = []
    u_partial_returns = []

    count = 0
    steps = 0
    itr = 10000
    for i in range(itr):
        print(f"{i} ", end="", flush=True)
        ag = Agent_u_partial(input_environment=env, env_u_star=u_star_obj, collect_data=True)
        k = ag.move()
        if k[0] == 1:
            count += 1
        u_partial_combos += ag.u_partial_combos
        u_partial_returns += ag.u_partial_returns
        steps += k[1]

    np.save('v_partial_training_combos.npy', u_partial_combos)
    np.save('v_partial_training_returns.npy', u_partial_returns)

    

    print('---------------------------')
    print('Success count :' + str(count))
    print('Steps:', steps/itr)
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