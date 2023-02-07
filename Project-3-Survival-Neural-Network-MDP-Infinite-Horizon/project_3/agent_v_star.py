from itertools import cycle
import random
import predator
import prey
import get_optimal_node
import environment
import numpy as np
import dill

class Agent_v_star:

    def __init__(self, input_predator = None, input_prey = None, input_environment = None, input_pos = None, verbose = False, env_v_star = None) -> None:
        #Sets predator object if not specified
        if input_predator is None:
            self.predator = predator.Predator()
        else: 
            self.predator = input_predator
        #Sets prey object if not specified
        if input_prey is None:
            self.prey = prey.Prey()
        else:
            self.prey = input_prey
        #Sets environment if not specified
        if input_environment is None:
            self.environment = environment.Env(50)
        else:
            self.environment = input_environment

        if env_v_star is None:
            raise Exception("v_star cannot be none")
        else:
            self.env_v_star = env_v_star

        #Sets agent position if not specified
        if input_pos is None:
            self.pos = random.choice(range(0,49))
        else:
            self.pos = input_pos

        self.steps = 0

        #make sure agent doesnt start in occupied node
        while self.prey.pos == self.pos or self.predator.pos == self.pos:
            self.pos = random.choice(range(0,49))

        #Values for keeping track of positions for animation
        self.agent_steps = [self.pos]
        self.prey_steps = [self.prey.pos]
        self.predator_steps = [self.prey.pos]
        self.actual_prey_steps =self.prey_steps
        self.actual_predator_steps = self.predator_steps

    def move(self):
        #runs for 50 steps else returns false
        while self.steps <= 5000:
            self.steps += 1
            predator_pos = self.predator.pos
            
            prey_pos = self.prey.pos

            new_pos = self.get_next_move()
            self.pos = new_pos

            #Keep track of positions for animations
            self.predator_steps.append(self.predator.pos)
            self.prey_steps.append(self.prey.pos)
            self.agent_steps.append(self.pos)
            self.actual_prey_steps =self.prey_steps
            self.actual_predator_steps = self.predator_steps

            #returns 0 if moves into predator or predator moves into it
            if predator_pos == self.pos: 
                return 0, self.steps, self.agent_steps, self.prey_steps, self.predator_steps, self.actual_prey_steps, self.actual_predator_steps
            #returns 1 if moves into prey 
            if prey_pos == self.pos:
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


    def get_next_move(self):
        predator_index = self.predator.pos
        prey_index = self.prey.pos
        agent_index = self.pos

        environment = self.environment

        #predator movements with 0.4 p of occuring
        if environment.lis[predator_index].degree == 2:
            unfocused_options = [environment.lis[predator_index].left_node_index,  environment.lis[predator_index].right_node_index]
        else:
            unfocused_options = [environment.lis[predator_index].left_node_index,  environment.lis[predator_index].right_node_index,  environment.lis[predator_index].other_node_index]

        #all prey possible movement
        if environment.lis[prey_index].degree == 2:
            prey_options = [environment.lis[prey_index].index, environment.lis[prey_index].left_node_index,  environment.lis[prey_index].right_node_index]
        else:
            prey_options = [environment.lis[prey_index].index, environment.lis[prey_index].left_node_index,  environment.lis[prey_index].right_node_index,  environment.lis[prey_index].other_node_index]

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
            prey_options_prob = 1/len(prey_options)
            focused_options_prob = 1/len(focused_options) * 0.6
            unfocused_options_prob = 1/len(unfocused_options) * 0.4

            if not (ag_index == prey_index and predator_index != ag_index):
                if not(predator_index == ag_index):
                    for prey_ind in prey_options:
                        for pred_focus_ind in focused_options:
                            if ag_index == pred_focus_ind:
                                total_value = np.Inf
                                break
                            total_value += focused_options_prob * prey_options_prob * self.env_v_star.get_value(pred_focus_ind, prey_ind, ag_index)
                            #
                        for pred_unfocus_ind in unfocused_options:
                            if ag_index == pred_unfocus_ind:
                                total_value = np.Inf
                                break
                            total_value += unfocused_options_prob * prey_options_prob * self.env_v_star.get_value(pred_unfocus_ind, prey_ind, ag_index)
                            #
                else:
                    total_value = np.Inf

            agent_options_vals[itr] = total_value + 1

        return agent_options[agent_options_vals.index(min(agent_options_vals))]

def main():
    with open("v_star_env.pkl", 'rb') as file:
        v_star = dill.load(file)
    with open("u_star_env.pkl", 'rb') as file:
        env, u_star = dill.load(file)

    count = 0
    for _ in range(100):
        ag = Agent_v_star(env_v_star=v_star, input_environment=env)
        k = ag.move()
        if k[0] == 1:
            count += 1 
        #print(k)
    print('---------------------------')
    print('Success count :' + str(count))
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
