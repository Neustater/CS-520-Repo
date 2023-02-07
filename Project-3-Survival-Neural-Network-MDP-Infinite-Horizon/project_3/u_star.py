import numpy as np
import environment
import math
import dill

class u_star:

    def __init__(self, environment, number_of_nodes = 50) -> None:
        self.number_of_nodes = number_of_nodes
        self.utility_matrix = np.zeros((number_of_nodes,number_of_nodes,number_of_nodes),dtype=float)
        self.environment = environment

        self.options_matrix = [[[0 for _ in range(number_of_nodes)] for _ in range(number_of_nodes)] for _ in range(number_of_nodes)]

        print("Initilizing Options Matrix")
        self.initilize_options_matrix()
        print("Finished Initilizing Options!")
        self.value_iteration()


        self.policy_matrix = np.zeros((self.number_of_nodes,self.number_of_nodes,self.number_of_nodes), dtype=int)
        self.create_policies()

    def initilize_options_matrix(self):
        for i in range(self.number_of_nodes):
            print(i+1, "/", self.number_of_nodes)
            for j in range(self.number_of_nodes):
                for k in range(self.number_of_nodes):
                    predator_index = i
                    prey_index = j
                    agent_index = k

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
                    self.options_matrix[i][j][k] = (agent_options, focused_options_given_agent, unfocused_options, prey_options)

    def value_iteration_helper(self, i, j, k):
        predator_index = i
        prey_index = j
        agent_index = k

        environment = self.environment

        if agent_index == predator_index:
            return np.Inf

        if agent_index == prey_index:
            return 0

        agent_options, focused_options_given_agent, unfocused_options, prey_options = self.options_matrix[i][j][k]
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
                            total_value += focused_options_prob * prey_options_prob * self.utility_matrix[pred_focus_ind][prey_ind][ag_index]
                        for pred_unfocus_ind in unfocused_options:
                            total_value += unfocused_options_prob * prey_options_prob * self.utility_matrix[pred_unfocus_ind][prey_ind][ag_index]
                else:
                    total_value = np.Inf
            agent_options_vals[itr] = total_value + 1

        return min(agent_options_vals)


    def value_iteration(self):
        
        diff = np.ones((self.number_of_nodes,self.number_of_nodes,self.number_of_nodes), dtype=float) * np.Inf
        max_diff = np.Inf
        np.seterr('ignore') #disables seterr because it doesnt play nice with infinity subtraction from infinity
        while not np.isclose(max_diff, 0, atol=1e-08):
            new_utility_matrix = np.ones((self.number_of_nodes,self.number_of_nodes,self.number_of_nodes), dtype=float)
            
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):
                    for k in range(self.number_of_nodes):
                        new_utility_matrix[i][j][k] = self.value_iteration_helper(i, j, k)
            diff = np.abs(self.utility_matrix - new_utility_matrix)
            diff = np.where(np.isnan(diff), 0, diff)            
            max_diff = np.nanmax(diff)
            self.utility_matrix = new_utility_matrix
            print("Max Difference", max_diff) #converges when difference is 0
        
        print("Converged!")
        print(self.utility_matrix)
        np.seterr('print')
        
        return

    def policy_creation_helper(self, i, j, k):
        predator_index = i
        prey_index = j
        agent_index = k

        if agent_index == predator_index:
            return agent_index

        if agent_index == prey_index:
            return agent_index

        agent_options, focused_options_given_agent, unfocused_options, prey_options = self.options_matrix[i][j][k]
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
                            total_value += focused_options_prob * prey_options_prob * self.utility_matrix[pred_focus_ind][prey_ind][ag_index]
                        for pred_unfocus_ind in unfocused_options:
                            total_value += unfocused_options_prob * prey_options_prob * self.utility_matrix[pred_unfocus_ind][prey_ind][ag_index]
                else:
                    total_value = np.Inf

            agent_options_vals[itr] = total_value + 1


        return agent_options[agent_options_vals.index(min(agent_options_vals))]

    def create_policies(self):
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                for k in range(self.number_of_nodes):
                    self.policy_matrix[i][j][k] = self.policy_creation_helper(i, j, k)


def main():
    test_environment = environment.Env(50)
    test_u_star = u_star(test_environment,50)

    dump_obj = (test_environment, test_u_star)

    with open('u_star_env.pkl', 'wb') as file:
        # A new file will be created
        dill.dump(dump_obj, file)

if __name__ == '__main__':
    main()
    