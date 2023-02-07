from itertools import cycle
import random
import predator
import prey
import get_optimal_node
import environment
import u_star

class Agent_u_star:

    def __init__(self, input_predator = None, input_prey = None, input_environment = None, input_pos = None, verbose = False, env_u_star = None) -> None:
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

        if env_u_star is None:
            u_star_result = u_star.u_star(self.environment)
            self.policy_matrix = u_star_result.policy_matrix
        else:
            self.policy_matrix = env_u_star.policy_matrix

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
        self.actual_prey_steps =self.predator_steps
        self.actual_predator_steps = self.predator_steps

    def move(self):
        #runs for 50 steps else returns false
        while self.steps <= 5000:
            self.steps += 1
            predator_pos = self.predator.pos
            
            prey_pos = self.prey.pos

            new_pos = self.policy_matrix[predator_pos][prey_pos][self.pos]
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

def main():
    count = 0
    for _ in range(1):
        ag = Agent_u_star(verbose=True)
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
