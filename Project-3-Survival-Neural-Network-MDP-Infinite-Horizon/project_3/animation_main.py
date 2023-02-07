import environment
import pygame
import predator
import prey
import random
import math
import networkx as nx
import copy
import agent_1 as ag1
import agent_2 as ag2
import agent_u_star as au
import dill


import animation as an


def main():
    
    with open("u_star_env.pkl", 'rb') as file:
        loaded_tuple = dill.load(file)
	
    input_environment, env_u_star = loaded_tuple

    input_predator = predator.Predator()
    input_prey = prey.Prey()
    input_pos = random.choice(range(0,49))

    #make sure agent doesnt start in occupied node
    while input_prey.pos == input_pos or input_predator.pos == input_pos:
        input_pos = random.choice(range(0,49))

    
    agent = ag1.Agent_1(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos)
    k = agent.move()
    prey_steps = agent.prey_steps
    predator_steps = agent.predator_steps
    agent_steps = agent.agent_steps
    actual_prey_steps = agent.actual_prey_steps
    actual_predator_steps = agent.actual_predator_steps
    test = an.Animation(input_environment, prey_steps, predator_steps, agent_steps, actual_prey_steps, actual_predator_steps)

    agent = ag2.Agent_2(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos)
    k = agent.move()
    prey_steps = agent.prey_steps
    predator_steps = agent.predator_steps
    agent_steps = agent.agent_steps
    actual_prey_steps = agent.actual_prey_steps
    actual_predator_steps = agent.actual_predator_steps
    test = an.Animation(input_environment,prey_steps, predator_steps, agent_steps, actual_prey_steps, actual_predator_steps)

    agent = au.Agent_u_star(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos, env_u_star=env_u_star)
    k = agent.move()
    prey_steps = agent.prey_steps
    predator_steps = agent.predator_steps
    agent_steps = agent.agent_steps
    actual_prey_steps = agent.actual_prey_steps
    actual_predator_steps = agent.actual_predator_steps
    test = an.Animation(input_environment,prey_steps, predator_steps, agent_steps, actual_prey_steps, actual_predator_steps)


if __name__ == '__main__':
    main()