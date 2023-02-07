import u_star
import v_star
import environment
import dill
import numpy as np
import agent_u_partial

def main():
    """Creates new environment and respective U*"""
    new_environment = environment.Env(50)
    new_u_star = u_star.u_star(new_environment)

    dump_obj = (new_environment, new_u_star)
    with open('u_star_env.pkl', 'wb') as file:
        dill.dump(dump_obj, file)

    """Trains V* Based on U*"""
    v_star.v_train()

    """Run U_partial to collect data and train v_partial on it using v_star as a base"""
    agent_u_partial.generate_train_data()
    v_star.v_partial_train()


if __name__ == '__main__':
	main()
    