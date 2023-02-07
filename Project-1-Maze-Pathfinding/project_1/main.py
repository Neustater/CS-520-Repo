import pickle
import subprocess
import time

max_ghosts = 121  # define max number of ghosts to test to (starts at 0)
threads = 121  # specify number of threads (subprocesses) more is faster, but much more intensive (only use more than 16 on iLabs, must be less than number of ghosts)
thread_div = round(max_ghosts / threads)
num_iterations = 100
def main():

    start = time.perf_counter()

    # create subprocess to create mazes and store to array

    proc = subprocess.Popen("python3 maze_generator.py", shell=True)
    proc.wait()

    processes_batch = []

    """Creates subprocesses corresponding to number of threads in range of ghosts of i to i + thread_div (not inclusive)"""

    i = 0
    while i < max_ghosts:
        if i + thread_div < max_ghosts:
            processes_batch.append(f"python3 ag_subprocess.py {i} {i + thread_div}")
            i += thread_div
        else:
            processes_batch.append(f"python3 ag_subprocess.py {i} {max_ghosts}")
            i = max_ghosts

    procs = [subprocess.Popen(i, shell=True) for i in processes_batch]
    for p in procs:
        p.wait()
    print(f"Time since start main: {((time.perf_counter() - start) / 60):0.2f} minute(s)")

    """Parsing results of subprocesses into greater array of data"""
    result_1 = []
    result_2 = []
    result_3 = []
    result_4 = []
    result_4_blind = []
    result_5 = []

    i = 0
    while i < max_ghosts:
        if i + thread_div < max_ghosts:
            with open(f'agent_1_testing/agent_1_{i}_{i + thread_div}.pickle', 'rb') as handle:
                result_1 = result_1 + pickle.load(handle)
            with open(f'agent_2_testing/agent_2_{i}_{i + thread_div}.pickle', 'rb') as handle:
                result_2 = result_2 + pickle.load(handle)
            with open(f'agent_3_testing/agent_3_{i}_{i + thread_div}.pickle', 'rb') as handle:
                result_3 = result_3 + pickle.load(handle)
            with open(f'agent_4_testing/agent_4_{i}_{i + thread_div}.pickle', 'rb') as handle:
                result_4 = result_4 + pickle.load(handle)
            with open(f'agent_4_testing/agent_4_blind_{i}_{i + thread_div}.pickle', 'rb') as handle:
                result_4_blind = result_4_blind + pickle.load(handle)
            with open(f'agent_5_testing/agent_5_{i}_{i + thread_div}.pickle', 'rb') as handle:
                result_5 = result_5 + pickle.load(handle)
            i += thread_div
        else:
            with open(f'agent_1_testing/agent_1_{i}_{max_ghosts}.pickle', 'rb') as handle:
                result_1 = result_1 + pickle.load(handle)
            with open(f'agent_2_testing/agent_2_{i}_{max_ghosts}.pickle', 'rb') as handle:
                result_2 = result_2 + pickle.load(handle)
            with open(f'agent_3_testing/agent_3_{i}_{max_ghosts}.pickle', 'rb') as handle:
                result_3 = result_3 + pickle.load(handle)
            with open(f'agent_4_testing/agent_4_{i}_{max_ghosts}.pickle', 'rb') as handle:
                result_4 = result_4 + pickle.load(handle)
            with open(f'agent_4_testing/agent_4_blind_{i}_{max_ghosts}.pickle', 'rb') as handle:
                result_4_blind = result_4_blind + pickle.load(handle)
            with open(f'agent_5_testing/agent_5_{i}_{max_ghosts}.pickle', 'rb') as handle:
                result_5 = result_5 + pickle.load(handle)
            i = max_ghosts

    """Seperating data arrays into time and survival rates"""

    survival_1 = []
    survival_2 = []
    survival_3 = []
    survival_4 = []
    survival_4_blind = []
    survival_5 = []

    time_1 = []
    time_2 = []
    time_3 = []
    time_4 = []
    time_4_blind = []
    time_5 = []

    for i in range(len(result_1)):
        survival_1.append(result_1[i][1])
        survival_2.append(result_2[i][1])
        survival_3.append(result_3[i][1])
        survival_4.append(result_4[i][1])
        survival_4_blind.append(result_4_blind[i][1])
        survival_5.append(result_5[i][1])
        time_1.append(result_1[i][2])
        time_2.append(result_2[i][2])
        time_3.append(result_3[i][2])
        time_4.append(result_4[i][2])
        time_4_blind.append(result_4_blind[i][2])
        time_5.append(result_5[i][2])

    print(f"Agent 1 | Avg Survival: {sum(survival_1) / len(survival_1)} | Avg Time Per {num_iterations} Iterations: {sum(time_1) / len(time_1)}")
    print(f"Agent 2 | Avg Survival: {sum(survival_2) / len(survival_2)} | Avg Time Per {num_iterations} Iterations: {sum(time_2) / len(time_2)}")
    print(f"Agent 3 | Avg Survival: {sum(survival_3) / len(survival_3)} | Avg Time Per {num_iterations} Iterations: {sum(time_3) / len(time_3)}")
    print(f"Agent 4 | Avg Survival: {sum(survival_4) / len(survival_4)} | Avg Time Per {num_iterations} Iterations: {sum(time_4) / len(time_4)}")
    print(f"Agent 4 Blind | Avg Survival: {sum(survival_4_blind) / len(survival_4_blind)} | Avg Time Per {num_iterations} Iterations: {sum(time_4_blind) / len(time_4_blind)}")
    print(f"Agent 5 | Avg Survival: {sum(survival_5) / len(survival_5)} | Avg Time Per {num_iterations} Iterations: {sum(time_5) / len(time_5)}")


if __name__ == "__main__":
    main()
