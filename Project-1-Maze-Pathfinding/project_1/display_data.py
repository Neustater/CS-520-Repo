"""Parsing results of subprocesses into greater array of data"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

from main import max_ghosts, thread_div, num_iterations

def display_data():
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

    print(
        f"Agent 1 | Avg Survival: {sum(survival_1) / len(survival_1)} | Avg Time Per {num_iterations} Iterations: {sum(time_1) / len(time_1)}")
    print(
        f"Agent 2 | Avg Survival: {sum(survival_2) / len(survival_2)} | Avg Time Per {num_iterations} Iterations: {sum(time_2) / len(time_2)}")
    print(
        f"Agent 3 | Avg Survival: {sum(survival_3) / len(survival_3)} | Avg Time Per {num_iterations} Iterations: {sum(time_3) / len(time_3)}")
    print(
        f"Agent 4 | Avg Survival: {sum(survival_4) / len(survival_4)} | Avg Time Per {num_iterations} Iterations: {sum(time_4) / len(time_4)}")
    print(
        f"Agent 4 Blind | Avg Survival: {sum(survival_4_blind) / len(survival_4_blind)} | Avg Time Per {num_iterations} Iterations: {sum(time_4_blind) / len(time_4_blind)}")
    print(
        f"Agent 5 | Avg Survival: {sum(survival_5) / len(survival_5)} | Avg Time Per {num_iterations} Iterations: {sum(time_5) / len(time_5)}")

    x = [i for i in range(len(result_1))]

    """Creates effectiveness scatterplot/ graphs"""
    plt.scatter(x, survival_1, alpha=0.25)
    fit = np.poly1d(np.polyfit(np.array(x), np.array(survival_1), 7))
    plt.plot(x, fit(np.array(x)), label='agent_1')
    plt.scatter(x, survival_2, alpha=0.25)
    fit = np.poly1d(np.polyfit(np.array(x), np.array(survival_2), 7))
    plt.plot(x, fit(np.array(x)), label='agent_2')
    plt.scatter(x, survival_3, alpha=0.25)
    fit = np.poly1d(np.polyfit(np.array(x), np.array(survival_3), 7))
    plt.plot(x, fit(np.array(x)), label='agent_3')
    plt.scatter(x, survival_4, alpha=0.25, color="red")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(survival_4), 7))
    plt.plot(x, fit(np.array(x)), label='agent_4', color="red")
    plt.scatter(x, survival_4_blind, alpha=0.25, color="pink")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(survival_4_blind), 7))
    plt.plot(x, fit(np.array(x)), label='agent_4 blind', color="pink")
    plt.scatter(x, survival_5, alpha=0.25, color="purple")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(survival_5), 7))
    plt.plot(x, fit(np.array(x)), label='agent_5', color="purple")
    plt.xlabel("Number of Ghosts")
    plt.ylabel(f"Survival Rate (per {num_iterations} iterations)")
    plt.title("Effectiveness")
    plt.yticks([x / 100 for x in range(0, 105, 10)], [f"{x}%" for x in range(0, 105, 10)])
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.25)

    plt.legend()
    plt.savefig('effectiveness.png', dpi=175)
    plt.show()

    """Converting time per 100 iterations to time per iteration """

    time_1 = [time_1[x] / num_iterations for x in range(len(time_1))]
    time_2 = [time_2[x] / num_iterations for x in range(len(time_2))]
    time_3 = [time_3[x] / num_iterations for x in range(len(time_3))]
    time_4 = [time_4[x] / num_iterations for x in range(len(time_4))]
    time_4_blind = [time_4_blind[x] / num_iterations for x in range(len(time_4_blind))]
    time_5 = [time_5[x] / num_iterations for x in range(len(time_5))]

    """Creates the scatterplot/ graphs for performance"""

    plt.scatter(x, time_1, alpha=0.25)
    fit = np.poly1d(np.polyfit(np.array(x), np.array(time_1), 5))
    plt.plot(x, fit(np.array(x)), label='agent_1')
    plt.xlabel("Number of Ghosts")
    plt.ylabel("Avg Time (seconds per attempt)")
    plt.title("Agent 1 Performance")
    plt.legend()
    plt.savefig('a1_performance.png', dpi=175)
    plt.show()
    plt.scatter(x, time_2, alpha=0.25, color="orange")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(time_2), 5))
    plt.plot(x, fit(np.array(x)), label='agent_2', color="orange")
    plt.xlabel("Number of Ghosts")
    plt.ylabel("Avg Time (seconds per attempt)")
    plt.title("Agent 2 Performance")
    plt.legend()
    plt.savefig('a2_performance.png', dpi=175)
    plt.show()
    plt.scatter(x, time_3, alpha=0.25, color="green")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(time_3), 5))
    plt.plot(x, fit(np.array(x)), label='agent_3', color="green")
    plt.xlabel("Number of Ghosts")
    plt.ylabel("Time (seconds per attempt)")
    plt.title("Agent 3 Performance")
    plt.legend()
    plt.savefig('a3_performance.png', dpi=175)
    plt.show()
    plt.scatter(x, time_4, alpha=0.25, color="red")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(time_4), 5))
    plt.plot(x, fit(np.array(x)), label='agent_4', color="red")
    plt.xlabel("Number of Ghosts")
    plt.ylabel("Avg Time (seconds per attempt)")
    plt.title("Agent 4 Performance")
    plt.legend()
    plt.savefig('a4_performance.png', dpi=175)
    plt.show()
    plt.scatter(x, time_4_blind, alpha=0.25, color="red")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(time_4_blind), 5))
    plt.plot(x, fit(np.array(x)), label='agent_4 blind', color="pink")
    plt.xlabel("Number of Ghosts")
    plt.ylabel("Avg Time (seconds per attempt)")
    plt.title("Agent 4 Blind Performance")
    plt.legend()
    plt.savefig('a4_blind_performance.png', dpi=175)
    plt.show()
    plt.scatter(x, time_5, alpha=0.25, color="purple")
    fit = np.poly1d(np.polyfit(np.array(x), np.array(time_5), 5))
    plt.plot(x, fit(np.array(x)), label='agent_5', color="purple")
    plt.xlabel("Number of Ghosts")
    plt.ylabel("Avg Time (seconds per attempt)")
    plt.title("Agent 5 Performance")
    plt.legend()
    plt.savefig('a5_performance.png', dpi=175)
    plt.show()

display_data()




