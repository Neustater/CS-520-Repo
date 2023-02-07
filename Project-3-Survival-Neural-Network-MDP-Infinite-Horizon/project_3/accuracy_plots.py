import numpy as np
import v_star
import u_star
import dill
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''IF YOU DOWNLOADED THIS RUN generate_v_partial_training_data.py FIRST!'''

def mse(y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

def se(y_true, y_pred):
        return np.power(y_true-y_pred, 2)

with open("v_partial_env.pkl", 'rb') as file:
    v_partial = dill.load(file)
with open("u_star_env.pkl", 'rb') as file:
    env, u_star = dill.load(file)
with open("v_star_env.pkl", 'rb') as file:
    v_star = dill.load(file)

v_star_mse = v_star.mse_arr
labels = [x+1 for x in range(len(v_star_mse))]

plt.plot(labels, v_star_mse)
plt.title('Loss of V*')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.savefig('accuracy_plots/v_star.png', bbox_inches='tight')
plt.clf()

x = np.load('v_partial_training_combos.npy')
y = np.load('v_partial_training_returns.npy')

v_partial_mse = v_partial.mse_arr
print("MSE for Training Before First Iteration: ", mse(y[:100000], v_star.predict(x[:100000])))
v_partial_testing_mse = v_partial.testing_mse_arr
print("MSE for Testing Before First Iteration: ", mse(y[100000:120000], v_star.predict(x[100000:120000])))
labels = [x + 1 for x in range(len(v_partial_mse))]

print("Min of Training", np.min(v_partial_mse))
print("Min of Testing", np.min(v_partial_testing_mse))

plt.plot(labels, v_partial_mse, label="Training Set")
plt.plot(labels, v_partial_testing_mse, label="Testing Set")
plt.title('Loss of V Partial')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.savefig('accuracy_plots/v_partial.png', bbox_inches='tight')
plt.clf()

x = x[-10000:]
y = y[-10000:]

V_partial_results = np.array(v_partial.predict(x))
U_V_partial_results = []

def U_V_partial(i_vector, vector, k_vector):
    i = np.argmax(i_vector)
    k = np.argmax(k_vector)
    predator_index = i
    agent_index = k

    agent_options, _, _, _ = u_star.options_matrix[i][0][k]
    agent_options_vals = [0] * len(agent_options)
    for prey_index, belief in enumerate(vector):
        _, focused_options_given_agent, unfocused_options, prey_options = u_star.options_matrix[i][prey_index][k]
        
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
                        total_value += focused_options_prob * prey_options_prob * v_star.get_value(pred_focus_ind, prey_index, ag_index)
                    for pred_unfocus_ind in unfocused_options:
                        #if prey_ind == ag_index and agent_index != pred_unfocus_ind:
                            #continue
                        total_value += unfocused_options_prob * prey_options_prob * v_star.get_value(pred_unfocus_ind, prey_index, ag_index)
                else:
                    total_value = np.Inf

                if belief != 0:
                    total_value = total_value * belief
                else:
                    total_value = 0
                agent_options_vals[itr] += total_value

    return min(agent_options_vals) + 1


def U_partial(i_vector, vector, k_vector):
    i = np.argmax(i_vector)
    k = np.argmax(k_vector)
    predator_index = i
    agent_index = k

    agent_options, _, _, _ = u_star.options_matrix[i][0][k]
    agent_options_vals = [0] * len(agent_options)
    for prey_index, belief in enumerate(vector):
        _, focused_options_given_agent, unfocused_options, prey_options = u_star.options_matrix[i][prey_index][k]
        
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
                        total_value += focused_options_prob * prey_options_prob * u_star.utility_matrix[pred_focus_ind][prey_index][ag_index]
                    for pred_unfocus_ind in unfocused_options:
                        #if prey_ind == ag_index and agent_index != pred_unfocus_ind:
                            #continue
                        total_value += unfocused_options_prob * prey_options_prob * u_star.utility_matrix[pred_unfocus_ind][prey_index][ag_index]
                else:
                    total_value = np.Inf

                if belief != 0:
                    total_value = total_value * belief
                else:
                    total_value = 0
                agent_options_vals[itr] += total_value

    return min(agent_options_vals) + 1

U_partial_results = []

j = 0

for i in x:
    one_hot = i[0]
    U_V_partial_results.append(U_V_partial(one_hot[:50],one_hot[50:100], one_hot[100:150]))
    #U_partial_results.append(U_partial(one_hot[:50],one_hot[50:100], one_hot[100:150]))
    j += 1
    print(j,end=' ', flush=True)

U_V_partial_results = np.array(U_V_partial_results)

msevp = mse(y.flatten(), V_partial_results.flatten())
mseuvp = mse(y.flatten(), U_V_partial_results.flatten())
print("\nMSE of V Partial", mse(y.flatten(), V_partial_results.flatten()))
print("MSE of U V* Partial", mse(y.flatten(), U_V_partial_results.flatten()))

plt.figure(figsize=(8,10))

"""
v_partial_se = se(y.flatten(), V_partial_results.flatten())
v_partial_se = v_partial_se[(v_partial_se > np.mean(v_partial_se) - 3 * np.std(v_partial_se))]
v_partial_se = v_partial_se[(v_partial_se < np.mean(v_partial_se) + 3 * np.std(v_partial_se))]

u_v_partial_se = se(y.flatten(), U_V_partial_results.flatten())
u_v_partial_se = u_v_partial_se[(u_v_partial_se > np.mean(u_v_partial_se) - 3 * np.std(u_v_partial_se))]
u_v_partial_se = u_v_partial_se[(u_v_partial_se < np.mean(u_v_partial_se) + 3 * np.std(u_v_partial_se))]

plt.figure(figsize=(7,10))
plt.boxplot([v_partial_se, u_v_partial_se])
plt.title('V Partial and U V* Partial Accuracy (Squared Error)')
plt.xticks([1, 2], ['V Partial', 'U V* Partial'])
plt.savefig('accuracy_plots/partials.png', bbox_inches='tight')
plt.clf()"""

#plt.boxplot([v_partial_se, u_v_partial_se])
#plt.scatter([0 for _ in range(len(v_partial_se))], v_partial_se)
#plt.scatter([1 for _ in range(len(u_v_partial_se))], u_v_partial_se)
plt.bar(['V Partial', 'U V* Partial'],[msevp, mseuvp])
plt.title('V Partial and U V* Partial Accuracy (Squared Error)')
#plt.xticks([1, 2], ['V Partial', 'U V* Partial'])
plt.savefig('accuracy_plots/partials.png', bbox_inches='tight')
plt.clf()

with open("u_star_env.pkl", 'rb') as file:
        loaded_tuple = dill.load(file)


env, u_star_obj = loaded_tuple

print("Max Value of U*:", np.where(np.isinf(u_star_obj.utility_matrix),-np.Inf,u_star_obj.utility_matrix).max())

x_train = []
y_train = []

matrix_len = round(len(u_star_obj.utility_matrix))
for i in range(matrix_len):
    for j in range(matrix_len):
        for k in range(len(u_star_obj.utility_matrix)):
            #remove inf from training set
            if u_star_obj.utility_matrix[i][j][k] == np.Inf:
                continue
            
            #create one-hot vectors
            i_env = np.zeros(50)
            i_env[i] = 1
            j_env = np.zeros(50)
            j_env[j] = 1
            k_env = np.zeros(50)
            k_env[k] = 1

            util = np.abs(u_star_obj.utility_matrix[i][j][k])
            
            u_arr = [util]
            input = np.concatenate((i_env, j_env))
            input = np.concatenate((input,k_env))
            x_train.append([input])
            y_train.append([u_arr])

x = np.array(x_train)
y = np.array(y_train)

y_pred = np.array(v_star.predict(x))

print("MSE of V*", mse(y.flatten(), y_pred.flatten()))

#sqerr = se(y.flatten(), y_pred.flatten())

#print("Arg max act=", y[np.argpartition(sqerr, -10)[-10:]])
#print("Arg pred=", y_pred[np.argpartition(sqerr, -10)[-10:]])

#plt.figure(figsize=(10, 5))
#print(y.flatten())
#print(V_partial_results.flatten())
#print(U_V_partial_results.flatten())