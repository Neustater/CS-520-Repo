import numpy as np
import dill
import agent_v_star
import datetime
import agent_v_partial

"""Neural Network to model U"""
class v_star:
    def __init__(self):

        #Initilize weights randomly between -0.5 and 0.5
        #layer size (input size, output size)
        #First Hidden
        self.W1 = np.random.rand(150, 150) - 0.5 #Weights
        self.B1 = np.random.rand(150, 1) - 0.5 #Bias
        self.Z1 = [] #Intermediate step of forward prop used in back prop
        self.Out1 = []  #output of layer (input to next layer)
        #Second Hidden
        self.W2 = np.random.rand(150, 150) - 0.5
        self.B2 = np.random.rand(150, 1) - 0.5
        self.Z2 = []
        self.Out2 = []
        #Output Hidden
        self.W3 = np.random.rand(150, 1) - 0.5
        self.B3 = np.random.rand(1, 1) - 0.5
        self.Z3 = []
        self.Out3 = []

        self.mse_arr = []
        self.testing_mse_arr = []


    
    def back_prop(self, x, y, step_size):
        x = x.T
        y = y.T
        #back_prop on Output
        grad = self.mse_prime(y, self.Out3) * self.ReLU_prime(self.Z3) 
        # update weights and bias
        self.W3 = self.W3 - step_size * self.Out2.dot(grad.T)
        self.B3 = self.B3 - step_size * grad

        #back_prop on second to last layer
        grad = self.tanh_prime(self.Z2) * np.dot(self.W3, grad)
        self.W2 = self.W2 - step_size * self.Out1.dot(grad.T)
        self.B2 = self.B2 - step_size * grad

        #back_prop on third to last layer
        grad = self.tanh_prime(self.Z1) * np.dot(self.W2, grad)
        self.W1 = self.W1 - step_size * x.dot(grad.T)
        self.B1 = self.B1 - step_size * grad


    def forward_prop(self, x):
        x = x.T
        #forward propegation through layers using activation functions
        self.Z1 = np.dot(self.W1.T, x) + self.B1
        self.Out1 = np.tanh(self.Z1)

        self.Z2 = np.dot(self.W2.T, self.Out1, ) + self.B2
        self.Out2 = np.tanh(self.Z2)

        self.Z3 = np.dot(self.W3.T, self.Out2) + self.B3
        self.Out3 = self.ReLU(self.Z3)
        #since Out3 is the final single node it is the output

    # Function to train network on inputs
    def train(self, x_train, y_train, iterations, step_size):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        min_loss = np.Inf
        min_weights_bias = (self.W1, self.B1, self.W2, self.B2, self.W3, self.B3)

        self.mse_arr = []
        self.testing_mse_arr = []

        #shuffles inputs each iteration
        range_arr = np.arange(len(x_train), dtype=int)
        # Gradient Descent
        for i in range(iterations):
            np.random.shuffle(range_arr)
            for j in range_arr:
                self.forward_prop(x_train[j])
                self.back_prop(x_train[j], y_train[j], step_size)

            cur_loss =  self.mse(self.predict(x_train), y_train)
            #loss for all inputs
            print(f'Iteration {i+1}/{iterations} MSE:{cur_loss}')

            self.mse_arr.append(cur_loss)

            #save weights if min loss seen
            if cur_loss < min_loss:
                min_weights_bias = (self.W1, self.B1, self.W2, self.B2, self.W3, self.B3)

        self.W1, self.B1, self.W2, self.B2, self.W3, self.B3 = min_weights_bias
           
    
    def train_partial(self, x_train, y_train, x_test, y_test, iterations, step_size):
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        min_testing_loss = np.Inf
        min_weights_bias = (self.W1, self.B1, self.W2, self.B2, self.W3, self.B3)
        itr_since_min = 0

        self.mse_arr = []
        self.testing_mse_arr = []

        #shuffles inputs each iteration
        range_arr = np.arange(len(x_train), dtype=int)
        # Graident Descent
        for i in range(iterations):
            np.random.shuffle(range_arr)
            for j in range_arr:
                self.forward_prop(x_train[j])
                self.back_prop(x_train[j], y_train[j], step_size)
            
            cur_testing_loss =  self.mse(self.predict(x_test), y_test)
            cur_loss = self.mse(self.predict(x_train), y_train)
            #saves wegihts if min testing loss
            if cur_testing_loss < min_testing_loss:
                min_weights_bias = (self.W1, self.B1, self.W2, self.B2, self.W3, self.B3)
                itr_since_min = 0
            
            self.mse_arr.append(cur_loss)
            self.testing_mse_arr.append(cur_testing_loss)
            print(f'Iteration {i+1}/{iterations}')
            print("MSE of Training Set:", cur_loss)
            print("MSE of Testing Set:",cur_testing_loss)
            #early stop
            if itr_since_min >= 10:
                print("Early stop to prevent overfit!")
                break
            itr_since_min += 1
        
        self.W1, self.B1, self.W2, self.B2, self.W3, self.B3 = min_weights_bias

    #function similar to predict to return predictions but converts input to combo
    def get_value(self,i,j,k):
        i_arr = np.zeros(50)
        i_arr[i] = 1
        j_arr = np.zeros(50)
        j_arr[j] = 1
        k_arr = np.zeros(50)
        k_arr[k] = 1

        combo = np.concatenate((i_arr, j_arr))
        combo = np.concatenate((combo,k_arr))
        combo = [[combo]]

        combo = np.array(combo)

        self.forward_prop(combo[0])
        result = self.Out3[0][0]

        return result

    #function similar to predict to return predictions but converts input to combo (takes belief vector)
    def get_value_partial(self,i,vector,k):
        i_arr = np.zeros(50)
        i_arr[i] = 1
        j_arr = vector
        k_arr = np.zeros(50)
        k_arr[k] = 1

        combo = np.concatenate((i_arr, j_arr))
        combo = np.concatenate((combo,k_arr))
        combo = [[combo]]

        combo = np.array(combo)

        self.forward_prop(combo[0])
        result = self.Out3[0][0]

        return result

    def predict(self, inputs):
        output = []
        inputs = np.array(inputs)
        for i in range(len(inputs)):
            self.forward_prop(inputs[i])
            output.append(self.Out3)
        return output

    def predict_batch(self, inputs):
        inputs = np.array(inputs)
        self.forward_prop(inputs)
        return self.Out3.T

    # Activation functions and their derivatives
    def tanh_prime(self, Z):
        return 1-np.tanh(Z)**2

    def ReLU(self, Z):
        return np.where(Z>0, Z, 0.1*Z)

    def ReLU_prime(self, Z):
        return np.where(Z>0, 1, 0.1)

    # Loss MSE function and its derivative
    def mse(self, y, pred):
        return np.mean(np.power(y-pred, 2))

    def mse_prime(self, y, pred):
        return 2 * (pred-y) / y.size


"""Function to intitiate training of v_partial model using v* as basis"""

def v_partial_train():
    with open("u_star_env.pkl", 'rb') as file:
        env, _ = dill.load(file)

    with open("v_star_env.pkl", 'rb') as file:
        loaded_tuple = dill.load(file)

    v_star_env = loaded_tuple

    x = np.load('v_partial_training_combos.npy')
    y = np.load('v_partial_training_returns.npy')

    m = len(x)

    print("loaded")

    print("Number of samples collected:", m)
    print("Samples used for training: 100,000")
    print("Samples used for testing: 20,000")
    print("(Not all samples are used)")

    
    x_train = np.array(x)[:100000]
    y_train = np.array(y)[:100000]

    x_test = np.array(x)[100000:120000]
    y_test = np.array(y)[100000:120000]

    print("Starting to train")

    v_star_env.train_partial(x_train, y_train, x_test, y_test, iterations=200, step_size=0.0005)

    #checking predictions
    out = v_star_env.predict(x_test)
    print(np.array(out)[:10])
    print(y_test[:10])
    print(np.average(np.abs(np.array(out) - y_test)))

    dump_obj = v_star_env

    with open('v_partial_env.pkl', 'wb') as file:
        # A new file will be created
        dill.dump(dump_obj, file)

    print("v_partial dumped to file")

    """
    steps = 0
    count = 0
    for i in range(100):
        ag = agent_v_partial.Agent_v_partial(input_environment = env, env_v_partial = v_star_env)
        k = ag.move()
        if k[0] == 1:
            count += 1 
        steps += k[1]
    print('---------------------------')
    print('Success count :' + str(count))
    print('Steps:', steps/count)

    now = datetime.datetime.now()
    print(now.time())"""

"""Function to intitiate training of v* requires u* to be created"""

def v_train():
    with open("u_star_env.pkl", 'rb') as file:
        loaded_tuple = dill.load(file)

    print("loaded")
    

    env, u_star_obj = loaded_tuple
    
    print(np.where(np.isinf(u_star_obj.utility_matrix),-np.Inf,u_star_obj.utility_matrix).max())

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

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    v_star_env = v_star()
    v_star_env.train(x_train, y_train, iterations=200, step_size=0.0005)

    # checking predictions
    out = v_star_env.predict(x_train[:10])
    print(np.array(out))
    print(y_train[:10])
    print(np.average(np.abs(np.array(out) - y_train[:10])))

    dump_obj = v_star_env

    with open('v_star_env.pkl', 'wb') as file:
        # A new file will be created
        dill.dump(dump_obj, file)

    print("v_star dumped to file")
    """
    count = 0
    steps = 0
    for i in range(100):
        ag = agent_v_star.Agent_v_star(input_environment = env, env_v_star = v_star_env)
        k = ag.move()
        if k[0] == 1:
            count += 1 
        steps += k[1]
        print(i)
    print('---------------------------')
    print('Success count :' + str(count))
    print('Steps:', steps/count)

    now = datetime.datetime.now()
    print(now.time())"""

def main():
    v_train()
    v_partial_train()

if __name__ == '__main__':
    main()
    