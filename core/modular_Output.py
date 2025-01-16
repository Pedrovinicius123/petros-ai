import numpy as np
from scipy.special import expit

class InnerModel:
    def __init__(self, x:np.ndarray, y:np.ndarray, learning_rate:float, n_inner_neurons:int):
        self.x = x
        self.y = []
        
        max_y = np.max(y)

        for item in y:
            value = np.zeros(max_y+1).astype(int)
            value[item] = 1

            self.y.append(value)
        self.y = np.array(self.y)

        self.learning_rate = learning_rate
        self.n_inner_neurons = n_inner_neurons

        self.input_neurons = self.x.shape[1]
        self.output_neurons = np.max(self.y)+1

        self.W1 = np.random.randn(self.input_neurons, n_inner_neurons)
        self.W2 = np.random.randn(n_inner_neurons, self.output_neurons)

        self.B1 = np.ones((1, n_inner_neurons))
        self.B2 = np.ones((1, self.output_neurons))

    def loss(self, softmax):
        # Cross Entropy
        pred = np.zeros(self.y.shape[0])
        for i, correct_index in enumerate(self.y):
            predicted = softmax[i][correct_index]
            pred[i] = predicted

        log_prob = -np.log(predicted)
        return log_prob/self.y.shape[0]

    def backpropagation(self):
        Delta2 = (self.A2 - self.y) * (self.sigmoidal_deriv(self.A2))
        W2_copy = np.copy(self.W2)
        self.W2 = Delta2.T.dot(self.A1)

        Delta1 = Delta2.dot(W2_copy.T)
        self.W1 = Delta1.T.dot(self.x)
    
    def sigmoidal_deriv(self, value):
        return expit(1 - expit(value))

    def forward(self, inp:np.ndarray):
        self.O1 = self.x.dot(self.W1) + self.B1
        self.A1 = expit(np.copy(self.O1))

        self.O2 = self.A1.dot(self.W2) + self.B2
        self.A2 = expit(np.copy(self.O2))

        return self.A2

    def fit(self, epochs:int):
        for epoch in range(epochs):            
            result = self.forward(self.x)
            loss = self.loss(result)

            self.backpropagation()
            print(f'EPOCHS {epoch}; ACCURACY {accuracy_score(result, self.y)}; LOSS {loss}')
