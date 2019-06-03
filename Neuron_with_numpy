from numpy import dot,array

class NeuralNetwork():
    
    def __init__(self):
        random.seed(1)
        ## Creates random weights with 3 rows and 1 column matrix.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
    
    def __sigmoid(self,x):
        ## used as activation function , it squashes the value b/w 0 to 1.
        return 1/(1+exp(-x))
    
    def __sigmoid_derivative(self,x):
        ## Gradient of activation function.
        return x*(1-x)
    
    def process(self,inputs):
        return self.__sigmoid(dot(inputs,self.synaptic_weights))
    
    def train(self,input_features,actual_output,number_of_iterations):
        for i in range(number_of_iterations):
            pred_output = self.process(input_features)
            errors = actual_output-pred_output
            adjustment = dot(input_features.T, errors* self.__sigmoid_derivative(pred_output))
            self.synaptic_weights += adjustment        
            
            

neural_network = NeuralNetwork()
## Weights before learning.
print("Before weights",neural_network.synaptic_weights.T)

trainingset = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
actual_ouput = array([[0, 1, 1,0]]).T
neural_network.train(trainingset, actual_ouput, 10000)

## Weights after learning.
print("After training weights",neural_network.synaptic_weights.T)
## Predicting for new entry.
print("Predicted out put",neural_network.process(array([1, 1, 0])))
               
