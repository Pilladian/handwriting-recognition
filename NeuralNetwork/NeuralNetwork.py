# Python 3.8

import scipy.special
import numpy


# Class Neural Network
class HaWiRe:

    # initialization of the network
    def __init__(self):
        # load learning rate
        try:
            with open("NeuralNetwork/data/learning_rate", 'r') as learning_rate:
                self.lr = float(learning_rate.readlines()[0])
        except Exception as e:
            print("Error occurred:\n" + str(e))

        # load weights
        try:
            self.wih = numpy.load('NeuralNetwork/data/wih.npy')
            self.who = numpy.load('NeuralNetwork/data/who.npy')
        except Exception as e:
            print("Error occurred:\n" + str(e))

    # activation function
    @staticmethod
    def activation(x):
        return scipy.special.expit(x)

    # return index of biggest value
    @staticmethod
    def find_max(array):
        maximum = 0
        ind = 0
        for i in range(len(array)):
            if array[i] > maximum:
                maximum = array[i]
                ind = i
        return ind

    # calculate the sum of an array
    @staticmethod
    def calculate_sum(array):
        s = 0
        for a in range(len(array)):
            s += abs(array[a][0])
        return s

    # query neural network
    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        hidden_input = numpy.dot(self.wih, inputs)
        hidden_output = self.activation(hidden_input)

        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation(final_input)

        # interpret results
        max_index = self.find_max(final_output)

        return max_index

    # train the neural network
    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # query the neural network
        hidden_input = numpy.dot(self.wih, inputs)
        hidden_output = self.activation(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation(final_input)

        # calculate error
        output_error = targets - final_output
        hidden_error = numpy.dot(self.who.T, output_error)

        # print target - prediction
        target = self.find_max(target_list)
        prediction = self.find_max(final_output)
        if target == prediction:
            print(" [ t | p ] \t" + str(target) + " ( + ) " + str(prediction))
        else:
            print(" [ t | p ] \t" + str(target) + " ( - ) " + str(prediction))

        # update weights
        self.who += self.lr * numpy.dot((output_error * final_output * (1.0 - final_output)), numpy.transpose(hidden_output))
        self.wih += self.lr * numpy.dot((hidden_error * hidden_output * (1.0 - hidden_output)), numpy.transpose(inputs))

    # save weights
    def save_weights(self):
        numpy.save("NeuralNetwork/data/wih.npy", self.wih)
        numpy.save("NeuralNetwork/data/who.npy", self.who)


# create weights for the neural network
def create_weights():
    # import node numbers
    with open("NeuralNetwork/data/nodes", 'r') as nodes:
        lines = nodes.readlines()
        input_nodes = int(lines[0][:-1], 10)
        hidden_nodes = int(lines[1][:-1], 10)
        output_nodes = int(lines[2], 10)

    # create weights
    wih = numpy.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
    who = numpy.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))

    numpy.save("NeuralNetwork/data/wih.npy", wih)
    numpy.save("NeuralNetwork/data/who.npy", who)
