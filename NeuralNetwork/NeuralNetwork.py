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
            self.wih1 = numpy.load('NeuralNetwork/data/wih1.npy')
            self.wh1h2 = numpy.load('NeuralNetwork/data/wh1h2.npy')
            self.wh2o = numpy.load('NeuralNetwork/data/wh2o.npy')
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

    # query neural network
    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        first_hidden_input = numpy.dot(self.wih1, inputs)
        first_hidden_output = self.activation(first_hidden_input)

        second_hidden_input = numpy.dot(self.wh1h2, first_hidden_output)
        second_hidden_output = self.activation(second_hidden_input)

        final_input = numpy.dot(self.wh2o, second_hidden_output)
        final_output = self.activation(final_input)

        # interpret results
        max_index = self.find_max(final_output)

        return max_index

    # train the neural network
    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # query the neural network
        first_hidden_input = numpy.dot(self.wih1, inputs)
        first_hidden_output = self.activation(first_hidden_input)
        second_hidden_input = numpy.dot(self.wh1h2, first_hidden_output)
        second_hidden_output = self.activation(second_hidden_input)
        final_input = numpy.dot(self.wh2o, second_hidden_output)
        final_output = self.activation(final_input)

        # calculate error
        output_error = targets - final_output
        second_hidden_error = numpy.dot(self.wh2o.T, output_error)
        first_hidden_error = numpy.dot(self.wh1h2.T, second_hidden_error)

        # update weights
        if self.find_max(input_list) != self.find_max(target_list):
            self.wh2o += self.lr * numpy.dot((output_error * final_output * (1.0 - final_output)), numpy.transpose(second_hidden_output))
            self.wh1h2 += self.lr * numpy.dot((second_hidden_error * second_hidden_output * (1.0 - second_hidden_output)), numpy.transpose(first_hidden_output))
            self.wih1 += self.lr * numpy.dot((first_hidden_error * first_hidden_output * (1.0 - first_hidden_error)), numpy.transpose(inputs))

    # save weights
    def save_weights(self):
        numpy.save("NeuralNetwork/data/wih1", self.wih1)
        numpy.save("NeuralNetwork/data/wh1h2", self.wh1h2)
        numpy.save("NeuralNetwork/data/wh2o", self.wh2o)


# create weights for the neural network
def create_weights():
    # import node numbers
    with open("NeuralNetwork/data/nodes", 'r') as nodes:
        lines = nodes.readlines()
        input_nodes = int(lines[0][:-1], 10)
        first_hidden_nodes = int(lines[1][:-1], 10)
        second_hidden_nodes = int(lines[2][:-1], 10)
        output_nodes = int(lines[3], 10)

    # create weights
    wih1 = numpy.random.normal(0.0, pow(first_hidden_nodes, -0.5), (first_hidden_nodes, input_nodes))
    wh1h2 = numpy.random.normal(0.0, pow(second_hidden_nodes, -0.5), (second_hidden_nodes, first_hidden_nodes))
    wh2o = numpy.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, second_hidden_nodes))

    numpy.save("NeuralNetwork/data/wih1", wih1)
    numpy.save("NeuralNetwork/data/wh1h2", wh1h2)
    numpy.save("NeuralNetwork/data/wh2o", wh2o)
