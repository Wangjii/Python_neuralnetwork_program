import numpy
# scipy.special for the sigmoid function expit()
import scipy.special


# neural network class definition
class neuralNetwork:

    # 创建一个类对象
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set the number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights insides the arrays are w_i_j,
        # where link is from node i to node j in the next layer
        # w11 w12 w21 w22 etc

        # simple random number
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        # Normal distribution
        # average = 0
        # Standard deviation = 1/evolution of number of nodes passed in
        '''
        self.wih = numpy.random.normal(0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))
        '''

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        # 利用传输矩阵wih，计算隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        # 计算隐藏层输出，激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        # 利用传输矩阵who，计算输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights,
        # recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        # wj,k = learningrate * error * sigmoid(ok) * (1 - sigmoid(ok)) · oj^T
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))

        pass

    # 定义神经网络
    # query the neural network
    def query(self, inputs_list):

        # convert inputs list to 2D array
        # 输入矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        # 利用传输矩阵wih，计算隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        # 计算隐藏层输出，激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        # 利用传输矩阵who，计算输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
