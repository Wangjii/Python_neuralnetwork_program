# coding=utf-8

# =============================== #
#  Project: 手写数字识别
#  Date: 2019年4月12日
#  Resource:Python 神经网络编程
# =============================== #

from model import neuralNetwork
import numpy

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# Learning rate is 0.1
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("NeuralNetwork/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the nerual network
'''
# go through all records in the training data set
for record in training_data_list:
    # split the record by the ',' commas
    # 通过','将数分段
    all_values = record.split(',')
    # scale and shift the inputs
    # 将所有的像素点的值转换为0.01-1.00
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
    # creat the target output values
    # 创建标签输出值
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    # 10个输出值，对应的为0.99，其他为0.01
    targets[int(all_values[0])] = 0.99
    # 传入网络进行训练
    n.train(inputs, targets)
    pass
'''

# 对训练过程进行循环
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        # split the record by the ',' commas
        # 通过','将数分段
        all_values = record.split(',')
        # scale and shift the inputs
        # 将所有的像素点的值转换为0.01-1.00
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
        # creat the target output values
        # 创建标签输出值
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        # 10个输出值，对应的为0.99，其他为0.01
        targets[int(all_values[0])] = 0.99
        # 传入网络进行训练
        n.train(inputs, targets)
        pass
    pass

# test the nerual network
testing_data_file = open("NeuralNetwork/mnist_test.csv", 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

# 创建一个空白的计分卡
scorecard = []
# 遍历测试数据
for record in testing_data_list:
    all_values = record.split(',')
    # 提取正确的标签
    correct_label = int(all_values[0])
    # print(correct_label, 'correct label')
    # 读取像素值并转换
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
    # 通过神经网络得出结果
    outputs = n.query(inputs)
    # 结果
    label = numpy.argmax(outputs)
    # print(label, "network's answer")
    # 标签相同，计分卡加一，否则加零
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
# 输出计分卡
# print(scorecard)
# 输出分数
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
