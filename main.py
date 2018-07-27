import numpy as np
from neural_network import NeuralNetwork

# number of input, hidden and output nodes
input_nodes = 1024
hidden_nodes = 300
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the devnagri training data CSV file into a list
training_data_file = open("train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# go through all records in the training data set
for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    neural_network.train(inputs, targets)
    pass

# load the devnagri test data CSV file into a list
test_data_file = open("test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    print(correct_label, "Correct label")
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = neural_network.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    print(label, "Network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)


