from dnn_lib import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


train_X, train_Y, test_X, test_Y = load_2D_dataset()

parameters, costs = train_deep_fully_connected_model(train_X, train_Y, [train_X.shape[0],20,3,1], learning_rate=0.3, num_iterations=30000, print_cost=True, initialization='xavier', lambd=0.1, keep_prob=[0.8,0.9])
plot_costs(costs, learning_rate=0.3)
train_predictions = predict(train_X, parameters, 0.5)
train_accuracy = calculate_accuracy(train_predictions, train_Y)
print ("Training set accuracy: {}".format(train_accuracy))

plot_decision_boundary(parameters, train_X, train_Y, padding=0.1)

test_predictions = predict(test_X, parameters, 0.5)
test_accuracy = calculate_accuracy(test_predictions, test_Y)
print ("Training set accuracy: {}".format(test_accuracy))