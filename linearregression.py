import sys
import numpy as np

# simple float list
x =[]
y =[]

# use second argument to open the correct file and split into x and y floats
filename = sys.argv[1]
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float (i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

# 80% of data used to train the linear regressor, 20% used to test the model
num_training = int(0.8 *len(x))
num_test = len(x) - num_training

# Training data
x_train = np.array(x[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data
x_test = np.array(x[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

from sklearn import linear_model

# Create linear regression object
linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(x_train, y_train)


import matplotlib.pyplot as plt

y_train_predict = linear_regressor.predict(x_train)
plt.figure()
# plt.scatter(x_train, y_train, color='green')
# plt.plot(x_train, y_train_predict, color='black', linewidth=4)
# plt.title('Training data')
# plt.show()

y_test_predict = linear_regressor.predict(x_test)

plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_test, y_test_predict, color='black', linewidth=4)
plt.title('Test data')
plt.show()