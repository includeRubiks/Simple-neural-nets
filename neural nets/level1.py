# Level 1 difficulty to understand
# Simple network simulating NOT gate
# 2 layers, 2 neurons
import numpy as np
import matplotlib as mpl
x = 1 # Input for model
y = 1 # Output for model
learning_rate = 0.01 # The learning rate for the model (Google has a good course outlining stuff like this)
epochs = 100 # The number of times you train the model
w = np.random.randint(0,10)*0.01 # The one weight needed for the network
b = np.random.randint(0,10)*0.01 # The one bias needed for the network
trainingx = [1,0] # Input for training model (trainingx[0]'s output should be aprox. to trainingy[0])
trainingy = [0,1] # Output the model is aiming for
def sigmoid(a):
    # Defining sigmoid function. If you don't know what that is 3Blue1Brown has a youtube video on neural networks, which is very good to learn the math.
    return 1/(1+np.exp(-a))
def output(xi,wi,bi):
    # Defining the output function of the model
    return sigmoid(wi * xi + bi)
# Definitions done!
# On to training
for i in range(epochs):
    #Train the weights
    for i in range(len((trainingx+trainingy)/2)):
        l2 = (output(trainingx[i],w,b)-trainingy[i])**2 # Find loss squared
        if l2 > (output(trainingx[i],w+learning_rate,b)-trainingy[i])**2: # Compare the two losses
            w += learning_rate # If it decreases loss, permanantly add to w so that it did
        elif l2 > (output(trainingx[i],w-learning_rate,b)-trainingy[i])**2: # Compare the two losses
            w -= learning_rate # If it decreases loss, permanantly subtract from w so that it did
        