# Level 1 difficulty to understand
# Simple network simulating NOT gate
# 2 layers, 2 neurons
import numpy as np
import matplotlib.pyplot as mpl
x = 1 # Input for model
learning_rate = 0.09 # The learning rate for the model (Google has a good course outlining stuff like this, but just google it.)
epochs = 100 # The number of times you train the model
w = np.random.randint(0,10)*0.01 # The one weight needed for the network
b = np.random.randint(0,10)*0.01 # The one bias needed for the network
trainingx = [1,0] # Input for training model (trainingx[0]'s output should be aprox. to trainingy[0])
trainingy = [0,1] # Output the model is aiming for (I'm not using sigmoid because I don't need it)
l2graph = [] # The y plots for the l2 graph
def sigmoid(a):
    # Defining sigmoid function. If you don't know what that is 3Blue1Brown has a youtube video on neural networks, which is very good to learn the math.
    return 1/(1+np.exp(-a))
def output(xi,wi,bi):
    # Defining the output function of the model
    return sigmoid((wi * xi + bi))
# Definitions done!
# On to training
for i in range(epochs):
    for i in range(len(trainingx)):
        # Train the weight
        l2 = (output(trainingx[i],w,b)-trainingy[i])**2 # Find loss squared
        if l2 > (output(trainingx[i],w+learning_rate,b)-trainingy[i])**2: # Compare the two losses
            w += learning_rate # If it decreases loss, permanantly add to w so that it did
        elif l2 > (output(trainingx[i],w-learning_rate,b)-trainingy[i])**2: # Compare the two losses
            w -= learning_rate # If it decreases loss, permanantly subtract from w so that it did
        l2graph.append((output(trainingx[i],w,b)-trainingy[i])**2)
    for i in range(len(trainingx)):
        # Train the bias
        if l2 > (output(trainingx[i],w,b+learning_rate)-trainingy[i])**2: # Compare the two losses
            b += learning_rate # If it decreases loss, permanantly add to w so that it did
        elif l2 > (output(trainingx[i],w,b-learning_rate)-trainingy[i])**2: # Compare the two losses
            b -= learning_rate # If it decreases loss, permanantly subtract from w so that it did
        l2graph.append((output(trainingx[i],w,b)-trainingy[i])**2)
# Training done!
print(f'Output: {output(x,w,b)}') # Print the output
print(f'w: {w}') # Print the weight
print(f'b: {b}') # Print the bias
mpl.plot(l2graph)
mpl.xlabel('Epoch')
mpl.ylabel('L2')
mpl.title('L2')
mpl.show()