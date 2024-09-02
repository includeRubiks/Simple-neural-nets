# Simple neural nets
## The simplest networks around
## Training
Trained using a method very similar to gradient descent, where I test the loss of the result of adding  `learning rate` to each weight, and if loss is less, add learning rate.
It's like gradient descent, but no calculus (which I don't understand).
## Networks
### Level 1
Level 1 is a simple neural network with 2 neurons and and 2 layers. (Including input and output).
It can be trained differently by changing the variables in `trainingy` because the model trains itself so that the 
input value of `trainingx[i]` gives an output close to `trainingy[i]`. The file is in the neural nets folder and 
is called level1.py.