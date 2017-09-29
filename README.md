# README.md
# Schroedinger equation deep learning project
# Version 1: 25 September 2017
# Steven K. Steinke
# steven.k.steinke@gmail.com
This project uses Tensorflow to first generate random 1-D potentials, then solve them using
gradient descent. These potentials and solutions are partitioned into training and validation data.
Next, the training data are fed into a simple neural network with 2 hidden layers that use
the softplus activation function. The mean squared distance between the “correct” solutions
and the output of the neural network is the cost function; gradient descent on the network
“solves” the problem. There are various simple tools included to visualize the output of this process.

# Files included:
genpotential.py   Generates the random potentials and solves them individually

schroedinger_nn.py   Sets up a simple neural network to solve the 1D Sch. Eqn.

display_nnout.py    Plots a single potential and its actual and predicted solutions

save_nn.py     Saves the weights and biases of the network for later recovery

visualize_nn.py   Creates bitmaps of the weights and biases of the network after sorting for spatial correlation
