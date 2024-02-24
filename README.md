# NN_Scratch
Neural network from scratch 

Data: https://drive.google.com/file/d/1YELZmJaFO2B1dk2Wjs969u89rFperXyP/view?usp=drive_link

To implement the linear transformation function, without using the in-built Python functions, we need to understand how a layer works. I have considered 2 layers and 2 different activation functions.

The forward propagation equations are as follows:
--> Z(1) = W(1)X + b(1)
--> A(1) = σ1(Z(1))
--> Z(2) = W(2)A(1) + b(2)
--> A(2) = σ2(Z(2))

Here sigmoid is for the output activation function σ2 and ReLU is for the hidden layer activation function σ1.

The input dimensions are (10000, 3072). We have 10000 examples and 3072 input neurons. The
weights have (3072, hidden units) dimensions. So, the output would be (10000, hidden units) dimensions.
Bias is (1, hidden units).

x.W1 + b1 : (10000, 3072) · (3072, 15) + (1, 15) = (10000, 15)

We apply non linearity to the above layer then during the second layer we apply weights with following
dimensions (15, 1) and bias as (1, 1) then we get an output with dimensions as (10000, 1).

x.W2 + b2 : (10000, 15) · (15, 1) + (1, 1) = (10000, 1)

In the above example we take hidden units as 15, we could consider different hidden layers.

For backward Propagation: we calculate gradients of weights and biases and add l2 regularization
to update weights and biases.

∂L/∂W(1) + λW(1),
∂L/∂W(2) + λW(2),
∂L/∂b(1) + λb(1),
∂L/∂b(2) + λb(2),

grad-weights1 : (3072, 50) · (50, 15) + (1, 15) = (3072, 15)
grad-bias1 : (1, 15)
grad-weights2 : (15, 50) · (50, 1) = (15, 1)
grad-bias2 : (1, 1)

I have included the compute loss function separately in the net class. The derivative of L2 regularization
gives us the value of l2-penalty*(W) and l2-penalty*b. the same has been added to
gradient weight and gradient bias.
loss = −1/N sum(y(i) log(a(2)(i)) + (1 − y(i))*log(1 − a(2)(i)))

Here the velocity is defined to speed up the convergence of mini batch stochastic gradient descent
optimization algorithm by incorporating a fraction of the previous update steps. The velocity
is updated at each step to update the weights and biases.
vW = momentum · vW + (1 − momentum) · dW,
vb = momentum · vb + (1 − momentum) · db.
Accordingly weights have been updated as follows:
W = W − learning rate · vW,
b = b − learning rate · vb.


Hyperparameters:
1. No of hidden units = 15
2. Learning rate = 0.01
3. Mini-batch size = 200
4. Momentum parameter = 0.8
5. L2 penalty regularizer = 0.0001

