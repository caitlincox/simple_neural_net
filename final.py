import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, random, tree_map
from jax.scipy.special import logsumexp
from jax.nn import relu, log_softmax, softmax
from functools import partial

#old scale 1e-2
#Starting values for weight and bias. Function is from Jax docs for training simple NN. 
#orig scale 1e-2
def initWeightBias(weight_size, layer_size, key, scale=1):
  w_key, b_key = random.split(key)
  struct = []
  for i, j in zip(weight_size, layer_size):
    wts = jnp.array(scale * random.normal(w_key, (int(j),int(i))))
    bias = jnp.array(scale * random.normal(b_key, (int(j),)))
    struct.append([wts,bias])
  return struct


#How many weights does each layer have given the
#  size of the data and the sizes of all the NN
#  layers (as a tuple)
def getWeightSizes(layer_sizes, input_size):
    wts = [i for i in layer_sizes[:-1]]
    wts.insert(0, input_size)
    return wts


#Initializes all weights and biases for nodes in the network,
# calling initWeightBias on one layer
# at a time. Returns array of node params. 
# @layer_sizes is a python array containing the number of nodes per layer
def initNodes(layer_sizes, prng_key, input_size):
    weights_sizes = getWeightSizes(layer_sizes, input_size)
    return initWeightBias(weights_sizes, layer_sizes, prng_key)


#Node function... you pass a layer wt matrix, bias & input vector. 
#Returns whether or not each activated.
def nodeFunction(weights, bias, input):
    x = vmap(jnp.dot, (None, 0))(weights, input) + bias
    return x


#assumed 1D jax array is the format of the data
def dataPrep(training_data, input_size, num_data):
    if (num_data == 1):
        return jnp.array(training_data)
    struct = jnp.reshape(training_data, (num_data, input_size))
    return struct


#Loop through layers of NN and output the logit for given input
def predict(current_params, num_layers, input):
    current_input = input 
    lin_op = 0 #Rare occurance of needing to declare in Python
    for j in range(num_layers):
        lin_op = nodeFunction(current_params[j][0], current_params[j][1], current_input)
        current_input = relu(lin_op)
    b = log_softmax(lin_op, axis = 1)
    return log_softmax(lin_op, axis = 1)


#log loss function
def logLoss(targets, log_probs):  
    ones = log_probs[:,0] * targets
    zeros = log_probs[:,1] * jnp.array(targets < 1)
    cross_entropy = -jnp.mean(ones + zeros)  
    return cross_entropy #mean cross-entropy loss


#network function that we want gradients of with backprop
def compGraph(current_params, num_layers, training_data, targets):
    log_predictions = predict(current_params, num_layers, training_data)
    return logLoss(targets, log_predictions)


#This is the other function I copied more or less exactly 
# from the JAX neural net guide
@partial(jit, static_argnums=3)
def update(params, targets, training_data, num_layers, step_size):
  grads = grad(compGraph)(params, num_layers, training_data, targets)
  updated_params = tree_map(lambda x, y: x - (y * step_size), params, grads)
  return updated_params

#Do one epoch.
def epoch(current_params, num_layers, input_size, training_data, targets):
    current_params = update(current_params, targets, training_data, num_layers, 0.01)
    return current_params

#Nyoom
def run(current_params, num_layers, input_size, training_data, targets, num_epochs):
    for i in range(num_epochs):
        current_params = epoch(current_params, num_layers, input_size, training_data, targets)
    return current_params

#TESTS

sizes = [3,2,3,4,2]
key = random.PRNGKey(12345)
h = jnp.array(key)
k = random.split(h)
nn_params = initNodes(sizes, key, 5)
print(nn_params)
training_data = jnp.asarray(random.normal(key = key, shape = (100,5)))
#a = jnp.asarray(dataPrep(training_data, 5, 2))
a = jnp.asarray(training_data)
print(a)
print(jnp.exp(predict(nn_params, 5, a)))

zeros = jnp.zeros(shape = (100))
ind = jnp.where(jnp.sum(a, axis = 1) > 1)
tar = zeros.at[ind].set(1)
gradient = grad(compGraph)(nn_params, 5, a, tar)
print(tar)
gradient = grad(compGraph)(nn_params, 5, a, tar)
print(gradient)
layer_sizes = jnp.split(jnp.asarray(sizes), 5)
y = epoch(nn_params, 5, 5, a, tar)
y = run(nn_params, 5, 5, a, tar, 10000)
print(y)
print("predict is ")
print(predict(y,5,a))
z = jnp.exp(predict(y, 5, a))
print(z)
print(z[:,0] - tar)
print(logLoss(tar, predict(y,5,a)))

gradient = grad(compGraph)(y, 5, a, tar)
print(gradient)



"""
key = random.PRNGKey(12345)
sizes = [1]
net = initNodes(sizes, key, 5)
print(net)
dat = jnp.array([jnp.ones(5)])
target = jnp.array([1])
thing = compGraph(net, 1, dat, target)
print(thing)
gr = grad(compGraph)(net, 1, dat, target)
print(gr)
print(predict(net,1,dat))
print (logLoss(target, predict(net, 1, dat)))

print(net[0][0][0])

arr = jnp.array([[0.0],[0.0]])
log_softmax(arr, axis = 0)
softmax(arr)"""