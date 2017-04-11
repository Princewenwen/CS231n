import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               use_batchnorm = False, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim

    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros((1, num_filters))
    self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W/4, hidden_dim)
    self.params['b2'] = np.zeros((1, hidden_dim))
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros((1, num_classes))
    if self.use_batchnorm:
      self.params['gamma1'] = np.ones((1, num_filters))
      self.params['beta1'] = np.zeros((1, num_filters))
      self.params['gamma2'] = np.ones((1, hidden_dim))
      self.params['beta2'] = np.zeros((1, hidden_dim))
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = {'mode': 'train'}

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    N, C, H, W = X.shape
    if self.use_batchnorm:
      gamma1 = self.params['gamma1']
      beta1 = self.params['beta1']
      gamma2 = self.params['gamma2']
      beta2 = self.params['beta2']
      c1, cache1 = conv_forward_im2col(X, W1, b1, conv_param)
      #print 'c1',c1.shape
      bn1, cache2 = spatial_batchnorm_forward(c1, gamma1, beta1, self.bn_params)
      r1, cache3 = relu_forward(bn1)
      h1, cache4 = max_pool_forward_naive(r1, pool_param)
      a2, cache5 = affine_forward(h1, W2, b2)
      #print 'a2',a2.shape
      bn2, cache6 = batchnorm_forward(a2, gamma2, beta2, self.bn_params)
      h2, cache7 = relu_forward(bn2)
      out, cache = affine_forward(h2, W3, b3)
      scores = out
    else:
      h1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
      h2, cache2 = affine_relu_forward(h1, W2, b2)
      out, cache = affine_forward(h2, W3, b3)
      scores = out
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    #reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(W3 * W3)
    reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])  # trick
    loss = data_loss + reg_loss
    if self.use_batchnorm:
      dh2, dW3, db3 = affine_backward(dscores, cache)
      dbn2 = relu_backward(dh2, cache7)
      da2, dgamma2, dbeta2 = batchnorm_backward(dbn2, cache6)
      dh1, dW2, db2 = affine_backward(da2, cache5)
      dr1 = max_pool_backward_naive(dh1, cache4)
      dbn1 = relu_backward(dr1, cache3)
      dc1, dgamma1, dbeta1 = spatial_batchnorm_backward(dbn1, cache2)
      dX, dW1, db1 = conv_backward_im2col(dc1, cache1)
    else:
      dh2, dW3, db3 = affine_backward(dscores, cache)
      dh1, dW2, db2 = affine_relu_backward(dh2, cache2) 
      dX, dW1, db1 = conv_relu_pool_backward(dh1, cache1)

    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3  

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}  # trick
    #grads['dx'] = dx
    #grads['dW1'] = dW1
    #grads['db1'] = db1
    #grads['dW2'] = dW2
    #grads['db2'] = db2
    #grads['dW3'] = dW3
    #grads['db3'] = db3
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
