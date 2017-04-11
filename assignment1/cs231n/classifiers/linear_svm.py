import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)  # X[i] one pictrue scores(1,C)
    correct_class_score = scores[y[i]]  #The correct class' score
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin        # get the loss
        dW[:,y[i]] += -X[i,:] # compute the correct_class gradients
        dW[:,j] += X[i,:]     # compute the wrong_class gradients

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  

  '''
  # compute the loss and the gradient
  h = 0.0001
  for m in xrange(W.shape[0]):
    for n in xrange(W.shape[1]):
      W_h = W
      W_h[m][n] = W[m][n] + h
      num_classes = W.shape[1]
      num_train = X.shape[0]
      loss_h = 0.0
      for i in xrange(num_train):
        scores_h = X[i].dot(W_h)
        #correct_class_score = scores[y[i]]
        correct_class_score_h = scores_h[y[i]]
        for j in xrange(num_classes):
          if j == y[i]:
            continue
          #margin = scores[j] - correct_class_score + 1 # note delta = 1
          margin_h = scores_h[j] - correct_class_score_h + 1 # note delta = 1
          #if margin > 0:
          #  loss += margin
          if margin_h > 0:
            loss_h += margin_h

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
      #loss /= num_train
      loss_h /= num_train

  # Add regularization to the loss.
      #loss += 0.5 * reg * np.sum(W * W)
      loss_h += 0.5 * reg * np.sum(W_h * W_h)
      dW[m][n] = (loss_h - loss)/h
  '''
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)  # scores(N,C)
  scores_correct = scores[np.arange(num_train),y]  # The use of np.arange()
  scores_correct = np.reshape(scores_correct,(num_train,1))  # reshape
  margins = np.maximum(0,scores - np.tile(scores_correct,(1,num_classes)) + 1) # Broadcasting
  margins[np.arange(num_train),y] = 0  # Let the correct class value is 0
  loss += np.sum(margins)/num_train  # loss_sum
  loss += 0.5 * reg * np.sum(W * W)
  margins[margins > 0] = 1  # ???
  row_sum = np.sum(margins,1)  # ???
  margins[np.arange(num_train),y] = - row_sum  # ???
  dW += np.dot(X.T,margins)/num_train + reg * W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
