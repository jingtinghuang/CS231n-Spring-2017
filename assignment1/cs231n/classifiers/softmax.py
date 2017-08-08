import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        softmax = np.exp(scores) / np.sum(np.exp(scores),keepdims = True)
        for j in xrange(num_classes):
            if j == y[i]:
                loss += -(np.log(softmax[j]))
                dW[:,j] += (softmax[j]-1) * X[i].T
            else:
                dW[:,j] += (softmax[j]) * X[i].T
                
  loss = loss / num_train          
  loss += reg*np.sum(W*W)
  dW = dW / num_train
  dW = dW + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  stab = np.amax(scores, axis = 1)
  scores -= stab.reshape(num_train,1)
  softmax = np.exp(scores) / np.sum(np.exp(scores),axis=1)[:,None]
  loss = -np.sum(np.log(softmax[range(num_train),y]))
  softmax[range(num_train), y] -=1
  dW = X.T.dot(softmax)
  loss = loss / num_train          
  loss += reg*np.sum(W*W)
  dW = dW / num_train
  dW = dW + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

