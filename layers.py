from builtins import range
import numpy as np

def affine_forward(x, w, b):
    out = None

    N = x.shape[0]
    M = np.prod(x.shape[1:])
    x_reshape = x.reshape(N,M)
    out = x_reshape.dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
  
    dx = dout.dot(w.T).reshape(x.shape)
    db = np.sum(dout, axis = 0)
    dw = x.reshape(x.shape[0],-1).T.dot(dout)

    return dx, dw, db


def relu_forward(x):
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    out = np.maximum(0,x)
    out[out>0] = 1
    dx = out*dout
    return dx

def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def softmax_loss(x, y):

    loss, dx = None, None

    num_train = x.shape[0]
    prob_score = np.exp(x - np.max(x, axis = 1, keepdims=True))/np.sum(np.exp(x- np.max(x, axis = 1, keepdims=True)), axis = 1, keepdims= True)
    
    loss = np.sum(-np.log(prob_score[np.arange(num_train),y])) / num_train
    
    dx = prob_score.copy()
    dx[np.arange(num_train), y] -= 1
    dx /= num_train

    return loss, dx