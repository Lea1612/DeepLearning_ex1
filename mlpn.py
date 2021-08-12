import numpy as np

from loglinear import softmax


def classifier_output(x, params):
    global dot_b
    out = x.copy()
    for W, b in zip(params[::2], params[1::2]):
        dot_b = np.dot(out, W) + b
        out = np.tanh(dot_b)

    probs = softmax(dot_b)
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """

    gradients = []

    y_pred, feed_forwards = perceptron_feed_forwards(x, params)
    loss = -np.log(y_pred[y])

    # We want y_tag to be a one hot vector
    y_one_hot = np.zeros(len(y_pred))
    y_one_hot[y] = 1

    feed_forwards.reverse()
    params.reverse()

    dz = 0
    subs = y_pred - y_one_hot

    for i in range(len(feed_forwards) + 1):
        W = params[2 * i + 1]

        if i == len(feed_forwards):
            z_x = x
        else:
            z_x = feed_forwards[i][1]
            dz = 1 - np.tanh(feed_forwards[i][0]) ** 2

        gW_i = np.outer(z_x, subs)
        gb_i = subs
        gradients += [gb_i, gW_i]

        if i != len(feed_forwards):
            subs = np.dot(W, subs) * dz

    params.reverse()
    return loss, gradients[::-1]


def perceptron_feed_forwards(x, params):
    feed_forwards = []
    x_copy = x.copy()
    layers = int((len(params) + 1) / 2)

    for i in range(0, layers - 1, 1):
        W, b = params[2 * i], params[2 * i + 1]
        dot_b = np.dot(x_copy, W) + b
        out = np.tanh(dot_b)
        feed_forwards.append((dot_b, out))
        x_copy = feed_forwards[-1][1]

    U, b_tag = params[-2], params[-1]

    return softmax(np.dot(x_copy, U) + b_tag), feed_forwards


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []

    for i in range(len(dims) - 1):
        epsilon = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
        W = np.random.uniform(-epsilon, epsilon, (dims[i], dims[i + 1]))
        b = np.random.uniform(-epsilon, epsilon, dims[i + 1])
        params.append(W)
        params.append(b)
    return params


if __name__ == '__main__':
    from grad_check import gradient_check

    params = create_classifier([3, 5, 4])
    [W, b, U, b_tag] = params


    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W, U, b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]


    for _ in range(10):
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
