import numpy as np

from loglinear import softmax


def classifier_output(x, params):
    [W, b, U, b_tag] = params
    layer = np.dot(x, W) + b
    probs = softmax(np.tanh(layer).dot(U) + b_tag)
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    [W, b, U, b_tag] = params

     # Compute y 
    y_tag = classifier_output(x, params)

    # Compute loss
    loss = -np.log(y_tag[y])

    # We want y_tag to be a one hot vector
    y_one_hot = np.zeros(len(y_tag))
    y_one_hot[y] = 1

    activation_neuron = np.tanh(np.dot(x, W) + b)

    gb_tag = y_tag - y_one_hot
    gU = np.outer(activation_neuron, gb_tag)

    dl_dz = np.dot(U, gb_tag)
    dz_dh = 1.0 - activation_neuron ** 2
    dl_dh = dl_dz * dz_dh

    gW = np.outer(x, dl_dh)
    gb = dl_dh

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    epsilon = np.sqrt(6.0 / (in_dim + hid_dim))
    W = np.random.uniform(-epsilon, epsilon, (in_dim, hid_dim))
    b = np.random.uniform(-epsilon, epsilon, hid_dim)

    epsilon = np.sqrt(6.0 / (out_dim + hid_dim))
    U = np.random.uniform(-epsilon, epsilon, (hid_dim, out_dim))
    b_tag = np.random.uniform(-epsilon, epsilon, out_dim)

    params = [W, b, U, b_tag]
    return params


if __name__ == '__main__':
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3, 5, 4)


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
        global b, U, W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]


    in_dim = 3
    hid_dim = 5
    out_dim = 4
    for _ in range(10):
        W = np.random.randn(in_dim, hid_dim)
        b = np.random.randn(hid_dim)
        U = np.random.randn(hid_dim, out_dim)
        b_tag = np.random.randn(out_dim)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
