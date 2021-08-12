import loglinear as ll
from utils import *
import numpy as np
import random


def feats_to_vec(features):
    # Should return a numpy vector of features.

    vec = np.zeros(len(F2I))
    for feature in features:
        if feature in F2I:
            index = F2I[feature]
            vec[index] += 1
    return vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:

        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)

        y_pred = ll.predict(feats_to_vec(features), params)

        if y_pred == L2I[label]:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss

            # update the parameters according to the gradients
            # and the learning rate.

            for i in range(len(grads)):
                params[i] = params[i] - learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def pred_file(data, params):
    with open("../test.pred", "w") as file:
        for tag, feature in data:
            # getting predicted value index and find it value
            y_pred = ll.predict(feats_to_vec(feature), params)

            for bigram, index in L2I.items():
                if y_pred == index:
                    tag = bigram
                    break
            file.write(tag + "\n")


def main():
    in_dim = len(F2I)
    out_dim = len(L2I)
    num_iterations = 40
    learning_rate = 0.0015
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)
    pred_file(TEST, trained_params)


if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    main()
