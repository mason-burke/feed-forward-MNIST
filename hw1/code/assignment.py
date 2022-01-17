from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
from preprocess import get_next_batch

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # hyperparameter initialization
        self.input_size = 784
        self.num_classes = 10
        self.batch_size = 100
        self.learning_rate = 0.5

        # weights and biases initialization
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros(self.num_classes)

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # compute logits
        logits = np.matmul(inputs, self.W) + self.b
        # exponentiate
        arr = np.exp(logits)
        # apply softmax
        return np.divide(arr, np.sum(arr, axis=1, keepdims = True))

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step).
        NOTE: This function is not actually used for gradient descent
        in this assignment, but is a sanity check to make sure model
        is learning.
        :param probabilities: matrix that contains the probabilities
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # grab probabilities of correct answers
        p_correct = probabilities[np.arange(self.batch_size), labels]
        # apply negative logarithm
        loss = - np.log(p_correct)
        # compute scalar loss
        return np.sum(loss) / self.batch_size

    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. The learning
        algorithm for updating weights and biases mentioned in
        class works for one image, but because we are looking at
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # create one-hot vector
        one_hot = np.zeros((self.batch_size, self.num_classes))
        rows = np.arange(self.batch_size)
        one_hot[rows, labels] = 1
        # compute gradients of weights and biases
        gradW = np.matmul(inputs.T, (probabilities -  one_hot)) / self.batch_size
        gradB = np.sum((probabilities - one_hot), axis = 0) / self.batch_size
        return gradW, gradB

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # get indices of greatest probability
        indices = np.argmax(probabilities, axis = 1)
        # if an index matches the corresponding label, we have a correct prediction
        num_correct = np.sum(indices == labels)
        return num_correct / labels.size

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # apply gradient descent to weights and biases
        self.W -= self.learning_rate * gradW
        self.b -= self.learning_rate * gradB

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    size = model.batch_size
    # iterate over batches
    for i in range(int(np.shape(train_inputs)[0] / size)):
        # get a batch
        (i_batch, l_batch) = get_next_batch(train_inputs, train_labels, size, i)
        # compute softmax probabilities
        probs = model.call(i_batch)
        # compute gradients
        (gradW, gradB) = model.back_propagation(i_batch, probs, l_batch)
        # apply gradient descend to model weights and biases
        model.gradient_descent(gradW, gradB)

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # compute softmax
    probs = model.call(test_inputs)
    # return accuracy
    return model.accuracy(probs, test_labels)

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far,
    and how do you know that?
    Optionally, add your answers to README!
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''
    # load training and testing inputs and labels
    (train_inputs, train_labels) = get_data('MNIST-data/train-images.gz', 'MNIST-data/train-labels.gz', 60000)
    (test_inputs, test_labels) = get_data('MNIST-data/test-images.gz', 'MNIST-data/test-labels.gz', 10000)
    # create model
    model = Model()
    # train the model
    train(model, train_inputs, train_labels)
    # print out accuracy of model on testing data
    print("The accuracy of the model is", test(model, test_inputs, test_labels))

if __name__ == '__main__':
    main()
