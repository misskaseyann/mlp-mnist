import numpy as np


class Perceptron(object):
    """
    Perceptron built for the MNIST dataset.
    """
    def __init__(self):
        """
        Initialize MLP for MNIST dataset.
        Takes in 784 data points for the number of pixels.
        Hidden nodes is set at 20.
        Output is ten for values 0-9.
        """
        self.input_layer_size = 784
        self.hidden_layer_size = 20
        self.output_layer_size = 10
        self.wh, self.wo = self.init_weights()
        self.bh, self.bo = self.init_bias()

    def init_weights(self):
        """
        Initialize weights.
        Modified from the original algorithm from
        Machine Learning: An Algorithmic Perspective by Stephen Marsland.

        :return: initialized weight matrixes: hidden and outer.
        """
        wh = (np.random.randn(self.input_layer_size, self.hidden_layer_size) *
              2 / np.sqrt(self.input_layer_size))
        wo = (np.random.randn(self.hidden_layer_size, self.output_layer_size) *
              2 / np.sqrt(self.hidden_layer_size))
        return wh, wo

    def init_bias(self):
        """
        Initialize the bias node and insert into the network.

        :return: bias for hidden and bias for outer, initialized at -1.
        """
        bh = np.full((1, self.hidden_layer_size), -1)
        bo = np.full((1, self.output_layer_size), -1)
        return bh, bo

    def forward_prop(self, x, y):
        """
        Forward feed of the neural network.

        :param x: input vector.
        :param y: target vector.
        :return: network predictions/guess for input given.
        """
        #  Move through hidden layer.
        zh = np.dot(x, self.wh) + self.bh
        #  Activate hidden layer.
        h = self.sigmoid(zh)
        #  Move through outer layer.
        zo = np.dot(h, self.wo) + self.bo
        #  Activate outer layer.
        yhat = self.sigmoid(zo)
        return yhat, h

    def sigmoid(self, z):
        """
        Sigmoid activation function.

        :param z: weighted input matrix.
        :return: 0 to 1 activation matrix of weighted input.
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """
        Sigmoid prime backprop function.

        :param z: 0 to 1 activated input matrix.
        :return: derivative of the sigmoid matrix.
        """
        return z * (1 - z)

    def network_error(self, yhat, y):
        """
        Detect total network error.

        :param yhat: predictions made by the output of the network.
        :param y: target for the output of the network.
        :return: error of the network.
        """
        error = np.sum((yhat - y) ** 2) / 2.0
        return error

    def back_prop(self, yhat, y, x, h, rate):
        """
        Backward feed to detect where errors are.
        Modified algorithm from class notes and the book:
        Machine Learning: An Algorithmic Perspective by Stephen Marsland.

        :param yhat: output predictions of the network.
        :param y: target for the output of the network.
        :param x: data used for input.
        :param h: hidden activation layer.
        :param rate: learning rate (eta) multiple applied to weights.
        """
        eo = (yhat - y) * self.sigmoid_prime(yhat)
        eh = np.dot(eo, np.transpose(self.wo)) * self.sigmoid_prime(h)

        updatewo = (np.dot(np.transpose(h),eo))
        updatewh = (np.dot(np.transpose(x),eh))

        self.wh -= updatewh * rate
        self.wo -= updatewo * rate

    def train(self, data, labels, rate, epochs):
        """
        Train the neural network.

        :param data: all inputs to train the network on.
        :param labels: all labels to the inputs.
        :param rate: learning rate for the weights.
        :param epochs: number of epochs to train.
        """
        #  Number of data.
        datasize = data.shape[0]
        #  Prepare matrix of guesses for confusion matrix (whole training).
        guesses = np.zeros((datasize, 10))
        #  Start looping through epochs
        for i in range(epochs):
            numtimes = 0
            #  Prepare matrix of guesses for confusion matrix (current epoch).
            epochguesses = np.zeros((datasize, 10))
            #  Start sequential training.
            for x, y in zip(data, labels):
                x = np.array([x])
                y = np.array([y])
                yhat, h = self.forward_prop(x, y)
                #  Add guesses and targets for confusion matrix.
                guesses[numtimes, np.argmax(yhat)] = 1
                epochguesses[numtimes, np.argmax(yhat)] = 1
                #  Correct errors.
                self.back_prop(yhat, y, x, h, rate)
                numtimes += 1
            print("Epoch %d" % i)
            self.confusion_matrix(labels, epochguesses)
        print("------DONE TRAINING------")
        self.confusion_matrix(labels, guesses)
        self.save_model()

    def predict(self, data, labels):
        """
        Predict using current loaded model.

        :param data: all inputs to train the network on.
        :param labels: all labels to the inputs.
        """
        print("------PREDICT DATA------")
        #  Number of data.
        datasize = data.shape[0]
        #  Prepare matrix of guesses for confusion matrix.
        guesses = np.zeros((datasize, 10))
        numtimes = 0
        #  Start reading through each data point and predict.
        for x, y in zip(data, labels):
            x = np.array([x])
            y = np.array([y])
            yhat, h = self.forward_prop(x, y)
            print("Predicted a: " + str(np.argmax(yhat)))
            print("Supposed to be a: " + str(np.argmax(y)))
            guesses[numtimes, np.argmax(yhat)] = 1
            numtimes += 1
        print("------DONE RUNNING------")
        self.confusion_matrix(labels, guesses)

    def save_model(self):
        """
        Save weights for current model to a .csv file for loading later.
        They are saved to the local folder. Each layer of weights is saved
        individually.
        """
        np.savetxt("weighth.csv", self.wh, delimiter=",")
        np.savetxt("weighto.csv", self.wo, delimiter=",")

    def load_model(self, weighthpath, weightopath):
        """
        Load weights for a trained model from a .csv file.
        Perceptron is then initialized with these values.

        :param weighthpath: path to the hidden weights.
        :param weightopath: path to the outer weights.
        """
        self.wh = np.genfromtxt(weighthpath, delimiter=",")
        self.wo = np.genfromtxt(weightopath, delimiter=",")
        self.bh, self.bo = self.init_bias()

    def confusion_matrix(self, labels, output):
        """
        Create confusion matrix based on the predictions and
        actual targets of the data being fed through the
        neural network.

        :param labels: target data from the inputs.
        :param output: guesses made by the network.
        """
        #  10x10 confusion matrix.
        cm = np.zeros((10, 10))
        #  Populate matrix with x values the guesses and y values the targets.
        for guess, target in zip(output, labels):
            column = np.argmax(guess)
            row = np.argmax(target)
            cm[row, column] = cm[row, column] + 1
        print(cm)
        #  Calculate accuracy metrics for each value.
        for i in range(10):
            #  True negative.
            tn = 0
            #  False positive.
            fp = 0
            #  False negative.
            fn = 0
            #  True positive.
            tp = cm[i][i]
            for x in range(10):
                if x != i:
                    fp += cm[x][i]
                for y in range(10):
                    if x == i and y != i:
                        fn += cm[i][y]
                    if x != i and y != i:
                        tn += cm[x][y]
            print("Accuracy for: %d" % i)
            accuracy = (tp + fp) / (tp + fp + tn + fn)
            print(str(accuracy))
            print("Sensitivity for: %d" % i)
            sensitivity = tp / (tp + fn)
            print(str(sensitivity))
            print("Specificity for %d" % i)
            specificity = tn / (tn + fp)
            print(str(specificity))
            print("Precision for %d" % i)
            precision = tp / (tp + fp)
            print(str(precision))
            print("Recall for %d" % i)
            recall = tp / (tp + fn)
            print(str(recall))
            print("F1 Measure for %d" % i)
            f1 = tp / (tp + (fn + fp) / 2)
            print(str(f1))
            print("------------------------------")
        #  Percentage accuracy.
        perc = np.trace(cm) / np.sum(cm) * 100
        print("---TOTAL ACCURACY---")
        print(str(perc))