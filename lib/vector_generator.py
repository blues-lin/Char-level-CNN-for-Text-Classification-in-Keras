""" Generate vector. """
import os.path
import random
import numpy as np
from lib import Vectorizer

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CHAR_PATH = os.path.join(BASE_PATH, 'data/charTable.txt')
LABEL_PATH = os.path.join(BASE_PATH, 'data/label.txt')


class VectGenerator:
    """
    Generate vector from list of tuples.
    tuple[0] is text and tuple[1] is label.
    """

    def __init__(
            self, dataList, textLength,
            forPredict=None,
            training=0.7,
            testing=0.15,
            validation=0.15,
            backend="th"):

        self._vectorizer = Vectorizer(CHAR_PATH, LABEL_PATH)
        random.shuffle(dataList)
        self._textLength = textLength
        self._backend = backend
        if forPredict is None:
            dataLen = len(dataList)
            testLen = int(dataLen * testing)
            validLen = int(dataLen * validation)
            self._testList = dataList[0:testLen]
            self._validList = dataList[testLen:(testLen + validLen)]
            self._trainingList = dataList[(testLen + validLen):]
        else:
            self._predictList = dataList[:]
        if self._backend == "th":
            self._inputShape = (
                1, self._vectorizer.getCharSpace(), self._textLength)
        else:
            self._inputShape = (
                self._vectorizer.getCharSpace(), self._textLength, 1)

    def getVectorizer(self):
        return self._vectorizer

    def _dataGenerator(self, dataSet, batch_size):
        x_batch = []
        y_batch = []
        while 1:
            for data in dataSet:
                x_batch.append(
                    self._vectorizer.vectorize(data[0], self._textLength))
                y_batch.append(
                    self._vectorizer.vectorizeLabel(data[1]))
                if (len(x_batch) == batch_size):
                    X_vect_batch = np.array(x_batch)
                    y_vect_batch = np.array(y_batch)
                    if self._backend == "th":
                        X_vect_batch = X_vect_batch.reshape(
                                            X_vect_batch.shape[0], 1,
                                            self._vectorizer.getCharSpace(),
                                            self._textLength)
                    else:
                        X_vect_batch = X_vect_batch.reshape(
                                            X_vect_batch.shape[0],
                                            self._vectorizer.getCharSpace(),
                                            self._textLength, 1)

                    yield (X_vect_batch, y_vect_batch)
                    x_batch = []
                    y_batch = []

    def trainGenerator(self, batch_size):
        for data in self._dataGenerator(self._trainingList, batch_size):
            yield data

    def testGenerator(self, batch_size):
        for data in self._dataGenerator(self._testList, batch_size):
            yield data

    def validGenerator(self, batch_size):
        for data in self._dataGenerator(self._validList, batch_size):
            yield data

    def predictGenerator(self, batch_size):
        for data in self._dataGenerator(self._predictList, batch_size):
            yield data[0]

    def nb_train_samples(self):
        if self._trainingList is not None:
            return len(self._trainingList)

    def nb_test_samples(self):
        if self._testList is not None:
            return len(self._testList)

    def nb_val_samples(self):
        if self._validList is not None:
            return len(self._validList)

    def nb_predict_samples(self):
        if self._predictList is not None:
            return len(self._predictList)

    def nb_char_space(self):
        return self._vectorizer.getCharSpace()

    def nb_classes(self):
        return self._vectorizer.getClassSpace()

    def input_shape(self):
        return self._inputShape
