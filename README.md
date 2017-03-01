# Char-level-CNN-for-Chinese-Text-Classification-in-Keras

Train character-level convolutional network for text classification. Based on "Character-level Convolutional Networks for Text Classification" by Xiang Zhang, [link here](https://arxiv.org/abs/1509.01626).

## About this model

Attempt to build a machine can classify term through the text information found about that term. Using character-level as input minimizing the need of preprocessing, model could apply on different languages like Chinese.

## Dependencies

* Python3
* [Keras](http://keras.io/)
* [TensorFlow](https://www.tensorflow.org). Model ran and test in version 0.12

## Runtime model

* Runtime model pre-calculated all text. Result data build in Mongodb. Require [PyMongo](https://api.mongodb.com/python/current/) and a Mongodb server to run.
