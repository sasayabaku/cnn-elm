# coding: utf-8

import hpelm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from keras.utils import np_utils

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Flatten
from keras.utils import plot_model
from keras.callbacks import EarlyStopping

from keras.utils import np_utils

import numpy as np
import plotly
import plotly.figure_factory as ff


NUM_CLASS = 10
ELM_HIDDEN_NEURONS = 1500


def load_mnist_2d():
    digits = load_digits()

    data = digits.data
    target = digits.target

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.33)

    data_train_2d = np.reshape(data_train, (data_train.shape[0], 8, 8, 1))
    data_test_2d = np.reshape(data_test, (data_test.shape[0], 8, 8, 1))

    target_train_oh = np_utils.to_categorical(target_train, NUM_CLASS)
    target_test_oh = np_utils.to_categorical(target_test, NUM_CLASS)

    return data_train_2d, data_test_2d, target_train, target_test


def cnn_generate(data_train_2d, target_train_oh):

    """
    CNNモデルの構築
    """

    cnn_model = Sequential()
    cnn_model.add(Conv2D(20, 2, input_shape=(data_train_2d.shape[1], data_train_2d.shape[2], 1)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPool2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(10, 2))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPool2D(pool_size=(2, 2)))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(200))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dense(target_train_oh.shape[1], activation='softmax'))

    adam = Adam(lr=1e-3)

    cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    history = cnn_model.fit(data_train_2d, target_train_oh, batch_size=20, epochs=100, verbose=1, validation_split=0.2)

    return cnn_model


# CNNの中間層の出力を取得するモデルの構築

def hidden_layer_generate(cnn_model):

    """
    CNNの中間層の出力を取得するモデルの構築
    :param cnn_model: CNNモデル
    :return:
    """

    layer_name = 'flatten_1'
    hidden_layer_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(layer_name).output)

    cnn_train_result = hidden_layer_model.predict(data_train_2d)

    return hidden_layer_model, cnn_train_result


def elm_model_generate(train_data):

    """
    ELMモデルの構築
    """

    target_train_oh = np_utils.to_categorical(target_test, NUM_CLASS)

    elm_model = hpelm.elm.ELM(cnn_train_result.shape[1], NUM_CLASS)
    elm_model.add_neurons(ELM_HIDDEN_NEURONS, func='sigm')

    elm_model.train(train_data, target_train_oh, 'c')

    # elm_model.confusion(target_train_oh, elm_model.predict(cnn_train_result))

    return elm_model


def cnn_elm_evaluation(cnn_part, elm_part, data_test, target_test, file_name="plot_result.html"):

    """
    CNN-ELMモデルの評価

    :param cnn_part: CNN Model
    :param elm_part: ELM Model
    :return: Result Score
    """

    target_test_oh = np_utils.to_categorical(target_test, NUM_CLASS)

    cnn_result = cnn_part.predict(data_test)
    elm_result = elm_part.predict(cnn_result)

    elm_result_class = np.array([np.argmax(r) for r in elm_result])

    confusion = elm_model.confusion(target_test_oh, elm_result_class)

    # Convert one-hot to class
    trace = plotly.graph_objs.Heatmap(z=confusion, colorscale=[[0, '#E6E6E6'], [1, '#04B486']])
    plot_data = [trace]
    plotly.offline.plot(plot_data, filename=file_name)

    # Confusion Matrix Plot
    trace = ff.create_annotated_heatmap(z=confusion, colorscale=[[0, '#E6E6E6'], [1, '#04B486']])
    for i in range(len(trace.layout.annotations)):
        trace.layout.annotations[i].font.size = 20
    plotly.offline.plot(trace, filename=file_name)

    precision, recall, fscore, support = precision_recall_fscore_support(target_test, elm_result_class)

    return precision, recall, fscore, support


if __name__ == '__main__':

    data_train_2d, data_test_2d, target_train, target_test = load_mnist_2d()

    cnn_model = cnn_generate(data_train_2d, target_train)

    hidden_layer_model, cnn_train_result = hidden_layer_generate(cnn_model)

    elm_model = elm_model_generate(cnn_train_result)

    precision, recall, fscore, support = cnn_elm_evaluation(hidden_layer_model, elm_model, data_test_2d, target_test)