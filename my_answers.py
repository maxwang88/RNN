import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append([series[i + window_size]])

    return np.array(X),np.array(y)

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = set(['!', ',', '.', ':', ';', '?', '\'', '-'])
    alphabets = set('abcdefghijklmnopqrstuvwxyz')
    unique_chars = set(text)
    for c in unique_chars:
        if (c not in alphabets) and (c not in punctuation):
            text = text.replace(c, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    i = 0
    while i + window_size < len(text):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size


    return np.array(inputs), np.array(outputs)

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model

