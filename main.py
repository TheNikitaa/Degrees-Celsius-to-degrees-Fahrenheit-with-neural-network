import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    c = np.array([-40, -10, 0, 8, 15, 22, 38])  # degrees Celsius
    f = np.array([-40, 14, 32, 46, 59, 72, 100])  # degrees Fahrenheit

    model = keras.models.Sequential()  # creating a multi-layered neural network (sequential)

    # creating a layer of neurons that will be connected to other layers
    model.add(Dense(units=1, activation='linear', input_shape=(1,)))

    # compile and create initial weights
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

    history = model.fit(c, f, epochs=500, verbose=0)  # training

    plt.plot(history.history['loss'])
    plt.grid(True)
    plt.show()

    print(model.predict([100]))  # 100 Celsius degrees there should be 212 Fahrenheit degrees (we have 211)
    print(model.get_weights())  # all weights (for 1 neuron in my case)


if __name__ == '__main__':
    main()