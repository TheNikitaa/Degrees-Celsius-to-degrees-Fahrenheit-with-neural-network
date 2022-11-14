import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.datasets import mnist  # training sample
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_result(model, x_test, index_of_picture=0):
    # Introducing an image in 3d tensor, axis=0 -> create new axis
    x = np.expand_dims(x_test[index_of_picture], axis=0)

    res = model.predict(x)
    all_predict = model.predict(x_test)

    print(f'Number of correctly recognized images: {all_predict.shape[0]}')
    print(f'Recognized digit: {np.argmax(res)}')

    plt.imshow(x_test[index_of_picture], cmap=plt.cm.binary)
    plt.show()


def check_not_correct_result(model, x_test, y_test, count_of_picture=1):
    predict = model.predict(x_test)
    predict_argmax = np.argmax(predict, axis=1)  # 0 or 1
    mask = predict_argmax == y_test

    x_false = x_test[~mask]
    y_false = predict_argmax[~mask]

    print(f'Number of incorrectly recognized images: {x_false.shape[0]}')

    for i in range(count_of_picture):
        print('Recognized digit:', str(y_false[i]))
        plt.imshow(x_false[i], cmap=plt.cm.binary)
        plt.show()


def main():
    # x_train, x_test -> images; y_train, y_test -> values
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Standardization input values { x -> (0, 1) }
    x_train = x_train / 255
    x_test = x_test / 255

    y_train_categorical = keras.utils.to_categorical(y_train, 10)  # converting input values into vectors by category
    y_test_categorical = keras.utils.to_categorical(y_test, 10)  # converting input values into vectors by category

    # plt.figure(figsize=(10, 5))
    #  for i in range(25):
    #    plt.subplot(5, 5, i + 1)       # draw training sample
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.imshow(x_train[i], cmap=plt.cm.binary)  # displaying the image
    # plt.show()

    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    print(model.summary())  # output of the NN structure to console

    # Compile NN
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Starting train (80% - training sample, 20% - validation sample)
    model.fit(x_train, y_train_categorical, batch_size=32, epochs=5, validation_split=0.2)

    # Checking the test sample
    model.evaluate(x_test, y_test_categorical)  # loss -> quality criteria

    check_result(model, x_test, 19)
    check_not_correct_result(model, x_test, y_test, 5)


if __name__ == '__main__':
    main()
