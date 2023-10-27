import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import os


def let_me_see(history):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def process_x(x):
    # TODO 1.1
    # Normalize x to be in range [0, 1]
    x_normalized = x / 255.0
    return x_normalized


def process_y(y):
    # TODO 1.2
    # Convert y to one-hot vector
    # Hint: You may consider using
    # keras.utils.to_categorical: https://keras.io/utils/#to_categorical-function
    y_onehot = keras.utils.to_categorical(y)
    return y_onehot


def BaselineModel(img_shape):
    # Create a sequential model
    model = Sequential()

    # Flatten the input image
    model.add(Flatten(input_shape=img_shape))

    # Add the output layer with 10 units (one for each class) and softmax activation
    model.add(Dense(10, activation='softmax'))

    return model


def myModel(img_shape):
    # Create a sequential model
    model = Sequential()

    # TODO 2.1
    # Build your own model with model.add() and Dense layers
    # Hint: you may consider using
    # Flatten(): https://keras.io/api/layers/reshaping_layers/flatten/
    # Dense(): https://keras.io/api/layers/core_layers/dense/
    # Dropout(): https://keras.io/api/layers/regularization_layers/dropout/
    # BatchNormalization(): https://keras.io/api/layers/normalization_layers/batch_normalization/

    # Flatten the input image
    model.add(Flatten(input_shape=img_shape))

    # Add a dense layer with 256 units and ReLU activation
    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.2))

    # Add another dense layer with 128 units and Tanh activation
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.2))

    # Add another dense layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    # Add the output layer with 10 units (one for each class) and Softmax activation
    model.add(Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)  # create a new folder "models" to save your model
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Perform data normalization
    x_train = process_x(x_train)
    x_test = process_x(x_test)
    # x_train: (N, h, w) = (60000, 28, 28)
    # x_test: (N, h, w) = (10000, 28, 28)

    # Convert the labels to one-hot encoded vectors
    y_train = process_y(y_train)
    y_test = process_y(y_test)
    # y_train: (N, 10) = (60000, 10)
    # y_test: (N, 10) = (10000, 10)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i], cmap=plt.cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"label:{np.argwhere(y_train[i] == 1)[0][0]}")
    plt.show()
    # Create the model
    baseline = BaselineModel(img_shape=(28, 28))

    # Compile the model
    baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    baseline_history = baseline.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    # Evaluate the model on the test data
    test_loss, test_accuracy = baseline.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {test_accuracy}')
    let_me_see(baseline_history)

    # Create the model
    model = myModel(img_shape=(28, 28))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create a callback that saves the model's weights
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("models", "weights.hdf5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[checkpointer])

    # load the best model
    model = keras.models.load_model(os.path.join('models', 'weights.hdf5'))
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {test_accuracy}')

    let_me_see(history)
