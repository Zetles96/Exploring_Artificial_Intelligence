from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D
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


def Data_augmentation(img_shape):
    # Create a sequential model
    model = Sequential()

    # TODO 1.1
    # Build your own model with model.add() and following layers
    # Hint: you may consider using
    # RandomFlip(mode): https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_flip/
    # RandomRotation(factor): https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_rotation/
    # RandomZoom(factor): https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_zoom/
    # Add the first layer with input shape
    # Add data augmentation layers
    model.add(RandomFlip(mode='horizontal', input_shape=img_shape))
    model.add(RandomRotation(factor=0.1))
    model.add(RandomZoom(height_factor=0.1, width_factor=0.1))

    return model


def myModel(img_shape, data_aug):
    # Create a sequential model
    model = Sequential()

    # Add the data augmentation layer
    model.add(data_aug)

    # TODO 2.1
    # Build your own model with model.add() and following layers
    # Hint: you may consider using
    # Conv2D(filters, kernel_size, activation='relu'): https://keras.io/api/layers/convolution_layers/convolution2d/
    # MaxPooling2D(pool_size): https://keras.io/api/layers/pooling_layers/max_pooling2d/
    # Flatten(): https://keras.io/api/layers/reshaping_layers/flatten/
    # Dense(units, activation='relu'): https://keras.io/api/layers/core_layers/dense/
    # Dropout(rate): https://keras.io/api/layers/regularization_layers/dropout/
    # BatchNormalization(): https://keras.io/api/layers/normalization_layers/batch_normalization/
    # GlobalAveragePooling2D(): https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Flatten layer to transition from convolutional to dense layers
    model.add(Flatten())

    # Dense layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))  # Dropout for regularization
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes for classification

    return model


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)  # create a new folder "models" to save your model
    # Load the MNIST dataset and split it into training and testing sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Perform data normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert the labels to one-hot encoded vectors
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # x_train: (N, h, w, c) = (50000, 32, 32, 3)
    # y_train: (N, 10) = (50000, 10)
    # x_test: (N, h, w, c) = (10000, 32, 32, 3)
    # y_test: (N, 10) = (10000, 10)
    # There are 10 classes in CIFAR10 dataset
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    data_aug = Data_augmentation(img_shape=(32, 32, 3))

    # visualize the picture in x_train
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i])
        plt.xticks([])
        plt.yticks([])
        plt.title(f"class{np.argwhere(y_train[i] == 1)[0][0]}: {classes[np.argwhere(y_train[i] == 1)[0][0]]}")

    # Create the model
    model = myModel(img_shape=(32, 32, 3), data_aug=data_aug)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create a callback that saves the model's weights
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("models", "weights.hdf5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[checkpointer])

    # load the best model
    model = keras.models.load_model(os.path.join('models', 'weights.hdf5'))
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {test_accuracy}')
    let_me_see(history)
