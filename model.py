from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization

class sudokuNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu"))
        model.add(BatchNormalization(-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(-1))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(-1))

        model.add(Dense(classes, activation="softmax"))
        return model

# model = sudokuNet.build(28, 28, 1, 10)
# model.summary()