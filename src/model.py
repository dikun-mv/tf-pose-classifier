from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout


def get_model(nc):
    model = Sequential()

    model.add(Conv1D(64, 5, activation='relu', input_shape=(50, 36)))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(nc, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    print(get_model(3).summary())
