from keras.models import Sequential
from keras.layers import LSTM, Dense


def get_model(nc):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(50, 36)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(nc, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    print(get_model(2).summary())
