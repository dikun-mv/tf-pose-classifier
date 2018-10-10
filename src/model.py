from keras.models import Sequential
from keras.layers import LSTM, Dense


def get_model(nc):
    model = Sequential()

    model.add(LSTM(36, input_shape=(50, 36), return_sequences=True))
    model.add(LSTM(72, return_sequences=False, activation='relu'))
    model.add(Dense(72, activation='relu'))
    model.add(Dense(nc, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    print(get_model(2).summary())
