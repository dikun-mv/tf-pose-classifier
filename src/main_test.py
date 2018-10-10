import numpy as np

from keras.models import load_model
from sklearn.utils import shuffle

from train import load_dataset, make_vect, load_batch
from model import get_model

if __name__ == '__main__':
    dataset = load_dataset()

    X_test = []
    Y_test = []
    X_real = []
    Y_real = []

    for idx, (name, data) in enumerate(dataset.items()):
        print(
            'Idx: {}\n'.format(idx) +
            'Class: {}\n'.format(name) +
            'Test batch: {}\n'.format(data['test'].shape) +
            'Real batch: {}\n'.format(data['real'].shape)
        )

        X_test.append(data['test'])
        Y_test.append(np.array([make_vect(idx, len(dataset)) for _ in range(data['test'].shape[0])]))
        X_real.append(data['real'])
        Y_real.append(np.array([make_vect(idx, len(dataset)) for _ in range(data['real'].shape[0])]))

    X_test, Y_test = shuffle(np.concatenate(X_test), np.concatenate(Y_test))
    X_real, Y_real = np.concatenate(X_real), np.concatenate(Y_real)

    # model = get_model(len(dataset))
    # model.load_weights('model-data/model.019-0.677.hdf5')
    model = load_model('model-data/pose-classifier.h5')
    print(model.summary())

    # Z_test = model.predict(X_test)
    Z_real = model.predict(X_real)

    # total = 0
    #
    # for i in range(len(Y_test)):
    #     y = np.argmax(Y_test[i])
    #     z = np.argmax(Z_test[i])
    #
    #     if y == z:
    #         total += 1
    #
    #     print('{} - {}'.format(y, z))
    #
    # print('{}/{}'.format(total, len(Y_test)))

    total = 0

    for i in range(len(Y_real)):
        y = np.argmax(Y_real[i])
        z = np.argmax(Z_real[i])

        if y == z:
            total += 1

        print('{}({}) - {}({})'.format(y, Y_real[i][y], z, Z_real[i][z]))

    print('{}/{}'.format(total, len(Y_real)))
