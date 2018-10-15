import numpy as np

from keras.models import load_model
from sklearn.utils import shuffle

from train import load_dataset, make_vect, load_batch
from model import get_model

if __name__ == '__main__':
    dataset = load_dataset()

    X_test = []
    Y_test = []

    for idx, (name, data) in enumerate(dataset.items()):
        print(
            'Idx: {}\n'.format(idx) +
            'Class: {}\n'.format(name) +
            'Test batch: {}\n'.format(data['test'].shape)
        )

        X_test.append(data['test'])
        Y_test.append(np.array([make_vect(idx, len(dataset)) for _ in range(data['test'].shape[0])]))

    X_test, Y_test = shuffle(np.concatenate(X_test), np.concatenate(Y_test))

    model = load_model('models/posec-3cn-conv-3k.h5')
    print(model.summary())

    Z_test = model.predict(X_test)

    total = 0

    for i in range(len(Y_test)):
        y = np.argmax(Y_test[i])
        z = np.argmax(Z_test[i])

        if y == z:
            total += 1

        print('{}({}) - {}({})'.format(y, Y_test[i][y], z, Z_test[i][z]))

    print('{}/{}'.format(total, len(Y_test)))
