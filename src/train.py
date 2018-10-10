import numpy as np
import json

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils import shuffle
from model import get_model


def load_batch(path):
    batch = []

    with open(path, 'r') as metadata_file:
        for file_path in metadata_file:
            with open(file_path.replace('\n', ''), 'r') as data_file:
                data = json.load(data_file)

                point_seq = np.array([np.array(data[i]).flatten() for i in range(0, 100, 2) if i < len(data)])
                vect_seq = np.array([point_seq[i + 1] - point_seq[i] for i in range(len(point_seq) - 1)])

                placeholder = np.tile(0., (50, 36))
                placeholder[:len(vect_seq)] = vect_seq

                batch.append(placeholder)

    return np.array(batch)


def load_class_data(name):
    return {batch: load_batch('data/{}/{}.txt'.format(name, batch)) if name != 'none' else np.tile(0., (32, 50, 36))
            for batch in ['test', 'training', 'validation', 'real']}


def load_dataset():
    return {name: load_class_data(name) for name in ['none', 'clapping', 'waving']}


def make_vect(idx, len):
    v = np.zeros(len)
    v[idx] = 1.
    return v


if __name__ == '__main__':
    dataset = load_dataset()

    X_test = []
    Y_test = []
    X_training = []
    Y_training = []
    X_validation = []
    Y_validation = []

    for idx, (name, data) in enumerate(dataset.items()):
        print(
            'Idx: {}\n'.format(idx) +
            'Class: {}\n'.format(name) +
            'Test batch: {}\n'.format(data['test'].shape) +
            'Training batch: {}\n'.format(data['training'].shape) +
            'Validation batch: {}\n'.format(data['validation'].shape)
        )

        X_test.append(data['test'])
        Y_test.append(np.array([make_vect(idx, len(dataset)) for _ in range(data['test'].shape[0])]))
        X_training.append(data['training'])
        Y_training.append(np.array([make_vect(idx, len(dataset)) for _ in range(data['training'].shape[0])]))
        X_validation.append(data['validation'])
        Y_validation.append(np.array([make_vect(idx, len(dataset)) for _ in range(data['validation'].shape[0])]))

    X_test, Y_test = shuffle(np.concatenate(X_test), np.concatenate(Y_test))
    X_training, Y_training = shuffle(np.concatenate(X_training), np.concatenate(Y_training))
    X_validation, Y_validation = shuffle(np.concatenate(X_validation), np.concatenate(Y_validation))

    model = get_model(len(dataset))
    print(model.summary())

    model.fit(
        X_training, Y_training,
        validation_data=(X_validation, Y_validation),
        epochs=100,
        batch_size=8,
        callbacks=[
            ModelCheckpoint(filepath='model-data/model.{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True),
            EarlyStopping(patience=10),
            CSVLogger('model-data/training.log')
        ]
    )
    loss, acc = model.evaluate(
        X_test, Y_test,
        batch_size=8
    )
    print(
        'Test loss: {}\n'.format(loss) +
        'Test acc: {}\n'.format(acc)
    )

    model.save('model-data/pose-classifier.h5')
