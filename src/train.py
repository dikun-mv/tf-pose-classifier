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

                batch.append(np.array(data))

    return np.array(batch)


def load_class_data(name):
    return {batch: load_batch('dataset/{}/data.txt'.format(name)) for batch in ['training', 'test']}


def load_dataset():
    return {name: load_class_data(name) for name in ['stand', 'hand-1', 'hand-2']}


def make_vect(idx, len):
    v = np.zeros(len)
    v[idx] = 1.
    return v


if __name__ == '__main__':
    dataset = load_dataset()

    X_training = []
    Y_training = []

    for idx, (name, data) in enumerate(dataset.items()):
        print(
            'Idx: {}\n'.format(idx) +
            'Class: {}\n'.format(name) +
            'Training batch: {}\n'.format(data['training'].shape)
        )

        X_training.append(data['training'])
        Y_training.append(np.array([make_vect(idx, len(dataset)) for _ in range(data['training'].shape[0])]))

    X_training.append(np.tile(.0, (10, 50, 36)))
    Y_training.append(np.tile(.0, (10, 3)))

    X_training, Y_training = shuffle(np.concatenate(X_training), np.concatenate(Y_training))

    model = get_model(len(dataset))
    print(model.summary())

    model.fit(
        X_training, Y_training,
        validation_split=0.3,
        epochs=100,
        batch_size=1,
        callbacks=[
            ModelCheckpoint(filepath='model-data/model.{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1,
                            save_best_only=True),
            EarlyStopping(patience=10),
            CSVLogger('model-data/training.log')
        ]
    )
