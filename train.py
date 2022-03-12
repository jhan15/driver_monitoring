import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.callbacks import ModelCheckpoint

from dms_utils.dms_utils import sampling_data, plot_sample_distribution, \
    load_and_preprocess_image, display4images, create_ds, plot_his_metrics, \
    ACTIONS, AUTOTUNE, SESSIONS
from net import MobileNet


def model_checkpoint_cb(file_path):
    return ModelCheckpoint(
        file_path, monitor='val_accuracy', mode='max',
        save_best_only=True, save_weights_only=True)

def train(args):
    
    rootdir = args.data_path
    modeldir = args.save_path
    trainer = args.trainer
    batch_size = args.batch_size
    epochs = args.epochs

    all_image_paths, all_image_labels = sampling_data(rootdir)
    nsamples = all_image_labels.shape[0]

    all_image_paths, all_image_labels = shuffle(
        all_image_paths, all_image_labels, random_state=42)
    
    if args.trainer == 'random':
        image_train, image_test, label_train, label_test = train_test_split(
        all_image_paths, all_image_labels, test_size=0.2, random_state=42)
        image_train, image_valid, label_train, label_valid = train_test_split(
            image_train, label_train, test_size=0.2, random_state=42)
        
        train_image_label_ds = create_ds(image_train, label_train)
        valid_image_label_ds = create_ds(image_valid, label_valid)
        test_image_label_ds = create_ds(image_test, label_test)

        ntrain = label_train.shape[0]
        nvalid = label_valid.shape[0]
        ntest = label_test.shape[0]

        train_ds = train_image_label_ds.cache().shuffle(ntrain).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        valid_ds = valid_image_label_ds.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        test_ds = test_image_label_ds.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)

        image_batch, label_batch = next(iter(train_ds))

        model = MobileNet()
        cp = model_checkpoint_cb('models/model_random.h5')

        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=valid_ds,
            callbacks=[cp])
        
        plot_his_metrics(history, 'images/loss_random.png')

        model.load_weights('models/model_random.h5')
        model.evaluate(test_ds)

    else:
        test_split = SESSIONS[5]
        test_indices = None

        for s in test_split:
            indices, = np.where(np.char.find(all_image_paths, s)>0)
            test_indices = np.concatenate((test_indices,indices), axis=0) \
                if test_indices is not None else indices

        test_image_paths = all_image_paths[test_indices]
        test_image_labels = all_image_labels[test_indices]
        train_image_paths = np.delete(all_image_paths, test_indices)
        train_image_labels = np.delete(all_image_labels, test_indices)

        image_train, image_valid, label_train, label_valid = train_test_split(
            train_image_paths, train_image_labels, test_size=0.2, random_state=42)

        ntrain = label_train.shape[0]
        nvalid = label_valid.shape[0]

        train_image_label_ds = create_ds(image_train, label_train)
        valid_image_label_ds = create_ds(image_valid, label_valid)
        test_image_label_ds = create_ds(test_image_paths, test_image_labels)

        train_ds = train_image_label_ds.cache().shuffle(ntrain).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        valid_ds = valid_image_label_ds.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        test_ds = test_image_label_ds.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)

        image_batch, label_batch = next(iter(train_ds))

        model = MobileNet()
        cp = model_checkpoint_cb('models/model_split.h5')

        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=valid_ds,
            callbacks=[cp])
        
        plot_his_metrics(history, 'images/loss_split.png')

        model.load_weights('models/model_split.h5')
        model.evaluate(test_ds)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, help='Data directory')
    p.add_argument('--save_path', type=str, help='Path/file to save trained model')
    p.add_argument('--trainer', type=str, default='random', help='Training type')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--epochs', type=int, default=20, help='Training epochs')
    args = p.parse_args()

    train(args)
