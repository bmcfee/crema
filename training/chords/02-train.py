#!/usr/bin/env python
'''CREMA structured chord model'''

import argparse
import os
import sys
from glob import glob
import six
import pickle

import pandas as pd
import keras as K

from sklearn.model_selection import ShuffleSplit

import pescador
import librosa
import crema.utils
import crema.layers
from jams.util import smkdirs

OUTPUT_PATH = 'resources'


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--max_samples', dest='max_samples', type=int,
                        default=128,
                        help='Maximum number of samples to draw per streamer')

    parser.add_argument('--patch-duration', dest='duration', type=float,
                        default=8.0,
                        help='Duration (in seconds) of training patches')

    parser.add_argument('--seed', dest='seed', type=int,
                        default='20170412',
                        help='Seed for the random number generator')

    parser.add_argument('--train-streamers', dest='train_streamers', type=int,
                        default=1024,
                        help='Number of active training streamers')

    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=32,
                        help='Size of training batches')

    parser.add_argument('--rate', dest='rate', type=int,
                        default=8,
                        help='Rate of pescador stream deactivation')

    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=100,
                        help='Maximum number of epochs to train for')

    parser.add_argument('--epoch-size', dest='epoch_size', type=int,
                        default=512,
                        help='Number of batches per epoch')

    parser.add_argument('--validation-size', dest='validation_size', type=int,
                        default=1024,
                        help='Number of batches per validation')

    parser.add_argument('--early-stopping', dest='early_stopping', type=int,
                        default=20,
                        help='# epochs without improvement to stop')

    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int,
                        default=10,
                        help='# epochs before reducing learning rate')

    parser.add_argument(dest='working', type=str,
                        help='Path to working directory')

    return parser.parse_args(args)


def make_sampler(max_samples, duration, pump, seed):

    n_frames = librosa.time_to_frames(duration,
                                      sr=pump['cqt'].sr,
                                      hop_length=pump['cqt'].hop_length)

    return pump.sampler(max_samples, n_frames, random_state=seed)


def data_sampler(fname, sampler):
    '''Generate samples from a specified h5 file'''
    for datum in sampler(crema.utils.load_h5(fname)):
        yield datum


def data_generator(working, tracks, sampler, k, augment=True, rate=8,
                   **kwargs):
    '''Generate a data stream from a collection of tracks and a sampler'''

    seeds = []

    for track in tracks:
        fname = os.path.join(working,
                             os.path.extsep.join([track, 'h5']))
        seeds.append(pescador.Streamer(data_sampler, fname, sampler))

        if augment:
            for fname in sorted(glob(os.path.join(working,
                                                  '{}.*.h5'.format(track)))):
                seeds.append(pescador.Streamer(data_sampler, fname, sampler))

    # Send it all to a mux
    return pescador.StochasticMux(seeds, k, rate, **kwargs)


def construct_model(pump):

    model_inputs = 'cqt/mag'

    # Build the input layer
    x = pump.layers()[model_inputs]

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x)

    # First convolutional filter: a single 5x5
    conv1 = K.layers.Convolution2D(1, (5, 5),
                                   padding='same',
                                   activation='relu',
                                   data_format='channels_last')(x_bn)

    # Second convolutional filter: a bank of full-height filters
    conv2 = K.layers.Convolution2D(12*6, (1, int(conv1.shape[2])),
                                   padding='valid', activation='relu',
                                   data_format='channels_last')(conv1)

    # Squeeze out the frequency dimension
    squeeze = crema.layers.SqueezeLayer(axis=2)(conv2)

    # BRNN layer
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(squeeze)

    rnn = K.layers.Bidirectional(K.layers.GRU(128,
                                              return_sequences=True))(rnn1)

    # 1: pitch class predictor
    pc = K.layers.Dense(pump.fields['chord_struct/pitch'].shape[1],
                        activation='sigmoid')

    pc_p = K.layers.TimeDistributed(pc, name='chord_pitch')(rnn)

    # 2: root predictor
    root = K.layers.Dense(13, activation='softmax')
    root_p = K.layers.TimeDistributed(root, name='chord_root')(rnn)

    # 3: bass predictor
    bass = K.layers.Dense(13, activation='softmax')
    bass_p = K.layers.TimeDistributed(bass, name='chord_bass')(rnn)

    # 4: merge layer
    codec = K.layers.concatenate([rnn, pc_p, root_p, bass_p])

    p0 = K.layers.Dense(len(pump['chord_tag'].vocabulary()),
                        activation='softmax',
                        bias_regularizer=K.regularizers.l2())

    tag = K.layers.TimeDistributed(p0, name='chord_tag')(codec)

    model = K.models.Model(x, [tag, pc_p, root_p, bass_p])
    model_outputs = ['chord_tag/chord',
                     'chord_struct/pitch',
                     'chord_struct/root',
                     'chord_struct/bass']

    return model, [model_inputs], model_outputs


def train(working, max_samples, duration, rate,
          batch_size, epochs, epoch_size, validation_size,
          early_stopping, reduce_lr, seed):
    '''
    Parameters
    ----------
    working : str
        directory that contains the experiment data (h5)

    max_samples : int
        Maximum number of samples per streamer

    duration : float
        Duration of training patches

    batch_size : int
        Size of batches

    rate : int
        Poisson rate for pescador

    epochs : int
        Maximum number of epoch

    epoch_size : int
        Number of batches per epoch

    validation_size : int
        Number of validation batches

    early_stopping : int
        Number of epochs before early stopping

    reduce_lr : int
        Number of epochs before reducing learning rate

    seed : int
        Random seed
    '''

    # Load the pump
    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb') as fd:
        pump = pickle.load(fd)

    # Build the sampler
    sampler = make_sampler(max_samples, duration, pump, seed)

    # Build the model
    model, inputs, outputs = construct_model(pump)

    # Load the training data
    idx_train_ = pd.read_json('index_train.json')

    # Split the training data into train and validation
    splitter_tv = ShuffleSplit(n_splits=1, test_size=0.25,
                               random_state=seed)
    train, val = next(splitter_tv.split(idx_train_))

    idx_train = idx_train_.iloc[train]
    idx_val = idx_train_.iloc[val]

    gen_train = data_generator(working,
                               idx_train['id'].values, sampler, epoch_size,
                               augment=True,
                               rate=rate,
                               mode='with_replacement',
                               random_state=seed)

    gen_train = pescador.maps.keras_tuples(pescador.maps.buffer_stream(gen_train(),
                                                                       batch_size,
                                                                       axis=0),
                                           inputs=inputs,
                                           outputs=outputs)

    gen_val = data_generator(working,
                             idx_val['id'].values, sampler, len(idx_val),
                             augment=True,
                             rate=rate,
                             mode='with_replacement',
                             random_state=seed)

    gen_val = pescador.maps.keras_tuples(gen_val(),
                                         inputs=inputs,
                                         outputs=outputs)

    loss = {'chord_tag': 'sparse_categorical_crossentropy'}
    metrics = {'chord_tag': 'sparse_categorical_accuracy'}

    loss.update(chord_pitch='binary_crossentropy',
                chord_root='sparse_categorical_crossentropy',
                chord_bass='sparse_categorical_crossentropy')
    monitor = 'val_chord_tag_loss'

    model.compile(K.optimizers.Adam(), loss=loss, metrics=metrics)

    # Store the model
    model_spec = K.utils.serialize_keras_object(model)
    with open(os.path.join(OUTPUT_PATH, 'model_spec.pkl'), 'wb') as fd:
        pickle.dump(model_spec, fd)

    # Construct the weight path
    weight_path = os.path.join(OUTPUT_PATH, 'model.h5')

    # Build the callbacks
    cb = []
    cb.append(K.callbacks.ModelCheckpoint(weight_path,
                                          save_best_only=True,
                                          verbose=1,
                                          monitor=monitor))

    cb.append(K.callbacks.ReduceLROnPlateau(patience=reduce_lr,
                                            verbose=1,
                                            monitor=monitor))

    cb.append(K.callbacks.EarlyStopping(patience=early_stopping,
                                        verbose=1,
                                        monitor=monitor))

    # Fit the model
    model.fit_generator(gen_train, epoch_size, epochs,
                        validation_data=gen_val,
                        validation_steps=validation_size,
                        callbacks=cb)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    smkdirs(OUTPUT_PATH)

    version = crema.utils.increment_version(os.path.join(OUTPUT_PATH,
                                                         'version.txt'))

    print('{}: training'.format(__doc__))
    print('Model version: {}'.format(version))
    print(params)

    train(params.working,
          params.max_samples, params.duration,
          params.rate,
          params.batch_size,
          params.epochs, params.epoch_size,
          params.validation_size,
          params.early_stopping,
          params.reduce_lr,
          params.seed)
