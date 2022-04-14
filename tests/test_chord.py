#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for chord model'''

import pytest
import numpy as np
import librosa

import crema.models.chord


@pytest.fixture
def SIGNAL():
    y, sr = librosa.load(librosa.ex('brahms'),
                         sr=None)
    return y, sr


@pytest.fixture
def AUDIOFILE():
    return librosa.ex('brahms')


@pytest.fixture
def CHORD_MODEL():
    return crema.models.chord.ChordModel()


def _validate_prediction(chord_model, ann):

    assert ann.namespace == 'chord'

    ann.validate()

    # Validate the version string matches the chord model
    assert 'CREMA' in ann.annotation_metadata.annotation_tools
    assert ann.annotation_metadata.version == chord_model.version
    assert ann.annotation_metadata.data_source == 'program'


def test_chord_predict_signal(CHORD_MODEL, SIGNAL):

    y, sr = SIGNAL
    ann = CHORD_MODEL.predict(y=y, sr=sr)

    _validate_prediction(CHORD_MODEL, ann)


def test_chord_outputs_signal(CHORD_MODEL, SIGNAL):

    y, sr = SIGNAL
    data = CHORD_MODEL.outputs(y=y, sr=sr)
    _validate_outputs(data)


def _validate_outputs(data):
    # Check the actual desired outputs here
    # chord_pitch, chord_bass, chord_root, chord_tag
    assert set(data.keys()) == set(['chord_pitch', 'chord_bass', 'chord_root', 'chord_tag'])
    assert data['chord_pitch'].shape[1] == 12
    assert np.all(data['chord_pitch'] >= 0) and np.all(data['chord_pitch'] <= 1)
    assert data['chord_root'].shape[1] == 13
    assert np.all(data['chord_root'] >= 0) and np.allclose(np.sum(data['chord_root'], axis=1), 1)
    assert data['chord_bass'].shape[1] == 13
    assert np.all(data['chord_bass'] >= 0) and np.allclose(np.sum(data['chord_bass'], axis=1), 1)
    assert data['chord_tag'].shape[1] == 170
    assert np.all(data['chord_tag'] >= 0) and np.allclose(np.sum(data['chord_tag'], axis=1), 1)
    assert data['chord_pitch'].shape[0] == data['chord_root'].shape[0]
    assert data['chord_pitch'].shape[0] == data['chord_bass'].shape[0]
    assert data['chord_pitch'].shape[0] == data['chord_tag'].shape[0]


def test_chord_predict_file(CHORD_MODEL, AUDIOFILE):

    ann = CHORD_MODEL.predict(filename=AUDIOFILE)
    _validate_prediction(CHORD_MODEL, ann)


@pytest.mark.xfail(raises=NotImplementedError)
def test_chord_transform(CHORD_MODEL, SIGNAL):
    y, sr = SIGNAL
    CHORD_MODEL.transform(y=y, sr=sr)
