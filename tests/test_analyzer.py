#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Analyzer'''

import pytest

import librosa
import crema.analyze


@pytest.fixture
def SIGNAL():
    y, sr = librosa.load(librosa.ex('trumpet'),
                         sr=None)
    return y, sr


@pytest.fixture
def AUDIOFILE():
    return librosa.ex('trumpet')



def _validate_analysis(jam):

    assert jam.annotations['chord']
    jam.validate()


def test_analyze_signal(SIGNAL):
    y, sr = SIGNAL
    jam = crema.analyze.analyze(y=y, sr=sr)
    _validate_analysis(jam)


def test_analyze_filename(AUDIOFILE):
    jam = crema.analyze.analyze(filename=AUDIOFILE)
    _validate_analysis(jam)
