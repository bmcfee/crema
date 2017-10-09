#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Test helper utilities'''

import tempfile
import os
import numpy as np

import pytest

import crema.utils
from crema.exceptions import DataError


def test_git_version():

    ver = crema.utils.git_version()

    assert ver == 'Unknown' or len(ver) == 7


@pytest.fixture
def version_file():

    _, ver_file = tempfile.mkstemp()
    os.unlink(ver_file)

    yield ver_file

    os.unlink(ver_file)


def test_increment_version(version_file):

    # Increment / create the version file
    v1 = crema.utils.increment_version(version_file)

    with open(version_file, 'r') as fd:
        saved_ver = fd.read()

    assert v1 == saved_ver
    assert v1[-1] == '0'

    v2 = crema.utils.increment_version(version_file)

    with open(version_file, 'r') as fd:
        saved_ver2 = fd.read()

    assert saved_ver2 == v2
    assert v2[-1] == '1'
    assert v1[:v1.rindex('.')] == v2[:v2.rindex('.')]


@pytest.fixture
def h5_file():

    _, h5f = tempfile.mkstemp(suffix='h5')
    os.unlink(h5f)
    yield h5f
    os.unlink(h5f)


def test_save_load_h5(h5_file):

    datum = dict(x=np.random.randn(5, 6, 7),
                 y=np.random.randn(128))

    crema.utils.save_h5(h5_file, **datum)

    datum2 = crema.utils.load_h5(h5_file)

    assert datum.keys() == datum2.keys()

    for k in datum:
        assert np.allclose(datum[k], datum2[k])


def test_base():

    base = crema.utils.base('/path/to/my.file.ext')
    assert base == 'my.file'


@pytest.fixture
def paired_files():

    root = tempfile.mkdtemp()

    files = [[root, 'file1.mp3'], [root, 'file1.jams'],
             [root, 'file2.ogg'], [root, 'file2.jams'],
             [root, 'file3.wav'], [root, 'file3.jams'],
             [root, 'file4.flac'], [root, 'file4.jams']]

    files = [os.sep.join(_) for _ in files]

    for fname in files:
        with open(fname, 'w'):
            pass

    yield root, files

    for fname in files:
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

    os.rmdir(root)


def test_get_ann_audio(paired_files):

    root, files = paired_files

    results = crema.utils.get_ann_audio(root)

    results_expected = list(zip(files[::2], files[1::2]))

    assert results == results_expected


@pytest.mark.xfail(raises=DataError)
def test_get_ann_audio_fail(paired_files):

    root, files = paired_files

    os.remove(files[-1])
    crema.utils.get_ann_audio(root)
