#!/usr/bin/env python
'''CREMA utilities'''

import os
import subprocess
import h5py
from librosa.util import find_files

from .exceptions import DataError


def git_version():
    '''Return the git revision as a string

    Returns
    -------
    git_version : str
        The current git revision
    '''
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'

        output = subprocess.check_output(cmd,
                                         stderr=subprocess.DEVNULL,
                                         env=env)
        return output

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', '--verify', '--quiet', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except subprocess.CalledProcessError:
        GIT_REVISION = 'UNKNOWN'

    return GIT_REVISION


def increment_version(filename):
    '''Increment a model version identifier.

    Parameters
    ----------
    filename : str
        The file containing the model version

    Returns
    -------
    model_version : str
        The new model version.
        This version will also be written out to `filename`.
    '''

    gv = git_version()
    iteration = 0

    try:
        with open(filename, 'r') as fd:
            line = fd.read()
            old_gv, old_iteration = line.split('.', 2)
            old_iteration = int(old_iteration)

            if old_gv == gv:
                iteration = old_iteration + 1
    except FileNotFoundError:
        pass

    version = '{}.{}'.format(gv, iteration)

    with open(filename, 'w') as fd:
        fd.write(version)

    return version


def save_h5(filename, **kwargs):
    '''Save data to an hdf5 file.

    Parameters
    ----------
    filename : str
        Path to the file

    kwargs
        key-value pairs of data

    See Also
    --------
    load_h5
    '''
    with h5py.File(filename, 'w') as hf:
        hf.update(kwargs)


def load_h5(filename):
    '''Load data from an hdf5 file created by `save_h5`.

    Parameters
    ----------
    filename : str
        Path to the hdf5 file

    Returns
    -------
    data : dict
        The key-value data stored in `filename`

    See Also
    --------
    save_h5
    '''
    data = {}

    def collect(k, v):
        if isinstance(v, h5py.Dataset):
            data[k] = v[...]

    with h5py.File(filename, mode='r') as hf:
        hf.visititems(collect)

    return data


def base(filename):
    '''Identify a file by its basename:

    /path/to/base.name.ext => base.name

    Parameters
    ----------
    filename : str
        Path to the file

    Returns
    -------
    base : str
        The base name of the file
    '''
    return os.path.splitext(os.path.basename(filename))[0]


def get_ann_audio(directory):
    '''Get a list of annotations and audio files from a directory.

    This also validates that the lengths match and are paired properly.

    Parameters
    ----------
    directory : str
        The directory to search

    Returns
    -------
    pairs : list of tuples (audio_file, annotation_file)
    '''

    audio = find_files(directory)
    annos = find_files(directory, ext=['jams', 'jamz'])

    paired = list(zip(audio, annos))

    if len(audio) != len(annos) or any([base(aud) != base(ann) for aud, ann in paired]):
        raise DataError('Unmatched audio/annotation data in {}'.format(directory))

    return paired
