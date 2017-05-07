#!/usr/bin/env python
'''CREMA analyzer interface'''

import argparse
import sys

import librosa
import jams

from . import models

__MODELS__ = []

__all__ = ['analyze', 'main']


def analyze(filename=None, y=None, sr=None):
    '''Analyze a recording for all tasks.

    Parameters
    ----------
    filename : str, optional
        Path to audio file

    y : np.ndarray, optional
    sr : number > 0, optional
        Audio buffer and sampling rate

    .. note:: At least one of `filename` or `y, sr` must be provided.

    Returns
    -------
    jam : jams.JAMS
        a JAMS object containing all estimated annotations
    '''

    jam = jams.JAMS()
    # populate file metadata

    jam.file_metadata.duration = librosa.get_duration(y=y, sr=sr,
                                                      filename=filename)

    for model in __MODELS__:
        jam.annotations.append(model.predict(filename=filename, y=y, sr=sr))

    return jam


def parse_args(args):

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('filename',
                        type=str,
                        help='Audio file to process')

    parser.add_argument('-o', '--output', dest='output',
                        type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='Path to store JAMS output')

    return parser.parse_args(args)


def main():
    params = parse_args(sys.argv[1:])
    jam = analyze(params.filename)
    jam.save(params.output)


# Populate models array
__MODELS__.append(models.chord.ChordModel())
__MODELS__.append(models.beat.BeatModel())

if __name__ == '__main__':
    main()
