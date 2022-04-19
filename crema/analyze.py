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

    Examples
    --------
    >>> from crema.analyze import analyze
    >>> import librosa
    >>> jam = analyze(filename=librosa.ex('brahms'))
    >>> jam
    <JAMS(file_metadata=<FileMetadata(...)>,
          annotations=[1 annotation],
          sandbox=<Sandbox(...)>)>
    >>> # Get the chord estimates
    >>> chords = jam.annotations['chord', 0]
    >>> chords.to_dataframe().head(5)
            time  duration    value  confidence
    0   0.000000  3.622313    G:min    0.767835
    1   3.622313  1.207438  C:min/5    0.652179
    2   4.829751  1.207438    D:maj    0.277913
    3   6.037188  4.086712    G:min    0.878656
    4  10.123900  1.486077   D#:maj    0.608746
    '''

    _load_models()

    jam = jams.JAMS()
    # populate file metadata

    jam.file_metadata.duration = librosa.get_duration(y=y, sr=sr,
                                                      filename=filename)

    for model in __MODELS__:
        jam.annotations.append(model.predict(filename=filename, y=y, sr=sr))

    return jam


def parse_args(args):  # pragma: no cover

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('filename',
                        type=str,
                        help='Audio file to process')

    parser.add_argument('-o', '--output', dest='output',
                        type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='Path to store JAMS output')

    return parser.parse_args(args)


def main():  # pragma: no cover
    params = parse_args(sys.argv[1:])
    jam = analyze(params.filename)
    jam.save(params.output)


# Populate models array
def _load_models():
    '''This helper builds and caches the model objects.'''
    global __MODELS__

    if not __MODELS__:
        __MODELS__.append(models.chord.ChordModel())


if __name__ == '__main__':  # pragma: no cover
    main()
