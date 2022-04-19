#!/usr/bin/env python
'''CREMA beats and downbeats'''

import argparse
import sys
import os
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed

from jams.util import smkdirs
import muda

import crema.utils

OUTPUT_PATH = 'resources'


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--stretch', dest='stretch', type=float, default=0.333,
                        help='Relative time stretch range')

    parser.add_argument('--n-stretch', dest='n_stretch', type=int, default=4,
                        help='Number of stretched examples')

    parser.add_argument('--audio-ext', dest='audio_ext', type=str,
                        default='ogg',
                        help='Output format for audio (ogg, flac, etc)')

    parser.add_argument('--jams-ext', dest='jams_ext', type=str,
                        default='jams',
                        help='Output format for annotations (jams, jamz)')

    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of jobs to run in parallel')

    parser.add_argument('input_path', type=str,
                        help='Path for input (audio, jams) pairs')

    parser.add_argument('output_path', type=str,
                        help='Path to store augmented data')

    return parser.parse_args(args)


def augment(afile, jfile, deformer, outpath, audio_ext, jams_ext):
    '''Run the data through muda'''
    jam = muda.load_jam_audio(jfile, afile)

    base = crema.utils.base(afile)

    outfile = os.path.join(outpath, base)
    for i, jam_out in enumerate(deformer.transform(jam)):
        muda.save('{}.{}.{}'.format(outfile, i, audio_ext),
                  '{}.{}.{}'.format(outfile, i, jams_ext),
                  jam_out, strict=False)


def make_muda(stretch, n_stretch):
    '''Construct a MUDA time stretcher'''

    shifter = muda.deformers.LogspaceTimeStretch(n_samples=n_stretch, lower=-stretch, upper=stretch)

    smkdirs(OUTPUT_PATH)
    with open(os.path.join(OUTPUT_PATH, 'muda.pkl'), 'wb') as fd:
        pickle.dump(shifter, fd)

    return shifter


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    smkdirs(OUTPUT_PATH)
    smkdirs(params.output_path)

    print('{}: setup'.format(__doc__))
    print(params)

    # Build the deformer
    deformer = make_muda(params.stretch, params.n_stretch)

    # Get the file list
    stream = tqdm(crema.utils.get_ann_audio(params.input_path),
                  desc='Augmenting training data')

    # Launch the job
    Parallel(n_jobs=params.n_jobs)(delayed(augment)(aud, ann, deformer,
                                                    params.output_path,
                                                    params.audio_ext,
                                                    params.jams_ext)
                                   for aud, ann in stream)
