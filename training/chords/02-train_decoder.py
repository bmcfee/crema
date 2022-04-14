#!/usr/bin/env python
'''CREMA structured chord model: HMM parameters'''

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

import crema.utils
from jams.util import smkdirs

OUTPUT_PATH = 'resources'


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(dest='working', type=str,
                        help='Path to working directory')

    parser.add_argument('--pseudo-count', dest='pseudocount',
                        type=float, default=0.5,
                        help='Pseudo-count for self-transitions prior')

    return parser.parse_args(args)


def self_transitions(fname):
    data = crema.utils.load_h5(fname)

    n_total, n_self = 0, 0

    # we might have multiple annotations per file
    for tags in data['chord_tag/chord']:
        n_total += len(tags) - 1
        n_self += np.sum(tags[1:] == tags[:-1])

    return n_self, n_total


def train(working, pseudocount):
    '''
    Parameters
    ----------
    working : str
        directory that contains the experiment data (h5)
    '''

    # Load the pump
    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb') as fd:
        pump = pickle.load(fd)

    # Load the training data
    idx_train = pd.read_json('index_train.json')

    n_self, n_total = 0., 0.

    for track in tqdm(idx_train['id']):
        fname = os.path.join(working, os.path.extsep.join([track, 'h5']))
        i_self, i_total = self_transitions(fname)

        n_self += i_self
        n_total += i_total

    p_self = (n_self + pseudocount) / (n_total + pseudocount)
    print('# frames = {} + {}'.format(n_total, pseudocount))
    print('# self = {} + {}'.format(n_self, pseudocount))
    print('P = {:.3g}'.format(p_self))

    pump['chord_tag'].set_transition(p_self)

    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'wb') as fd:
        pickle.dump(pump, fd)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    smkdirs(OUTPUT_PATH)

    print('{}: training HMM parameters'.format(__doc__))
    print(params)

    train(params.working, params.pseudocount)
