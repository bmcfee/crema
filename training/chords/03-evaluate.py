#!/usr/bin/env python
'''CREMA structured chord model'''

import argparse
import sys
import os

import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed

import jams

import crema
import crema.utils

OUTPUT_PATH = 'resources'


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of parallel jobs')

    parser.add_argument('input_path', type=str,
                        help='Path for directory containing (audio, jams)')

    return parser.parse_args(args)


def track_eval(ann, aud):

    model = crema.models.chord.ChordModel()

    # Load the audio and make a prediction
    track = crema.utils.base(ann)

    jam_ref = jams.load(ann, validate=False)

    est = model.predict(filename=aud)
    try:
        scores = jams.eval.chord(jam_ref.annotations['chord', 0],
                                 est)
    except ValueError as exc:
        print(ann)
        print(est.to_dataframe())
        print(jam_ref.annotations['chord', 0].to_dataframe())
        raise exc

    return (track, dict(scores))


def evaluate(input_path, n_jobs):

    aud, ann = zip(*crema.utils.get_ann_audio(input_path))

    test_idx = set(pd.read_json('index_test.json')['id'])

    # drop anything not in the test set
    ann = [ann_i for ann_i in ann if crema.utils.base(ann_i) in test_idx]
    aud = [aud_i for aud_i in aud if crema.utils.base(aud_i) in test_idx]

    stream = tqdm(zip(ann, aud), desc='Evaluating test set', total=len(ann))

    results = Parallel(n_jobs=n_jobs)(delayed(track_eval)(ann_i, aud_i)
                                      for ann_i, aud_i in stream)
    df = pd.DataFrame.from_dict(dict(results), orient='index')

    print('Results')
    print('-------')
    print(df.describe().T.sort_index())

    df.to_json(os.path.join(OUTPUT_PATH, 'test_scores.json'))


if __name__ == '__main__':

    params = process_arguments(sys.argv[1:])
    print('{}: evaluation'.format(__doc__))
    print(params)

    evaluate(params.input_path, params.n_jobs)
