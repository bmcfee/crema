#!/usr/bin/env python
'''CREMA Chord model'''

import numpy as np
from scipy.stats import gmean
from librosa import time_to_frames
import mir_eval

from .base import CremaModel


SEMITONE_TO_SCALE_DEGREE = ['1', 'b2', '2', 'b3', '3',
                            '4', 'b5', '5', 'b6', '6', 'b7', '7']


class ChordModel(CremaModel):

    def __init__(self):
        self._instantiate('chord')

    def predict(self, filename=None, y=None, sr=None, outputs=None):

        if outputs is None:
            outputs = self.outputs(filename=filename, y=y, sr=sr)

        output_key = self.model.output_names[0]
        pump_op = self.pump[output_key]

        ann = super(ChordModel, self).predict(y=y, sr=sr, filename=filename,
                                              outputs=outputs)

        bass_pred = outputs['chord_bass']

        # Handle inversion estimation
        for obs in ann.pop_data():
            value = obs.value

            if obs.value not in ('N', 'X'):
                start, end = time_to_frames([obs.time, obs.time + obs.duration],
                                            sr=pump_op.sr,
                                            hop_length=pump_op.hop_length)

                mean_bass = gmean(bass_pred[start:end+1])

                bass_pc = np.argmax(mean_bass)
                root_pc, pitches, _ = mir_eval.chord.encode(obs.value)

                bass_rel = 0
                if bass_pc < 12:
                    bass_rel = np.mod(bass_pc - root_pc, 12)

                if bass_rel and pitches[bass_rel]:
                    value = '{}/{}'.format(value,
                                           SEMITONE_TO_SCALE_DEGREE[bass_rel])

            ann.append(time=obs.time,
                       duration=obs.duration,
                       value=value,
                       confidence=obs.confidence)

        return ann
