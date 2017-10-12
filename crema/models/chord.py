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
        '''Chord prediction

        Parameters
        ----------
        filename : str
            Path to the audio file to analyze

        y, sr : np.ndarray, number>0

            Audio signal in memory to analyze

        outputs : dict `{str: np.ndarray}`

            Pre-computed model outputs, as given by ``ChordModel.outputs``.

        .. note:: At least one of `filename`, `y, sr`, or `outputs`
            must be provided.

        Returns
        -------
        jams.Annotation, namespace='chord'

            The chord estimate for the given signal.

        Examples
        --------
        >>> import crema
        >>> import librosa
        >>> model = crema.models.chord.ChordModel()
        >>> chord_est = model.predict(filename=librosa.util.example_audio_file())
        >>> chord_est
        <Annotation(namespace='chord',
                    time=0,
                    duration=61.4,
                    annotation_metadata=<AnnotationMetadata(...)>,
                    data=<45 observations>,
                    sandbox=<Sandbox(...)>)>
        >>> chord_est.to_dataframe().head(5)
               time  duration  value  confidence
        0  0.000000  0.092880  E:maj    0.336977
        1  0.092880  0.464399    E:7    0.324255
        2  0.557279  1.021678  E:min    0.448759
        3  1.578957  2.693515  E:maj    0.501462
        4  4.272472  1.486077  E:min    0.287264
        '''
        if outputs is None:
            outputs = self.outputs(filename=filename, y=y, sr=sr)

        output_key = self.model.output_names[0]
        pump_op = self.pump[output_key]

        ann = super(ChordModel, self).predict(y=y, sr=sr, filename=filename,
                                              outputs=outputs)

        bass_pred = outputs['chord_bass']

        # Handle inversion estimation
        for obs in ann.pop_data():
            start, end = time_to_frames([obs.time, obs.time + obs.duration],
                                        sr=pump_op.sr,
                                        hop_length=pump_op.hop_length)

            value = obs.value
            if obs.value not in ('N', 'X'):
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
