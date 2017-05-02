#!/usr/bin/env python
'''Base CremaModel class definition'''

import pickle
import os
from pkg_resources import resource_filename

import keras
import librosa

from ..version import version as version


class CremaModel(object):

    def predict(self, filename=None, y=None, sr=None):
        '''Predict chord labels

        Parameters
        ----------
        filename : str (optional)
            Path to audio file

        y, sr : (optional)
            Audio buffer and sample rate

        .. note:: At least one of `filename` or `y, sr` must be provided.

        Returns
        -------
        jams.Annotation, namespace='chord'
            The predicted chord annotation
        '''

        # Pump the input features
        data = self.pump.transform(audio_f=filename, y=y, sr=sr)

        # Line up input variables with the data
        pred = self.model.predict([data[_] for _ in self.model.input_names])

        # Invert the prediction.  This is always the first output layer.
        ann = self.pump[self.model.output_names[0]].inverse(pred[0][0])

        # Populate the metadata
        ann.annotation_metadata.version = self.version
        ann.annotation_metadata.annotation_tools = 'CREMA {}'.format(version)
        ann.annotation_metadata.data_source = 'program'
        ann.duration = librosa.get_duration(y=y, sr=sr, filename=filename)

        return ann

    def predict_proba(self, filename=None, y=None, sr=None):
        raise NotImplementedError

    def transform(self, filename=None, y=None, sr=None):
        raise NotImplementedError

    def _instantiate(self, rsc):

        # First, load the pump
        with open(resource_filename(__name__,
                                    os.path.join(rsc, 'pump.pkl')),
                  'rb') as fd:
            self.pump = pickle.load(fd)

        # Now load the model
        with open(resource_filename(__name__,
                                    os.path.join(rsc, 'model_spec.pkl')),
                  'rb') as fd:
            spec = pickle.load(fd)
            self.model = keras.models.model_from_config(spec)

        # And the model weights
        self.model.load_weights(resource_filename(__name__,
                                                  os.path.join(rsc,
                                                               'model.h5')))

        # And the version number
        with open(resource_filename(__name__,
                                    os.path.join(rsc, 'version.txt')),
                  'r') as fd:
            self.version = fd.read().strip()
