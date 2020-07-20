#!/usr/bin/env python
'''Base CremaModel class definition'''

import pickle
import os
from pkg_resources import resource_filename

from keras.models import model_from_config
import librosa

from ..version import version as version
from .. import layers


CORE_CUSTOM_OBJECTS = {k: layers.__dict__[k] for k in layers.__all__}


class CremaModel(object):
    name = None
    models_dir = None
    custom_objects = {}

    def __init__(self, name=None, models_dir=None):
        self.name = name or self.name
        self.models_dir = models_dir or self.models_dir
        if self.name:
            self._instantiate(self.name)

    def predict(self, filename=None, y=None, sr=None, outputs=None):
        '''Predict annotations

        Parameters
        ----------
        filename : str (optional)
            Path to audio file

        y, sr : (optional)
            Audio buffer and sample rate

        outputs : (optional)
            Pre-computed model outputs as produced by `CremaModel.outputs`.
            If provided, then predictions are derived from these instead of
            `filename` or `(y, sr)`.


        .. note:: At least one of `filename`, `y, sr` must be provided.

        Returns
        -------
        jams.Annotation
            The predicted annotation
        '''

        # Pump the input features
        output_key = self.model.output_names[0]

        if outputs is None:
            outputs = self.outputs(filename=filename, y=y, sr=sr)

        # Invert the prediction.  This is always the first output layer.
        ann = self.pump[output_key].inverse(outputs[output_key])

        # Populate the metadata
        ann.annotation_metadata.version = self.version
        ann.annotation_metadata.annotation_tools = 'CREMA {}'.format(version)
        ann.annotation_metadata.data_source = 'program'
        ann.duration = librosa.get_duration(y=y, sr=sr, filename=filename)

        return ann

    def outputs(self, filename=None, y=None, sr=None):
        '''Return the model outputs (e.g., class likelihoods)

        Parameters
        ----------
        filename : str (optional)
            Path to audio file

        y, sr : (optional)
            Audio buffer and sample rate

        .. note:: At least one of `filename` or `y, sr` must be provided.

        Returns
        -------
        outputs : dict, {str: np.ndarray}
            Each key corresponds to an output name,
            and the value is the model's output for the given input
        '''

        # Pump the input features
        data = self.pump.transform(audio_f=filename, y=y, sr=sr)

        # Line up input variables with the data
        pred = self.model.predict([data[_] for _ in self.model.input_names])

        # Invert the prediction.  This is always the first output layer.
        return {k: pred[i][0] for i, k in enumerate(self.model.output_names)}

    def transform(self, filename=None, y=None, sr=None):
        '''Feature transformation'''
        raise NotImplementedError

    def resource_file(self, f='', *fname):
        return (
            os.path.join(self.models_dir, f, *fname)
            if self.models_dir is not None else
            resource_filename(__name__, os.path.join(f, *fname)))

    def _instantiate(self, rsc):

        # First, load the pump
        with open(self.resource_file(rsc, 'pump.pkl'), 'rb') as fd:
            self.pump = pickle.load(fd)

        # Now load the model
        custom_objects = dict(CORE_CUSTOM_OBJECTS, **self.custom_objects)
        with open(self.resource_file(rsc, 'model_spec.pkl'), 'rb') as fd:
            spec = pickle.load(fd)
            self.model = model_from_config(spec, custom_objects=custom_objects)

        # And the model weights
        self.model.load_weights(self.resource_file(rsc, 'model.h5'))

        # And the version number
        with open(self.resource_file(rsc, 'version.txt'), 'r') as fd:
            self.version = fd.read().strip()
