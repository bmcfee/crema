.. _models:

Model reference
===============
This section describes the implementation of models provided by `crema`.

Chord recognition
-----------------

The chord recognition model is based on the structured prediction model of McFee and Bello [1]_.
The implementation here has been enhanced to support inversion (bass) tracking, and predicts chords out of an effective vocabulary of 602 classes.

.. [1] McFee, Brian, Juan Pablo Bello.
    "Structured training for large-vocabulary chord recognition."
    In ISMIR, 2017.

.. module:: crema.models.chord

.. autoclass:: ChordModel
    :inherited-members:

