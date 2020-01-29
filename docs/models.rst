.. _models:

Model reference
===============
This section describes the implementation of models provided by `crema`.

Chord recognition
-----------------

The chord recognition model is based on the structured prediction model of McFee and Bello [1]_.
The implementation here has been enhanced to support inversion (bass) tracking, and predicts chords out of an effective vocabulary of 602 classes. Chord class names are based on an extended version of Harte's [2]_ grammar: `N` corresponds to "no-chord" and `X` corresponds to out-of-gamut chords (usually power chords).

.. [1] McFee, Brian, Juan Pablo Bello.
    "Structured training for large-vocabulary chord recognition."
    In ISMIR, 2017.
    
.. [2] Harte, Christopher, Mark B. Sandler, Samer A. Abdallah, and Emilia Gómez.
    "Symbolic Representation of Musical Chords: A Proposed Syntax for Text Annotations."
    In ISMIR, vol. 5, pp. 66-71. 2005.

.. module:: crema.models.chord

.. autoclass:: ChordModel
    :inherited-members:
    :members:

