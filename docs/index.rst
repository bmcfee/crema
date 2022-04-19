.. crema documentation master file, created by
   sphinx-quickstart on Mon Oct  9 12:09:02 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _crema:
.. toctree::
   :maxdepth: 3
   :caption: Contents:


Convolutional and Recurrent Estimators for Music Analysis
=========================================================

The `crema` package provides pre-trained statistical models for a variety of musical analysis tasks.
All tasks are provided under a unified interface, which can be accessed by the :ref:`analyze`
functionality.

Currently, only chord recognition is supported, but more features will be introduced over
time.

The `crema` analyzer can operate on either audio files stored on disk, or audio buffers
stored in `numpy` arrays.
The results of the analyzer are stored in a `JAMS <https://jams.readthedocs.org/>`_ object.


Installation
------------

`crema` can be installed directly from GitHub by issuing the following command:

.. code:: shell

    pip install -e git+https://github.com/bmcfee/crema.git

or from PyPI by:

.. code:: shell

    pip install crema

Quick start
-----------

The simplest way to apply `crema` is via the command line:

.. code:: shell

    python -m crema.analyze -o my_song.jams /path/to/my_song.ogg

.. note:: 
    Any audio format supported by `librosa <https://librosa.org/>`_ will work here.

This command will apply all analyzers to `mysong.ogg` and store the outputs as
`my_song.jams`.

For processing multiple recordings, the above will be inefficient because it will have to
instantiate the models repeatedly each time.
If you need to process a batch of recordings, it is better to do so directly in Python:

.. code:: python

    from crema.analyze import analyze

    jam1 = analyze(filename='song1.ogg')
    jam1.save('song1.jams')

    jam2 = analyze(filename='song2.ogg')
    jam2.save('song2.jams')

    ...


API Reference
-------------

.. toctree::
    :maxdepth: 2

    analyze
    models
    utils


Release notes
-------------

.. toctree::
    :maxdepth: 1

    changes

Contribute
----------
- `Issue Tracker <http://github.com/bmcfee/crema/issues>`_
- `Source Code <http://github.com/bmcfee/crema>`_

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
