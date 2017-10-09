# crema
convolutional and recurrent estimators for music analysis

[![Build Status](https://travis-ci.org/bmcfee/crema.svg?branch=master)](https://travis-ci.org/bmcfee/crema)
[![Coverage Status](https://coveralls.io/repos/github/bmcfee/crema/badge.svg?branch=master)](https://coveralls.io/github/bmcfee/crema?branch=master)
[![Documentation Status](https://readthedocs.org/projects/crema/badge/?version=latest)](http://crema.readthedocs.io/en/latest/?badge=latest)


Usage options
-------------

From the command-line, print to the screen in JAMS format:

```
python -m crema.analyze file.mp3
```

or save to a file:

```
python -m crema.analyze file.mp3 -o file.jams
```


From within python:

```python
import crema.analyze

jam = crema.analyze(filename='/path/to/file.mp3')
```

or if you have an audio buffer in memory, librosa-style:

```python
jam = crema.analyze(y=y, sr=sr)
```
