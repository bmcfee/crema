# crema
convolutional and recurrent estimators for music analysis

[![GitHub license](https://img.shields.io/badge/license-BSD-blue.svg)](https://raw.githubusercontent.com/bmcfee/crema/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/crema/badge/?version=latest)](http://crema.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1010486.svg)](https://doi.org/10.5281/zenodo.1010486)


Usage options
-------------

From the command-line, print to the screen in [JAMS](https://github.com/marl/jams) format:

```
python -m crema.analyze file.mp3
```

or save to a file:

```
python -m crema.analyze file.mp3 -o file.jams
```


From within python:

```python
from crema.analyze import analyze

jam = analyze(filename='/path/to/file.mp3')
```

or if you have an audio buffer in memory, librosa-style:

```python
jam = analyze(y=y, sr=sr)
```
