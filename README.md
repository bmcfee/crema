# crema
convolutional and recurrent estimators for music analysis


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
