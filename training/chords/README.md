Structured chord estimation
===========================

This model is based on the paper by McFee and Bello, originally implemented in
[this repository](https://github.com/bmcfee/ismir2017_chords/), with a few extensions:

- Input uses the harmonic CQT representation
- Filters have been extended from 36 to 72 channels
- Output is post-processed by viterbi decoding
- Inversion tracking is implemented using the `chord/root` output of the structured model


Model architecture
------------------

The model accepts as input a harmonic constant-Q spectrogram:

- `HCQT(name='cqt', sr=44100, hop_length=4096, n_octaves=6,
        harmonics=[1, 2, 3], over_sample=3, log=True, conv='channels_last')`

The model has 9 layers (not counting reshaping/merging) defined as follows:

- `Input`
- `BatchNorm`
- `Conv2D(3, (5, 5), activation='relu', padding='same')`
- `Conv2D(72, (1, 216), activation='relu', padding='valid')`
- `Squeeze`
- `Bidirectional(GRU(128))`
- `Bidirectional(GRU(128))`  <-- the encoding layer
- `Root, Pitches, Bass`      <-- the structured output
- `Chord`                    <-- the decoded chord label output

The output chord vocabulary is pumpp's `3567s` set (170).

Training
--------
We use MUDA for data augmentation: +- 6 semitones for each training track.

Batches are managed by pescador with the following parameters:

- 8-second patches
- 128 samples per active streamer
- rate=8
- 1024 active streamers
- 16 patches per training batch
- 1024 batches per epoch
- 1024 batches per validation
- learning rate reduction at 10
- early stopping at 20
- max 100 epochs
