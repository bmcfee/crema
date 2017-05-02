#!/usr/bin/env python
'''CREMA Chord model'''

from .base import CremaModel


class ChordModel(CremaModel):

    def __init__(self):
        self._instantiate('chord')
