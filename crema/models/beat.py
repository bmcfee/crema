#!/usr/bin/env python
'''CREMA Beat model'''

from .base import CremaModel


class BeatModel(CremaModel):

    def __init__(self):
        self._instantiate('beat')
