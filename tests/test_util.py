#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Test helper utilities'''

import pytest
import crema.utils


def test_git_version():

    ver = crema.utils.git_version()

    assert ver == 'Unknown' or len(ver) == 7
