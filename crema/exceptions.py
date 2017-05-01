#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Exception classes for crema'''


class CremaError(Exception):
    '''The root crema exception class'''
    pass


class DataError(CremaError):
    '''Exceptions for when data are mal-formatted'''
    pass
